import numpy as np
import matplotlib as mpl
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, iirdesign, zpk2tf, freqz
import h5py
import json
import matplotlib.pyplot as plt
from matplotlib import mlab
import os

def read_hdf5(filename, readstrain=True):
    dataFile = h5py.File(filename, 'r')
    if readstrain:
        strain = dataFile['strain']['Strain'][...]
    else:
        strain = 0
    ts = dataFile['strain']['Strain'].attrs['Xspacing']
    dqInfo = dataFile['quality']['simple']
    qmask = dqInfo['DQmask'][...]
    shortnameArray = dqInfo['DQShortnames'][()]
    shortnameList  = list(shortnameArray)
    injInfo = dataFile['quality/injections']
    injmask = injInfo['Injmask'][...]
    injnameArray = injInfo['InjShortnames'][()]
    injnameList  = list(injnameArray)
    meta = dataFile['meta']
    gpsStart = meta['GPSstart'][()]    
    dataFile.close()
    return strain, gpsStart, ts, qmask, shortnameList, injmask, injnameList

def loaddata(filename, ifo=None, tvec=True, readstrain=True):
    strain, gpsStart, ts, qmask, shortnameList, injmask, injnameList = read_hdf5(filename, readstrain)
    gpsEnd = gpsStart + len(qmask)
    if tvec:
        time = np.arange(gpsStart, gpsEnd, ts)
    else:
        meta = {}
        meta['start'] = gpsStart
        meta['stop']  = gpsEnd
        meta['dt']    = ts
    channel_dict = {}  #-- 1 Hz, mask
    for flag in shortnameList:
        bit = shortnameList.index(flag)
        if isinstance(flag, bytes): flag = flag.decode("utf-8")
        channel_dict[flag] = (qmask >> bit) & 1
    for flag in injnameList:
        bit = injnameList.index(flag)
        if isinstance(flag, bytes): flag = flag.decode("utf-8")
        channel_dict[flag] = (injmask >> bit) & 1
    try:
        channel_dict['DEFAULT'] = ( channel_dict['DATA'] )
    except:
        print("Warning: Failed to calculate DEFAULT data quality channel")
    if tvec:
        return strain, time, channel_dict
    else:
        return strain, meta, channel_dict

def whiten(strain, interp_psd, dt):
    Nt = len(strain)
    freqs = np.fft.rfftfreq(Nt, dt)
    hf = np.fft.rfft(strain)
    norm = 1./np.sqrt(1./(dt*2))
    white_hf = hf / np.sqrt(interp_psd(freqs)) * norm
    white_ht = np.fft.irfft(white_hf, n=Nt)
    return white_ht

eventname = 'GW170817'
tevent = 1187008882.43
fs = 4096

try:
    strain_H1, time_H1, chan_dict_H1 = loaddata('H-H1_LOSC_CLN_4_V1-1187007040-2048.hdf5', 'H1')
    strain_L1, time_L1, chan_dict_L1 = loaddata('L-L1_LOSC_CLN_4_V1-1187007040-2048.hdf5', 'L1')
except:
    print("Cannot find data files!")
    print("You can download them from https://losc.ligo.org/s/events/"+eventname)
    print("Quitting.")
    quit()

time = time_H1
dt = time[1] - time[0]
print('time_H1: len, min, mean, max = ', len(time_H1), time_H1.min(), time_H1.mean(), time_H1.max() )
print('strain_H1: len, min, mean, max = ', len(strain_H1), strain_H1.min(),strain_H1.mean(),strain_H1.max())
print('strain_L1: len, min, mean, max = ', len(strain_L1), strain_L1.min(),strain_L1.mean(),strain_L1.max())
bits = chan_dict_H1['DATA']
print("For H1, {0} out of {1} seconds contain usable DATA".format(bits.sum(), len(bits)))
bits = chan_dict_L1['DATA']
print("For L1, {0} out of {1} seconds contain usable DATA".format(bits.sum(), len(bits)))

deltat = 30
indxt = np.where((time >= tevent-deltat) & (time < tevent+deltat))
# print(tevent)
plt.figure()
plt.plot(time[indxt]-tevent,strain_H1[indxt],'r',label='H1 strain')
plt.plot(time[indxt]-tevent,strain_L1[indxt],'g',label='L1 strain')
plt.axis([-1.5, 1.5, -1e-17, 1e-17])
plt.xlabel('time (s) since '+str(tevent))
plt.ylabel('strain')
plt.legend(loc='lower right')
plt.title('Advanced LIGO strain data near '+eventname)

NFFT = 4*fs
Pxx_H1, freqs = mlab.psd(strain_H1, Fs = fs, NFFT = NFFT)
Pxx_L1, freqs = mlab.psd(strain_L1, Fs = fs, NFFT = NFFT)
psd_H1 = interp1d(freqs, Pxx_H1)
psd_L1 = interp1d(freqs, Pxx_L1)
f_min = 20.
f_max = 2000.
plt.figure()
plt.loglog(freqs, np.sqrt(Pxx_L1),'g',label='L1 strain')
plt.loglog(freqs, np.sqrt(Pxx_H1),'r',label='H1 strain')
plt.axis([f_min, f_max, 1e-23, 1e-20])
plt.grid('on')
plt.ylabel('ASD (strain/rtHz)')
plt.xlabel('Freq (Hz)')
plt.legend(loc='upper center')
plt.title('Advanced LIGO strain data near '+eventname)

fband = [120.,300.]
strain_H1_whiten = whiten(strain_H1,psd_H1,dt)
strain_L1_whiten = whiten(strain_L1,psd_L1,dt)
bb, ab = butter(4, [fband[0]*2./fs, fband[1]*2./fs], btype='band')
normalization = np.sqrt((fband[1]-fband[0])/(fs/2))
strain_H1_whitenbp = filtfilt(bb, ab, strain_H1) / normalization
strain_L1_whitenbp = filtfilt(bb, ab, strain_L1) / normalization

# plt.figure()
# plt.plot(time[indxt]-tevent,strain_H1_whiten[indxt],'r',label='H1 strain')
# plt.plot(time[indxt]-tevent,strain_L1_whiten[indxt],'g',label='L1 strain')
# plt.axis([-0.01, 0.01, -3, 3])
# plt.xlabel('time (s) since '+str(tevent))
# plt.ylabel('strain')
# plt.legend(loc='lower right')
# plt.title('Advanced LIGO strain whitened data near '+eventname)

# NFFT = 4*fs
# Pxx_H1, freqs = mlab.psd(strain_H1_whiten, Fs = fs, NFFT = NFFT)
# Pxx_L1, freqs = mlab.psd(strain_L1_whiten, Fs = fs, NFFT = NFFT)
# psd_H1 = interp1d(freqs, Pxx_H1)
# psd_L1 = interp1d(freqs, Pxx_L1)
# f_min = 20.
# f_max = 2000.
# plt.figure()
# plt.loglog(freqs, np.sqrt(Pxx_L1),'g',label='L1 strain')
# plt.loglog(freqs, np.sqrt(Pxx_H1),'r',label='H1 strain')
# plt.axis([f_min, f_max, 2e-2, 4e-2])
# plt.grid('on')
# plt.ylabel('ASD (strain/rtHz)')
# plt.xlabel('Freq (Hz)')
# plt.legend(loc='upper center')
# plt.title('Advanced LIGO strain whitened data near '+eventname)

NFFT = int(fs/16.0)
NOVL = int(NFFT*15/16.0)
window = np.blackman(NFFT)
spec_cmap='viridis'
plt.figure()
spec_H1, freqs, bins, im = plt.specgram(strain_H1_whiten[indxt], NFFT=NFFT,\
    Fs=fs, window=window, noverlap=NOVL, cmap=spec_cmap, xextent=[-deltat,deltat])
plt.xlabel('time (s) since '+str(tevent))
plt.ylabel('Frequency (Hz)')
plt.colorbar(im.set_clim(-60, -20))
plt.axis([-6, 2, 0, 500])
plt.title('aLIGO H1 strain data near '+eventname)
plt.figure()
spec_H1, freqs, bins, im = plt.specgram(strain_L1_whiten[indxt], NFFT=NFFT,\
    Fs=fs, window=window, noverlap=NOVL, cmap=spec_cmap, xextent=[-deltat,deltat])
plt.xlabel('time (s) since '+str(tevent))
plt.ylabel('Frequency (Hz)')
plt.colorbar(im.set_clim(-60, -20))
plt.axis([-6, 2, 0, 500])
plt.title('aLIGO L1 strain data near '+eventname)

import gwpy
from gwpy.timeseries import TimeSeries
from gwosc.datasets import event_gps
gps = event_gps('GW170817')
segment = (int(gps) - 30, int(gps) + 2)
hdata = TimeSeries.fetch_open_data('H1', *segment, cache=True)
hq = hdata.q_transform(frange=(30, 500), qrange=(100, 110))
plot = hq.plot()
ax = plot.gca()
ax.set_epoch(gps)
ax.set_yscale('log')
ax.colorbar(clim=(0, 20), label="Normalised energy")
ldata = TimeSeries.fetch_open_data('L1', *segment, cache=True)
lq = ldata.q_transform(frange=(30, 500), qrange=(100, 110))
plot = lq.plot()
ax = plot.gca()
ax.set_epoch(gps)
ax.set_yscale('log')
ax.colorbar(clim=(0, 20), label="Normalised energy")

plt.show()