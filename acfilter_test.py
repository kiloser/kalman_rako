# -*- coding: utf-8 -*-
#this file is used to verify the butterworth filter implented on accelerator.
#the feature is sucessfully realized on stm32,so this file is not useful anymore.
#but data still can be ploted for eassy 
import matplotlib.pyplot as plt
import scipy.fftpack
import numpy as np
import scipy
def simple_low_pass(acceldata):
    udata=0
    lastdata=0
    dt=0.001
    fcut=10
    a=(2*np.pi*fcut*dt)/(2*np.pi*fcut*dt+1)
    data=[]
    for tmp in acceldata:
        if udata==0:
            data.append(tmp)
            udata=tmp
            lastdata=tmp
        else:
            udata=(1-a)*udata+a*lastdata
            lastdata=tmp-udata
            data.append(lastdata)
    return data

fd=open('data\\raw_accel2.txt','r')
accelxdata=[]
accelydata=[]
accelzdata=[]
for line in fd.readlines():
    data=line.strip('\n').split('\t')
    data=[float(ii) for ii in data]
    accelxdata.extend([data[0]])
    accelydata.extend([data[1]])
    accelzdata.extend([data[2]])

fd.close()
# Number of samplepoints
N = len(accelxdata)
# sample spacing
T = 1.0 / 100.0
x = np.linspace(0.0, N*T, N)
y = [ii for ii in accelxdata]
fig1=plt.figure(1)
ax=fig1.add_subplot(111)
ax.plot(x,y)#plt raw data

yf = scipy.fftpack.fft(y)
xf = np.linspace(0.0, 1.0/(2.0*T), N/2)
fig2=plt.figure(2)
ax2=fig2.add_subplot(111)
ax2.plot(xf[1:], 2.0/N * np.abs(yf[:N//2][1:]))

nyq = 0.5 * 100
low = 10 / nyq
b, a = scipy.signal.butter(2, low, btype='low')
y = scipy.signal.lfilter(b, a, y)
ax1=fig1.add_subplot(111)
ax1.plot(x,y)#plt filtered data

yf = scipy.fftpack.fft(y)
xf = np.linspace(0.0, 1.0/(2.0*T), N/2)
ax2=fig2.add_subplot(111)
ax2.plot(xf[1:], 2.0/N * np.abs(yf[:N//2][1:]))


plt.show()

