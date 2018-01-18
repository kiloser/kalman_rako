# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import scipy.fftpack
import numpy as np
import scipy
import serial
import threading
use_serial=1
readevent=threading.Event()
def simple_low_pass(acceldata):
    udata=0
    dt=0.001
    fcut=10
    a=(2*np.pi*fcut*dt)/(2*np.pi*fcut*dt+1)
    data=[]
    if not type(acceldata) is list:
        acceldata=[acceldata]
    for tmp in acceldata:
        if udata==0:
            data.append(tmp)
            udata=tmp
        else:
            udata=(1-a)*udata+a*tmp
            data.append(tmp-udata)
    return data

class serial_thread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.ser=serial.Serial()
        self.ser.baudrate=115200
        self.ser.port='COM4'
        self.thread_stop = False
        self.message=''
    def run(self):
        if not self.ser.isOpen():  
            self.ser.open()
            print('serial open sucessfully\n')
            self.ser.flushInput()
            self.ser.flushOutput()
        
        while not self.thread_stop:
            self.message=self.ser.readline()
            readevent.set()
    def stop(self):
        self.thread_stop=True
        self.ser.close()        
        print('closed serial thread\n')
        
class getcmd_thread(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)
        self.stop_serial=0
    def run(self):
        while not self.stop_serial:        
            mycmd=input('waiting for stop cmd\n')
            if mycmd=='stop':
                print('get stop cmd\n')
                self.stop_serial=1
                readevent.set()
    def stop(self):        
        print('closed cmd thread\n')

#class dataprocess_thread(threading.Thread):
#    def __init__(self):
#        threading.Thread.__init__(self)
#        self.wdaccelx=[]
#        self.wdaccely=[]
#        self.dataidx=0
#        self.x_lim=500
#        plt.ion()
#        self.fig1=plt.figure(1,figsize=(7,4))
#        self.ax_wdaccelx=plt.subplot(2,1,1)
#        self.ax_wdaccelx.set_xlim(0,self.x_lim)
#        self.ax_wdaccely=plt.subplot(2,1,2)
#        self.ax_wdaccely.set_xlim(0,self.x_lim)
#
#        plt.pause(.001)    
#    def run(self):
#        while True:
#            readevent.wait()
#            readevent.clear()
#            wdacceldata=serial1.message.decode().strip('\r\n').split('\t')
#            wdacceldata=[float(i) for i in wdacceldata]
#            self.wdaccelx.append(wdacceldata[0])
#            self.wdaccely.append(wdacceldata[1])
#            self.dataidx+=1
#            if self.dataidx>self.x_lim:
#                self.ax_wdaccelx.set_xlim(self.dataidx-self.x_lim,self.dataidx)
#                x_range=range(self.dataidx-self.x_lim,self.dataidx)
#            else:
#                x_range=range(len(self.wdaccelx))
#            self.ax_wdaccelx.plot(x_range,self.wdaccelx,'b')
#            self.ax_wdaccelx.plot(x_range,np.zeros(len(self.wdaccelx)),'r')
#            self.ax_wdaccelx.set_ylim(np.min(self.wdaccelx)-0.2,np.max(self.wdaccelx)+0.2)
#            plt.pause(.001)            
#                     
#    def stop(self):
#        print('cloesd dataprocess thread\n')


if use_serial==0:        
    fd=open('data\\wd_raw_accel.txt','r')
    accelxdata=[]
    accelydata=[]
    for line in fd.readlines():
        data=line.strip('\n').split('\t')
        data=[float(ii) for ii in data]
        accelxdata.extend([data[0]])
        accelydata.extend([data[1]])
    
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
    plt.show()
else:
    serial1=serial_thread()
    serial1.start()
    cmdthread=getcmd_thread()
    cmdthread.start()
#    dataprocess=dataprocess_thread()
#    dataprocess.start()
    wdaccelx=[]
    wdaccely=[]
    dataidx=0
    tmpcnt=0
    x_lim=500
    plt.ion()
    fig1=plt.figure(1,figsize=(14,8))
    ax_wdaccelx=plt.subplot(2,1,1)
    ax_wdaccelx.set_xlim(0,x_lim)
    ax_wdaccely=plt.subplot(2,1,2)
    ax_wdaccely.set_xlim(0,x_lim)   
    while not cmdthread.stop_serial:
        readevent.wait()
        readevent.clear()
        wdacceldata=serial1.message.decode().strip('\r\n').split('\t')
        print(wdacceldata)
        wdacceldata=[float(i) for i in wdacceldata]
#        wdaccelx.append(wdacceldata[0])
#        wdaccely.append(wdacceldata[1])
        dataidx+=1
        if dataidx>x_lim:
            wdaccelx.append(wdacceldata[0])
            wdaccely.append(wdacceldata[1])
            wdaccelx=wdaccelx[1:]
            wdaccely=wdaccelx[1:]
            x_range=range(dataidx-x_lim,dataidx)
            ax_wdaccelx.set_xlim(dataidx-x_lim,dataidx)
        else:
            wdaccelx.append(wdacceldata[0])
            wdaccely.append(wdacceldata[1])
            x_range=range(len(wdaccelx))
        tmpcnt+=1
        if tmpcnt==100:
            if dataidx>x_lim:
                x_range=range(dataidx-x_lim,dataidx)
                ax_wdaccelx.set_xlim(dataidx-x_lim,dataidx)
                ax_wdaccely.set_xlim(dataidx-x_lim,dataidx)
            else:
                x_range=range(len(wdaccelx))
            ax_wdaccelx.plot(x_range,wdaccelx,'b')
            ax_wdaccelx.plot(x_range,np.zeros(len(wdaccelx)),'r')
            ax_wdaccelx.set_ylim(np.min(wdaccelx)-1,np.max(wdaccelx)+1)
            ax_wdaccely.plot(x_range,wdaccely,'b')
            ax_wdaccely.plot(x_range,np.zeros(len(wdaccely)),'r')
            ax_wdaccely.set_ylim(np.min(wdaccely)-1,np.max(wdaccely)+1)            
            plt.pause(.001) 
            tmpcnt=0

    serial1.stop()
    cmdthread.stop()
#    dataprocess.stop()




