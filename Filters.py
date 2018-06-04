# -*- coding: utf-8 -*-
'''
滤波代码，stm32已经实现了二阶低通滤波
'''
__author__='racode'
import numpy as np
from scipy.signal import lfilter,butter
def noisefilter(data,NP,NPS,NPL):
    valueF=data[0];
    dataF=[]
    for v in data:
        if v<valueF-NP or v>valueF+NP:
            valueF=v
        else:
            if v<valueF-NPS or v> valueF+NPS :
                valueF=(v+valueF)/2
            else:
                valueF=(1-NPL)*valueF+NPL*v
        dataF.append(valueF)
    return dataF
lastval=[0,0,0]
def filter_raw_data(data,thread):
    factor=0.75
    global lastval
    data_ret=[[],[],[]]
    datashape=np.shape(data)
    dataarray=np.array(data)
    for n in range(0,datashape[1]):
        acceldata=dataarray[:,n]
        for i in range(0,datashape[0]):
            if lastval[i]==0:
                lastval[i]=acceldata[i]
                data_ret[i].append(acceldata[i])
            else:
                if abs(acceldata[i]-lastval[i])<thread:
                    lastval[i]=lastval[i]*(1-factor)+acceldata[i]*factor
                    data_ret[i].append(lastval[i])
                else:
                    lastval[i]=acceldata[i]
                    data_ret[i].append(lastval[i])
    return data_ret


def creat_lowpass_filter(fcut,fs,order):
    initdata=[0,0]
    nyq=0.5*fs
    low=fcut/nyq
    def butter_filter(data):
        b,a=butter(order,low,btype='lowpass')
        data=initdata+data
        data_f = lfilter(b, a, data)
        datalen=len(data_f)
        initdata[1]=data_f[datalen-1]
        initdata[0]=data_f[datalen-2]
        return data_f
    return butter_filter

    
