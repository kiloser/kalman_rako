# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 13:15:33 2017

@author: Administrator
"""
import struct
import numpy as np
import matplotlib.pyplot as plt
#calib data
accelscale=[1,    1,    1]
accelbias=[-755.40186823, -359.90878696, -874.79820164]#cacued fram some files
magsens=[1.1914,1.1875,1.1445]# this is got from MPU fuse ROM
gyrobias=[-22.37,-29.88,28.28]

K=np.matrix([[ 1.04654758,  0.        , -0.009884  ],
             [ 0.05359849,  1.02752393,  0.03621744],
             [ 0.        ,  0.        ,  1.04087807]])
B=np.matrix([[-17.28529639],
             [ 66.32626503],
             [-50.01458553]])
#frame struct
num_AC=4 
datalen_AC=num_AC*4
freq_mpu=10
single_len_mpu=18
total_len_mpu=single_len_mpu*freq_mpu
info_len=2
divider_len=2
framelen_std=total_len_mpu+datalen_AC+info_len+divider_len#加两个界定符
        
def get_frameread_fun(filename):
    fd=open('data\\'+filename,'r')
    num=int(fd.seek(0,2)/(framelen_std*3))
    fd.seek(0,0)
    frame_idx=[0]
    def frameread():
        frame_idx[0]=frame_idx[0]+1
        gyro_array=np.zeros((freq_mpu,3),dtype=np.float)
        accel_array=np.zeros((freq_mpu,3),dtype=np.float)
        mag_array=np.zeros((freq_mpu,3),dtype=np.float)
        disuwb_array=np.zeros((1,num_AC),dtype=np.float)
        info_array=np.zeros((1,2),dtype=np.int)
        if frame_idx[0]>num:
            fd.close()
            return -1;
        temp=fd.read(framelen_std*3)
        temp=bytes.fromhex(temp)
        disuwb_array=struct.unpack('<4f',temp[0:datalen_AC])
        for i in range(0,freq_mpu):
            mpudata=list(struct.unpack('<9h',temp[datalen_AC+i*single_len_mpu:single_len_mpu+datalen_AC+i*single_len_mpu]))
            gyro_array[i,:]=[float(v) for v in mpudata[0:3]]#gyroscopes data rad per second
            accel_array[i,:]=[ kk for kk in mpudata[3:6]]#accelerometer data
            magtmp=[ kk  for kk in mpudata[6:9]]#quaternion data
            mag_array[i,:]=magtmp
        info_array=struct.unpack('<2B',temp[-4:-2])
#        devidestr=struct.unpack('<2s',temp[-2:])
        return disuwb_array,accel_array,gyro_array,mag_array,info_array
    return frameread

#def readframe(fd):
#    for j in range(0,num):
#        temp=f.read(framelen_std*3)
#        if len(temp) < framelen_std*3:
#            print('end of file \n')
#            break    
#        temp=bytes.fromhex(temp)
#        disuwb_array[j]=struct.unpack('<4f',temp[0:datalen_AC])
#        for i in range(0,num_mpudata):
#            mpudata=list(struct.unpack('<9h',temp[datalen_AC+i*single_len_mpu:single_len_mpu+datalen_AC+i*single_len_mpu]))
#            mpudata_array[j,i,0:3]=[float(v)/16.384 for v in mpudata[0:3]]#gyroscopes data
#            mpudata_array[j,i,3:6]=[ kk*9.8/16384 for kk in list(np.multiply(np.add(mpudata[3:6],accelbias),accelscale))]#accelerometer data
#            mpudata_array[j,i,6:9]=[ kk*0.15  for kk in list(np.multiply(np.add(mpudata[6:9],magbias),magscale))]#quaternion data
#            magtemp=mpudata_array[j,i,6]
#            mpudata_array[j,i,6]=mpudata_array[j,i,7] 
#            mpudata_array[j,i,7]=magtemp
#            mpudata_array[j,i,8]=-mpudata_array[j,i,8]
#        info_array[j]=struct.unpack('<2B',temp[-4:-2])
#        devidestr=struct.unpack('<2s',temp[-2:])
        
if __name__=='__main__':
    filename="data\\testdata1.txt"        
    frameread=get_frameread_fun(filename)
    plt.figure(1)
    accel_ax1=plt.subplot(311)
    plt.title('accelx')
    accel_ax2=plt.subplot(312)
    plt.title('accely')
    accel_ax3=plt.subplot(313)
    plt.title('accelz')
    plt.figure(2)
    gyro_ax1=plt.subplot(311)
    plt.title('gyrox')
    gyro_ax2=plt.subplot(312)
    plt.title('gyroy')
    gyro_ax3=plt.subplot(313)
    plt.title('gyroz')
    plt.figure(3)
    mag_ax1=plt.subplot(311)
    plt.title('magx')
    mag_ax2=plt.subplot(312)
    plt.title('magy')
    mag_ax3=plt.subplot(313)
    plt.title('magz')
    
    data_idx=0
    try:
        while 1:
            disuwb_array,accel_array,gyro_array,mag_array,info_array=frameread()
            x_ax=range(data_idx*10,10+data_idx*10)
            accel_ax1.plot(x_ax,accel_array[:,0],'b')
            accel_ax2.plot(x_ax,accel_array[:,1],'b')
            accel_ax3.plot(x_ax,accel_array[:,2],'b')
            gyro_ax1.plot(x_ax,gyro_array[:,0],'b')
            gyro_ax2.plot(x_ax,gyro_array[:,1],'b')
            gyro_ax3.plot(x_ax,gyro_array[:,2],'b')
            mag_ax1.plot(x_ax,mag_array[:,0],'b')
            mag_ax2.plot(x_ax,mag_array[:,1],'b')
            mag_ax3.plot(x_ax,mag_array[:,2],'b')
            data_idx+=1
    except TypeError:
        print("reach the file end\n")
    

