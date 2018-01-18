# -*- coding: utf-8 -*-
import Madgwick
import Calibfun
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import serial
import struct
import quaternion
import Filters
# =============================================================================
#frame struct
num_AC=4 
datalen_AC=num_AC*4
freq_mpu=10
single_len_mpu=18
total_len_mpu=single_len_mpu*freq_mpu
info_len=2
divider_len=2
framelen_std=total_len_mpu+datalen_AC+info_len+divider_len#加两个界定符
# =============================================================================
gyro_array=np.zeros((freq_mpu,3),dtype=np.float)
accel_array=np.zeros((freq_mpu,3),dtype=np.float)
mag_array=np.zeros((freq_mpu,3),dtype=np.float)
gyro_array_raw=np.zeros((freq_mpu,3),dtype=np.float)
accel_array_raw=np.zeros((freq_mpu,3),dtype=np.float)
mag_array_raw=np.zeros((freq_mpu,3),dtype=np.float)
disuwb_array=np.zeros((1,num_AC),dtype=np.float)
info_array=np.zeros((1,2),dtype=np.int)
quat=[1,0,0,0]
# =============================================================================
ser=serial.Serial()
ser.baudrate=115200
ser.port='COM4'
if not ser.isOpen():
    ser.open()    
ser.flushInput()
ser.flushOutput()


titlelist=[['pitch','roll','yaw'],
           ['q1','q2','q3','q4'],
           ['world_accelx','world_accely','world_accelz'],
           ['accelx','accely','accelz']]
showeular=1
showquat=0
showaccel=1
showfilaccel=0
eulardata=[[],[],[]]
quatdata=[[],[],[],[]]
accelwddata=[[],[],[]]
accel_fdata=[[],[],[]]
accel_caldata=[[],[],[]]
x_lim=200
data_idx=0
quat=[1,0,0,0]
plt.ion()
if showeular==1:
    ax_eular=[]
    fig1=plt.figure(1,figsize=(14,8))
    for j in range(0,3):
        axtemp=plt.subplot(3,2,2*j+1)
        axtemp.set_title(titlelist[0][j])
        axtemp.set_xlim(0,x_lim)
        ax_eular.append(axtemp)
        plt.pause(.05)
        
if showquat==1:
    ax_quat=[]
    fig2=plt.figure(2,figsize=(8,6))
    for j in range(0,4):
        axtemp=plt.subplot(4,1,j+1)
        axtemp.set_title(titlelist[1][j])
        axtemp.set_xlim(0,x_lim)
        ax_quat.append(axtemp)
        plt.pause(.05)
            
if showaccel==1:
    ax_accel=[]
#    fig=plt.figure(3,figsize=(8,6))
    for j in range(0,3):
        axtemp=plt.subplot(3,2,2*j+2)
        axtemp.set_title(titlelist[2][j])
        axtemp.set_xlim(0,x_lim)
        ax_accel.append(axtemp)
        plt.pause(.05)
if showfilaccel==1:
    ax_filaccel=[]
    fig=plt.figure(4,figsize=(8,6))
    for j in range(0,3):
        axtemp=plt.subplot(3,1,j+1)
        axtemp.set_title(titlelist[3][j])
        axtemp.set_xlim(0,x_lim)
        ax_filaccel.append(axtemp)
        plt.pause(.05)
libc=Madgwick.SelDll("quat.dll")

while True:
    
    while ser.inWaiting()!=framelen_std:
        pass
    temp=ser.read(framelen_std)
    data_idx+=1
    disuwb_array=struct.unpack('<4f',temp[0:datalen_AC])
    for i in range(0,freq_mpu):
        mpudata=list(struct.unpack('<9h',temp[datalen_AC+i*single_len_mpu:single_len_mpu+datalen_AC+i*single_len_mpu]))
        gyro_data=[float(v) for v in mpudata[0:3]]#gyroscopes data
        accel_data=[ kk for kk in mpudata[3:6]]#accelerometer data
        mag_data=[ kk  for kk in mpudata[6:9]]#mag data
        accel=Calibfun.calibaccel(accel_data)
        gyro=Calibfun.calibgyro(gyro_data)
        mag=Calibfun.calibmag(mag_data)
        
#===============================================================================
        gyro_array_raw[i,:]=mpudata[0:3]
        accel_array_raw[i,:]=mpudata[3:6]
        mag_array_raw[i,:]=mpudata[6:9]
        gyro_array[i,:]=accel
        accel_array[i,:]=gyro
        mag_array[i,:]=mag
#===============================================================================
        quat,eular=Madgwick.MadgwickQuat(accel,gyro,mag,quat,0.1,libc)
        quat_q=quaternion.as_quat_array(quat)
        accel_wd=quat_q*(quaternion.as_quat_array([0]+accel))*quat_q.inverse()
        accel_wd=quaternion.as_float_array(accel_wd)
        accel_wd=accel_wd[1::]
        accel_f=Filters.filter_raw_data([[kk] for kk in accel],0.3)
        for j in range(0,3):
            if data_idx*10>x_lim:
                eulardata[j]=eulardata[j][1::]
                eulardata[j].extend([eular[j]])
            else:
                eulardata[j].extend([eular[j]])
        for j in range(0,3):
            if data_idx*10>x_lim:
                accelwddata[j]=accelwddata[j][1::]
                accelwddata[j].extend([accel_wd[j]])
            else:
                accelwddata[j].extend([accel_wd[j]])
        for j in range(0,4):
            if data_idx*10>x_lim:
                quatdata[j]=quatdata[j][1::]
                quatdata[j].extend([quat[j]])
            else:
                quatdata[j].extend([quat[j]])
        for j in range(0,3):
            if data_idx*10>x_lim:
                accel_fdata[j]=accel_fdata[j][1::]
                accel_fdata[j].extend([accel_f[j]])
            else:
                accel_fdata[j].extend([accel_f[j]])
        for j in range(0,3):
            if data_idx*10>x_lim:
                accel_caldata[j]=accel_caldata[j][1::]
                accel_caldata[j].extend([accel[j]])
            else:
                accel_caldata[j].extend([accel[j]])
                
    if showeular==1:
        for j in range(0,3):
            if data_idx*10>x_lim:
                ax_eular[j].set_xlim(data_idx*10-x_lim,data_idx*10)
                x_range=range(data_idx*10-x_lim,data_idx*10)
            else:
                x_range=range(len(eulardata[j]))
            ax_eular[j].plot(x_range,eulardata[j],'b')
            ax_eular[j].plot(x_range,np.zeros(len(eulardata[j])),'r')
            ax_eular[j].set_ylim(-200,200)
#            ax_eular[j].set_ylim(np.min(eulardata[j])-5,np.max(eulardata[j])+5)
            plt.pause(.01)       
    if showaccel==1:
        for j in range(0,3):
            if data_idx*10>x_lim:
                ax_accel[j].set_xlim(data_idx*10-x_lim,data_idx*10)
                x_range=range(data_idx*10-x_lim,data_idx*10)
            else:
                x_range=range(len(accelwddata[j]))
            ax_accel[j].plot(x_range,accelwddata[j],'b')
            ax_accel[j].plot(x_range,np.zeros(len(accelwddata[j])),'r')
            ax_accel[j].set_ylim(np.min(accelwddata[j])-1,np.max(accelwddata[j])+1)
            plt.pause(.01)
    if showquat==1:
        for j in range(0,4):
            if data_idx*10>x_lim:
                ax_quat[j].set_xlim(data_idx*10-x_lim,data_idx*10)
                x_range=range(data_idx*10-x_lim,data_idx*10)
            else:
                x_range=range(len(quatdata[j]))
            ax_quat[j].plot(x_range,quatdata[j],'b')
            ax_quat[j].plot(x_range,np.zeros(len(quatdata[j])),'r')
            ax_quat[j].set_ylim(np.min(quatdata[j])-0.2,np.max(quatdata[j])+0.2)
            plt.pause(.01)    
    if showfilaccel==1:
        for j in range(0,3):
            if data_idx*10>x_lim:
                ax_filaccel[j].set_xlim(data_idx*10-x_lim,data_idx*10)
                x_range=range(data_idx*10-x_lim,data_idx*10)
            else:
                x_range=range(len(accel_fdata[j]))
            ax_filaccel[j].plot(x_range,accel_fdata[j],'b')
            ax_filaccel[j].plot(x_range,accel_caldata[j],'g')
            ax_filaccel[j].plot(x_range,np.zeros(len(accel_fdata[j])),'r')
            ax_filaccel[j].set_ylim(np.min(accel_fdata[j])-0.2,np.max(accel_fdata[j])+0.2)
            plt.pause(.01)   
    info_array=struct.unpack('<2B',temp[-4:-2])