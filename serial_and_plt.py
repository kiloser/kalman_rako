# -*- coding: utf-8 -*-
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import serial
import struct
import math
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
# =============================================================================
#calib data
accelscale=[0.98674348,    0.95364379,    1.00778921]
magscale=[0.91566725,    0.98295111,    0.88824092]
accelbias=[-755.40186823, -359.90878696, -874.79820164]
magbias=[-21.55728099, 104.87445398,  -70.38780774]
q30=float(1073741824)
magsens=[1.1914,1.1875,1.1445]
K=np.matrix([[ 1.03146487,  0.        , -0.01745165],
             [ 0.05334139,  1.01507258,  0.03478666],
             [ 0.        ,  0.        ,  1.02887899]])
B=np.matrix([[-22.85868902],
             [ 62.03290237],
             [-44.29979761]])
def calibmag(magdata):
    mag=np.matrix(magdata).T
    mag_rec=K*(mag-B)
    mag_rec=mag_rec.T
    mag_rec=np.asarray(mag_rec)*0.15
    mag_rec=mag_rec[0].tolist()
    return mag_rec
# =============================================================================
gyro_array=np.zeros((freq_mpu,3),dtype=np.float)
accel_array=np.zeros((freq_mpu,3),dtype=np.float)
mag_array=np.zeros((freq_mpu,3),dtype=np.float)
accel_fdata=np.zeros((freq_mpu,3),dtype=np.float)
gyro_array_raw=np.zeros((freq_mpu,3),dtype=np.float)
accel_array_raw=np.zeros((freq_mpu,3),dtype=np.float)
mag_array_raw=np.zeros((freq_mpu,3),dtype=np.float)
disuwb_array=np.zeros((1,num_AC),dtype=np.float)
info_array=np.zeros((1,2),dtype=np.int)
data_idx=1

ser=serial.Serial()
ser.baudrate=115200
ser.port='COM4'
titlelist=[['accelx','accely','accelz','mod'],
            ['gyrox','gyroy','gyroz','mod'],
            ['magx','magy','magz','mod']]

ax=[list(),list(),list(),list()]
accelplotdata=[[],[],[],[]]
gyroplotdata=[[],[],[],[]]
magplotdata=[[],[],[],[]]
accelfplotdata=[[],[],[],[]]
plotdata=[accelplotdata,gyroplotdata,magplotdata]
x_lim=500
for i in range(0,1):
    plt.figure(i+1,figsize=(8,6))
    for j in range(0,4):
        axtmp=plt.subplot(4,1,j+1)
        ax[i].append(axtmp)
        axtmp.set_title(titlelist[i][j])
        axtmp.set_xlim(0,x_lim)
        plt.pause(.05)

#plt.figure(4)
#ax3d=plt.subplot(111,projection='3d')
#plt.pause(.05)
if not ser.isOpen():
    ser.open()    
ser.flushInput()
ser.flushOutput()
lowpass_filter=Filters.creat_lowpass_filter(10,10,2)
while True:
    
    while ser.inWaiting()!=framelen_std:
        pass
    temp=ser.read(framelen_std)
    disuwb_array=struct.unpack('<4f',temp[0:datalen_AC])
    for i in range(0,freq_mpu):
        mpudata=list(struct.unpack('<9h',temp[datalen_AC+i*single_len_mpu:single_len_mpu+datalen_AC+i*single_len_mpu]))
        gyro_array[i,:]=[float(v) for v in mpudata[0:3]]#gyroscopes data
        accel_array[i,:]=[ kk for kk in list(np.multiply(np.add(mpudata[3:6],accelbias),accelscale))]#accelerometer data
#        magtmp=[ kk*0.15  for kk in list(np.multiply(np.add(mpudata[6:9],magbias),magscale))]#mag data
        magtmp=[ kk  for kk in mpudata[6:9]]#mag data
        magtmp=calibmag(magtmp)
        tmp=magtmp[0]
        magtmp[0]=magtmp[1]
        magtmp[1]=tmp
        magtmp[2]=-magtmp[2]
        mag_array[i,:]=magtmp
        
        gyro_array_raw[i,:]=mpudata[0:3]
        accel_array_raw[i,:]=mpudata[3:6]
        mag_array_raw[i,:]=mpudata[6:9]
        accel_fdata[i,:]=lowpass_filter(accel_array[i,:].tolist())
        
    info_array=struct.unpack('<2B',temp[-4:-2])
        
    mpudata_list=[accel_array,gyro_array,mag_array]
    mpudata_list_raw=[accel_array_raw,gyro_array_raw,mag_array_raw]
    for i in range(0,1):
        datatemp=mpudata_list[i]
        modvalue=list(map(lambda x,y,z: math.sqrt(x**2+y**2+z**2),datatemp[:,0],datatemp[:,1],datatemp[:,2]))
        modvalue=list(zip(modvalue))
        datatemp=np.append(datatemp,modvalue,1)
        for j in range(0,4):
            if data_idx*10>500:
                plotdata[i][j]=plotdata[i][j][10::]
                plotdata[i][j].extend(list(datatemp[:,j]))
                ax[i][j].set_xlim(data_idx*10-500,data_idx*10)
                x_ax=range(data_idx*10-500,data_idx*10)                   
            else:
                plotdata[i][j].extend(list(datatemp[:,j]))
                x_ax=range(len(plotdata[i][j]))    
            ax[i][j].plot(x_ax,plotdata[i][j],'b')
            ax[i][j].plot(x_ax,np.zeros(len(plotdata[i][j])),'r',)
            ax[i][j].set_ylim(np.min(plotdata[i][j])-5,np.max(plotdata[i][j])+5)
            plt.pause(.001)
    data_idx+=1
#    ax3d.scatter(mag_array[:,0],mag_array[:,1],mag_array[:,2],c='blue')
#    plt.pause(.001)
