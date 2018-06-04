# -*- coding: utf-8 -*-
'''
用来矫正加速度非正交误差，不过数据有些问题
'''
import MPUdataread
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
import scipy.optimize
from mpl_toolkits.mplot3d import Axes3D

def accelfunc(p,x,y,z):
    a,b,c,x0,y0,z0=p
    return a**2*(x+x0)**2+b**2*(y+y0)**2+c**2*(z+z0)**2-16384**2

def residuals(p,x,y,z):
    return accelfunc(p,x,y,z)

filename="acceldata1.txt"
frameread=MPUdataread.get_frameread_fun(filename)
accellist=list()
try:
    while True:
        disuwb_array,accel_array,gyro_array,mag_array,info_array=frameread()
        accellist.extend(accel_array.tolist())
except TypeError:
    print('file end\n')
#plt.ion()
fig = plt.figure(1)  
ax1 = fig.add_subplot(111,projection='3d')
ax1.legend('accel')  
for accel in accellist:
    ax1.scatter(accel[0], accel[1], accel[2],c='b',marker='.')
#    plt.pause(0.01)

accelarray=np.array(accellist)
p0=[1,1,1,0,0,0]
accelx=accelarray[:,0]
accely=accelarray[:,1]
accelz=accelarray[:,2]
paras1=scipy.optimize.leastsq(residuals,p0,args=(accelx,accely,accelz))
fig=plt.figure(2)
ax2 = fig.add_subplot(111,projection='3d') 
for i in range(0,len(accelx)):
    accelxr=(accelx[i]+paras1[0][3])*paras1[0][0]
    accelyr=(accely[i]+paras1[0][4])*paras1[0][1]
    accelzr=(accelz[i]+paras1[0][5])*paras1[0][2]
    ax2.scatter(accelxr, accelyr, accelzr,marker='.',c='b') 
plt.show()