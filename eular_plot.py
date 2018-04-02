# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import math
plt.rc('font',size=14)
filename='data.txt'
fd=open('data\\'+filename,'r')
pitchdata=[]
rolldata=[]
yawdata=[]
accelx=[]
accely=[]
tmp=np.load("eular and wdaccel plot data.npy")
eular1=tmp[0]
eular2=tmp[1]
eular3=tmp[2]
ac1=tmp[3]
ac2=tmp[4]
#
#for line in fd.readlines():
#    accel=line.strip('\r\n').split('\t')
#    accel=list(map(float,accel))
#
##    q0=quat[0]
##    q1=quat[1]
##    q2=quat[2]
##    q3=quat[3]
##    norm=np.sqrt(q0**2+q1**2+q2**2+q3**2)
##    q0/=norm
##    q1/=norm
##    q2/=norm
##    q3/=norm
##    roll = math.asin(2*q0*q2-2*q1*q3)*57.3;
##    pitch = math.atan2(2*q2*q3+2*q0*q1,-(2*q1*q1+2*q2*q2)+1)*57.3;
##    yaw = math.atan2(2*q1*q2+2*q0*q3,-(2*q2*q2+2*q3*q3)+1)*57.3;     
##    pitchdata.append(pitch)
##    rolldata.append(roll)
##    yawdata.append(yaw)
#    accelx.append(accel[0])
#    accely.append(accel[1])
#
#fig1=plt.figure(1)
#ax1=fig1.add_subplot(211)
#ax2=fig1.add_subplot(212)
#ax1.set_ylim([-5,5])
#ax2.set_ylim([-5,5])
#ax1.plot(accelx)
#ax2.plot(accely)
#fd.close()    


change=eular3
pitchdata=change[0]
rolldata=change[1]
yawdata=change[2]
fig1=plt.figure(1)
ax1=fig1.add_subplot(311)
ax2=fig1.add_subplot(312)
ax3=fig1.add_subplot(313)
datalen=len(yawdata)
x=np.linspace(0,0.01*datalen,datalen,endpoint=False)
ax1.plot(x,pitchdata)
ax2.plot(x,rolldata)
ax3.plot(x,yawdata)
ax1.set_ylim((-100,50))
ax2.set_ylim((-50,100))
ax3.set_ylim((0,150))
ax1.set_xlabel('time (s)',fontsize=14)
ax2.set_xlabel('time (s)',fontsize=14)
ax3.set_xlabel('time (s)',fontsize=14)
ax1.set_ylabel('pitch (degree)',fontsize=14)
ax2.set_ylabel('roll (degree)',fontsize=14)
ax3.set_ylabel('yaw (degree)',fontsize=14)
plt.tight_layout()
ax1.set_title('Pitch angle',fontsize=14)
ax2.set_title('Roll angle',fontsize=14)
ax3.set_title('Yaw angle',fontsize=14)


data=ac2
xdata=data[0]
ydata=data[1]
datalen=len(data[0])
fig2=plt.figure(2)
ax1=fig2.add_subplot(211)
ax2=fig2.add_subplot(212)
x=np.linspace(0,0.01*datalen,datalen,endpoint=False)

ax1.plot(x,xdata)
ax2.plot(x,ydata)
ax1.set_ylim((-5,5))
ax2.set_ylim((-5,5))
ax1.set_xlabel('time (s)')
ax2.set_xlabel('time (s)')
ax1.set_ylabel('North Acceleration (m/s^2)')
ax2.set_ylabel('East Acceleration (m/s^2)')
plt.tight_layout()
ax1.set_title('World frame Acceleration')
ax2.set_title('World frame Acceleration')
