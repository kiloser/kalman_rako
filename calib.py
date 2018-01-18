# -*- coding: utf-8 -*-

import numpy as np
import struct
import matplotlib.pyplot as plt
import scipy
import math
from mpl_toolkits.mplot3d import Axes3D
def files(maxnum):
    n=1
    while n<=maxnum:
        filename='pos'+str(n)+'.txt'
        yield filename
        n+=1
def noisefilter(data,NP,NPS,NPL):
    valueF=data[0];
    dataF=[]
    for v in data:
        if v<valueF+NP or v>valueF-NP:
            valueF=v
        else:
            if v<valueF-NPS or v> valueF+NPS :
                valueF=(v+valueF)/2
            else:
                valueF=(1-NPL)*valueF+NPL*v
        dataF.append(valueF)
    return dataF

def filter_raw_data(data,thread):
    i=len(data)
    mean=0
    dataF=[]
    for k in range(0,i):
        if mean==0:
            mean=data[k]
            dataF.append(data[k])
        else:
            mean=np.mean(dataF)
            if np.abs(data[k]-mean)>thread:
                dataF.append(mean)
            else:
                dataF.append(data[k])
    return dataF
                
def magfunc(p,x,y,z):
    #a(x-x0)^2+b(y-y0)^2+c(z-z0)^2
#    a1,a2,a3,a4,a5,a6,x0,y0,z0=p
#    return a1*(x-x0)**2+a2*(y-y0)**2+a3*(z-z0)**2+a4*(x-x0)*(y-y0)+a5*(x-x0)*(z-z0)+a6*(y-y0)*(z-z0)    
    a1,a2,a3,a4,a5,a6,a7,a8,a9=p
#    A=np.matrix([[a1,a4,a5],
#               [a4,a2,a6],
#               [a5,a6,a3]])
#    Ainv=A.I
    return a1*x**2+a2*y**2+a3*z**2+2*a4*x*y+2*a5*x*z+2*a6*y*z+2*a7*x+2*a8*y+2*a9*z-1

def accelfunc(p,x,y,z):
    a,b,c,x0,y0,z0=p
    return a**2*(x-x0)**2+b**2*(y-y0)**2+c**2*(z-z0)**2-16384**2

def magresiduals(p,x,y,z):
    return magfunc(p,x,y,z)

def accelresiduals(p,x,y,z):
    return accelfunc(p,x,y,z)

def KBcacu(paras):
    A=np.matrix([[paras[0],paras[3],paras[4]],[paras[3],paras[1],paras[5]],[paras[4],paras[5],paras[2]]])
    Ainv=A.I
    B=-Ainv*np.matrix([paras[6],paras[7],paras[8]]).T
    ap=Ainv[0,0]
    dp=Ainv[0,1]
    ep=Ainv[0,2]
    cp=Ainv[2,2]
    bp=Ainv[1,1]
    fp=Ainv[1,2]
    kx=math.sqrt(ap)/magmod
    ky=math.sqrt(bp)/magmod
    kz=math.sqrt(cp)/magmod
    alphe=math.asin(ep/math.sqrt(ap*cp))
    temp=((dp*cp-ep*fp)/math.sqrt((ap*cp-ep**2)*(bp*cp-fp**2)))
    beta=math.asin(temp)
    gama=math.asin(fp/math.sqrt(bp*cp))
    K=np.matrix([[1/kx, 0, -alphe/kz],
                 [-beta/kx, 1/ky, -gama/kz],
                 [0, 0, 1/kz]])
    return K,B

accelxM=[]
accelyM=[]
accelzM=[]
magxM=[]
magyM=[]
magzM=[]
framelen_std=200
num_mpudata=10
filecnt=72
magmod=345#unit is 0.15uT

plt.ion()
for filename in files(filecnt):
    try:
        fd=open('data\\'+filename,'r')
        accelx=[]
        accely=[]
        accelz=[]
        gyrox=[]
        gyroy=[]
        gyroz=[]
        magx=[]
        magy=[]
        magz=[]
        num=int(fd.seek(0,2)/(framelen_std*3))
        fd.seek(0,0)
        for j in range(0,num):
            temp=fd.read(framelen_std*3)
            if len(temp) < framelen_std*3:
                print('end of file \n')
                break    
            temp=bytes.fromhex(temp)
            dis=struct.unpack('<4f',temp[0:16])
            for i in range(0,num_mpudata):
                mpudata=list(struct.unpack('<9h',temp[16+i*18:34+i*18]))
                gyrox.append(mpudata[0])
                gyroy.append(mpudata[1])
                gyroz.append(mpudata[2])
                accelx.append(mpudata[3])
                accely.append(mpudata[4])
                accelz.append(mpudata[5])
                magx.append(mpudata[6])
                magy.append(mpudata[7])
                magz.append(mpudata[8])
        NP=200#corresponding NP/16384*9.8m/s^2
        NPL=0.0405#Fc=NPL/(1-NPL)*2pi*dT
        NPS=150#corresponding NPS/16384*9.8m/s^2
        accelxM.append(np.mean(filter_raw_data(accelx,400)))
        accelyM.append(np.mean(filter_raw_data(accely,400)))
        accelzM.append(np.mean(filter_raw_data(accelz,400)))
        magxM.append(np.mean(filter_raw_data(magx,40)))
        magyM.append(np.mean(filter_raw_data(magy,40)))
        magzM.append(np.mean(filter_raw_data(magz,40)))
    #    accelxF=noisefilter(accelx,NP,NPS,NPL)    
    #    accelyF=noisefilter(accely,NP,NPS,NPL)
    #    accelzF=noisefilter(accelz,NP,NPS,NPL)
    except IOError as e:
        print(e)
        filecnt=filecnt-1
    finally:
        fd.close()
fig = plt.figure(1)
ax1 = fig.add_subplot(111,projection='3d') 
ax1.scatter(accelxM, accelyM, accelzM,marker='.')
plt.pause(0.01)
ax1.legend('accel')
fig = plt.figure(2)  
ax2 = fig.add_subplot(111,projection='3d') 
ax2.scatter(magxM, magyM, magzM,marker='.')
plt.pause(0.01)
ax2.legend('mag')  
plt.show() 
print('starting optimizing\n')
accelxM=np.array(accelxM)
accelyM=np.array(accelyM)
accelzM=np.array(accelzM)
magxM=np.array(magxM)
magyM=np.array(magyM)
magzM=np.array(magzM)
p0=[1,1,1,-755.40186823, -359.90878696, -874.79820164]
print('accel resault\r\n')
paras1=scipy.optimize.leastsq(accelresiduals,p0,args=(accelxM,accelyM,accelzM))
print(paras1[0])
p0=[0,0,0,0,0,0,0,0,0]
print('mag resault\r\n')
paras2=scipy.optimize.leastsq(magresiduals,p0,args=(magxM,magyM,magzM))
print(paras2[0])
K,B=KBcacu(paras2[0])
fig=plt.figure(3)
ax = fig.add_subplot(111,projection='3d') 
for i in range(0,len(magxM)):
    Magm=np.matrix([magxM[i],magyM[i],magzM[i]]).T
    Magr=K*(Magm-B)
    ax.scatter(Magr[0], Magr[1], Magr[2]) 
#out=open('data\\outdata','wb')
#buff=struct.pack(str(filecnt)+'d',*accelxM)
#out.write(buff)
#buff=struct.pack(str(filecnt)+'d',*accelyM)
#out.write(buff)
#buff=struct.pack(str(filecnt)+'d',*accelzM)
#out.write(buff)
#buff=struct.pack(str(filecnt)+'d',*magxM)
#out.write(buff)
#buff=struct.pack(str(filecnt)+'d',*magyM)
#out.write(buff)
#buff=struct.pack(str(filecnt)+'d',*magzM)
#out.write(buff)
#out.close()