# -*- coding: utf-8 -*-
import MPUdataread
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
from mpl_toolkits.mplot3d import Axes3D
magmod=345
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
def residuals(p,x,y,z):
    return magfunc(p,x,y,z)
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
filename="magdata1.txt"
frameread=MPUdataread.get_frameread_fun(filename)
maglist=list()
try:
    while True:
        disuwb_array,accel_array,gyro_array,mag_array,info_array=frameread()
        maglist.extend(mag_array.tolist())
except TypeError:
    print('file end\n')
#plt.ion()
fig = plt.figure(1)  
ax1 = fig.add_subplot(111,projection='3d')
ax1.legend('mag')  
for mag in maglist:
    ax1.scatter(mag[0], mag[1], mag[2],c='b',marker='.')
#    plt.pause(0.01)

magarray=np.array(maglist)
p0=[0,0,0,0,0,0,0,0,0]
magx=magarray[:,0]
magy=magarray[:,1]
magz=magarray[:,2]
paras2=scipy.optimize.leastsq(residuals,p0,args=(magx,magy,magz))
K,B=KBcacu(paras2[0])
fig=plt.figure(2)
ax2 = fig.add_subplot(111,projection='3d') 
for i in range(0,len(magx)):
    Magm=np.matrix([magx[i],magy[i],magz[i]]).T
    Magr=K*(Magm-B)
    ax2.scatter(Magr[0], Magr[1], Magr[2],marker='.',c='b') 
plt.show()

