# -*- coding: utf-8 -*-
'''
在得到相关参数后，用来矫正惯导数据的一些函数

'''
import numpy as np
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
def calibmag(magdata):
    mag=np.matrix(magdata).T
    mag_rec=K*(mag-B)#calib it
    mag_rec=mag_rec.T
    mag_rec=np.asarray(mag_rec)*0.15
    mag_rec=mag_rec[0].tolist()
    tmp=mag_rec[0]#rotate it to accel axis
    mag_rec[0]=mag_rec[1]
    mag_rec[1]=tmp
    mag_rec[2]=-mag_rec[2]
    return mag_rec

def calibaccel(acceldata):
    if type(acceldata) is list:
        acceldata=np.array(acceldata)
    acceldata=(acceldata+np.array(accelbias))*np.array(accelscale)
    acceldata=(acceldata/16384*9.8).tolist()
    return acceldata

def calibgyro(gyrodata):
    if type(gyrodata) is list:
        gyrodata=np.array(gyrodata)
    gyrodata=(gyrodata)/32.768/57.296 # (+-1000) degreePersec to rad per sec
    gyrodata=gyrodata.tolist()
    return gyrodata