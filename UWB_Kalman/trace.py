# -*- coding: utf-8 -*-
import numpy as np
def imutrace(initstat,accelarray):
    '''
    用来在只用加速度数据的情况下绘制路径
    '''
    n=int(len(accelarray)/100)
    laststatus=[0,0,0,0,0,0]
    laststatus[:4]=initstat
    tagpos=[]
    for i in range(n):
        tracefunc(laststatus,accelarray[100*i:100*(i+1),:])
        tagpos.append(laststatus[:2])
    return tagpos
    
def tracefunc(laststatus,accel_array):
    '''
    输入100组加速度数据，返回一组坐标速度的状态数据
    因为采样率是100hz，所以对应的一百组加速度数据才会有一组坐标数据
    '''
    lastx=laststatus[0]
    lasty=laststatus[1]
    lastvx=laststatus[2]
    lastvy=laststatus[3]
    lastaccel=np.array([0,0])
    lastaccel[0]=laststatus[4]
    lastaccel[1]=laststatus[5]
    vx=[]
    vy=[]
    dt_imu=0.01
    if lastaccel[0]==0:
        lastaccel=accel_array[0]
    for accel in accel_array:
        lastvx=lastvx+(accel[0]+lastaccel[0])*dt_imu/2
        lastaccel[0]=accel[0]
        vx.append(lastvx)
        lastvy=lastvy+(accel[1]+lastaccel[1])*dt_imu/2
        lastaccel[1]=accel[1]
        vy.append(lastvy)

    for vel in vx:
        lastx=lastx+(vel+lastvx)*dt_imu/2
        lastvx=vel
        
    for vel in vy:
        lasty=lasty+(vel+lastvy)*dt_imu/2
        lastvy=vel
        
    laststatus[0]=lastx
    laststatus[1]=lasty
    laststatus[2]=lastvx
    laststatus[3]=lastvy
    laststatus[4]=lastaccel[0]
    laststatus[5]=lastaccel[1]    