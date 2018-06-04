#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
姿态求解算法的C回调，用于实验验证。后期已经在stm32上实现
'''
__author__='racode'
Dllname=''
import ctypes
import os
floatArray3=ctypes.c_float*3
class MPUDATA_st(ctypes.Structure):
    _fields_=[("accel",floatArray3),
              ("gyro",floatArray3),
              ("mag",floatArray3)]
class RETV_st(ctypes.Structure):
    _fields_=[("q0",ctypes.c_float),
             ("q1",ctypes.c_float),
             ("q2",ctypes.c_float),
             ("q3",ctypes.c_float),
             ("pitch",ctypes.c_float),
             ("roll",ctypes.c_float),
             ("yaw",ctypes.c_float)]
Mpudata=MPUDATA_st()
Retval=RETV_st()
pMpudata=ctypes.pointer(Mpudata)
pRetval=ctypes.pointer(Retval)
accel_c_type=floatArray3()
gyro_c_type=floatArray3()
mag_c_type=floatArray3()
beta=ctypes.c_float(0.8)
def SelDll(name):
    Dllname=name
    if os.path.exists(Dllname):
        libc = ctypes.WinDLL(Dllname)
        print('dll file is loaded\n')
    else:
        print('dll file is not exist\n')
    return libc
        
def MadgwickQuat(accel,gyro,mag,q_last,delt,libc):
    for i in range(0,3):
        accel_c_type[i]=accel[i]
        gyro_c_type[i]=gyro[i]
        mag_c_type[i]=mag[i]
    Retval.q0=ctypes.c_float(q_last[0])
    Retval.q1=ctypes.c_float(q_last[1])
    Retval.q2=ctypes.c_float(q_last[2])
    Retval.q3=ctypes.c_float(q_last[3])
    Mpudata.accel=accel_c_type
    Mpudata.gyro=gyro_c_type
    Mpudata.mag=mag_c_type
    delt_c=ctypes.c_float(delt)
    libc.MadgwickQuaternionUpdate(pMpudata,pRetval,delt_c,beta)
    quat=[Retval.q0,Retval.q1,Retval.q2,Retval.q3]
    eular=[Retval.pitch,Retval.roll,Retval.yaw]
    return quat,eular

def DelDll(libc):
    del libc
if __name__=='__main__':
    libc = ctypes.WinDLL('quat.dll')
    print("this is a quaternion algrithom\n")
    Mpudata.accel[0]=-1.6816246509552002
    Mpudata.accel[1]=1.6126521825790405
    Mpudata.accel[2]=8.33885383605957
    Mpudata.gyro[0]=0
    Mpudata.gyro[1]=0
    Mpudata.gyro[2]=0
    Mpudata.mag[0]=18.503929138183594
    Mpudata.mag[1]=-34.66197967529297
    Mpudata.mag[2]=-34.50738525390625
    Retval.q0=1
    Retval.q1=0
    Retval.q2=0
    Retval.q3=0
    delt_c=ctypes.c_float(0.1)

    libc.MadgwickQuaternionUpdate(pMpudata,pRetval,delt_c)



