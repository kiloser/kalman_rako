# -*- coding: utf-8 -*-
import Madgwick
import MPUdataread
import Calibfun
filename=""
quat=[0.0,0.0,0.0,0.0]
frameread=MPUdataread.get_frameread_fun(filename)
try:
    while True:
        disuwb_array,accel_array,gyro_array,mag_array,info_array=frameread()
        accel=Calibfun.calibaccel(accel_array)
        gyro=Calibfun.calibgyro(gyro_array)
        mag=Calibfun.calibmag(mag_array)
        libc=Madgwick.SelDll("quat.dll")
        quat,eular=Madgwick.MadgwickQuat(accel,gyro,mag,quat,0.1,libc)
        
except TypeError:
    print('file end\n')
