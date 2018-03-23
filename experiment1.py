# -*- coding: utf-8 -*-
# TOA experiment
import UWB_Kalman
import numpy as np

def readdata(TS,accel):
    fd=open('experiment_data\\TOAdata1.txt','r')
    try: 
        while True:
            Tstmp=np.zeros(4)    
            for i in range(4):
                s=fd.readline()
                Tstmp[i]=int(s.split()[3])
            TS.append(Tstmp)    
            acceltmp=np.zeros((100,2))    
            for i in range(100):
                s=fd.readline().split()
                acceltmp[i,0]=float(s[0])
                acceltmp[i,1]=float(s[1])
            accel.append(acceltmp)
    except (EOFError,IndexError):
        pass
    fd.close()
    
def checkmpudata(accel):
    cnt=0
    for i in accel:
        if abs(i)<0.0001:
            cnt+=1
    if cnt>10:
        return False
    else:
        return True

def checkuwbdata(dis):
    cnt=0
    for i in dis:
        if i>0.001:
            cnt+=1
    if cnt>=3:
        return True
    else:
        return False
    
DIS=[]
ACCEL=[]
readdata(DIS,ACCEL)
datacnt=len(DIS)

dt_IMU=0.01
dt_UBW=1
Anchor_num=4
Anchor_pos=np.array([[0,0],
                 [10,0],
                 [0,10],
                 [10,10]])
std_a=0.02
std_r=0.1
sigma_a=std_a**2
sigma_r=std_r**2

ekf=UWB_Kalman.EKF.RAKOEKF(sigma_a,sigma_r,Anchor_pos,Anchor_num,dt_IMU,dt_UBW)


for i in range(datacnt):
    Dis=DIS[i]
    Accel=ACCEL[i]
    mpuvalid=checkmpudata(Accel)
    uwbvalid=checkuwbdata(Dis)
    plot_x=[]
    plot_y=[]
    if mpuvalid and uwbvalid:
        postmp=ekf.ekffilter(Accel,Dis)
        plot_x.append(postmp[0])
        plot_y.append(postmp[1])
    elif not mpuvalid and uwbvalid:
        postmp=ekf.LSQ_TOA(Dis)
        ekf.StatusLast[0,0]=postmp[0]
        ekf.StatusLast[1,0]=postmp[1]
        plot_x.append(postmp[0])
        plot_y.append(postmp[1])        
        
    elif not mpuvalid and not uwbvalid:
        pass
    elif mpuvalid and not uwbvalid:
        pass
        
    
    