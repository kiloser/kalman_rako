# -*- coding: utf-8 -*-
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
import numpy as np
from sympy import symbols,Matrix
from scipy import constants as C
import matplotlib.pyplot as plt

class UKF_rako(UKF):
    def __init__(self,sigma_a,sigma_r,Anchor_pos,Anchor_num,dt_imu,dt_uwb):
        x_dim=6
        z_dim=Anchor_num*(Anchor_num-1)/2
        sigmas=MerweScaledSigmaPoints(n=x_dim,alpha=.1, beta=2., kappa=3-x_dim)
        super(UKF_rako,self).__init__(x_dim,z_dim,dt_uwb,fx=self.f_cv,hx=self.h_cv,points=sigmas)
        
        self.w=np.eye(2)*sigma_a#输入噪声
        self.Q_t=np.mat([[dt_uwb**2/2,0],
                       [0,dt_uwb**2/2],
                       [dt_uwb,0],
                       [0,dt_uwb],
                       [1,0],
                       [0,1]])#状态转移方程
        self.Q=np.array(self.Q_t*self.w*self.Q_t.T)
        self.R=np.eye(Anchor_num)*sigma_r
            
        self.acpos=Anchor_pos
        self.acnum=Anchor_num
        self.StatusLast=np.zeros(4)
        self.x=np.zeros(6)
        self.P=np.diag([.1, .1, .1, .1, .1, .1])#初始化协方差    
    def f_cv(self,x, dt):
        F=np.matrix([[1,0,dt,0,0,0],
                   [0,1,0,dt,0,0],
                   [0,0,1,0,dt,0],
                   [0,0,0,1,0,dt],
                   [0,0,0,0,1,0],
                   [0,0,0,0,0,1]])#状态转移方程
        x=np.matrix(x).T
        ans=np.array(F*x).ravel()
        return ans
    def h_cv(self,x,*h_args):
        x_imu=h_args[0]
        y_imu=h_args[1]
        xerror=x[0]
        yerror=x[1]
        dimu=np.array([])
        derr=np.array([])
        for pos in self.acpos:
            dimu.append(np.sqrt((pos[0]-x_imu)**2+(pos[1]-y_imu)**2))
            derr.append(np.sqrt((pos[0]-x_imu-xerror)**2+(pos[1]-y_imu-yerror)**2))
        z=np.array([])
        for i in range(0,self.acnum-1):
            for j in range(i+1,self.acnum):
                z.append((derr[j]-derr[i])-(dimu[j]-dimu[i]))
        return z
                
    def imupos(self,accel_array):
        '''
        计算根据加速度数据得到的坐标位置，并且更新laststatus作为运动系统最新的运动状态估计
        '''
        lastx=self.StatusLast[0]
        lasty=self.StatusLast[1]
        lastvx=self.StatusLast[2]
        lastvy=self.StatusLast[3]
        vx=[]
        vy=[]
        tag_pos_imu=[0,0]
        if self.lastaccel[0]==0:
            self.lastaccel=accel_array[0]
        for accel in accel_array:
            lastvx=lastvx+(accel[0]+self.lastaccel[0])*self.dt_imu/2
            self.lastaccel[0]=accel[0]
            vx.append(lastvx)
            lastvy=lastvy+(accel[1]+self.lastaccel[1])*self.dt_imu/2
            self.lastaccel[1]=accel[1]
            vy.append(lastvy)
        lastvx=self.StatusLast[2]
        lastvy=self.StatusLast[3]   
        for vel in vx:
            lastx=lastx+(vel+lastvx)*self.dt_imu/2
            lastvx=vel
            
        for vel in vy:
            lasty=lasty+(vel+lastvy)*self.dt_imu/2
            lastvy=vel
        tag_pos_imu[0]=lastx
        tag_pos_imu[1]=lasty
        self.StatusLast[0]=lastx
        self.StatusLast[1]=lasty
        self.StatusLast[2]=lastvx
        self.StatusLast[3]=lastvy
        
        return tag_pos_imu        
    def chan_algorithm(self,arrivetime):
        idx=np.argsort(arrivetime)
        arrivetime=np.take(arrivetime,idx)
        base_pos=np.zeros(self.acnum,self.acnum)
        for i in range(self.acnum):
            base_pos[i,:]=self.acpos[idx[i],:]
        evVal=np.concatenate((np.mat(arrivetime).T,base_pos),axis=1)
        row, column = evVal.shape  # 行，列
        baseX = evVal[:, 1]  # 列向量
        baseY = evVal[:, 2]
    
        ri1 = C.c*(evVal[:, 0] - evVal[0, 0])[:-1]  # 第i个基站和第一个基站之间的距离gui
        xi1 = (baseX - baseX[0])[1:]
        yi1 = (baseY - baseY[0])[1:]
        Standaraddeviation = 3.5e-2
    
        k = np.zeros(row)
        for i in range(0, row):
            k[i] = baseX[i] ** 2 + baseY[i] ** 2
        k = np.mat(k).T
    
        h = np.zeros((3, 1))
        for i in range(0, 3):
            h[i, 0] = 0.5 * ((ri1[i]) ** 2 - k[i + 1] + k[0])
        h = np.mat(h)
    
#        Ga = -np.bmat("xi1 yi1 ri1")
        Ga = -np.concatenate((xi1,yi1,ri1),axis=1)
        Q = np.zeros((row - 1, row - 1))
        Q = np.mat(Q)
        for i in range(0, row - 1):
            Q[i, i] = (Standaraddeviation) ** 2
    
        Za = (Ga.T * Q.I * Ga).I * Ga.T * Q.I * h
    
        B1 = np.zeros((row - 1, row - 1))
        for i in range(0, row - 1):
            B1[i, i] = np.sqrt((baseX[i + 1] - Za[0]) ** 2 + (baseY[i + 1] - Za[1]) ** 2)
    
        B1 = np.mat(B1)
    
        P1 = C.c ** 2 * B1 * Q * B1
        Za1 = (Ga.T * P1.I * Ga).I * Ga.T * P1.I * h
        C0 = (Ga.T * P1.I * Ga).I
    
        h1 = np.zeros((3, 1))
        h1[0] = (Za1[0] - baseX[0]) ** 2
        h1[1] = (Za1[1] - baseY[0]) ** 2
        h1[2] = (Za1[2]) ** 2
        h1 = np.mat(h1)
    
        Ga1 = np.mat([[1, 0], [0, 1], [1, 1]])
        r1 = np.sqrt((baseX[0] - Za1[0]) ** 2 + (baseY[0] - Za1[1]) ** 2)
    
        B2 = np.zeros((3, 3))
        B2[0, 0] = Za1[0] - baseX[0]
        B2[1, 1] = Za1[1] - baseY[0]
        B2[2, 2] = r1
        B2 = np.mat(B2)
    
        P2 = 4 * B2 * C0 * B2
        Za2 = (Ga1.T * P2.I * Ga1).I * Ga1.T * P2.I * h1
    
        ms0 = np.sqrt(np.abs(Za2))
        ms0[0] = ms0[0] + baseX[0]
        ms0[1] = ms0[1] + baseY[0]
    
        return ms0

    def Z_observe(self,arrive_time,tag_pos_imu):
        '''
        根据给定的到达距离差和IMU推测距离来计算观测向量
        '''
        '''
        disdiff是表达各个基站之间到达距离差的矩阵
        disdiff=[[0,1]
                [-1,0]]
        代表了
        1号基站的到达时间减去2号基站的到达时间1s
        2号基站的到达时间减去1号基站的到达时间-1s
        第i列代表各个基站减去第i基站的时间差
        '''
        dis_diff=np.zeros((Anchor_num,Anchor_num))
        for i in range(self.acnum):
            dis_diff[:,i]=arrive_time-arrive_time[i]
        x_imu=tag_pos_imu[0]
        y_imu=tag_pos_imu[1]
        dimu=np.array([])
        z_obs=np.array([])
        for pos in self.acpos:
            dimu.append(np.sqrt((pos[0]-x_imu)**2+(pos[1]-y_imu)**2))
        for i in range(0,self.acnum-1):
            for j in range(i+1,self.acnum):
                z_obs.append((dis_diff[j,i])-(dimu[j]-dimu[i]))
        return z_obs        
    def ukf_filter(self,accel_array,arrive_time):
        if self.StatusLast[0]==0:
            
            #计算位置，使用TOA
            tag_pos=self.chan_algorithm(arrive_time)
            self.StatusLast[0,0]=tag_pos[0]
            self.StatusLast[1,0]=tag_pos[1]
            return tag_pos
        tag_pos_imu=self.imupos(accel_array)
        Zobs=self.Z_observe(arrive_time,tag_pos_imu)
        '''
        每次的误差变量初始值为0
        '''
        self.x=np.zeros(6)
        self.predict()
        self.update(Zobs,hx_args=(tag_pos_imu[0],tag_pos_imu[1]))
        self.StatusLast=self.StatusLast+self.x[:4]
        tag_pos=[0,0]
        tag_pos[0]=self.StatusLast[0]
        tag_pos[1]=self.StatusLast[1]
        return tag_pos

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
    
dt_IMU=0.001
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
idx=input("想进行哪个仿真？\n\
      1.变加速度\n\
      2.圆周\n\
      3.水平匀速\n\
      4.垂直匀速\n\
      5.定点分析\n")

if idx=='1':
    fig4=plt.figure(4)
    ax4=fig4.add_subplot(111)
    ax4.scatter(Anchor_pos[:,0],Anchor_pos[:,1],marker='o',c='black',s=6)
    tagposlist=[]
    tagpoint_len=200
    acceldata=np.zeros((tagpoint_len*100,2))
    acceldata[:1000,0]=acceldata[:1000,0]+0.1*np.ones(1000)
    acceldata[19000:20000,0]=acceldata[19000:20000,0]-0.1*np.ones(1000)
    acceldata[3000:4000,1]=acceldata[3000:4000,1]+0.1*np.ones(1000)
    acceldata[6000:7000,1]=acceldata[6000:7000,1]-0.1*np.ones(1000)
    acceldata[9000:10000,1]=acceldata[9000:10000,1]-0.2*np.ones(1000)
    acceldata[12000:13000,1]=acceldata[12000:13000,1]+0.2*np.ones(1000)
    acceldata[14000:15000,1]=acceldata[14000:15000,1]+0.1*np.ones(1000)
    acceldata[18000:19000,1]=acceldata[18000:19000,1]-0.1*np.ones(1000)
    acceldata=np.concatenate((acceldata,acceldata),axis=0)
    acceldata=np.concatenate((acceldata,acceldata),axis=0)
    initstat=[-35,5]+[0,0]
    tagposlist=imutrace(initstat,acceldata)
    tagpoint_len=len(tagposlist)

ukf=UKF_rako(sigma_a,sigma_r,Anchor_pos,Anchor_num,dt_IMU,dt_UBW)

'''
静止运动
'''
if idx=='5':
    fig1=plt.figure(1)
    ax1=fig1.add_subplot(111)
    ax1.scatter(Anchor_pos[:,0],Anchor_pos[:,1],marker='o',c='black',s=6)  
    tagposlist=[]
    for i in np.linspace(-10,20,1):
        for j in np.linspace(-10,20,1):
            tagposlist.append([i,j])
    
    for tagpos in tagposlist:
        #============================================================    
        #let's make some fake data
        #============================================================
        tgpos=tagpos
        realdis=np.zeros(Anchor_num)
        uwbdis_data=np.zeros((50,Anchor_num))
        arrivetime_data=np.zeros((50,Anchor_num))
        acceldata=np.zeros((50*100,2))
        for i in range(Anchor_num):
            realdis[i]=np.sqrt((tgpos[0]-Anchor_pos[i][0])**2+(tgpos[1]-Anchor_pos[i][1])**2)
            uwbdis_data[:,i]=realdis[i]+np.random.normal(0,std_r,50)
            arrivetime_data[:,i]=uwbdis_data[:,i]/C.c
        for i in range(2):
            acceldata[:,i]=acceldata[:,i]+np.random.normal(0,std_a,5000)
        #============================================================    
        #end
        #============================================================

