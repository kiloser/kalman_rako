# -*- coding: utf-8 -*-
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
import numpy as np
from sympy import symbols,Matrix
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
    def chan_algorithm(self,dis_diff):
        chan_pos=[1,1]
        return chan_pos
    def Z_observe(self,dis_diff,tag_pos_imu):
        '''
        根据给定的到达距离差和IMU推测距离来计算观测向量
        '''
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
    def ukf_filter(self,accel_array,dis_diff):
        if self.StatusLast[0]==0:
            
            #计算位置，使用TOA
            tag_pos=self.chan_algorithm(dis_diff)
            self.StatusLast[0,0]=tag_pos[0]
            self.StatusLast[1,0]=tag_pos[1]
            return tag_pos
        tag_pos_imu=self.imupos(accel_array)
        Zobs=self.Z_observe(dis_diff,tag_pos_imu)
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
    
dt_IMU=0.001
dt_UBW=1
Anchor_num=4
Anchor_pos=np.array([[0,0],
                     [10,0],
                     [0,10],
                     [10,10]])
sigma_a=0.02**2
sigma_r=0.2**2
ukf=UKF_rako(sigma_a,sigma_r,Anchor_pos,Anchor_num,dt_IMU,dt_UBW)
'''
disdiff是表达各个基站之间到达距离差的矩阵
disdiff=[[0,1]
        [-1,0]]
代表了
1号基站的到达时间减去2号基站的到达时间乘以光速是1m
2号基站的到达时间减去1号基站的到达时间乘以光速是-1m
'''
disdiff=np.zeros((Anchor_num,Anchor_num))

