# -*- coding: utf-8 -*-
import sympy 
from sympy import symbols,Matrix
from filterpy.common import dot3
import numpy as np
from numpy import dot, array

class RAKOEKF():
    def __init__(self,sigma_a,sigma_r,Anchor_pos,num_Anchor,dt_imu,dt_uwb):
        #EKF.__init__(self, 4, 3, 2)#class filterpy.kalman.ExtendedKalmanFilter(dim_x, dim_z, dim_u=0)
        dt=symbols('dt')
        self.F=Matrix([[1,0,dt,0,0,0],
                       [0,1,0,dt,0,0],
                       [0,0,1,0,dt,0],
                       [0,0,0,1,0,dt],
                       [0,0,0,0,1,0],
                       [0,0,0,0,0,1]])#状态转移方程
        self.w=sympy.eye(2)*sigma_a#输入噪声
        self.Q=Matrix([[dt**2/2,0],
                       [0,dt**2/2],
                       [dt,0],
                       [0,dt],
                       [1,0],
                       [0,1]])#状态转移方程
        self.Q=self.Q*self.w*self.Q.T
        self.R=np.mat(np.eye(num_Anchor)*sigma_r)
        
        x_error,y_error,vx_error,vy_error,x_imu,y_imu,ax_error,ay_error=symbols('x_error,y_error,vx_error,vy_error,x_imu,y_imu,ax_error,ay_error')
        self.Z_f=sympy.zeros(num_Anchor,1)
        for i in range(num_Anchor):
            derr=sympy.sqrt((x_imu+x_error-Anchor_pos[i,0])**2 + (y_imu+y_error-Anchor_pos[i,1])**2)
            dimu=sympy.sqrt((x_imu-Anchor_pos[i,0])**2 + (y_imu-Anchor_pos[i,1])**2)
            self.Z_f[i,0]=derr-dimu
        self.H_jac=self.Z_f.jacobian(Matrix([x_error, y_error, vx_error, vy_error,ax_error,ay_error]))#得到观测方程的雅可比矩阵
        
        #保存一些要用的数据，比如上次的值以及一些常量
        self.XLastEsti=array([[0,0,0,0,0,0]]).T
        self.StatusLast=sympy.zeros(4,1)
        self.lastaccel=[0,0]
        self.dt_imu=dt_imu
        self.dt_uwb=dt_uwb
        self.step=dt_uwb/dt_imu
        self.P=np.diag([.1, .1, .1, .1, .1, .1])#初始化协方差    
        self.ac_num=num_Anchor
        self.ac_pos=Anchor_pos
#        self.A_mat=np.mat(np.zeros((num_Anchor-1,2)))
#        self.B_mat=np.mat(np.zeros((num_Anchor-1,1)))
#        self.realACdis=np.zeros(num_Anchor-1)
#        for i in range(num_Anchor-1):
#            self.A_mat[i]=self.ac_pos[i+1]-self.ac_pos[0]
#            self.realACdis[i]=np.sqrt((self.ac_pos[i+1][0]-self.ac_pos[0][0])**2+(self.ac_pos[i+1][1]-self.ac_pos[0][1])**2)
    def LSQ_TOA(self,uwbdis):
#            for i in range(self.ac_num-1):
#                self.B_mat[i][0]=(uwbdis[0]**2+self.realACdis[i]**2-uwbdis[i+1]**2)/2
#            tmp=((self.A_mat.T*self.A_mat).I*self.A_mat.T*self.B_mat).T+self.ac_pos[0]
#            toapos=[0,0]
#            toapos[0]=tmp[0,0]
#            toapos[1]=tmp[0,1]
#            return toapos
        notzeroidx=[idx for idx, e in enumerate(uwbdis) if e!=0]
        zerocount=list(uwbdis).count(0)
        ac_pos=self.ac_pos[notzeroidx]
        uwbdis=uwbdis[notzeroidx]      
        num_Anchor=self.ac_num-zerocount
        realACdis=np.zeros(num_Anchor-1)
        A_mat=np.mat(np.zeros((num_Anchor-1,2)))
        for i in range(num_Anchor-1):
            A_mat[i]=ac_pos[i+1]-ac_pos[0]
            realACdis[i]=np.sqrt((ac_pos[i+1][0]-ac_pos[0][0])**2+(ac_pos[i+1][1]-ac_pos[0][1])**2)
        B_mat=np.mat(np.zeros((num_Anchor-1,1)))        
        for i in range(num_Anchor-1):
            B_mat[i][0]=(uwbdis[0]**2+realACdis[i]**2-uwbdis[i+1]**2)/2
        tmp=((A_mat.T*A_mat).I*A_mat.T*B_mat).T+ac_pos[0]
        toapos=[0,0]
        toapos[0]=tmp[0,0]
        toapos[1]=tmp[0,1]
        return toapos
    def H_jac_cacu(self,Xpre,imu_pos):#输入的参数是状态变量的先验估计，返回雅可比矩阵
        x_error,y_error,vx_error,vy_error=symbols('x_error,y_error,vx_error,vy_error')
        x_imu,y_imu=symbols('x_imu,y_imu')
        dic={x_error:Xpre[0],y_error:Xpre[1],vx_error:Xpre[2],vy_error:Xpre[3],x_imu:imu_pos[0],y_imu:imu_pos[1]}
        H_jaco_nummat = array(self.H_jac.evalf(subs=dic)).astype(float)
        return H_jaco_nummat
    def imupos(self,accel_array):
        lastx=self.StatusLast[0,0]
        lasty=self.StatusLast[1,0]
        lastvx=self.StatusLast[2,0]
        lastvy=self.StatusLast[3,0]
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
        lastvx=self.StatusLast[2,0]
        lastvy=self.StatusLast[3,0]   
        for vel in vx:
            lastx=lastx+(vel+lastvx)*self.dt_imu/2
            lastvx=vel
            
        for vel in vy:
            lasty=lasty+(vel+lastvy)*self.dt_imu/2
            lastvy=vel
        tag_pos_imu[0]=lastx
        tag_pos_imu[1]=lasty
        self.StatusLast[0,0]=lastx
        self.StatusLast[1,0]=lasty
        self.StatusLast[2,0]=lastvx
        self.StatusLast[3,0]=lastvy
        
        return tag_pos_imu
    def Z_observe(self,uwb_dis,tag_pos_imu):
        Zobs=sympy.zeros(self.ac_num,1)
        for i in range(self.ac_num):
            imudiff2=(tag_pos_imu[0]-self.ac_pos[i,0])**2+(tag_pos_imu[1]-self.ac_pos[i,1])**2
            if uwb_dis[i]!=0:
                Zobs[i,0]=uwb_dis[i]-sympy.sqrt(imudiff2)
            else:
                Zobs[i,0]=0
#        Zobs[1,0]=0
        return Zobs
    def ekffilter(self,accel_array,uwb_dis):
        if self.StatusLast[0]==0:
            
            #计算位置，使用TOA
            tag_pos=self.LSQ_TOA(uwb_dis)
            self.StatusLast[0,0]=tag_pos[0]
            self.StatusLast[1,0]=tag_pos[1]
            return tag_pos
        tag_pos_imu=self.imupos(accel_array)
        
        Zobs=self.Z_observe(uwb_dis,tag_pos_imu)
        
        dt=symbols('dt')
        thesubs={dt:self.dt_uwb}
        F_nummat = array(self.F.evalf(subs=thesubs)).astype(float)
        '''
        这里需要进一步的分析，是重新归零初值还是继承上一时刻的值
        '''
        Xpre=array([[0,0,0,0,0,0]]).T
        #Xpre=self.XLastEsti
        Xpre=dot(F_nummat,Xpre)
        
        thesubs={dt:self.dt_uwb}
        F_nummat = array(self.F.evalf(subs=thesubs)).astype(float)
        Q_nummat = array(self.Q.evalf(subs=thesubs)).astype(float)     
        self.P=dot(F_nummat,self.P).dot(F_nummat.T)+Q_nummat#更新先验协方差
        H=self.H_jac_cacu(Xpre,tag_pos_imu)#cacu H
        S = dot3(H, self.P, H.T) + self.R
        K = dot3(self.P, H.T, S.I)#cacu K
        
        Zpre=dot(H,Xpre)
        Xest=Xpre+Matrix(K)*(Zobs-Zpre)
        self.StatusLast=self.StatusLast+Xest[:4,0]
        tag_pos=[0,0]
        tag_pos[0]=self.StatusLast[0,0]
        tag_pos[1]=self.StatusLast[1,0]
        I_KH=np.eye(6)-dot(K,H)
        self.P=dot(I_KH,self.P)
#        self.XLastEsti=np.array(Xest).astype(np.float64)
        return tag_pos