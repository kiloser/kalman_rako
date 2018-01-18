# -*- coding: utf-8 -*-
import sympy 
import scipy.linalg as linalg
from sympy import symbols,Matrix
from filterpy.common import dot3
import numpy as np
from numpy import dot, array

class RAKOEKF():
    def __init__(self,sigma_a,Anchor_pos,num_Anchor,dt_imu,dt_uwb):
        #EKF.__init__(self, 4, 3, 2)#class filterpy.kalman.ExtendedKalmanFilter(dim_x, dim_z, dim_u=0)
        dt=symbols('dt')
        self.F=Matrix([[1,0,dt,0],
                     [0,1,0,dt],
                     [0,0,1,0],
                     [0,0,0,1]])#状态转移方程
        self.B=Matrix([dt**2/2,0],
                   [0,dt**2/2],
                   [dt,0],
                   [0,dt])#控制方程
        self.w=Matrix([sigma_a[0],0],
                     [0,sigma_a[1]])#输入噪声
        self.Q=self.B*self.w*self.B.T#控制输入噪声转换到状态域
        x,y,dis_uwb1,dis_uwb2,dis_uwb3=symbols('x,y,dis_uwb1,dis_uwb2,dis_uwb3')
        self.Z_f=Matrix([[sympy.sqrt((x-Anchor_pos[0,0])**2 + (y-Anchor_pos[0,1])**2)-dis_uwb1],
                        [sympy.sqrt((x-Anchor_pos[1,0])**2 + (y-Anchor_pos[1,1])**2)-dis_uwb2],
                        [sympy.sqrt((x-Anchor_pos[2,0])**2 + (y-Anchor_pos[2,1])**2)-dis_uwb3]])#观测方程
        vx,vy=symbols('vx,vy')
        self.H_jac=self.Z_f.jacobian(Matrix([x, y, vx, vy]))#得到观测方程的雅可比矩阵
        #保存一些要用的数据，比如上次的值以及一些常量
        self.XLastEsti=array([0],
                             [0],
                             [0],
                             [0])
        self.dt_imu=dt_imu
        self.dt_uwb=dt_uwb
        self.step=dt_uwb/dt_imu
        self.K=0;
        self.P=0;
        self.R=0;
    def update(self,Xsim,uwb_dis):
        H=self.H_jaccacu(Xsim[0:2],Xsim[2:4])
        S = dot3(H, self.P, H.T) + self.R
        self.K = dot3(self.P, H.T, linalg.inv (S))
        residual=self.Z_fcacu(Xsim,uwb_dis)-dot(H,Xsim)#事实上Xsim是和pos_imu一致的，都是要用IMU数据得到
        Xesti=Xsim+self.K*residual
        I_KH=np.eye(4)-dot(self.K,H)
        self.P=dot(I_KH,self.P)
        #在filterpy中看到了另外一种计算协方差的计算公式
        #self.P=dot3(I_KH, self.P, I_KH.T) + dot3(self.K, self.R, self.K.T)
        self.XLastEsti=Xesti
        return Xesti
        
    def H_jaccacu(self,pos,vel):#输入的参数是状态变量的先验估计，返回雅可比矩阵的数值
        x,y=symbols('x,y')
        vx,vy=symbols('vx,vy')
        H_jaco_nummat = array(self.H_jac.evalf(subs={x:pos[0],y:pos[1],vx:vel[0],vy:vel[1]})).astype(float)
        return H_jaco_nummat
    def Z_fcacu(self,pos_imu,uwb_dis):#输入的是IMU推算的坐标和超宽带定位得到的距离
        x,y,dis_uwb1,dis_uwb2,dis_uwb3=symbols('x,y,dis_uwb1,dis_uwb2,dis_uwb3')
        thesubs={x:pos_imu[0],y:pos_imu[1],dis_uwb1:uwb_dis[0],dis_uwb2:uwb_dis[1],dis_uwb3:uwb_dis[2]}
        Z_f_nummat = array(self.Z_f.evalf(subs=thesubs)).astype(float)
        return Z_f_nummat
    def predict(self,accel_array):#预测新的状态变量并且更新协方差矩阵
        dt=symbols('dt')
        thesubs={dt:self.dt_imu}
        F_nummat = array(self.F.evalf(subs=thesubs)).astype(float)
        B_nummat = array(self.B.evalf(subs=thesubs)).astype(float)
        Xsim=self.XLastEsti
        for accel in accel_array:
            Xsim=dot(F_nummat,Xsim)+dot(B_nummat,accel)#根据IMU数据计算预测状态变量 
        thesubs={dt:self.dt_ubw}
        F_nummat = array(self.F.evalf(subs=thesubs)).astype(float)
        Q_nummat = array(self.Q.evalf(subs=thesubs)).astype(float)     
        self.P=dot(F_nummat,self.P).dot(F_nummat.T)+Q_nummat#更新先验协方差
        return Xsim

           
def get_accel(i):
    accel_array
    return accel_array 
def get_uwb_dis(i):
    uwb_dis
    return uwb_dis

dt_IMU=0.2
dt_UBW=1
num_Anchor=3
Anchor_pos=array([[0,0],
                 [0,0],
                 [0,0]])
sigma_a=array([0,0])#初始化加速度噪声
ekf=RAKOEKF(sigma_a,Anchor_pos,num_Anchor,dt_IMU,dt_UBW)
ekf.P = np.diag([.1, .1, .1])#初始化协方差
ekf.R = np.diag([0,0,0])#观测噪声
ekf.XLastEsti=array([[0,0,0]]).T#坐标初值

for i in range(100):
    accel_array=get_accel(i)
    uwb_dis=get_uwb_dis(i)
    Xsim=ekf.predict(accel_array)
    Xesti=ekf.update(Xsim,uwb_dis)
