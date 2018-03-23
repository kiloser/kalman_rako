# -*- coding: utf-8 -*-
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import JulierSigmaPoints
import numpy as np
from sympy import symbols,Matrix
from scipy import constants as C

class WirelessSync():
    '''
    以一号基站作为基准0时返回到达时间差
    逻辑优点复杂
    建议参考WirelssSync_Logic.xmind
    '''
    def __init__(self,acnum,acpos,tbpos,tbTref):
        self.maxtime=17207356974694.4*1e-12
        self.acnum=acnum
        self.Tref=np.zeros(acnum)
        self.LastTs=np.zeros(acnum)
        self.Tbias=np.zeros(acnum)
        self.isinit=0
        self.tbTref=tbTref
        for i in range(acnum):
            self.Tbias[i]=np.linalg.norm(acpos[i]-tbpos)/C.c
    def ts_sub(self,ts1,ts2):
        if ts1<ts2:
            return ts1+self.maxtime-ts2
        else:
            return ts1-ts2
    def ts_add(self,ts1,ts2):
        if ts1+ts2>=self.maxtime:
            return ts1+ts2-self.maxtime
        else:
            return ts1+ts2
    def ts_recter(self,tagid,Ts):
        if self.isinit:
            if tagid:
                Ts_rect=np.zeros(self.acnum)
                for i in range(self.acnum):
                    if Ts[i]:
                        Ts_rect[i]=self.ts_sub(Ts[i],self.LastTs[i])*self.tbTref/self.Tref[i]+self.Tbias[i]
                    else:
                        Ts_rect[i]=0
                return Ts_rect
            else:
                for i in range(self.acnum):
                    if Ts[i]:
                        self.Tref[i]=0.9*self.Tref[i]+0.1*Ts[i]
                        self.LastTs[i]=Ts[i]
                    else:
                        self.LastTs[i]=self.ts_add(self.LastTs[i],self.Tref[i])
                return 0
        else:
            if tagid:
                pass
            else:
                for i in range(self.acnum):
                    if self.LastTs[i]:
                        if self.Tref[i]:
                            if Ts[i]:
                                self.Tref[i]=0.9*self.Tref[i]+0.1*Ts[i]
                                self.LastTs[i]=Ts[i]
                            else:
                                self.LastTs[i]=self.ts_add(self.LastTs[i],self.Tref[i])
                        else:
                            if Ts[i]:
                                self.Tref[i]=self.ts_sub(Ts[i],self.LastTs[i])
                                self.LastTs[i]=Ts[i]
                            else:
                                self.LastTs[i]=0
                    else:
                        if self.Tref[i]:
                            print("it is impossible,something error")
                            return 1
                        else:
                            self.LastTs[i]=Ts[i]
            if list(self.Tref==0).count(True)==0:
                self.isinit=1
            return 0        
    
class UKF_rako(UKF):
    def __init__(self,sigma_a,sigma_r,Anchor_pos,Anchor_num,dt_imu,dt_uwb):
        x_dim=6
        z_dim=int(Anchor_num*(Anchor_num-1)/2)
        sigmas=MerweScaledSigmaPoints(n=x_dim,alpha=.1, beta=2., kappa=3-x_dim)
#        sigmas=JulierSigmaPoints(6,-3)
        super(UKF_rako,self).__init__(x_dim,z_dim,dt_uwb,fx=self.f_cv,hx=self.h_cv,points=sigmas)
        
        self.w=np.eye(2)*sigma_a#输入噪声
        self.Q_t=np.mat([[dt_uwb**2/2,0],
                       [0,dt_uwb**2/2],
                       [dt_uwb,0],
                       [0,dt_uwb],
                       [1,0],
                       [0,1]])#状态转移方程
        self.Q=np.array(self.Q_t*self.w*self.Q_t.T)
        self.R=np.eye(z_dim)*3*sigma_r+np.ones(z_dim)*sigma_r
        self.acpos=Anchor_pos
        self.acnum=Anchor_num
        self.StatusLast=np.zeros(4)
        self.x=np.zeros(6)
        self.lastaccel=[0,0]
        self.dt_imu=dt_imu
        self.dt_uwb=dt_uwb
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
        dimu=[]
        derr=[]
        for pos in self.acpos:
            dimu.append(np.sqrt((pos[0]-x_imu)**2+(pos[1]-y_imu)**2))
            derr.append(np.sqrt((pos[0]-x_imu-xerror)**2+(pos[1]-y_imu-yerror)**2))
        z=[]
        dimu=np.array(dimu)
        derr=np.array(derr)
        for i in range(0,self.acnum-1):
            for j in range(i+1,self.acnum):
                z.append((derr[j]-derr[i])-(dimu[j]-dimu[i]))
                
        return np.array(z)
                
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
        idx=[0]
        notzeroidx=[idx for idx, e in enumerate(arrivetime) if e!=0]
        zerocount=list(arrivetime).count(0)
        acnum=self.acnum-zerocount
        acpos=self.acpos[notzeroidx,:]
        arrivetime=arrivetime[notzeroidx]
        
        idx=np.argsort(arrivetime)
        arrivetime=np.take(arrivetime,idx)
        base_pos=np.zeros((acnum,2))
        for i in range(acnum):
            base_pos[i,:]=acpos[idx[i],:]
        for i in range(acnum-1,-1,-1):
            base_pos[i,:]=base_pos[i,:]-base_pos[0,:]
        evVal=np.concatenate((np.mat(arrivetime).T,base_pos),axis=1)
#        evVal=np.concatenate((np.mat(arrivetime).T,self.acpos),axis=1)
        row, column = evVal.shape  # 行，列
        baseX = evVal[:, 1]  # 列向量
        baseY = evVal[:, 2]
    
        ri1 = C.c*(evVal[:, 0] - evVal[0, 0])[1:]  # 第i个基站和第一个基站之间的距离gui
        xi1 = (baseX - baseX[0])[1:]
        yi1 = (baseY - baseY[0])[1:]
        Standaraddeviation = 0.1
    
        k = np.zeros(row)
        for i in range(0, row):
            k[i] = baseX[i] ** 2 + baseY[i] ** 2
        k = np.mat(k).T
    
        h = np.zeros((row-1, 1))
        for i in range(0, row-1):
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
    
#        P1 = C.c ** 2 * B1 * Q * B1
        P1 = B1 * Q * B1
        Za1 = (Ga.T * P1.I * Ga).I * Ga.T * P1.I * h
        C0 = (Ga.T * P1.I * Ga).I
    
        h1 = np.zeros((3, 1))
#        h1[0] = (Za1[0] - baseX[0]) ** 2
#        h1[1] = (Za1[1] - baseY[0]) ** 2
        h1[0] = (Za1[0]) ** 2
        h1[1] = (Za1[1]) ** 2      
        h1[2] = (Za1[2]) ** 2
        h1 = np.mat(h1)
    
        Ga1 = np.mat([[1, 0], [0, 1], [1, 1]])
#        r1 = np.sqrt((baseX[0] - Za1[0]) ** 2 + (baseY[0] - Za1[1]) ** 2)
    
        B2 = np.zeros((3, 3))
#        B2[0, 0] = Za1[0] - baseX[0]
#        B2[1, 1] = Za1[1] - baseY[0]
        B2[0, 0] = Za1[0]
        B2[1, 1] = Za1[1]    
#        B2[2, 2] = r1
        B2[2, 2] = np.sqrt(Za1[0]**2+Za1[1]**2)
        B2 = np.mat(B2)
    
        P2 = 4 * B2 * C0 * B2
        Za2 = (Ga1.T * P2.I * Ga1).I * Ga1.T * P2.I * h1
    
        ms0 = np.sqrt(np.abs(Za2))
        if Za[0]<0:
            ms0[0]=-abs(ms0[0])
        if Za[1]<0:
            ms0[1]=-abs(ms0[1])
            
        ms0[0] = ms0[0] + acpos[idx[0],0]
        ms0[1] = ms0[1] + acpos[idx[0],1]
    
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
        zeroidx=[idx for idx, e in enumerate(arrive_time) if e==0]
        dis_diff=np.zeros((self.acnum,self.acnum))
        for i in range(self.acnum):
            dis_diff[:,i]=arrive_time-arrive_time[i]
        dis_diff=C.c*dis_diff
        x_imu=tag_pos_imu[0]
        y_imu=tag_pos_imu[1]
        dimu=[]
        z_obs=[]
        for pos in self.acpos:
            dimu.append(np.sqrt((pos[0]-x_imu)**2+(pos[1]-y_imu)**2))
        dimu=np.array(dimu)
        dis_imu=np.zeros((self.acnum,self.acnum))
        for i in range(self.acnum):
            dis_imu[:,i]=dimu-dimu[i]
        diff_mat=dis_diff-dis_imu
        errsum=[]
        '''
        针对丢失的时间戳数据要令观测矩阵的相关项为0
        如果数据戳都存在则针对误差最大的那个数据我们选择丢弃
        '''
        if zeroidx==[]:
            for row in diff_mat:
                errsum.append(sum(map(abs,row)))
            idx=np.argsort(errsum)
    #        diff_mat[idx[0],:]=diff_mat[idx[0],:]/2
    #        diff_mat[:,idx[0]]=diff_mat[:,idx[0]]/2
            diff_mat[:,idx[0]]=0
            diff_mat[idx[0],:]=0
        else:
            diff_mat[:,zeroidx]=0
            diff_mat[zeroidx,:]=0
#        diff_mat=diff_mat/2
        for i in range(0,self.acnum-1):
            for j in range(i+1,self.acnum):
                
                z_obs.append(diff_mat[j,i])
#                z_obs.append(C.c*(arrive_time[j]-arrive_time[i])-(dimu[j]-dimu[i]))
        for i in range(len(z_obs)):
            if abs(z_obs[i])>11:
                z_obs[i]=0
        return np.array(z_obs)        
    def ukf_filter(self,accel_array,arrive_time):
        if self.StatusLast[0]==0:
            
            #计算位置，使用TOA
            tag_pos=self.chan_algorithm(arrive_time)
            tag_pos=np.array(tag_pos).ravel()
            self.StatusLast[0]=tag_pos[0]
            self.StatusLast[1]=tag_pos[1]
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