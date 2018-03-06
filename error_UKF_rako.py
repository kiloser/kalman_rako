# -*- coding: utf-8 -*-
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import JulierSigmaPoints
import numpy as np
from sympy import symbols,Matrix
from scipy import constants as C
import matplotlib.pyplot as plt
import math
import statsmodels.api as sm
def cdfsolve(p,cdffun):
    initx=0
    while cdffun(initx)<p:
        initx+=0.005
    return initx     
def get_TDOAGDOP(pos,acpos,acnum):
    cx=[]
    cy=[]
    for i in range(acnum):
        r=np.sqrt((acpos[i,0]-pos[0])**2+(acpos[i,1]-pos[1])**2)
        if r==0:
            cx.append(0)
            cy.append(0)
        else:
            cx.append((pos[0]-acpos[i,0])/r)
            cy.append((pos[1]-acpos[i,1])/r)
        
    C=np.zeros((acnum-1,2))
    for i in range(acnum-1):
        
        C[i,0]=cx[i+1]-cx[0] 
        C[i,1]=cy[i+1]-cy[0] 
    C=np.mat(C)    
    B=(C.T*C).I*C.T
    sigma=0.1**2
    P=np.eye(acnum-1)*3*sigma+np.ones(acnum-1)*sigma
#    P=np.eye(acnum-1)*sigma
    GDOP=B*np.mat(P)*B.T
    return np.sqrt(np.trace(GDOP))
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
        idx=np.argsort(arrivetime)
        arrivetime=np.take(arrivetime,idx)
        base_pos=np.zeros((self.acnum,2))
        for i in range(self.acnum):
            base_pos[i,:]=self.acpos[idx[i],:]
        for i in range(self.acnum-1,-1,-1):
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
            
        ms0[0] = ms0[0] + self.acpos[idx[0],0]
        ms0[1] = ms0[1] + self.acpos[idx[0],1]
    
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
        dis_diff=C.c*dis_diff
        x_imu=tag_pos_imu[0]
        y_imu=tag_pos_imu[1]
        dimu=[]
        z_obs=[]
        for pos in self.acpos:
            dimu.append(np.sqrt((pos[0]-x_imu)**2+(pos[1]-y_imu)**2))
        dimu=np.array(dimu)
        dis_imu=np.zeros((Anchor_num,Anchor_num))
        for i in range(self.acnum):
            dis_imu[:,i]=dimu-dimu[i]
        diff_mat=dis_diff-dis_imu
        errsum=[]
        '''
        针对误差最大的那个数据我们丢弃
        '''
        for row in diff_mat:
            errsum.append(sum(map(abs,row)))
        idx=np.argsort(errsum)
#        diff_mat[idx[0],:]=diff_mat[idx[0],:]/2
#        diff_mat[:,idx[0]]=diff_mat[:,idx[0]]/2
        diff_mat[:,idx[0]]=0
        diff_mat[idx[0],:]=0
        
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
    
dt_IMU=0.01
dt_UBW=1



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
    Anchor_num=5
    Anchor_pos=np.array([[0,0],
                         [50,0],
                         [65.4,47.5],
                         [25,76.9],
                         [-15.4,47.5]])
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

    for i in range(2):
        acceldata[:,i]=acceldata[:,i]+np.random.normal(0,std_a,tagpoint_len*100)
        
    uwbdis_data=np.zeros((tagpoint_len,Anchor_num))
    realdis_data=np.zeros((tagpoint_len,Anchor_num))
    arrivetime_data=np.zeros((tagpoint_len,Anchor_num))
    for j in range(len(tagposlist)):
        realdis=np.zeros(Anchor_num)
        tgpos=tagposlist[j]
        for i in range(Anchor_num):
            realdis[i]=np.sqrt((tgpos[0]-Anchor_pos[i][0])**2+(tgpos[1]-Anchor_pos[i][1])**2)    
            realdis_data[j,i]=realdis[i]
            uwbdis_data[j,i]=realdis[i]+np.random.normal(0,std_r,1)
            arrivetime_data[j,i]=uwbdis_data[j,i]/C.c
            
    ukf=UKF_rako(sigma_a,sigma_r,Anchor_pos,Anchor_num,dt_IMU,dt_UBW)
      
    plot_x=[]
    plot_y=[]
    
    plt.ion()
    #ax.set_xlim(-1,11)
    #ax.set_ylim(-1,11)
    for data in arrivetime_data:
        temp=ukf.chan_algorithm(data)
        temp=np.array(temp).ravel()
        plot_x.append(temp[0])
        plot_y.append(temp[1])
#        ax4.scatter(temp[0],temp[1],marker='^',c='blue',s=3,label='TDOA estimated position')
#            plt.pause(0.001)
#        ax4.scatter(plot_x,plot_y,marker='^',c='blue',s=3,label='TDOA estimated position')    
    ax4.plot(plot_x,plot_y,c='blue',linewidth=1,label='TDOA estimated position')    
    plot_x2=[]
    plot_y2=[]
    for i in range(len(tagposlist)):
        temp=ukf.ukf_filter(acceldata[100*i:100*(i+1),:],arrivetime_data[i])
        plot_x2.append(temp[0])
        plot_y2.append(temp[1])
#    ax4.scatter(plot_x2,plot_y2,marker='o',c='green',s=3,label='DF estimated position')       
#    ax4.scatter(np.array(tagposlist)[:,0],np.array(tagposlist)[:,1],marker='.',c='r',s=3,label='real position')
    ax4.plot(plot_x2,plot_y2,c='green',linewidth=1,label='DF estimated position')       
    ax4.plot(np.array(tagposlist)[:,0],np.array(tagposlist)[:,1],c='r',linewidth=1,label='real position')
    ax4.set_position([0.1,0.1,0.8,0.7])
    fig4.legend(loc='upper left')
    ax4.set_title("Acceleration motion analysis")        
    ax4.set_xlabel('x-axis(m)')
    ax4.set_ylabel('y-axis(m)')

if idx=='2':
#    Anchor_num=5
#    Anchor_pos=np.array([[0,0],
#                     [20,0],
#                     [0,20],
#                     [20,20],
#                     [15,6]])
    Anchor_num=5
    Anchor_pos=np.array([[0,0],
                         [50,0],
                         [65.4,47.5],
                         [25,76.9],
                         [-15.4,47.5]])  
    Anchor_pos=Anchor_pos/2
    fig5=plt.figure(5)
    ax5=fig5.add_subplot(111)
    ax5.scatter(Anchor_pos[:,0],Anchor_pos[:,1],marker='o',c='black',s=6)        
    tagposlist=[]
    tagpoint_len=200
    a=0.025
    v=a*tagpoint_len/(2*np.pi)
    t=np.linspace(0,2*np.pi,tagpoint_len*100)
    ax=a*np.sin(t)
    ay=a*np.cos(t)
    acceldata=np.zeros((tagpoint_len*100,2))
    acceldata[:,0]=ax
    acceldata[:,1]=ay
    initstat=[-12,0]+[-v,0]
    tagposlist=imutrace(initstat,acceldata)
    
    for i in range(2):
        acceldata[:,i]=acceldata[:,i]+np.random.normal(0,std_a,tagpoint_len*100)
    uwbdis_data=np.zeros((tagpoint_len,Anchor_num))
    realdis_data=np.zeros((tagpoint_len,Anchor_num))
    arrivetime_data=np.zeros((tagpoint_len,Anchor_num))
    for j in range(len(tagposlist)):
        realdis=np.zeros(Anchor_num)
        tgpos=tagposlist[j]
        for i in range(Anchor_num):
            realdis[i]=np.sqrt((tgpos[0]-Anchor_pos[i][0])**2+(tgpos[1]-Anchor_pos[i][1])**2)    
            realdis_data[j,i]=realdis[i]
            uwbdis_data[j,i]=realdis[i]+np.random.normal(0,std_r,1)
            arrivetime_data[j,i]=uwbdis_data[j,i]/C.c
            
    ukf=UKF_rako(sigma_a,sigma_r,Anchor_pos,Anchor_num,dt_IMU,dt_UBW)        
    plot_x=[]
    plot_y=[]
    GDOPvalue=[]
    plt.ion()
    #ax.set_xlim(-1,11)
    #ax.set_ylim(-1,11)
    for data in arrivetime_data:
        temp=ukf.chan_algorithm(data)
        temp=np.array(temp).ravel()
        plot_x.append(temp[0])
        plot_y.append(temp[1])
        
#            plt.pause(0.001)
#        ax1.scatter(plot_x,plot_y,marker='^',c='blue',s=3)    
    plot_x2=[]
    plot_y2=[]
    for i in range(len(tagposlist)):
        temp=ukf.ukf_filter(acceldata[100*i:100*(i+1),:],arrivetime_data[i])
        plot_x2.append(temp[0])
        plot_y2.append(temp[1])
        GDOPvalue.append(get_TDOAGDOP(tagposlist[i],Anchor_pos,Anchor_num))
#    ax5.scatter(plot_x,plot_y,marker='^',c='blue',s=3)
#    ax5.scatter(plot_x2,plot_y2,marker='o',c='green',s=3)       
#    ax5.scatter(np.array(tagposlist)[:,0],np.array(tagposlist)[:,1],marker='.',c='r',s=3)
    ax5.plot(plot_x,plot_y,linewidth=1,c='blue',label='TDOA estimated position')
    ax5.plot(plot_x2,plot_y2,linewidth=1,c='green',label='DF estimated position')       
    ax5.plot(np.array(tagposlist)[:,0],np.array(tagposlist)[:,1],linewidth=1,c='r',label='real position')
    ax5.legend()    
    ax5.set_title("Circular motion analysis")        
    ax5.set_xlabel('x-axis(m)')
    ax5.set_ylabel('y-axis(m)')    


    plot_x=np.array(plot_x,dtype='float')
    plot_y=np.array(plot_y,dtype='float')
    plot_x2=np.array(plot_x2,dtype='float')
    plot_y2=np.array(plot_y2,dtype='float')
    tagposlist=np.array(tagposlist,dtype='float')
    
    toa_xerr=plot_x-tagposlist[:,0]
    toa_yerr=plot_y-tagposlist[:,1]
    df_xerr=plot_x2-tagposlist[:,0]
    df_yerr=plot_y2-tagposlist[:,1]
    toa_rerr=np.sqrt(toa_xerr**2+toa_yerr**2)
    df_rerr=np.sqrt(df_xerr**2+df_yerr**2)

    
    fig1=plt.figure(1)
    ax1=fig1.add_subplot(111)
    ax1.plot(GDOPvalue,toa_xerr,label='tdoa_xerr')
    ax1.plot(GDOPvalue,df_xerr,label='df_xerr')
    ax1.set_title('Circular motion x-axis error')
    ax1.set_xlabel('GDOP value')
    ax1.set_ylabel('x error')
    ax1.legend(loc='upper left')
    
    fig3=plt.figure(3)
    ax3=fig3.add_subplot(111)
    ax3.plot(GDOPvalue,toa_yerr,label='tdoa_yerr')
    ax3.plot(GDOPvalue,df_yerr,label='df_yerr')
    ax3.set_title('Circular motion y-axis error')
    ax3.set_xlabel('GDOP value')
    ax3.set_ylabel('y error')
    ax3.legend(loc='upper left')
    
    fig4=plt.figure(4)
    ax4=fig4.add_subplot(111)
    ax4.plot(GDOPvalue,toa_rerr,label='tdoa_rerr')
    ax4.plot(GDOPvalue,df_rerr,label='df_rerr')
    ax4.set_title('Circular motion distance error')
    ax4.set_xlabel('GDOP value')
    ax4.set_ylabel('distance error')
    ax4.legend(loc='upper left')     
    
Anchor_num=5
Anchor_pos=np.array([[0,0],
                     [5,0],
                     [6.54,4.75],
                     [2.50,7.69],
                     [-1.54,4.75]])   
Anchor_num=5
Anchor_pos=np.array([[0,0],
                     [10,0],
                     [13.08,9.5],
                     [5.0,15.38],
                     [-3.08,9.5]])  
'''
水平无加速度运动
'''            
if idx=='3':
    
    fig2=plt.figure(2)
    ax2=fig2.add_subplot(111)
    ax2.scatter(Anchor_pos[:,0],Anchor_pos[:,1],marker='o',c='black',s=6)
    tagposlist=[]
    tagpoint_len=200
    for i in np.linspace(-30,30,tagpoint_len,endpoint=False):
        tagposlist.append([i,5])
    np.array(tagposlist)
    acceldata=np.zeros((tagpoint_len*100,2))
    
    for i in range(2):
        acceldata[:,i]=acceldata[:,i]+np.random.normal(0,std_a,tagpoint_len*100)
    uwbdis_data=np.zeros((tagpoint_len,Anchor_num))
    realdis_data=np.zeros((tagpoint_len,Anchor_num))
    arrivetime_data=np.zeros((tagpoint_len,Anchor_num))
    for j in range(len(tagposlist)):
        realdis=np.zeros(Anchor_num)
        tgpos=tagposlist[j]
        for i in range(Anchor_num):
            realdis[i]=np.sqrt((tgpos[0]-Anchor_pos[i][0])**2+(tgpos[1]-Anchor_pos[i][1])**2)    
            realdis_data[j,i]=realdis[i]
            uwbdis_data[j,i]=realdis[i]+np.random.normal(0,std_r,1)
            arrivetime_data[j,i]=uwbdis_data[j,i]/C.c

    ukf=UKF_rako(sigma_a,sigma_r,Anchor_pos,Anchor_num,dt_IMU,dt_UBW)        
    plot_x=[]
    plot_y=[]
    plot_x2=[]
    plot_y2=[]  
    GDOPvalue=[]
    plt.ion()
    #ax.set_xlim(-1,11)
    #ax.set_ylim(-1,11)
    for i in range(len(tagposlist)):
        data=arrivetime_data[i]
        temp=ukf.chan_algorithm(data)
        temp=np.array(temp).ravel()
        plot_x.append(temp[0])
        plot_y.append(temp[1])
#        ax2.scatter(temp[0],temp[1],marker='^',c='blue',s=3)
        
        temp=ukf.ukf_filter(acceldata[100*i:100*(i+1),:],arrivetime_data[i])
        plot_x2.append(temp[0])
        plot_y2.append(temp[1])
#        ax2.scatter(temp[0],temp[1],marker='o',c='green',s=3)
#        plt.pause(0.1)
#        ax1.scatter(plot_x,plot_y,marker='^',c='blue',s=3)    
        GDOPvalue.append(get_TDOAGDOP(tagposlist[i],Anchor_pos,Anchor_num))
        
    ax2.scatter(plot_x,plot_y,c='blue',s=3,label='TDOA estimated position')        
    ax2.scatter(plot_x2,plot_y2,c='limegreen',s=3,label='DF estimated position')
    ax2.scatter(np.array(tagposlist)[:,0],np.array(tagposlist)[:,1],s=3,c='r',label='real position')

    plot_x=np.array(plot_x,dtype='float')
    plot_y=np.array(plot_y,dtype='float')
    plot_x2=np.array(plot_x2,dtype='float')
    plot_y2=np.array(plot_y2,dtype='float')
    tagposlist=np.array(tagposlist,dtype='float')
    
    toa_xerr=plot_x-tagposlist[:,0]
    toa_yerr=plot_y-tagposlist[:,1]
    df_xerr=plot_x2-tagposlist[:,0]
    df_yerr=plot_y2-tagposlist[:,1]
    toa_rerr=np.sqrt(toa_xerr**2+toa_yerr**2)
    df_rerr=np.sqrt(df_xerr**2+df_yerr**2)
    
#    ax2.plot(np.array(tagposlist)[:,0],np.array(tagposlist)[:,1],linewidth=1,c='r',label='real position')
    ax2.set_position([0.1,0.1,0.8,0.7])
    ax2.set_title('uniform rectilinear motion analysis')
    ax2.set_xlabel('x-axis(m)')
    ax2.set_ylabel('y-axis(m)')
    fig2.legend(loc='upper left')
    
    fig1=plt.figure(1)
    ax1=fig1.add_subplot(111)
    ax1.plot(GDOPvalue,toa_xerr,label='tdoa_xerr')
    ax1.plot(GDOPvalue,df_xerr,label='df_xerr')
    ax1.set_title('uniform rectilinear motion x-axis error')
    ax1.set_xlabel('GDOP value')
    ax1.set_ylabel('x error')
    ax1.legend(loc='upper left')
    
    fig3=plt.figure(3)
    ax3=fig3.add_subplot(111)
    ax3.plot(GDOPvalue,toa_yerr,label='tdoa_yerr')
    ax3.plot(GDOPvalue,df_yerr,label='df_yerr')
    ax3.set_title('uniform rectilinear motion y-axis error')
    ax3.set_xlabel('GDOP value')
    ax3.set_ylabel('y error')
    ax3.legend(loc='upper left')
    
    fig4=plt.figure(4)
    ax4=fig4.add_subplot(111)
    ax4.plot(GDOPvalue,toa_rerr,label='tdoa_rerr')
    ax4.plot(GDOPvalue,df_rerr,label='df_rerr')
    ax4.set_title('uniform rectilinear motion distance error')
    ax4.set_xlabel('GDOP value')
    ax4.set_ylabel('distance error')
    ax4.legend(loc='upper left')                
'''
垂直无加速度运动
'''
if idx=='4':
    fig3=plt.figure(3)
    ax3=fig3.add_subplot(111)
    ax3.scatter(Anchor_pos[:,0],Anchor_pos[:,1],marker='o',c='black',s=6)
    tagposlist=[]
    tagpoint_len=200
    for i in np.linspace(-30,30,tagpoint_len,endpoint=False):
        tagposlist.append([5,i])
    np.array(tagposlist)
    acceldata=np.zeros((tagpoint_len*100,2))
    for i in range(2):
        acceldata[:,i]=acceldata[:,i]+np.random.normal(0,std_a,tagpoint_len*100)
    uwbdis_data=np.zeros((tagpoint_len,Anchor_num))
    realdis_data=np.zeros((tagpoint_len,Anchor_num))
    arrivetime_data=np.zeros((tagpoint_len,Anchor_num))
    for j in range(len(tagposlist)):
        realdis=np.zeros(Anchor_num)
        tgpos=tagposlist[j]
        for i in range(Anchor_num):
            realdis[i]=np.sqrt((tgpos[0]-Anchor_pos[i][0])**2+(tgpos[1]-Anchor_pos[i][1])**2)    
            realdis_data[j,i]=realdis[i]
            uwbdis_data[j,i]=realdis[i]+np.random.normal(0,std_r,1)
            arrivetime_data[j,i]=uwbdis_data[j,i]/C.c
    ax3.scatter(np.array(tagposlist)[:,0],np.array(tagposlist)[:,1],marker='.',c='r',s=3)
    ukf=UKF_rako(sigma_a,sigma_r,Anchor_pos,Anchor_num,dt_IMU,dt_UBW)              
    plot_x=[]
    plot_y=[]
    
    plt.ion()
    #ax.set_xlim(-1,11)
    #ax.set_ylim(-1,11)
    for data in arrivetime_data:
        temp=ukf.chan_algorithm(data)
        temp=np.array(temp).ravel()
        plot_x.append(temp[0])
        plot_y.append(temp[1])
        ax3.scatter(temp[0],temp[1],marker='^',c='blue',s=3)
#            plt.pause(0.001)
#        ax1.scatter(plot_x,plot_y,marker='^',c='blue',s=3)    
    plot_x2=[]
    plot_y2=[]
    for i in range(len(tagposlist)):
        temp=ukf.ukf_filter(acceldata[100*i:100*(i+1),:],arrivetime_data[i])
        plot_x2.append(temp[0])
        plot_y2.append(temp[1])
    ax3.scatter(plot_x2,plot_y2,marker='o',c='green',s=3)    
    
'''
静止运动
'''

if idx=='5':   
#    Anchor_num=5
#    Anchor_pos=np.array([[0,0],
#                         [4.2,0],
#                         [1.54,7.68],
#                         [6.2,6.84],
#                         [-0.25,3.5]])
    Anchor_num=4
    Anchor_pos=np.array([[0,0],
                         [4.2,0],
                         [1.54,7.68],
                         [6.2,6.84]])
            
    fig1=plt.figure(1)
    ax1=fig1.add_subplot(111)
    ax1.scatter(Anchor_pos[:,0],Anchor_pos[:,1],marker='o',c='black',s=6)  
    tagposlist=[]
    
    toa_xstderr=[]
    toa_ystderr=[]
    toa_meanerr=[]
    df_xstderr=[]
    df_ystderr=[]
    df_meanerr=[]
    GDOPvalue=[]
    toaCEP=[]
    dfCEP=[]
    
    for i in np.linspace(6.2,25,1,endpoint=True):
        for j in np.linspace(4.71,25,1,endpoint=True):
            tagposlist.append([i,j])
    
#    x=np.linspace(5,30,200,endpoint=True)
#    y=1.37*(x-5)+7.69
#    tagposlist=[]
#    for i in range(len(x)):
#        tagposlist.append([x[i],y[i]])
    
    plot_x2s=[]
    plot_y2s=[]
    plot_xs=[]
    plot_ys=[]
    
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
        ukf=UKF_rako(sigma_a,sigma_r,Anchor_pos,Anchor_num,dt_IMU,dt_UBW)
        
        #============================================================    
        #plot the  estimate resaults
        #============================================================
        
        plot_x=[]
        plot_y=[]
        
        plt.ion()
        #ax.set_xlim(-1,11)
        #ax.set_ylim(-1,11)
        for data in arrivetime_data:
            temp=ukf.chan_algorithm(data)
            temp=np.array(temp).ravel()
            plot_x.append(temp[0])
            plot_y.append(temp[1])
#            ax1.scatter(temp[0],temp[1],marker='^',c='blue',s=3)
#            plt.pause(0.001)
#        fig1_line1=ax1.scatter(plot_x,plot_y,marker='^',c='blue',s=3) 
        
        plot_x2=[]
        plot_y2=[]
        ukf.StatusLast[0]=tgpos[0]
        ukf.StatusLast[1]=tgpos[1]
        
        for i in range(50):
            
            temp=ukf.ukf_filter(acceldata[100*i:100*(i+1),:],arrivetime_data[i])
            plot_x2.append(temp[0])
            plot_y2.append(temp[1])
#            ax1.scatter(temp[0],temp[1],marker='o',c='green',s=3)
#            plt.pause(0.001)
        
        plot_xs.extend(plot_x)
        plot_ys.extend(plot_y)
        plot_x2s.extend(plot_x2)
        plot_y2s.extend(plot_y2)
        
#        fig1_line2=ax1.scatter(plot_x2,plot_y2,marker='o',c='limegreen',s=3)
#        fig1_line3=ax1.scatter(tgpos[0],tgpos[1],marker='+',c='r',s=8)
        
        plot_x=np.array(plot_x,dtype='float')
        plot_y=np.array(plot_y,dtype='float')
        plot_x2=np.array(plot_x2,dtype='float')
        plot_y2=np.array(plot_y2,dtype='float')   
        
        plot_x_mean=np.mean(plot_x)
        plot_y_mean=np.mean(plot_y)
        toa_meanerr.append(math.sqrt((plot_x_mean-tgpos[0])**2+(plot_y_mean-tgpos[1])**2))
        toa_rerr=np.sqrt((plot_x-tgpos[0])**2+(plot_y-tgpos[1])**2)
        toa_xstderr.append(np.std(plot_x))
        toa_ystderr.append(np.std(plot_y))
        ecdf = sm.distributions.ECDF(toa_rerr)
        toaCEP.append(cdfsolve(0.8,ecdf))
        
        plot_x2_mean=np.mean(plot_x2)
        plot_y2_mean=np.mean(plot_y2)
        df_meanerr.append(math.sqrt((plot_x2_mean-tgpos[0])**2+(plot_y2_mean-tgpos[1])**2))
        df_rerr=np.sqrt((plot_x2-tgpos[0])**2+(plot_y2-tgpos[1])**2)
        df_xstderr.append(np.std(plot_x2))#I don't know why, but I have to do this shit.
        df_ystderr.append(np.std(plot_y2))
        ecdf = sm.distributions.ECDF(df_rerr)
        dfCEP.append(cdfsolve(0.8,ecdf))
        
        GDOPvalue.append(get_TDOAGDOP(tgpos,Anchor_pos,Anchor_num))
    
    tagposlist=np.array(tagposlist)
    fig1_line1=ax1.scatter(plot_xs,plot_ys,marker='^',c='blue',s=3)      
    fig1_line2=ax1.scatter(plot_x2s,plot_y2s,marker='o',c='limegreen',s=3)
    fig1_line3=ax1.scatter(tagposlist[:,0],tagposlist[:,1],marker='+',c='r',s=6)    
    
    labels=['TDOA estimated position','DF estimated position','real position']
    handles=[fig1_line1,fig1_line2,fig1_line3]
    ax1.set_position([0.1,0.1,0.8,0.7])
    ax1.set_title("Fixed position analysis")        
    fig1.legend(handles,labels,loc="upper left")
    ax1.set_xlabel('x-axis(m)')
    ax1.set_ylabel('y-axis(m)')

    fig2=plt.figure(2)
    ax2=fig2.add_subplot(111)
    ax2.plot(GDOPvalue,toa_xstderr,label='tdoa_xstderr')
    ax2.plot(GDOPvalue,df_xstderr,label='df_xstderr')
    ax2.set_title("x-axis standard error")        
    ax2.set_xlabel('GDOP value')
    ax2.set_ylabel('standerd error')
    ax2.set_position([0.1,0.1,0.8,0.7])
    ax2.legend(loc="upper left")
    
    fig3=plt.figure(3)
    ax3=fig3.add_subplot(111)
    ax3.plot(GDOPvalue,toa_ystderr,label='tdoa_ystderr')
    ax3.plot(GDOPvalue,df_ystderr,label='df_ystderr')
    ax3.set_title("y-axis standard error")
    ax3.set_xlabel('GDOP value')
    ax3.set_ylabel('standerd error')
    ax3.set_position([0.1,0.1,0.8,0.7])
    ax3.legend(loc="upper left")
    
    fig4=plt.figure(4)
    ax4=fig4.add_subplot(111)
    ax4.plot(GDOPvalue,toa_meanerr,label='tdoa_meanerr')
    ax4.plot(GDOPvalue,df_meanerr,label='df_meanerr')
    ax4.set_title("Mean error")
    ax4.set_xlabel('GDOP value')
    ax4.set_ylabel('mean error')
    ax4.set_position([0.1,0.1,0.8,0.7])
    ax4.legend(loc="upper left")
    
    fig5=plt.figure(5)
    ax5=fig5.add_subplot(111)
    ax5.plot(GDOPvalue,toaCEP,label='tdoaCEP80%')
    ax5.plot(GDOPvalue,dfCEP,label='dfCEP80%')
    ax5.set_title("80% CEP analysis")
    ax5.set_xlabel('GDOP value')
    ax5.set_ylabel('80% CEP radius')
    ax5.set_position([0.1,0.1,0.8,0.7])
    ax5.legend(loc="upper left")
    
#    
#        print(tgpos)       
if idx=='6':
#    Anchor_num=5
#    Anchor_pos=np.array([[0,0],
#                         [5,0],
#                         [6.54,4.75],
#                         [2.5,7.69],
#                         [-1.54,4.75]])
    Anchor_num=5
    Anchor_pos=np.array([[0,0],
                         [4.2,0],
                         [1.54,7.68],
                         [6.2,6.84],
                         [-0.25,3.5]])
    fig4=plt.figure(4)
    ax4=fig4.add_subplot(111)
    ax4.scatter(Anchor_pos[:,0],Anchor_pos[:,1],marker='o',c='black',s=6)
    tagpoint_len=81
    tagposlist=np.array((tagpoint_len,2))
    acceldata=np.zeros((tagpoint_len*100,2))
    ac=[1]*100+[0]*1900+[-1]*100
    ac=np.array(ac)
    ac=ac*0.5
    hold=np.zeros((300,2))
    acceldata[:2100,0]=acceldata[:2100,0]+ac
    acceldata[2000:4100,1]=acceldata[2000:4100,1]+ac
    acceldata[4000:6100,0]=acceldata[4000:6100,0]-ac
    acceldata[6000:8100,1]=acceldata[6000:8100,1]-ac
    acceldata=np.concatenate((hold,acceldata,hold),axis=0)
    
    tagpoint_len+=6
    initstat=[-1,2.5]+[0,0]
    tagposlist=imutrace(initstat,acceldata)
    tagpoint_len=len(tagposlist)

    for i in range(2):
        acceldata[:,i]=acceldata[:,i]+np.random.normal(0,std_a,tagpoint_len*100)
        
    uwbdis_data=np.zeros((tagpoint_len,Anchor_num))
    realdis_data=np.zeros((tagpoint_len,Anchor_num))
    arrivetime_data=np.zeros((tagpoint_len,Anchor_num))
    for j in range(len(tagposlist)):
        realdis=np.zeros(Anchor_num)
        tgpos=tagposlist[j]
        for i in range(Anchor_num):
            realdis[i]=np.sqrt((tgpos[0]-Anchor_pos[i][0])**2+(tgpos[1]-Anchor_pos[i][1])**2)    
            realdis_data[j,i]=realdis[i]
            uwbdis_data[j,i]=realdis[i]+np.random.normal(0,std_r,1)
            arrivetime_data[j,i]=uwbdis_data[j,i]/C.c
            
    ukf=UKF_rako(sigma_a,sigma_r,Anchor_pos,Anchor_num,dt_IMU,dt_UBW)
      
    plot_x=[]
    plot_y=[]
    
    plt.ion()
    #ax.set_xlim(-1,11)
    #ax.set_ylim(-1,11)
    for data in arrivetime_data:
        temp=ukf.chan_algorithm(data)
        temp=np.array(temp).ravel()
        plot_x.append(temp[0])
        plot_y.append(temp[1])
#        ax4.scatter(temp[0],temp[1],marker='^',c='blue',s=3,label='TDOA estimated position')
#            plt.pause(0.001)
#        ax4.scatter(plot_x,plot_y,marker='^',c='blue',s=3,label='TDOA estimated position')    
    ax4.plot(plot_x,plot_y,c='blue',linewidth=1,label='TDOA estimated position')    
    plot_x2=[]
    plot_y2=[]
    for i in range(len(tagposlist)):
        temp=ukf.ukf_filter(acceldata[100*i:100*(i+1),:],arrivetime_data[i])
        plot_x2.append(temp[0])
        plot_y2.append(temp[1])
#    ax4.scatter(plot_x2,plot_y2,marker='o',c='green',s=3,label='DF estimated position')       
#    ax4.scatter(np.array(tagposlist)[:,0],np.array(tagposlist)[:,1],marker='.',c='r',s=3,label='real position')
    ax4.plot(plot_x2,plot_y2,c='green',linewidth=1,label='DF estimated position')       
#    ax4.plot(np.array(tagposlist)[:,0],np.array(tagposlist)[:,1],c='r',linewidth=1,label='real position')
    ax4.set_position([0.1,0.1,0.8,0.7])
    fig4.legend(loc='upper left')
    ax4.set_title("Acceleration motion analysis")        
    ax4.set_xlabel('x-axis(m)')
    ax4.set_ylabel('y-axis(m)')     
    
    