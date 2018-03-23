# -*- coding: utf-8 -*-
import sympy 
from sympy import symbols,Matrix
from filterpy.common import dot3
import numpy as np
from numpy import dot, array
import matplotlib.pyplot as plt
import math
import statsmodels.api as sm
def cdfsolve(p,cdffun):
    initx=0
    while cdffun(initx)<p:
        initx+=0.005
    return initx        
def get_TOAGDOP(pos,acpos,acnum):
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
    C=np.zeros((acnum,2))
    for i in range(acnum):
        C[i,0]=cx[i] 
        C[i,1]=cy[i]
    C=np.mat(C)    
    B=(C.T*C).I*C.T
    sigma=0.1**2
    P=np.eye(acnum)*sigma
#    P=np.eye(acnum-1)*sigma
    GDOP=B*np.mat(P)*B.T
    return np.sqrt(np.trace(GDOP))

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
        
    def LSQ_TOA(self,uwbdis):
        
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
             

dt_IMU=0.01
dt_UBW=1
Anchor_num=4
Anchor_pos=array([[0,0],
                 [10,0],
                 [0,10],
                 [10,10]])
plt.ion() 

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


'''
变加速度运动
'''
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
    
    for i in range(2):
        acceldata[:,i]=acceldata[:,i]+np.random.normal(0,std_a,tagpoint_len*100)
    uwbdis_data=np.zeros((tagpoint_len,Anchor_num))
    realdis_data=np.zeros((tagpoint_len,Anchor_num))
    for j in range(len(tagposlist)):
        realdis=np.zeros(Anchor_num)
        tgpos=tagposlist[j]
        for i in range(Anchor_num):
            realdis[i]=np.sqrt((tgpos[0]-Anchor_pos[i][0])**2+(tgpos[1]-Anchor_pos[i][1])**2)    
            realdis_data[j,i]=realdis[i]
            uwbdis_data[j,i]=realdis[i]+np.random.normal(0,std_r,1)
            
    ekf=RAKOEKF(sigma_a,sigma_r,Anchor_pos,Anchor_num,dt_IMU,dt_UBW)
    ekf.P = np.diag([.1, .1, .1, .1, .1, .1])#初始化协方差
    
    plot_x=[]
    plot_y=[]
    #ax.set_xlim(-1,11)
    #ax.set_ylim(-1,11)
    for data in uwbdis_data:
        temp=ekf.LSQ_TOA(data)
        plot_x.append(temp[0])
        plot_y.append(temp[1])
    ax4.plot(plot_x,plot_y,linewidth=1,c='blue',label='TOA estimated position')
    plt.show()   
    
    plot_x2=[]
    plot_y2=[]
    for i in range(len(tagposlist)):
        temp=ekf.ekffilter(acceldata[100*i:100*(i+1),:],uwbdis_data[i])
        plot_x2.append(temp[0])
        plot_y2.append(temp[1])
    ax4.plot(plot_x2,plot_y2,linewidth=1,c='limegreen',label='DF estimated position')
    ax4.plot(np.array(tagposlist)[:,0],np.array(tagposlist)[:,1],linewidth=1,c='r',label='real position')
    
    ax4.set_title('acceleration motion analysis')
    ax4.set_xlabel('x-axis(m)')
    ax4.set_ylabel('y-axis(m)')
    ax4.legend(loc='upper left')    

    
    plot_x=np.array(plot_x,dtype='float')
    plot_y=np.array(plot_y,dtype='float')
    plot_x2=np.array(plot_x2,dtype='float')
    plot_y2=np.array(plot_y2,dtype='float')
    tagposlist=np.array(tagposlist,dtype='float')
    
#    std1=np.std(np.array(plot_x).astype(np.float64)-np.array(tagposlist).astype(np.float64)[:,0])
#    std2=np.std(np.array(plot_y).astype(np.float64)-np.array(tagposlist).astype(np.float64)[:,1])
#    std3=np.std(np.array(plot_x2).astype(np.float64)-np.array(tagposlist).astype(np.float64)[:,0])
#    std4=np.std(np.array(plot_y2).astype(np.float64)-np.array(tagposlist).astype(np.float64)[:,1])
#    print('变加速度运动')
#    print('rawdata x std:'+str(std1))
#    print('rawdata y std:'+str(std2))
#    print('kalman x std:'+str(std3))
#    print('kalman y std:'+str(std4))        
#    

'''
圆周运动
'''
if idx=='2':
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
    for j in range(len(tagposlist)):
        realdis=np.zeros(Anchor_num)
        tgpos=tagposlist[j]
        for i in range(Anchor_num):
            realdis[i]=np.sqrt((tgpos[0]-Anchor_pos[i][0])**2+(tgpos[1]-Anchor_pos[i][1])**2)    
            realdis_data[j,i]=realdis[i]
            uwbdis_data[j,i]=realdis[i]+np.random.normal(0,std_r,1)
            
    ekf=RAKOEKF(sigma_a,sigma_r,Anchor_pos,Anchor_num,dt_IMU,dt_UBW)
    ekf.P = np.diag([.1, .1, .1, .1, .1, .1])#初始化协方差
    
    plot_x=[]
    plot_y=[]
    #ax.set_xlim(-1,11)
    #ax.set_ylim(-1,11)
    for data in uwbdis_data:
        temp=ekf.LSQ_TOA(data)
        plot_x.append(temp[0])
        plot_y.append(temp[1])
    ax5.plot(plot_x,plot_y,linewidth=1,c='blue',label='TOA estimated position')
    plt.show()   
    
    plot_x2=[]
    plot_y2=[]
    GDOPvalue=[]
    for i in range(len(tagposlist)):
        temp=ekf.ekffilter(acceldata[100*i:100*(i+1),:],uwbdis_data[i])
        plot_x2.append(temp[0])
        plot_y2.append(temp[1])
        GDOPvalue.append(get_TOAGDOP(tagposlist[i],Anchor_pos,Anchor_num))
    ax5.plot(plot_x2,plot_y2,linewidth=1,c='limegreen',label='DF estimated position')
    ax5.plot(np.array(tagposlist)[:,0],np.array(tagposlist)[:,1],linewidth=1,c='r',label='real position')
    
    ax5.set_title('circular motion analysis')
    ax5.set_xlabel('x-axis(m)')
    ax5.set_ylabel('y-axis(m)')
    ax5.legend(loc='upper left')  
    
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
    ax1.plot(GDOPvalue,toa_xerr,label='toa_xerr')
    ax1.plot(GDOPvalue,df_xerr,label='df_xerr')
    ax1.set_title('circular motion x-axis error')
    ax1.set_xlabel('GDOP value')
    ax1.set_ylabel('x error')
    ax1.legend(loc='upper left')
    
    fig3=plt.figure(3)
    ax3=fig3.add_subplot(111)
    ax3.plot(GDOPvalue,toa_yerr,label='toa_yerr')
    ax3.plot(GDOPvalue,df_yerr,label='df_yerr')
    ax3.set_title('circular motion y-axis error')
    ax3.set_xlabel('GDOP value')
    ax3.set_ylabel('y error')
    ax3.legend(loc='upper left')
    
    fig4=plt.figure(4)
    ax4=fig4.add_subplot(111)
    ax4.plot(GDOPvalue,toa_rerr,label='toa_rerr')
    ax4.plot(GDOPvalue,df_rerr,label='df_rerr')
    ax4.set_title('circular motion distance error')
    ax4.set_xlabel('GDOP value')
    ax4.set_ylabel('distance error')
    ax4.legend(loc='upper left')    
#    std1=np.std(np.array(plot_x).astype(np.float64)-np.array(tagposlist).astype(np.float64)[:,0])
#    std2=np.std(np.array(plot_y).astype(np.float64)-np.array(tagposlist).astype(np.float64)[:,1])
#    std3=np.std(np.array(plot_x2).astype(np.float64)-np.array(tagposlist).astype(np.float64)[:,0])
#    std4=np.std(np.array(plot_y2).astype(np.float64)-np.array(tagposlist).astype(np.float64)[:,1])
#    print('圆周运动')
#    print('rawdata x std:'+str(std1))
#    print('rawdata y std:'+str(std2))
#    print('kalman x std:'+str(std3))
#    print('kalman y std:'+str(std4))   
'''
水平无加速度运动
'''
if idx=='3':
#    Anchor_num=3
#    Anchor_pos=np.array([[0,0],
#                         [4.2,0],
#                         [1.54,7.68]])    
#    Anchor_pos=np.array([[0,4],
#                         [8,4],
#                         [0,6]])    
    fig2=plt.figure(2)
    ax2=fig2.add_subplot(111)
    ax2.scatter(Anchor_pos[:,0],Anchor_pos[:,1],marker='o',c='black',s=10)
    tagposlist=[]
    tagpoint_len=400
    for i in np.linspace(-60,60,tagpoint_len,endpoint=False):
        tagposlist.append([i,5])
    np.array(tagposlist)
    acceldata=np.zeros((tagpoint_len*100,2))
    
    for i in range(2):
        acceldata[:,i]=acceldata[:,i]+np.random.normal(0,std_a,tagpoint_len*100)
    uwbdis_data=np.zeros((tagpoint_len,Anchor_num))
    realdis_data=np.zeros((tagpoint_len,Anchor_num))
    for j in range(len(tagposlist)):
        realdis=np.zeros(Anchor_num)
        tgpos=tagposlist[j]
        for i in range(Anchor_num):
            realdis[i]=np.sqrt((tgpos[0]-Anchor_pos[i][0])**2+(tgpos[1]-Anchor_pos[i][1])**2)    
            realdis_data[j,i]=realdis[i]
            uwbdis_data[j,i]=realdis[i]+np.random.normal(0,std_r,1)
            
    ekf=RAKOEKF(sigma_a,sigma_r,Anchor_pos,Anchor_num,dt_IMU,dt_UBW)
    ekf.P = np.diag([.1, .1, .1, .1, .1, .1])#初始化协方差
    
    GDOPvalue=[]
    
    plot_x=[]
    plot_y=[]
    #ax.set_xlim(-1,11)
    #ax.set_ylim(-1,11)
    
    for data in uwbdis_data:
        temp=ekf.LSQ_TOA(data)
        plot_x.append(temp[0])
        plot_y.append(temp[1])
    ax2.plot(plot_x,plot_y,c='blue',linewidth=1,label='TOA estimated position')
    plt.show()   
    
    plot_x2=[]
    plot_y2=[]
    for i in range(len(tagposlist)):
        temp=ekf.ekffilter(acceldata[100*i:100*(i+1),:],uwbdis_data[i])
        plot_x2.append(temp[0])
        plot_y2.append(temp[1])
        GDOPvalue.append(get_TOAGDOP(tagposlist[i],Anchor_pos,Anchor_num))
        
        
    ax2.plot(plot_x2,plot_y2,c='limegreen',linewidth=1,label='DF estimated position')
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
    
    ax2.plot(np.array(tagposlist)[:,0],np.array(tagposlist)[:,1],linewidth=1,c='r',label='real position')
#    ax2.set_position([0.1,0.1,0.8,0.7])
    ax2.set_title('uniform rectilinear motion analysis')
    ax2.set_xlabel('x-axis(m)')
    ax2.set_ylabel('y-axis(m)')
    ax2.legend(loc='upper left')
    
    fig1=plt.figure(1)
    ax1=fig1.add_subplot(111)
    ax1.plot(GDOPvalue,toa_xerr,label='toa_xerr')
    ax1.plot(GDOPvalue,df_xerr,label='df_xerr')
    ax1.set_title('uniform rectilinear motion x-axis error')
    ax1.set_xlabel('GDOP value')
    ax1.set_ylabel('x error')
    ax1.legend(loc='upper left')
    
    fig3=plt.figure(3)
    ax3=fig3.add_subplot(111)
    ax3.plot(GDOPvalue,toa_yerr,label='toa_yerr')
    ax3.plot(GDOPvalue,df_yerr,label='df_yerr')
    ax3.set_title('uniform rectilinear motion y-axis error')
    ax3.set_xlabel('GDOP value')
    ax3.set_ylabel('y error')
    ax3.legend(loc='upper left')
    
    fig4=plt.figure(4)
    ax4=fig4.add_subplot(111)
    ax4.plot(GDOPvalue,toa_rerr,label='toa_rerr')
    ax4.plot(GDOPvalue,df_rerr,label='df_rerr')
    ax4.set_title('uniform rectilinear motion distance error')
    ax4.set_xlabel('GDOP value')
    ax4.set_ylabel('distance error')
    ax4.legend(loc='upper left')
    
#    std1=np.std(np.array(plot_x).astype(np.float64)-np.array(tagposlist).astype(np.float64)[:,0])
#    std2=np.std(np.array(plot_y).astype(np.float64)-np.array(tagposlist).astype(np.float64)[:,1])
#    std3=np.std(np.array(plot_x2).astype(np.float64)-np.array(tagposlist).astype(np.float64)[:,0])
#    std4=np.std(np.array(plot_y2).astype(np.float64)-np.array(tagposlist).astype(np.float64)[:,1])
#    print('水平无加速度运动')
#    print('rawdata x std:'+str(std1))
#    print('rawdata y std:'+str(std2))
#    print('kalman x std:'+str(std3))
#    print('kalman y std:'+str(std4))
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
    for j in range(len(tagposlist)):
        realdis=np.zeros(Anchor_num)
        tgpos=tagposlist[j]
        for i in range(Anchor_num):
            realdis[i]=np.sqrt((tgpos[0]-Anchor_pos[i][0])**2+(tgpos[1]-Anchor_pos[i][1])**2)    
            realdis_data[j,i]=realdis[i]
            uwbdis_data[j,i]=realdis[i]+np.random.normal(0,std_r,1)
            
    ekf=RAKOEKF(sigma_a,sigma_r,Anchor_pos,Anchor_num,dt_IMU,dt_UBW)
    ekf.P = np.diag([.1, .1, .1, .1, .1, .1])#初始化协方差
    
    plot_x=[]
    plot_y=[]
    #ax.set_xlim(-1,11)
    #ax.set_ylim(-1,11)
    for data in uwbdis_data:
        temp=ekf.LSQ_TOA(data)
        plot_x.append(temp[0])
        plot_y.append(temp[1])
    ax3.scatter(plot_x,plot_y,marker='^',c='blue',s=3)
    ax3.scatter(np.array(tagposlist)[:,0],np.array(tagposlist)[:,1],marker='.',c='r',s=3)
    plt.show()   
    
    plot_x2=[]
    plot_y2=[]
    for i in range(len(tagposlist)):
        temp=ekf.ekffilter(acceldata[100*i:100*(i+1),:],uwbdis_data[i])
        plot_x2.append(temp[0])
        plot_y2.append(temp[1])
    ax3.scatter(plot_x2,plot_y2,marker='o',c='green',s=3)
    
    std1=np.std(np.array(plot_x).astype(np.float64)-np.array(tagposlist).astype(np.float64)[:,0])
    std2=np.std(np.array(plot_y).astype(np.float64)-np.array(tagposlist).astype(np.float64)[:,1])
    std3=np.std(np.array(plot_x2).astype(np.float64)-np.array(tagposlist).astype(np.float64)[:,0])
    std4=np.std(np.array(plot_y2).astype(np.float64)-np.array(tagposlist).astype(np.float64)[:,1])
    print('垂直无加速度运动')
    print('rawdata x std:'+str(std1))
    print('rawdata y std:'+str(std2))
    print('kalman x std:'+str(std3))
    print('kalman y std:'+str(std4))

     
'''
静止运动
'''
if idx=='5':
#    Anchor_num=4
#    Anchor_pos=array([[5,5],
#                      [-5,5],
#                      [5,-5],
#                      [-5,-5]])
    Anchor_num=4
    Anchor_pos=np.array([[0,0],
                         [4.2,0],
                         [1.54,7.68],
                         [6.2,6.84]])
    samplecnt=50
    fig1=plt.figure(1)
    ax1=fig1.add_subplot(111)
    ax1.scatter(Anchor_pos[:,0],Anchor_pos[:,1],marker='o',c='black',s=8)  
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
#    for i in np.linspace(0,50,100,endpoint=False):
#        for j in np.linspace(0,50,1,endpoint=False):
#            tagposlist.append([i,j])
    tagposlist.append([8.42,3.87])    
    for tagpos in tagposlist:
        #============================================================    
        #let's make some fake data
        #============================================================
        tgpos=tagpos
        realdis=np.zeros(Anchor_num)
        uwbdis_data=np.zeros((samplecnt,Anchor_num))
        acceldata=np.zeros((samplecnt*100,2))
        for i in range(Anchor_num):
            realdis[i]=np.sqrt((tgpos[0]-Anchor_pos[i][0])**2+(tgpos[1]-Anchor_pos[i][1])**2)
            uwbdis_data[:,i]=realdis[i]+np.random.normal(0,std_r,samplecnt)
        for i in range(2):
            acceldata[:,i]=acceldata[:,i]+np.random.normal(0,std_a,samplecnt*100)
        #============================================================    
        #end
        #============================================================
        ekf=RAKOEKF(sigma_a,sigma_r,Anchor_pos,Anchor_num,dt_IMU,dt_UBW)
        ekf.P = np.diag([.1, .1, .1, .1, .1, .1])#初始化协方差
        
        #============================================================    
        #plot the LSQ estimate resaults
        #============================================================
        
        plot_x=[]
        plot_y=[]
        
        
        #ax.set_xlim(-1,11)
        #ax.set_ylim(-1,11)
        for data in uwbdis_data:
            temp=ekf.LSQ_TOA(data)
            plot_x.append(temp[0])
            plot_y.append(temp[1])
        fig1_line1=ax1.scatter(plot_x,plot_y,marker='^',c='blue',s=3)
        plt.show()
        #============================================================    
        #plot the kalman estimate resaults
        #============================================================
        
        plot_x2=[]
        plot_y2=[]
        for i in range(samplecnt):
            temp=ekf.ekffilter(acceldata[100*i:100*(i+1),:],uwbdis_data[i])
            plot_x2.append(temp[0])
            plot_y2.append(temp[1])
        fig1_line2=ax1.scatter(plot_x2,plot_y2,marker='o',c='limegreen',s=3)
        fig1_line3=ax1.scatter(tgpos[0],tgpos[1],marker='+',c='r',linewidths=0.05)
        
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
        
        GDOPvalue.append(get_TOAGDOP(tgpos,Anchor_pos,Anchor_num))
        
        print(tgpos)
    labels=['TOA estimated position','DF estimated position','real position']
    handles=[fig1_line1,fig1_line2,fig1_line3]
    ax1.set_title("fixed position analysis")        
    fig1.legend(handles,labels,loc="upper left")
    ax1.set_xlabel('x-axis(m)')
    ax1.set_ylabel('y-axis(m)')
    ax1.set_position([0.1,0.1,0.8,0.7])
    
    fig2=plt.figure(2)
    ax2=fig2.add_subplot(111)
    ax2.plot(GDOPvalue,toa_xstderr,label='toa_xstderr')
    ax2.plot(GDOPvalue,df_xstderr,label='df_xstderr')
    ax2.set_title("x-axis standard error")        
    ax2.set_xlabel('GDOP value')
    ax2.set_ylabel('standerd error')
    ax2.set_position([0.1,0.1,0.8,0.7])
    ax2.legend(loc="upper left")
    
    fig3=plt.figure(3)
    ax3=fig3.add_subplot(111)
    ax3.plot(GDOPvalue,toa_ystderr,label='toa_ystderr')
    ax3.plot(GDOPvalue,df_ystderr,label='df_ystderr')
    ax3.set_title("y-axis standard error")
    ax3.set_xlabel('GDOP value')
    ax3.set_ylabel('standerd error')
    ax3.set_position([0.1,0.1,0.8,0.7])
    ax3.legend(loc="upper left")
    
    fig4=plt.figure(4)
    ax4=fig4.add_subplot(111)
    ax4.plot(GDOPvalue,toa_meanerr,label='toa_meanerr')
    ax4.plot(GDOPvalue,df_meanerr,label='df_meanerr')
    ax4.set_title("mean error")
    ax4.set_xlabel('GDOP value')
    ax4.set_ylabel('mean error')
    ax4.set_position([0.1,0.1,0.8,0.7])
    ax4.legend(loc="upper left")
    
    fig5=plt.figure(5)
    ax5=fig5.add_subplot(111)
    ax5.plot(GDOPvalue,toaCEP,label='toaCEP80%')
    ax5.plot(GDOPvalue,dfCEP,label='dfCEP80%')
    ax5.set_title("80% CEP analysis")
    ax5.set_xlabel('GDOP value')
    ax5.set_ylabel('80% CEP radius')
    ax5.set_position([0.1,0.1,0.8,0.7])
    ax5.legend(loc="upper left")
#    ax1.set_xlim([-15,35])
#    ax1.set_ylim([-15,15])
    #std1=np.std(np.array(plot_x).astype(np.float64))
    #std2=np.std(np.array(plot_y).astype(np.float64))
    #std3=np.std(np.array(plot_x2).astype(np.float64))
    #std4=np.std(np.array(plot_y2).astype(np.float64))
    #print('rawdata x std:'+str(std1))
    #print('rawdata y std:'+str(std2))
    #print('kalman x std:'+str(std3))
    #print('kalman y std:'+str(std4))
    
if idx=='6':
    Anchor_num=4
    Anchor_pos=np.array([[0,0],
                         [4.2,0],
                         [1.54,7.68],
                         [6.2,6.84]])
    fig4=plt.figure(4)
    ax4=fig4.add_subplot(111)
    ax4.scatter(Anchor_pos[:,0],Anchor_pos[:,1],marker='o',c='black',s=6)
    tagpoint_len=161
    tagposlist=np.array((tagpoint_len,2))
    acceldata=np.zeros((tagpoint_len*100,2))
    ac=[1]*100+[0]*3900+[-1]*100
    ac=np.array(ac)
    ac=ac*0.5
    hold=np.zeros((300,2))
    acceldata[:4100,0]=acceldata[:4100,0]+ac
    acceldata[4000:8100,1]=acceldata[4000:8100,1]+ac
    acceldata[8000:12100,0]=acceldata[8000:12100,0]-ac
    acceldata[12000:16100,1]=acceldata[12000:16100,1]-ac
    acceldata=np.concatenate((hold,acceldata,hold),axis=0)
    
    tagpoint_len+=6
    initstat=[-1,2.5]+[0,0]
    tagposlist=imutrace(initstat,acceldata)
    tagpoint_len=len(tagposlist)
    
    for i in range(2):
        acceldata[:,i]=acceldata[:,i]+np.random.normal(0,std_a,tagpoint_len*100)
    uwbdis_data=np.zeros((tagpoint_len,Anchor_num))
    realdis_data=np.zeros((tagpoint_len,Anchor_num))
    for j in range(len(tagposlist)):
        realdis=np.zeros(Anchor_num)
        tgpos=tagposlist[j]
        for i in range(Anchor_num):
            realdis[i]=np.sqrt((tgpos[0]-Anchor_pos[i][0])**2+(tgpos[1]-Anchor_pos[i][1])**2)    
            realdis_data[j,i]=realdis[i]
            uwbdis_data[j,i]=realdis[i]+np.random.normal(0,std_r,1)
            
    ekf=RAKOEKF(sigma_a,sigma_r,Anchor_pos,Anchor_num,dt_IMU,dt_UBW)
    ekf.P = np.diag([.1, .1, .1, .1, .1, .1])#初始化协方差
    
    plot_x=[]
    plot_y=[]
    #ax.set_xlim(-1,11)
    #ax.set_ylim(-1,11)
    for data in uwbdis_data:
        temp=ekf.LSQ_TOA(data)
        plot_x.append(temp[0])
        plot_y.append(temp[1])
    ax4.plot(plot_x,plot_y,linewidth=1,c='blue',label='TOA estimated position')
    plt.show()   
    
    plot_x2=[]
    plot_y2=[]
    for i in range(len(tagposlist)):
        temp=ekf.ekffilter(acceldata[100*i:100*(i+1),:],uwbdis_data[i])
        plot_x2.append(temp[0])
        plot_y2.append(temp[1])
    ax4.plot(plot_x2,plot_y2,linewidth=1,c='limegreen',label='DF estimated position')
#    ax4.plot(np.array(tagposlist)[:,0],np.array(tagposlist)[:,1],linewidth=1,c='r',label='real position')
    
    ax4.set_title('acceleration motion analysis')
    ax4.set_xlabel('x-axis(m)')
    ax4.set_ylabel('y-axis(m)')
    fig4.legend(loc='upper left')    

    
    plot_x=np.array(plot_x,dtype='float')
    plot_y=np.array(plot_y,dtype='float')
    plot_x2=np.array(plot_x2,dtype='float')
    plot_y2=np.array(plot_y2,dtype='float')
    tagposlist=np.array(tagposlist,dtype='float')
    
#    std1=np.std(np.array(plot_x).astype(np.float64)-np.array(tagposlist).astype(np.float64)[:,0])
#    std2=np.std(np.array(plot_y).astype(np.float64)-np.array(tagposlist).astype(np.float64)[:,1])
#    std3=np.std(np.array(plot_x2).astype(np.float64)-np.array(tagposlist).astype(np.float64)[:,0])
#    std4=np.std(np.array(plot_y2).astype(np.float64)-np.array(tagposlist).astype(np.float64)[:,1])
#    print('变加速度运动')
#    print('rawdata x std:'+str(std1))
#    print('rawdata y std:'+str(std2))
#    print('kalman x std:'+str(std3))
#    print('kalman y std:'+str(std4))        
    ax4.set_position([0.1,0.1,0.8,0.7])