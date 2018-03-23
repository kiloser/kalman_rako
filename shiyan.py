# -*- coding: utf-8 -*-
#class father():
#    def __init__(self,func):
#        self.f=func
#        self.K=100
#    def runf(self,val):
#        self.f(val)
#class child(father):
#    def __init__(self):
#        super(child,self).__init__(self.func)
#        self.fuck=2
#        self.K=20
#    def func(self,val):
#        print('the fuck is '+str(self.fuck))
#        print('the val is '+str(val))
#        
#te=child()
#te.runf(223)
#
#def changelist(list1):
#    list1[0]=2
#    
#list1=[1,2]
#changelist(list1)

'''
import scipy

#chan 算法， 基站个数5个或者更多
def doChanForMore(D):
    c = 3e8
    #所有的基站坐标
    M = D[:,:3]
    #[-xn0, -yn0, -zn0, -Rn0]
    Ga = D[1:] - D[0]
    num = len(M)
    Q = scipy.matrix((0.5 * scipy.identity(num - 1)) + 0.5)

    E = -Ga.T * Q.I
    Fi = (E * -Ga).I

    R = Ga[:,3]
    R_squared = scipy.matrix(R.A * R.A)

    K = scipy.zeros((num,1))
    for n in range(num):
        K[n] = M[n] * M[n].T

    h = (R_squared - K[1:] + K[0]) / 2
    first_est = (Fi * E * h).A.squeeze()
    R0 = first_est[3]

    B = scipy.matrix(scipy.identity(num - 1) * (R.A + R0))
    Y = B * Q * B

    E = -Ga.T * (Y * (c ** 2)).I
    Fi = (E * -Ga).I

    second_est = (Fi * E * h).A.squeeze()

    return [second_est[:3], R0]
'''
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

#Anchor_num=4
#Anchor_pos=np.array([[5,5],
#                  [-5,5],
#                  [5,-5],
#                  [-5,-5]])
Anchor_num=5
Anchor_pos=np.array([[0,0],
                     [5,0],
                     [6.54,4.75],
                     [2.50,7.69],
                     [-1.54,4.75]])     
tagposlist=[]
x=np.linspace(-50,50,50,endpoint=False)
y=np.linspace(-50,50,50,endpoint=False)
GDOP=[] 
for i in np.linspace(-50,50,50,endpoint=False):
    for j in np.linspace(-50,50,50,endpoint=False):
        tagposlist.append([i,j])    
for pos in tagposlist:
        GDOP.append(get_TDOAGDOP(pos,Anchor_pos,Anchor_num))
        
        print(pos)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
pos=np.array(tagposlist)
x=pos[:,0]
y=pos[:,1]
ax.plot_trisurf(x, y, GDOP,cmap='rainbow')
line=ax.scatter3D(Anchor_pos[:,0], Anchor_pos[:,1], np.zeros(Anchor_num),color='black')
ax.set_xlabel('x-axis(m)')
ax.set_ylabel('y-axis(m)')
ax.set_zlabel('GDOP')
ax.legend([line],['Anchor position'])

