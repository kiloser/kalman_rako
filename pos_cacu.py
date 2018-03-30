# -*- coding: utf-8 -*-
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import math
def residuals(p):
    r=[]
    p1=np.array([0,0])
    p2=np.array([-4.13,-4.42])
    p3=np.array(p[0:2])
    p4=np.array(p[2:4])
    p5=np.array(p[4:6])
    p6=np.array(p[6:8])
    
    r.append(np.linalg.norm(p1-p2)-6.05)
    r.append(np.linalg.norm(p1-p3)-10.4)
    r.append(np.linalg.norm(p1-p4)-10.3)
    r.append(np.linalg.norm(p1-p5)-6.44)
    r.append(np.linalg.norm(p1-p6)-3.32)
    
    r.append(np.linalg.norm(p2-p3)-4.44)
    r.append(np.linalg.norm(p2-p4)-5.78)
    r.append(np.linalg.norm(p2-p5)-6.38)
    r.append(np.linalg.norm(p2-p6)-6.44)
    
    r.append(np.linalg.norm(p3-p4)-4.59)
    r.append(np.linalg.norm(p3-p5)-9)
    r.append(np.linalg.norm(p3-p6)-10.25)
    
    r.append(np.linalg.norm(p4-p5)-6.33)
    r.append(np.linalg.norm(p4-p6)-8.95)
    
    r.append(np.linalg.norm(p5-p6)-3.64)
    
    r=[abs(i) for i in r]
    return np.sum(r)

p=[0,0,-9.1,-4.8,-4.93,4.13,-1.66,2.87]
paras=optimize.minimize(residuals,p)
pos=paras.x
pos=pos.reshape((4,2))
pos=np.concatenate(([[0,0],[-4.13,-4.42]],pos),axis=0)
plt.scatter(pos[:,0],pos[:,1])
#x1,x2,y1,y2,r=sympy.symbols('x1,x2,y1,y2,r')
#f=sympy.sqrt((x1-y1)**2+(x2-y2)**2)-r

base1=np.matrix([2.5,1])
v1=np.matrix([0,-20])
v2=np.matrix([-12.5,0])
v3=np.matrix([-12.5,-20])
theta=38/180*math.pi
rotate=np.array([[math.cos(theta),-math.sin(theta)],
                  [math.sin(theta),math.cos(theta)]])
base2=base1+v1*rotate
base3=base1+v2*rotate
base4=base1+v3*rotate
base=np.concatenate((base1,base2,base3,base4))
plt.plot(base[0:2,0],base[0:2,1],color="black")
plt.plot([base[1,0],base[3,0]],[base[1,1],base[3,1]],color="black")
plt.plot(base[2:4,0],base[2:4,1],color="black")
plt.plot([base[0,0],base[2,0]],[base[0,1],base[2,1]],color="black")


