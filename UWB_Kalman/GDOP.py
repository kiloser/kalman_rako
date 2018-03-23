# -*- coding: utf-8 -*-
import numpy as np
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