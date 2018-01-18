# -*- coding: utf-8 -*-
import sympy
import numpy as np
import scipy
#def resifun_init(ac_pos,acnum):
#    x,y=sympy.symbols('x,y')
#    f=0
#    def dis(acnum):
#        n=1
#        while n<acnum:
#            yield sympy.symbols('dis'+str(n))
#            n+=1
#    dwdis=list(dis(acnum))
#    for i in range(acnum):
#        pos=ac_pos[i]
#        f+=dwdis[i]-sympy.sqrt((x-pos[0])**2+(y-pos[1])**2)
#    def residualfun(p,dw_dis):
#        posx,posy=p
#        residual=f.evalf(subs={x:posx,y:posy})
#        return residual
#    return residualfun
#
#ac_pos=np.array([[1,2],
#                 [3,4],
#                 [0,0]])
#residualfun=resifun_init(ac_pos,3)

acnum=3
pos=np.array([[0,12],
              [5,6],
              [0,0]])
A=np.zeros((2,2))
A[0]=pos[1]-pos[0]
A[1]=pos[2]-pos[0]
B=np.zeros((2,1))
realdis=[0.0,0.0,0.0]
realACdis=[0.0,0.0,0.0]
realpos=[3,7]
for i in range(3):
    realdis[i]=np.sqrt((realpos[0]-pos[i][0])**2+(realpos[1]-pos[i][1])**2)
for i in range(3):
    realACdis[i]=np.sqrt((pos[i][0]-pos[0][0])**2+(pos[i][1]-pos[0][1])**2)
refdis=realdis[0]
B[0]=(refdis**2+realACdis[1]**2-realdis[1]**2)/2
B[1]=(refdis**2+realACdis[2]**2-realdis[2]**2)/2
A=np.mat(A)
B=np.mat(B)
solvepos=((A.T*A).I*A.T*B).T+pos[0]
