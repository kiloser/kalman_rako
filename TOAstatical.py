# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt

dis=[[],[],[],[],[]]
with open('data\\TOAanalyze.txt','r') as fd:
    for line in fd:
        ACid=int(line.split()[1])
        ACdistmp=float(line.split()[3])
        dis[ACid-1].append(ACdistmp)


fig1=plt.figure(1)
ax1=fig1.add_subplot(211)
ax2=fig1.add_subplot(212)
datalen=len(dis[0])
x=range(datalen)
for data in dis:    
    ax1.plot(x,data)
    datatemp=data
    for idx,tmp in enumerate(datatemp):
        if abs(tmp-np.mean(datatemp))>1:
            datatemp[idx]=np.mean(datatemp)
    print(np.std(datatemp))
    ax2.plot(x,datatemp)