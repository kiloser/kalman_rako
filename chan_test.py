# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
'''
这应该是陶璐移植的matlab的TDOA程序。
'''

import matplotlib.pyplot as plt
from scipy import constants as C
import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import minimize
from numpy import linalg as la


TB2AC_dis=np.array([5.35,4.51,3.10,4.52])
TAG2AC_dis=np.array([4.16,4.44,3.63,3.62])
TB2AC_T=TB2AC_dis/C.c

def f(x):
    x0,x1,x2,x3,x4,x5 = x.tolist()
    return [x0**2 + x1**2 - 3.12**2 ,
            x2**2 + x3**2 - 7.79**2 ,
            x4**2 + x5**2 - 7.03**2 ,
            (x2-x0)**2 + x3**2 - 7.57**2 ,
            (x4-x0)**2 + x5**2 - 8.05**2 ,
            (x4-x2)**2 + (x5-x3)**2 - 3.07**2 ]

def chan_algorithm(arrivetime):
    acnum=4
    acpos=np.array([[0,0],
                     [10,0],
                     [0,10],
                     [10,10]])
    idx=np.argsort(arrivetime)
    arrivetime=np.take(arrivetime,idx)
    base_pos=np.zeros((acnum,2))
    for i in range(acnum):
        base_pos[i,:]=acpos[idx[i],:]
    evVal=np.concatenate((np.mat(arrivetime).T,base_pos),axis=1)
    row, column = evVal.shape  # 行，列
    baseX = evVal[:, 1]  # 列向量
    baseY = evVal[:, 2]

    ri1 = C.c*(evVal[:, 0] - evVal[0, 0])[1:]  # 第i个基站和第一个基站之间的距离gui
    xi1 = (baseX - baseX[0])[1:]
    yi1 = (baseY - baseY[0])[1:]
    
    k = np.zeros(row)
    for i in range(0, row):
        k[i] = baseX[i] ** 2 + baseY[i] ** 2
    k = np.mat(k).T    

# chan算法 ，计算标签坐标
def chan(evVal):
    row, column = evVal.shape  # 行，列
    baseX = evVal[:, 1]  # 列向量
    baseY = evVal[:, 2]

    ri1 = C.c * (evVal[:, 0] - evVal[0, 0])[:-1]  # 第i个基站和第一个基站之间的距离gui
    xi1 = (baseX - baseX[0])[1:]
    yi1 = (baseY - baseY[0])[1:]
    Standaraddeviation = 3.5e-2

    k = np.zeros(row)
    for i in range(0, row):
        k[i] = baseX[i] ** 2 + baseY[i] ** 2
    k = np.mat(k).T

    h = np.zeros((3, 1))
    for i in range(0, 3):
        h[i, 0] = 0.5 * ((ri1[i]) ** 2 - k[i + 1] + k[0])
    h = np.mat(h)

    Ga = -np.bmat("xi1 yi1 ri1")

    Q = np.zeros((row - 1, row - 1))
    Q = np.mat(Q)
    for i in range(0, row - 1):
        Q[i, i] = (Standaraddeviation) ** 2

    Za = (Ga.T * Q.I * Ga).I * Ga.T * Q.I * h

    B1 = np.zeros((row - 1, row - 1))
    for i in range(0, row - 1):
        B1[i, i] = np.sqrt((baseX[i + 1] - Za[0]) ** 2 + (baseY[i + 1] - Za[1]) ** 2)

    B1 = np.mat(B1)

    P1 = C.c ** 2 * B1 * Q * B1
    Za1 = (Ga.T * P1.I * Ga).I * Ga.T * P1.I * h
    C0 = (Ga.T * P1.I * Ga).I

    h1 = np.zeros((3, 1))
    h1[0] = (Za1[0] - baseX[0]) ** 2
    h1[1] = (Za1[1] - baseY[0]) ** 2
    h1[2] = (Za1[2]) ** 2
    h1 = np.mat(h1)

    Ga1 = np.mat([[1, 0], [0, 1], [1, 1]])
    r1 = np.sqrt((baseX[0] - Za1[0]) ** 2 + (baseY[0] - Za1[1]) ** 2)

    B2 = np.zeros((3, 3))
    B2[0, 0] = Za1[0] - baseX[0]
    B2[1, 1] = Za1[1] - baseY[0]
    B2[2, 2] = r1
    B2 = np.mat(B2)

    P2 = 4 * B2 * C0 * B2
    Za2 = (Ga1.T * P2.I * Ga1).I * Ga1.T * P2.I * h1

    ms0 = np.sqrt(np.abs(Za2))
    ms0[0] = ms0[0] + baseX[0]
    ms0[1] = ms0[1] + baseY[0]

    return ms0


def main():
    # 所有的都从0开始
    MAXTIME = 17207356974694.4 * 1e-12
    tmpi = 1
    locbuff = []
    Timestamp_lasttime = np.zeros(4)
    Tref = np.zeros(4)
    Tref_lasttime = np.zeros(4)
    k = 0.6
    syncpreiod = 0.233
    tagdatacnt = 0
    tagdatal = 0
    discardcnt = 0
    TBDATA = {}
    TAGDATA = {}

    # 求解实际标签、基站、同步基站坐标
    result = fsolve(f, [0, 0, 0, 0, 0, 0])
    coordi = np.array([[0, 0],
                       [3.12, 0],
                        [result[2], result[3]],
                        [result[4], result[5]]])
    func_tb = lambda x: (x[0] ** 2 + x[1] ** 2 - 5.35 ** 2) ** 2 + ((x[0] - 3.12) ** 2 + x[1] ** 2 - 4.51 ** 2) ** 2 + \
                        ((x[0] - result[2]) ** 2 + (x[1] - result[3]) ** 2 - 3.10 ** 2) ** 2 + \
                        ((x[0] - result[4]) ** 2 + (x[1] - result[5]) ** 2 - 4.52 ** 2) ** 2
    res_tb = minimize(func_tb, [0, 0])
    tbcoordi =np.mat(res_tb.x)

    func_tag = lambda x: (x[0] ** 2 + x[1] ** 2 - 4.16 ** 2) ** 2 + ((x[0] - 3.12) ** 2 + x[1] ** 2 - 4.44 ** 2) ** 2 + \
                         ((x[0] - result[2]) ** 2 + (x[1] - result[3]) ** 2 - 3.63 ** 2) ** 2 + \
                         ((x[0] - result[4]) ** 2 + (x[1] - result[5]) ** 2 - 3.62 ** 2) ** 2
    res_tag = minimize(func_tag, [0, 0])
    tagcoordi = np.mat(abs(res_tag.x))

    # coordi = np.array([[0,0],
    #                    [3.12,0],
    #                    [2.1096,7.5007],
    #                    [-0.9133,6.9687]])
    # tbcoordi = np.mat([1.2102,3.8821])
    # tagcoordi = np.mat([2.8804,4.5038])


    # 读写数据
    fin = open(r'233ms.dat', 'rb')
    while tagdatacnt is not None:
        #判断是否读完
        tag = fin.read(2)
        if tag == b'':
            break
        tagdatacnt = int.from_bytes(tag, byteorder='little', signed=False)

        # 同步基站时间戳数据
        for i in range(0, 4):            # 读数据类型
            fin.read(2)
            TBDATA[i] = {}
            TBDATA[i]['anchor'] = fin.read(2)  # 第几号基站的数据

            TBDATA[i]['tagID'] = fin.read(2)  # 第几号标签的数据
            TBDATA[i]['Idx'] = fin.read(2)
            TBDATA[i]['Timestamp'] = int.from_bytes(fin.read(8), byteorder='little', signed=False)
            TBDATA[i]['Timestamp'] = TBDATA[i]['Timestamp'] * 15.65e-12

        # 标签数据
        if tagdatacnt != 0:
            tagdatal = tagdatal + 1
            for i in range(0, tagdatacnt):
                fin.read(2)
                TAGDATA[i] = {}
                TAGDATA[i]['tagID'] = fin.read(2)
                TAGDATA[i]['Idx'] = fin.read(2)
                TAGDATA[i]['Validmask'] = fin.read(2)
                TAGDATA[i]['ACTimestamp']= {}
                for j in range(0,4):  # 基站到达时间戳数据  总共4个基站
                    TAGDATA[i]['ACTimestamp'].update({j:int.from_bytes(fin.read(8), byteorder='little', signed=False)})
                    TAGDATA[i]['ACTimestamp'][j] = TAGDATA[i]['ACTimestamp'][j] * 15.65e-12
                    # 运动传感器
                TAGDATA[i]['MPUDATA'] = {}
                for j in range(0,4):
                    TAGDATA[i]['MPUDATA'][j] = {}
                    TAGDATA[i]['MPUDATA'][j]['gyro'] = fin.read(6)
                    TAGDATA[i]['MPUDATA'][j]['accel'] = fin.read(6)
                    TAGDATA[i]['MPUDATA'][j]['quat'] = fin.read(16)

        # 定位数据的计算    TAGTStamp记得从0开始
        if tagdatacnt != 0:
            for i in range(0, tagdatacnt):
                TAGTStamp = np.zeros(4)  # 从0开始
                TStmpdiff = np.zeros(4)
                TStmpArr = np.zeros(4)
                tagVDcnt = 0
                meanTS = 0

                # 四个基站的时间戳处理
                for j in range(0, 4):
                    if TAGDATA[i]['ACTimestamp'] != 0:
                        TAGTStamp[j] = TAGDATA[i]['ACTimestamp'][j]
                        if TAGTStamp[j] - Timestamp_lasttime[j] < 0:
                            TStmpdiff[j] = MAXTIME + TAGTStamp[j] - Timestamp_lasttime[j]
                        else:
                            TStmpdiff[j] = TAGTStamp[j] - Timestamp_lasttime[j]

                            meanTS = meanTS + TStmpdiff[j]
                            tagVDcnt = tagVDcnt + 1

                meanTS = meanTS / tagVDcnt

                for j in range(0, 4):
                    if TStmpdiff[j] != 0:  #判断数据有没有污染
                        if np.abs(TStmpdiff[j] - meanTS) > 0.1:  #有 丢弃
                            TAGDATA[i]['ACTimestamp'][j] = 0
                            tagVDcnt = tagVDcnt - 1
                        else: #没有 修正
                            TStmpdiff[j] = TStmpdiff[j] * syncpreiod / Tref_lasttime[j]
                            TStmpArr[j] = TStmpdiff[j] + TB2AC_T[j]

                # 判断有没有数量的基站
                if tagVDcnt >= 4:
                    evVal = np.zeros((4,3))
                    evVal = np.mat(evVal)
                    for j in range(0, 4):
                        if TAGDATA[i]['ACTimestamp'][j] != 0:
                            coordi_mat =np.mat(coordi)
                            evVal[j,0] = TStmpArr[j]
                            evVal[j,1] = coordi_mat[j,0]
                            evVal[j,2] = coordi_mat[j,1]
                    loc = chan(evVal)
                    if (la.norm(loc.T-tagcoordi)<3) :
                        print(loc)
                        #loc_print = loc.A
                        #plt.scatter(loc_print[0], loc_print[1], c='b', marker='^', label='loc')
                    else:
                        discardcnt = discardcnt + 1

                else:
                    discardcnt = discardcnt + 1
        #基站状态更新
        for i in range(0,4):
            if TBDATA[i]['Timestamp'] != 0: #存在基站更新数据
                if Timestamp_lasttime[i] == 0: #判断是不是初始数据
                    Timestamp_lasttime[i] = TBDATA[i]['Timestamp']
                else:
                    if Tref[i] == 0: #判断是不是初始数据
                        if TBDATA[i]['Timestamp'] - Timestamp_lasttime[i] < 0:
                            Tref[i] = MAXTIME + TBDATA[i]['Timestamp'] - Timestamp_lasttime[i]
                        else:
                            Tref[i] = TBDATA[i]['Timestamp'] - Timestamp_lasttime[i]

                        Tref_lasttime[i] = Tref[i]
                        Timestamp_lasttime[i] = TBDATA[i]['Timestamp']
                    else:
                        if TBDATA[i]['Timestamp'] - Timestamp_lasttime[i]< 0 :
                            Tref[i] = MAXTIME + TBDATA[i]['Timestamp'] - Timestamp_lasttime[i]
                        else:
                            Tref[i] = TBDATA[i]['Timestamp']-Timestamp_lasttime[i]
                        if Tref[i] > syncpreiod + 0.02  or Tref[i] < syncpreiod - 0.02:
                            Tref_lasttime[i] = Tref_lasttime[i]
                            Timestamp_lasttime[i] = Timestamp_lasttime[i]+Tref_lasttime[i]
                        else:
                            Tref[i] = Tref_lasttime[i] * (1-k) + k * Tref[i]
                            Tref_lasttime[i] = Tref[i]
                            Timestamp_lasttime[i] = TBDATA[i]['Timestamp']
            else:
                Tref_lasttime[i] = Tref_lasttime[i]
                Timestamp_lasttime[i] = Timestamp_lasttime[i] + Tref_lasttime[i]


if __name__ == '__main__':
    main()
    plt.show()