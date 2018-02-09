# -*- coding: utf-8 -*-
import numpy as np

    
# chan算法 ，计算标签坐标
def chan_algorithm(self,arrivetime):
    idx=np.argsort(arrivetime)
    arrivetime=np.take(arrivetime,idx)
    
    row, column = evVal.shape  # 行，列
    baseX = evVal[:, 1]  # 列向量
    baseY = evVal[:, 2]

    ri1 = C.c*(evVal[:, 0] - evVal[0, 0])[:-1]  # 第i个基站和第一个基站之间的距离gui
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