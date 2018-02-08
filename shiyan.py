# -*- coding: utf-8 -*-
class father():
    def __init__(self,func):
        self.f=func
        self.K=100
    def runf(self,val):
        self.f(val)
class child(father):
    def __init__(self):
        super(child,self).__init__(self.func)
        self.fuck=2
        self.K=20
    def func(self,val):
        print('the fuck is '+str(self.fuck))
        print('the val is '+str(val))
        
te=child()
te.runf(223)

def changelist(list1):
    list1[0]=2
    
list1=[1,2]
changelist(list1)

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