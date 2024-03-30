import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.linalg import eigvalsh
from scipy import io
import os
import pandas as pd
path = os.getcwd()
base_path = os.path.join(path,'middle_result','Ada-AT-LISTA')
# 创建空行字符串
empty_row = ''
class Adaptive_ISTA_AT(nn.Module):
    def __init__(self, n, m, D=None, T=10, lambd = 0.5):
        super(Adaptive_ISTA_AT, self).__init__()
        self.n, self.m = n, m
        self.D = D
        self.T = T  # ISTA Iterations
        self.lambd = lambd  # Lagrangian Multiplier
        self.tao = None
        self.W1 = nn.Linear(n, n, bias=False)  # Weight Matrix
        self.W2 = nn.Linear(n, n, bias=False)  # Weight Matrix
        self.W1.weight.data = torch.eye(n)
        self.W2.weight.data = torch.eye(n)
        # ISTA Stepsizes
        self.etas = nn.Parameter(torch.ones(T + 1, 1, 1, 1), requires_grad=True)
        self.gammas = nn.Parameter(torch.ones(T + 1, 1, 1, 1), requires_grad=True)
        # self.theta = nn.Parameter(torch.tensor(1.0),requires_grad=True)
        
        # Initialization
        if D is not None:
            L =   round( float(eigvalsh(D.T @ D, eigvals=(m - 1, m - 1))))
            self.etas.data *= 1 / L
            self.gammas.data *= 1*self.lambd / L
        self.reinit_num = 0  # Number of re-initializations

    def _A(self, D, i):
        A_tmp = self.W1.weight @ D
        return self.etas[i, :, :, :] * A_tmp.transpose(1, 2)

    def _B(self, D, i):
        B_tmp = self.W2.weight @ D
        return self.etas[i, :, :, :] * B_tmp.transpose(1, 2) @ B_tmp

    # def _shrink(self, x, eta):
    #     return eta * F.softshrink(x / eta, lambd=self.lambd)
    def _shrink(self,x,theta):
        return torch.mul(torch.sign(x), torch.max((torch.abs(x) - theta), torch.zeros_like(x)))

    def forward(self, y, D, epoch):
        # y = y.unsqueeze(2)
        x = torch.zeros(y.shape[0], self.m, y.shape[2])
        # self.tao = 0.005*torch.max(y)
        self.tao = 10e-4
        if y.is_cuda:
            x = x.cuda()
        for i in range(0, self.T + 1):
            x = x - self._B(D, i) @ x + self._A(D, i) @ y
            theta = self.gammas[i,:,:,:]*self.tao/(torch.abs(x)/self.tao+1)
            x = self._shrink(x, theta)
        return x

    def reinit(self):
        reinit_num = self.reinit_num + 1
        self.__init__(n=self.n, m=self.m, D=self.D, T=self.T, lambd=self.lambd)
        self.reinit_num = reinit_num









class LISTA_Net(nn.Module):
    def __init__(self,n,m,LayerNum,lambd, L , Dict,tao = 0.001 ):
        super(LISTA_Net,self).__init__()
        self.n = n
        self.m = m
        self.LayerNum = LayerNum
        self.lambd = lambd 
        self.tao = tao
        self._W = nn.Linear(in_features = n, out_features = m)
        self._S = nn.Linear(in_features = m, out_features = m)
        self.L = L
        self.Dict = Dict
        self.weights_W = []
        self.weights_S = []

        # custom weights initialization called on network
    def weights_init(self):
        L = self.L
        S = torch.from_numpy(np.eye(self.Dict.shape[1]) - (1/L)*np.matmul(self.Dict.T, self.Dict)).float()
        B = torch.from_numpy((1/L)*self.Dict.T).float()
        self._S.weight = nn.Parameter(S)
        self._W.weight = nn.Parameter(B)

    def _shrink(self,x,theta):
        return torch.mul(torch.sign(x), torch.max((torch.abs(x) - theta), torch.zeros_like(x)))
    
    def forward(self, y):
        y = np.squeeze(y)
        thr = self.lambd*0.95/self.L
        x = self._shrink(self._W(y),thr)
        if y.is_cuda:
            x = x.cuda()
        for iter in range(self.LayerNum):  
            theta = (self.lambd*self.tao)/(torch.abs(x)+self.tao)
            x = self._W(y)+self._S(x)
            x = self._shrink(x, theta)
        return x