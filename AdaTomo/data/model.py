import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.linalg import eigvalsh
# from scipy import io
import os


def soft_threshold(input,theta):
    theta = torch.maximum(theta,torch.zeros(1).cuda())
    return torch.sign(input)*torch.maximum(torch.abs(input)-theta,torch.zeros_like(input))


class LISTA(nn.Module):
    def __init__(self,n,m,lambd,T,D):
        super(LISTA,self).__init__()
        self.n = n
        self.m = m
        self.T = T
        self.lambd = lambd
        self.D = D
        eig_value = eigvalsh(self.D.T@self.D)
        self.L = np.ceil(max(eig_value))
        self.W = nn.Linear(in_features = n, out_features = m,bias=False)
        self.S = nn.Linear(in_features = m, out_features = m,bias=False)
        self.theta = nn.ParameterList([nn.Parameter(torch.ones(1) * self.lambd/self.L) for i in range(self.T)])
        W2 = np.eye(self.D.shape[1])-1/self.L*np.matmul(self.D.T,self.D)
        W2 = torch.from_numpy(W2).float()
        W1 = torch.from_numpy((1/self.L)*self.D.T).float()
        self.W.weight = nn.Parameter(W1)
        self.S.weight = nn.Parameter(W2)

        
    
    def forward(self,y,D):
        y = y.squeeze(2)
        x = torch.zeros(y.shape[0], self.m)
        # thr = self.lambd/self.self.L*0.95
        # thr = torch.tensor(thr)
        if y.is_cuda:
            x = x.cuda()
            # thr = thr.cuda()
        for i in range(self.T):
            x = self.W(y)+self.S(x)
            thr = self.theta[i][0]
            x = soft_threshold(x,thr)
        return x.unsqueeze(2)
    
class Ada_LISTA(nn.Module):
    def __init__(self,n,m,lambd,T,D):
        super(Ada_LISTA,self).__init__()
        self.n, self.m = n, m
        self.D = D
        self.T = T 
        self.lambd = lambd
        self.W1 = nn.Linear(n, n, bias=False)
        self.W2 = nn.Linear(n, n, bias=False)
        L =  np.ceil(np.max(eigvalsh(self.D.T @ self.D)))
        self.L = L
        self.theta = nn.ParameterList([nn.Parameter(torch.ones(1) * self.lambd/self.L) for i in range(self.T)])   
        self.W1.weight.data = 1/L*torch.eye(self.n)
        self.W2.weight.data = 1/L*torch.eye(self.n)


    def _A(self, D, i):
        A_tmp = self.W1.weight @ D
        return  A_tmp.transpose(1, 2)

    def _B(self, D, i):
        B_tmp = self.W2.weight @ D
        return  B_tmp.transpose(1, 2) @ B_tmp   
    
    def forward(self,y,D):
        x = torch.zeros(y.shape[0], self.m, y.shape[2])
        if y.is_cuda:
            x = x.cuda()
        for i in range(0, self.T ):
            x = x- self._B(D, i) @ x + self._A(D, i) @ y
            x = soft_threshold(x, self.theta[i][0]) 
        return x
    
class Ada_AT_LISTA(nn.Module):
    def __init__(self,n:int,m:int,lambd:float,T:int,D) :
        super(Ada_AT_LISTA,self).__init__()
        self.n, self.m = n, m
        self.D = D
        self.T = T 
        self.lambd = lambd
        self.tao = None
        self.W1 = nn.Linear(n, n, bias=False)
        self.W2 = nn.Linear(n, n, bias=False)
        self.gammas = nn.Parameter(torch.ones(self.T + 1, 1, 1, 1),requires_grad=True)
        self.weights_init()

    def weights_init(self):
        L =  np.ceil(np.max(eigvalsh(self.D.T @ self.D)))
        self.W1.weight.data = 1/L*torch.eye(self.n)
        self.W2.weight.data = 1/L*torch.eye(self.n)
        self.gammas.data *= 1*self.lambd / L#

    def _A(self, D, i):
        A_tmp = self.W1.weight @ D
        return  A_tmp.transpose(1, 2)

    def _B(self, D, i):
        B_tmp = self.W2.weight @ D
        return  B_tmp.transpose(1, 2) @ B_tmp   
    
    def forward(self,y,D):
        x = torch.zeros(y.shape[0], self.m, y.shape[2])
        self.tao = torch.max(y)
        if y.is_cuda:
            x = x.cuda()
        for i in range(0, self.T + 1):
            x = x- self._B(D, i) @ x + self._A(D, i) @ y
            gamma_i = self.gammas[i,:,:,:]
            theta = gamma_i*self.tao/(torch.abs(x)/self.tao+1)
            x = soft_threshold(x, theta) 
        return x