import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import pandas as pd

def soft_thr(input_, theta_):
    return torch.max(torch.zeros_like(input_), input_ - theta_) - torch.max(torch.zeros_like(input_), -input_ - theta_)

path = os.getcwd()
base_path = os.path.join(path,'middle_result','LISTA')
# 创建空行字符串
empty_row = ''

class LISTA_Net(nn.Module):
    def __init__(self,n,m,T,lambd, L , D ):
        super(LISTA_Net,self).__init__()
        self.n = n
        self.m = m
        self.T = T
        self.lambd = lambd 
        self._W = nn.Linear(in_features = n, out_features = m)
        self._S = nn.Linear(in_features = m, out_features = m)
        self.L = L
        self.D = D
        self.weights_W = []
        self.weights_S = []

        # custom weights initialization called on network
    def weights_init(self):
        L = self.L
        S = torch.from_numpy(np.eye(self.D.shape[1]) - (1/L)*np.matmul(self.D.T, self.D)).float()
        B = torch.from_numpy((1/L)*self.D.T).float()
        self._S.weight = nn.Parameter(S)
        self._W.weight = nn.Parameter(B)
    def _shrink(self,x,theta):
        return torch.mul(torch.sign(x), torch.max((torch.abs(x) - theta), torch.zeros_like(x)))
    
    def forward(self, y,D,epoch):
        y=y.squeeze(2)
        thr = self.lambd*0.95/self.L
        x = self._shrink(self._W(y),thr)
        if y.is_cuda:
            x = x.cuda()
        # file_name = os.path.join(base_path,f'LISTA_epoch{epoch}_output.csv')
        # data =[]
        for i in range(self.T):  
            x = self._shrink(self._W(y)+self._S(x),thr)
            # data.append(x.detach().cpu().numpy().reshape(-1))
        # df = pd.DataFrame(data)
        # pd.set_option('display.float_format', '{:.15f}'.format)
        # df.to_csv(file_name, index=False, mode='a')
        # with open(file_name, 'a') as file:
        #     file.write(empty_row + '\n')
        return x

