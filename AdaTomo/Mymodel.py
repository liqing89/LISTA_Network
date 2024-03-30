import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.linalg import eigvalsh




class LISTA(nn.Module):
    def __init__(self, n, m, T, lambd, alpha, D=None):
        super(LISTA, self).__init__()
        self.n, self.m = n, m
        self.D = D
        self.T = T  # ISTA Iterations
        self.lambd = lambd  # Lagrangian Multiplier
        self.A = nn.Linear(n, m, bias=False)  # Weight Matrix
        self.B = nn.Linear(m, m, bias=False)  # Weight Matrix
        # ISTA Stepsizes eta = 1/L
        self.etas = nn.Parameter(torch.ones(T + 1, 1, 1), requires_grad=True)      # 步长
        self.gammas = nn.Parameter(torch.ones(T + 1, 1, 1), requires_grad=True) # * 0.1 / alpha #阈值
        # Initialization
        if D is not None:
            self.A.weight.data = D.t()
            self.B.weight.data = D.t() @ D
        self.reinit_num = 0  # Number of re-initializations

    def _shrink(self, x, eta):
        return eta * F.softshrink(x / eta, lambd=self.lambd)

    def forward(self, y, D=None):
        y = y.squeeze()
        x = self._shrink(self.gammas[0, :, :] * self.A(y), self.etas[0, :, :])
        for i in range(1, self.T + 1):
            # x = shrink(self.gammas[i, :, :] * (self.B(x) + self.A(y)), self.etas[i, :, :])
            x = self._shrink(
                x - self.gammas[i, :, :] * self.B(x) + self.gammas[i, :, :] * self.A(y),
                self.etas[i, :, :],
            )
        return x
    


def soft_thr(input_, theta_):
    return F.relu(input_-theta_)-F.relu(-input_-theta_)

class LISTA1(nn.Module):
    def __init__(self, m, n, Dict, numIter, alpha, device):
        super(LISTA1, self).__init__()
        self._W = nn.Linear(in_features = m, out_features = n, bias=False)
        self._S = nn.Linear(in_features = n, out_features = n,
                            bias=False)

        self.thr = nn.Parameter(torch.rand(numIter,1), requires_grad=True)
        self.numIter = numIter
        self.A = Dict
        self.alpha = alpha
        self.device = device
        
    # custom weights initialization called on network
    def weights_init(self):
        A = self.A
        alpha = self.alpha
        S = torch.from_numpy(np.eye(A.shape[1]) - (1/alpha)*np.matmul(A.T, A))
        S = S.float().to(self.device)
        B = torch.from_numpy((1/alpha)*A.T)
        B = B.float().to(self.device)
        
        thr = torch.ones(self.numIter, 1) * 0.1 / alpha
        
        self._S.weight = nn.Parameter(S)
        self._W.weight = nn.Parameter(B)
        self.thr.data = nn.Parameter(thr.to(self.device))


    def forward(self, y):
        x = []
        d = torch.zeros(y.shape[0], self.A.shape[1], device = self.device)
            
        for iter in range(self.numIter):
            d = soft_thr(self._W(y) + self._S(d), self.thr[iter])
            x.append(d)
        return x
    
 
class LISTA2(nn.Module):
    def __init__(self, n, m, W_e, max_iter, L, theta):
        """
        # Arguments
            n: int, dimensions of the measurement
            m: int, dimensions of the sparse signal
            W_e: array, dictionary
            max_iter:int, max number of internal iteration
            L: Lipschitz const 
            theta: Thresholding
        """
        
        super(LISTA2, self).__init__()
        self._W = nn.Linear(in_features=n, out_features=m, bias=False)
        self._S = nn.Linear(in_features=m, out_features=m,
                            bias=False)
        self.shrinkage = nn.Softshrink(theta)
        self.theta = theta
        self.max_iter = max_iter
        self.A = W_e
        self.L = L
        
    # weights initialization based on the dictionary
    def weights_init(self):
        A = self.A.cpu().numpy()
        L = self.L
        S = torch.from_numpy(np.eye(A.shape[1]) - (1/L)*np.matmul(A.T, A)) #权重矩阵W2计算
        S = S.float().to(device)
        W = torch.from_numpy((1/L)*A.T)    #权重矩阵W1计算
        W = W.float().to(device)
        
        self._S.weight = nn.Parameter(S)
        self._W.weight = nn.Parameter(W)


    def forward(self, y):
        x = self.shrinkage(self._W(y))

        if self.max_iter == 1 :
            return x

        for iter in range(self.max_iter):
            x = self.shrinkage(self._W(y) + self._S(x))

        return x
    
    
def shrinkage(x, theta):
    return np.multiply(np.sign(x), np.maximum(np.abs(x) - theta, 0))    #软阈值函数