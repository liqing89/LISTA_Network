import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from tqdm import tqdm
import sys
#for debug
from scipy.linalg import orth
import matplotlib.pyplot as plt
import pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class LISTA(nn.Module):
    # net = LISTA(n, m, W_d, max_iter=30, L=L, theta=a/L)
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
        
        super(LISTA, self).__init__()
        self._W = nn.Linear(in_features=n, out_features=m, bias=False)
        self._S = nn.Linear(in_features=m, out_features=m, bias=False)
        self.shrinkage = nn.Softshrink(theta)
        self.theta = theta # a/L
        self.max_iter = max_iter # max number of internal iteration
        self.A = W_e # dictionary
        self.L = L
        
    # weights initialization based on the dictionary
    def weights_init(self):
        A = self.A.cpu().numpy() 
        L = self.L
        S = torch.from_numpy(np.eye(A.shape[1]) - (1/L)*np.matmul(A.T, A))
        S = S.float().to(device)
        W = torch.from_numpy((1/L)*A.T)
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


def train_lista(Y, dictionary, a, L, max_iter=30):
    
    n, m = dictionary.shape # dictionary matrix:[n]256x[m]1000
    n_samples = Y.shape[0] # Y: 5000x256
    batch_size = 128 # 每个批次使用的样本数量
    steps_per_epoch = n_samples // batch_size # 步数 = 样本数量 / 批次容量
    
    # convert the data into tensors
    Y = torch.from_numpy(Y)
    Y = Y.float().to(device)
    
    W_d = torch.from_numpy(dictionary)
    W_d = W_d.float().to(device)

    net = LISTA(n, m, W_d, max_iter=30, L=L, theta=a/L)
    net = net.float().to(device)
    net.weights_init()

    # build the optimizer and criterion
    learning_rate = 1e-2
    criterion1 = nn.MSELoss()
    criterion2 = nn.L1Loss()
    all_zeros = torch.zeros(batch_size, m).to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate,  momentum=0.9)

    loss_list = []
    total = 1
    for epoch in tqdm(range(total),desc='training process'):
        time.sleep(0.1)
        index_samples = np.random.choice(a=n_samples, size=n_samples, replace=False, p=None)
        Y_shuffle = Y[index_samples]
        # 每个训练轮次的样本顺序都随机打乱过
        data = range(steps_per_epoch)
        progress_bar = tqdm(data, desc='current_steps')
        for step in progress_bar:
            time.sleep(0.1)
            Y_batch = Y_shuffle[step*batch_size:(step+1)*batch_size]
            # 选取该批次的训练数据
            optimizer.zero_grad()
            # 清除之前的梯度
            # get the outputs
            X_h = net(Y_batch)
            # 当我们自定义一个类继承自 nn.Module 时，PyTorch 会默认调用 forward 方法来定义模型的前向传播过程
            Y_h = torch.mm(X_h, W_d.T)
    
            # compute the losss
            loss1 = criterion1(Y_batch.float(), Y_h.float()) 
            loss2 = a * criterion2(X_h.float(), all_zeros.float())
            loss = loss1 + loss2
            
            loss.backward()
            # 通过损失函数计算梯度
            optimizer.step()  
            # 根据梯度更新模型参数
    
            with torch.no_grad():
                loss_list.append(loss.detach().data)
            
            
    return net, loss_list


if __name__ == '__main__':

    # 硬编码根路径
    root = 'tutorial_lista'

    # dimensions of the sparse signal, measurement and sparsity
    m, n, k = 100, 16, 5
    # number of test examples
    N = 5000

    # generate dictionary
    Psi = np.eye(m)
    Phi = np.random.randn(n, m)
    Phi = np.transpose(orth(np.transpose(Phi)))
    W_d = np.dot(Phi, Psi)

    # generate sparse signal Z and measurement X
    Z = np.zeros((N, m))
    X = np.zeros((N, n))
    for i in range(N):
        index_k = np.random.choice(a=m, size=k, replace=False, p=None)
        Z[i, index_k] = 5 * np.random.randn(k, 1).reshape([-1,])
        X[i] = np.dot(W_d, Z[i, :])

    # computing average reconstruction-SNR
    net, err_list = train_lista(X, W_d, 0.1, 2)

    with open('net.pkl', 'wb') as f:
        pickle.dump(f"{root}/net.pkl", f)
    with open('err_list.pkl', 'wb') as f:
        pickle.dump(f"{root}/err_list.pkl", f)

    Z = np.zeros((1, m))
    X = np.zeros((1, n))
    for i in range(1):
        index_k = np.random.choice(a=m, size=k, replace=False, p=None)
        Z[i, index_k] = 5 * np.random.randn(k, 1).reshape([-1,])
        X[i] = np.dot(W_d, Z[i, :])

    Z_recon = net(torch.from_numpy(X).float().to(device))
    Z_recon = Z_recon.detach().cpu().numpy()
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(X[0])
    plt.subplot(2,1,2)
    plt.plot(Z[0], label='real')
    plt.subplot(2,1,2)
    plt.plot(Z_recon[0], '.-', label='LISTA')
    plt.show()
    