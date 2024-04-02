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
    for epoch in tqdm(range(100),desc='training process'):
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

    
    # ------------------- 导入数据 并进行参数初始化 ------------------
    root = '/Users/liqing/Research/LISTA_Network/AdaTomo/data'
    X = np.load(f"{root}/X_train_org.npy") # 稀疏信号(100000, 100)
    Y = np.load(f"{root}/Y_train_org.npy") # 观测信号(100000, 16)
    D = np.load(f"{root}/original_D.npy") # 字典矩阵(16, 100)
    m, n, k = X.shape[1], Y.shape[1], 5
    N = X.shape[0] # or Y.shape[0]
    
    
    # -------------------- [压缩感知base] 制作字典矩阵 ----------------------
    Psi = np.eye(m)
    Phi = np.random.randn(n, m)
    Phi = np.transpose(orth(np.transpose(Phi))) # 为什么要对Phi进行正交化
    W_d = np.dot(Phi, Psi)

    # ------------ [训练集] 制作稀疏向量 并使用字典矩阵生成观测值 ----------
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42) # 拆分 X 和 Y 数组为训练集和测试集
    # Z = np.zeros((N, m))
    # X = np.zeros((N, n))
    N_train  = X_train.shape[0]
    Y_measure = np.zeros((N_train, n))
    for i in range(N_train):
    #     # how to generate sparse signal Z:
    #     index_k = np.random.choice(a=m, size=k, replace=False, p=None) # 1.randomly ick k elements for set the location of non-zero elements
    #     Z[i, index_k] = 5 * np.random.randn(k, 1).reshape([-1,]) # 2.randomly generate the altitude of non-zero elements
        # how to generate measurement X:
        # X[i] = np.dot(W_d, Z[i, :]) # X = Wd * Z
        Y_measure[i,:] = np.dot(W_d, X_train[i, :]) # Y = Wd * X

    # ----------------- [网络生成] 给出测量值和字典矩阵 生成重构网络net -----------------
    net, err_list = train_lista(Y_measure, W_d, 0.1, 2)


    # --------- [测试集] 重新生成同字典矩阵下的稀疏向量和对应观测值 】------------
    # Z = np.zeros((1, m))
    # X = np.zeros((1, n))
    # for i in range(1):
    #     index_k = np.random.choice(a=m, size=k, replace=False, p=None)
    #     Z[i, index_k] = 5 * np.random.randn(k, 1).reshape([-1,])
    #     X[i] = np.dot(W_d, Z[i, :])


    # ---------- [网络应用] 使用net对测试集的观测值进行重构 反演稀疏向量 -----------
    # Z = np.zeros((1, m))
    # Z_recon = net(torch.from_numpy(X).float().to(device))
    # Z_recon = Z_recon.detach().cpu().numpy()
    Y_recon = np.zeros((Y_test.shape[0], m))
    for i in range(Y_test.shape[0]):
        Y_recon[i,:] = net(torch.from_numpy(X_test[i,:]).float().to(device))
        Y_recon[i,:] =  Y_recon[i,:].detach().cpu().numpy()


    # ------------------------------ 画图 -----------------------------------
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(X[0])
    plt.subplot(2,1,2)
    plt.plot(Y_test[0], label='real')
    plt.subplot(2,1,2)
    plt.plot(Y_recon[0], '.-', label='LISTA')
    plt.show()