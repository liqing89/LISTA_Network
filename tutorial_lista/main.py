import torch
import torch.nn as nn
import numpy as np
import pickle
from scipy.linalg import orth
from LISTA import LISTA, train_lista
import matplotlib.pyplot as plt
import multiprocessing
from funcWrapper import dataPacking

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# if torch.cuda.is_available():
#     devices = [torch.device("cuda:1"), torch.device("cuda:2"),torch.device("cuda:3"),torch.device("cuda:4"),torch.device("cuda:5")]
# else:
#     devices = [torch.device("cpu")]

sourceRoot = 'AdaTomo/data'
saveRoot = 'tutorial_lista'
parasRoot = 'tutorial_lista/trainingResults'

# ---------------------- 读取数据 ----------------------
X = np.load(f"{sourceRoot}/X_train_org.npy") # 稀疏信号(100000, 100)
Y = np.load(f"{sourceRoot}/Y_train_org.npy") # 观测信号(100000, 16)
D = np.load(f"{sourceRoot}/original_D.npy") # 字典矩阵(16, 100)
m, n, k = X.shape[1], Y.shape[1], 5
N = X.shape[0] # or Y.shape[0]

# 字典矩阵
Psi = np.eye(m)
Phi = np.random.randn(n, m)
Phi = np.transpose(orth(np.transpose(Phi))) # 为什么要对Phi进行正交化
W_d = np.dot(Phi, Psi)

#
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42) # 拆分 X 和 Y 数组为训练集和测试集
N_train  = X_train.shape[0]
N_test = X_test.shape[0]
Y_measure = np.zeros((N_train, n))
# for i in range(N_train):
#     Y_measure[i,:] = np.dot(W_d, X_train[i, :]) # Y = Wd * X


# ---------------------- 引入网络 ----------------------
# # 1. 训练网络
# net, err_list = train_lista(Y_train, W_d, 0.1, 2) # learning rate = 0.1, epoch = 2
# # 存档训练结果
# with open(f"{parasRoot}/net.pkl", 'wb') as f:
#     pickle.dump(net,f)
# with open(f"{parasRoot}/err_list.pkl", 'wb') as f:
#     pickle.dump(err_list,f)

# 2. 现成网络
with open(f"{parasRoot}/net.pkl", 'rb') as f:
    net = pickle.load(f)
with open(f"{parasRoot}/err_list.pkl", 'rb') as f:
    err_list = pickle.load(f)


# ---------------------- 数据重建 ----------------------
X_recon = np.zeros((X_test.shape[0], n))
X_recon = net(torch.from_numpy(Y_test).float().to(device))
X_recon = X_recon.detach().cpu().numpy()


# ---------------------- 保存数据 ----------------------
dataSavePath = f"{saveRoot}/data"
dataPacking(X_recon,'X_recon',dataSavePath) # 重建数据的存储
dataPacking(X_test,'X_test',dataSavePath) # 对应真值的存储


# ---------------------- 作图 ----------------------
for i in range(10): # randomly generate 10 pics to show the effect of this network
    index = np.random.randint(0,N_test)
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(Y_test[index])
    plt.subplot(2,1,2)
    plt.plot(X_test[index], label='real')
    plt.subplot(2,1,2)
    plt.plot(X_recon[index], '.-', label='LISTA')
    plt.savefig("result.png")