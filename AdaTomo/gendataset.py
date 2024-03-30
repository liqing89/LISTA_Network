import numpy as np
from scipy.stats import rayleigh
from scipy import io
import os

path  = os.getcwd()
data_path = os.path.join(path,'data')

np.random.seed(12)
# 雷达参数设置
delta_B = 0.084
nums_antenna  = 8
bn = np.linspace(0,(nums_antenna-1)*delta_B,nums_antenna).reshape(8,1)
lambda_radar = 0.021
Rmin  = 1200
rou_s = lambda_radar*Rmin/2/bn[-1]
detla_samb = lambda_radar*Rmin/2/delta_B
sl = np.arange(1,51,1).reshape(1,50)
m = len(bn)
n = sl.shape[1]
k = 1
# 单散射生成
nums_sample_1  = 10000
# 初始化数据
D = np.zeros((m,n),dtype=complex)
Y_1 = np.zeros((nums_sample_1,m),dtype=complex)
X_1 = np.zeros((nums_sample_1,n),dtype=complex)
# 生成观测矩阵
D = np.exp(1j*4*np.pi*bn*sl/(lambda_radar*Rmin))
io.savemat(os.path.join(data_path,'D_complex.mat'),{'D':D}) 
D_real = np.real(D)
D_imag = np.imag(D)
original_D = np.vstack((np.hstack((D_real, -D_imag)), np.hstack((D_imag, D_real))))
np.save(os.path.join(data_path,'original_D.npy'),original_D) 
# 生成散射系数
sigma = 2.0  # 瑞利分布的尺度参数
size = (nums_sample_1*20,)  # 生成样本的数量
scale = 1 #缩放因子
# 生成符合瑞利分布的随机数
Amplitude = rayleigh.rvs(scale=sigma, size=size)
Amplitude = Amplitude*scale/max(Amplitude)
phi = np.random.uniform(0, 1, nums_sample_1*20)
choice_1 = np.arange(10,n)
for i in range(nums_sample_1):
    index = np.random.choice(choice_1,size = 1,replace = False,p=None).astype(int)
    X_1[i,index] = Amplitude[i]*np.exp(1j*2*np.pi*phi[i])
    Y_1[i] = D@X_1[i,:]
# 间距不同的双散射体生成
# 距离因子
alpha = 0.1*np.arange(1,19)
nums_sample_2 = 5000
X_train = X_1
Y_train = Y_1
for i in range(len(alpha)):
    n_2 = n - alpha[i]*rou_s
    n_2 = int(np.floor(n_2)-1)
    choice_2 = np.arange(1,n_2)
    X_2 = np.zeros((nums_sample_2,n),dtype=complex)
    Y_2 = np.zeros((nums_sample_2,m),dtype=complex)
    for j in range(nums_sample_2):
        index_k1 = np.random.choice(choice_2,size = 1,replace = True, p=None).astype(int)
        index_k2 = int(index_k1+np.floor(rou_s*alpha[i]))
        X_2[j,index_k1] = Amplitude[10000+10000*i+2*j-1]*np.exp(1j*2*np.pi*phi[10000+10000*i+2*j-1])
        X_2[j,index_k2] = Amplitude[10000+10000*i+2*j]*np.exp(1j*2*np.pi*phi[10000+10000*i+2*j])
        Y_2[j] = D@X_2[j,:]
    X_train = np.concatenate((X_train,X_2),axis=0)
    Y_train = np.concatenate((Y_train,Y_2),axis=0)
np.save(os.path.join(data_path,'X_train_complex_org_D.npy'),X_train)
np.save(os.path.join(data_path,'Y_train_complex_org_D.npy'),Y_train) 
filename_x1 = 'X_train_complex_org.mat'
filename_y1 = 'Y_train_complex_org.mat'
io.savemat(os.path.join(data_path,filename_x1),{'X':X_train})
io.savemat(os.path.join(data_path,filename_y1),{'Y':Y_train})
X_train_real = np.real(X_train)
X_train_imag = np.imag(X_train)
X_train = np.concatenate((X_train_real,X_train_imag),axis = 1)
Y_train_real = np.real(Y_train)
Y_train_imag = np.imag(Y_train)
Y_train = np.concatenate((Y_train_real,Y_train_imag),axis = 1)
filename_xtrain = 'X_train_org.npy'
filename_ytrain = 'Y_train_org.npy'
np.save(os.path.join(data_path,filename_xtrain),X_train)
np.save(os.path.join(data_path,filename_ytrain),Y_train)
