import numpy as np
from scipy.stats import rayleigh
from scipy import io
import os

path  = os.getcwd()
data_path = os.path.join(path,'data')

# 添加信噪比函数
def add_complex_gaussian_noise(signal, snr_dB):
    """
    给定复数信号添加高斯白噪声，生成具有指定信噪比的带噪声复数信号。
    参数：
        signal: 原始复数信号，一个 NumPy 数组。
        snr_dB: 信噪比，以分贝为单位。
    返回：
        noisy_signal: 带噪声的复数信号，一个 NumPy 数组。
    """
    # 将复数信号分成实部和虚部
    # 计算信噪比对应的噪声方差
    snr_linear = 10**(snr_dB / 10.0)  # 将分贝转换为线性信噪比
    signal_power = np.mean(np.abs(signal)**2)   # 计算信号的功率
    noise_power = signal_power / snr_linear  # 计算噪声的功率
    noise_real = np.random.normal(scale=np.sqrt(noise_power/2), size=signal.shape)
    noise_imag = np.random.normal(scale=np.sqrt(noise_power/2), size=signal.shape)
    # noise_std = np.sqrt(noise_power)  # 计算噪声标准差
    # noise = noise_std * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))
    # 将噪声添加到复数信号上
    noisy_signal = signal + noise_real + 1j * noise_imag

    return noisy_signal

np.random.seed(12)
# 雷达参数设置
delta_B = 0.084
nums_antenna  = 8
bn = np.linspace(0,(nums_antenna-1)*delta_B,nums_antenna).reshape(8,1)
lambda_radar = 0.021
Rmin  = 1200
rou_s = lambda_radar*Rmin/2/bn[-1]
detla_samb = lambda_radar*Rmin/2/delta_B
sl = np.arange(1,51).reshape(1,50)
m = len(bn)
n = sl.shape[1]
k = 1
nums_sample_1  = 10000
nums_sample_2 = 5000
# 生成观测矩阵
SNR = [10,20,30]
for SNR_i in SNR:
    D = np.exp(1j*4*np.pi*bn*sl/lambda_radar/Rmin)
    D_change_complex = add_complex_gaussian_noise(D,SNR_i)
    np.save(os.path.join(data_path,f'D_change_SNR{SNR_i}_complex.npy'),D_change_complex) 
    io.savemat(os.path.join(data_path,f'D_change_SNR{SNR_i}.mat'),{'D_SNR':D_change_complex})
    D_real = np.real(D_change_complex)
    D_imag = np.imag(D_change_complex)
    D_change  = np.vstack((np.hstack((D_real, -D_imag)), np.hstack((D_imag, D_real))))
    np.save(os.path.join(data_path,f'D_change_SNR{SNR_i}.npy'),D_change) 
    # 生成散射系数
    sigma = 2.0  # 瑞利分布的尺度参数
    size = (nums_sample_1*11,)  # 生成样本的数量
    scale = 4 #缩放因子
    # 生成符合瑞利分布的随机数
    Amplitude = rayleigh.rvs(scale=sigma, size=size)
    Amplitude = Amplitude*scale/max(Amplitude)
    phi = np.random.uniform(0, 1, nums_sample_1*11)
    # 生成不同信噪比下的训练数据
    for SNR_j in SNR:
        Y_1 = np.zeros((nums_sample_1,m),dtype=complex)
        X_1 = np.zeros((nums_sample_1,n),dtype=complex)
        for i in range(nums_sample_1):
            index = np.random.choice(a = n,size = 1,replace = False,p=None).astype(int)
            X_1[i,index] = Amplitude[i]*np.exp(1j*2*np.pi*phi[i])
            Y_1[i] = D_change_complex@X_1[i,:]
        Y_1 = add_complex_gaussian_noise(Y_1,SNR_j)
        # 间距不同的双散射体生成
        # 距离因子
        alpha = 0.1*np.arange(3,13)
        nums_sample_2 = 5000
        X_train = X_1
        Y_train = Y_1
        for i in range(len(alpha)):
            n_2 = n - alpha[i]*rou_s
            n_2 = int(np.floor(n_2)-1)
            X_2 = np.zeros((nums_sample_2,n),dtype=complex)
            Y_2 = np.zeros((nums_sample_2,m),dtype=complex)
            for j in range(nums_sample_2):
                index_k1 = np.random.choice(a = n_2,size = 1,replace = True, p=None).astype(int)
                index_k2 = int(index_k1+np.floor(rou_s*alpha[i]))
                X_2[j,index_k1] = Amplitude[10000+10000*i+2*j-1]*np.exp(1j*2*np.pi*phi[10000+10000*i+2*j-1])
                X_2[j,index_k2] = Amplitude[10000+10000*i+2*j]*np.exp(1j*2*np.pi*phi[10000+10000*i+2*j])
                Y_2[j] = D_change_complex@X_2[j,:]
            Y_2 = add_complex_gaussian_noise(Y_2,SNR_j)
            X_train = np.concatenate((X_train,X_2),axis=0)
            Y_train = np.concatenate((Y_train,Y_2),axis=0)
        np.save(os.path.join(data_path,f'X_train_complex_SNR{SNR_j}_DSNR{SNR_i}.npy'),X_train)
        np.save(os.path.join(data_path,f'Y_train_complex_SNR{SNR_j}_DSNR{SNR_i}.npy'),Y_train)
        filename_x1_mat =  f'X_train_complex_SNR{SNR_j}_DSNR{SNR_i}.mat'
        filename_y1_mat =  f'Y_train_complex_SNR{SNR_j}_DSNR{SNR_i}.mat'
        io.savemat(os.path.join(data_path,filename_x1_mat),{'X':X_train})
        io.savemat(os.path.join(data_path,filename_y1_mat),{'Y':Y_train})
        X_train_real = np.real(X_train)
        X_train_imag = np.imag(X_train)
        X_train = np.concatenate((X_train_real,X_train_imag),axis = 1)
        Y_train_real = np.real(Y_train)
        Y_train_imag = np.imag(Y_train)
        Y_train = np.concatenate((Y_train_real,Y_train_imag),axis = 1)
        filename_xtrain = f'X_train_DSNR{SNR_i}_SNR{SNR_j}.npy'
        filename_ytrain = f'Y_train_DSNR{SNR_i}_SNR{SNR_j}.npy'
        filename_x1 = f'X_train_DSNR{SNR_i}_SNR{SNR_j}.mat'
        filename_y1 = f'Y_train_DSNR{SNR_i}_SNR{SNR_j}.mat'
        np.save(os.path.join(data_path,filename_xtrain),X_train)
        np.save(os.path.join(data_path,filename_ytrain),Y_train)
        io.savemat(os.path.join(data_path,filename_x1),{'X':X_train})
        io.savemat(os.path.join(data_path,filename_y1),{'Y':Y_train})

