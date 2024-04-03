import numpy as np
import scipy as scipy
from scipy.stats import rayleigh
from scipy import io
import os

def dataPacking(X,X_name,folderPath):
    '''
        X: should be the data you want to save, in any format
        X_name: should be the name of the data, as a identifier in the format of <string>
    '''
    folder = f"{folderPath}/{X_name}" # 为每个类型的数据创造一个新的文件夹
    if not os.path.exists(folder):
        os.makedirs(folder)
    # 保存npy:观测向量集合Y和散射点空间向量集合X
    np.save(os.path.join(folder,f"{X_name}_complex_org_D.npy"),X)
    # 保存mat:观测向量集合Y和散射点空间向量集合X
    filename_x1 = f"{X_name}_complex_org.mat"
    io.savemat(os.path.join(folder,filename_x1),{'X':X})
    # 保存npy:观测向量集合Y和散射点空间向量集合X的实部和虚部拼接矩阵
    X_valid_real = np.real(X)
    X_valid_imag = np.imag(X)
    X = np.concatenate((X_valid_real,X_valid_imag),axis = 1)
    filename_xvalid = f"{X_name}_org.npy"
    np.save(os.path.join(folder,filename_xvalid),X)

if __name__ == "__main__":
    pass