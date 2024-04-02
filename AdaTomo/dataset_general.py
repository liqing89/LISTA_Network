import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import os
import os.path
import gc
from sklearn.model_selection import train_test_split
# 清除缓存
gc.collect()
path  = os.getcwd()
data_path = os.path.join(path,'data')
class Mytraindataset(Dataset):
    def __init__(self,root=data_path,test_size=0.3,train = True):
        self.root = data_path
        self.X = None
        self.Y = None
        self.D = None
        self.n = None
        self.m = None
        self.Y_train = None
        self.X_train = None
        self.Y_val = None
        self.X_val = None
        self.train = train
        self.load_data()
        self.split_data(test_size)

    def load_data(self):
        X = np.load(os.path.join(data_path,'X_train_org.npy')) # <class 'numpy.ndarray'>
        Y = np.load(os.path.join(data_path,'Y_train_org.npy'))
        D = np.load(os.path.join(data_path,'original_D.npy'))
        # NumPy数组转换为PyTorch张量
        self.X  = torch.tensor(X, dtype=torch.float32) # <class 'torch.Tensor'>
        self.Y  = torch.tensor(Y, dtype=torch.float32)
        self.D  = torch.tensor(D, dtype=torch.float32)
        self.num_pixels = self.Y.shape[0]
        [self.m,self.n] = self.D.shape
        self.Y = self.Y.reshape(-1,self.m,1) # torch.Size([100000, 16, 1])
        self.X = self.X.reshape(-1,self.n,1)

    def split_data(self,test_size):
        # 划分数据集为训练集和测试集
        X_train, X_val, Y_train, Y_val = train_test_split(self.X, self.Y, test_size=test_size, random_state=42)
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_val = X_val
        self.Y_val = Y_val

    def __len__(self):
        if self.train:
            return self.Y_train.shape[0]
        else:
            return self.Y_val.shape[0]
    
    def __getitem__(self, idx):
        if self.train:
            X = self.X_train[idx,:,:]
            Y = self.Y_train[idx,:,:]
        else:
            X = self.X_val[idx,:,:]
            Y = self.Y_val[idx,:,:]
        D = self.D
        return Y, D, X
    
class Myvaliddataset(Dataset):
    def __init__(self,root=data_path):
        self.root = data_path
        self.X = None
        self.Y = None
        self.D = None
        self.n = None
        self.m = None
        self.num_pixels = None
        self.batch_size = 64
        self.load_data()

    def load_data(self):
        # X = np.load(os.path.join(data_path,'gamma_valid_facade.npy'))
        Y = np.load(os.path.join(data_path,'gn_valid_paris.npy'))
        D = np.load(os.path.join(data_path,'D.npy'))
        # self.X  = torch.tensor(X,dtype=torch.float32)
        self.Y  = torch.tensor(Y,dtype=torch.float32)
        self.D  = torch.tensor(D,dtype=torch.float32)
        self.num_pixels = self.Y.shape[0]
        [self.n,self.m] = self.D.shape
        self.Y = self.Y.reshape(-1,self.n,1)
        # self.X = self.X.reshape(-1,self.m,1)
        

    def __len__(self):
        return self.Y.shape[0]
    
    def __getitem__(self, idx):
        # X = self.X[idx,:,:]
        Y = self.Y[idx,:,:]
        D = self.D
        return Y, D
    
# if __name__ == '__main__':
#     dset_Train = Mytraindataset(root= '/home/amax/Wcx/wangcx/code/AdaTomo/data' )
#     dset_Test = Myvaliddataset( root= '/home/amax/Wcx/wangcx/code/AdaTomo/data')