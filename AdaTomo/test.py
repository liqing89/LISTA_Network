import numpy as np

x = np.load('/home/amax/Wcx/wangcx/code/AdaTomo/data/X_train_org.npy')
y = np.load('/home/amax/Wcx/wangcx/code/AdaTomo/data/Y_train_org.npy')
D = np.load('/home/amax/Wcx/wangcx/code/AdaTomo/data/original_D.npy')
print(x.shape)
print(y.shape)
print(D.shape)