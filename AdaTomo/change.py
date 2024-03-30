from scipy import io
import numpy as np
import os

path  = os.getcwd()
data_path = os.path.join(path,'data')
gamma_valid = io.loadmat('/home/amax/Wcx/wangcx/code/AdaTomo/data/gamma_valid_facade.mat')
gn_valid = io.loadmat('/home/amax/Wcx/wangcx/code/AdaTomo/data/gn_valid_facade.mat')
gn = gn_valid['gn_valid_facade']
gamma = gamma_valid['gamma_valid_facade']
gn_real = np.real(gn)
gn_imag = np.imag(gn)
gamma_real = np.real(gamma)
gamma_imag = np.imag(gamma)
gn = np.concatenate((gn_real,gn_imag),axis = 1)
gamma = np.concatenate((gamma_real,gamma_imag),axis = 1)
filename_Y = 'gn_valid_facade.npy'
filename_X = 'gamma_valid_facade.npy'
np.save(os.path.join(data_path,filename_Y),gn)
np.save(os.path.join(data_path,filename_X),gamma)


