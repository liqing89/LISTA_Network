from scipy import io
import numpy as np
import os

path  = os.getcwd()
data_path = os.path.join(path,'data')
gamma_valid = io.loadmat('E:\code\AdaTomo\data\X_valid_complex_org.mat')
gn_valid = io.loadmat('E:\code\AdaTomo\data\Y_valid_complex_org.mat')
gn = gn_valid['Y']
gamma = gamma_valid['X']
gn_real = np.real(gn)
gn_imag = np.imag(gn)
gamma_real = np.real(gamma)
gamma_imag = np.imag(gamma)
gn = np.concatenate((gn_real,gn_imag),axis = 1)
gamma = np.concatenate((gamma_real,gamma_imag),axis = 1)
filename_Y = 'Y_valid.npy'
filename_X = 'X_valid.npy'
np.save(os.path.join(data_path,filename_Y),gn)
np.save(os.path.join(data_path,filename_X),gamma)


