import os
from scipy import io
import numpy as np
import torch
# 将复数数据转化为实数域数据
def complex_to_column_vector(complex_array):
    real_part = np.real(complex_array)
    imag_part = np.imag(complex_array)
    # column_vector = np.concatenate((real_part,imag_part),axis=1)
    column_vector = np.zeros((complex_array.shape[0],2*complex_array.shape[1],complex_array.shape[2]))
    for i in range(complex_array.shape[0]):
        for j in range(complex_array.shape[1]):
            column_vector[i,j*2,:] = real_part[i,j,:]
            column_vector[i,j*2+1,:] = imag_part[i,j,:]
    return column_vector

def transform_array(array):
    real_part = np.real(array)
    imag_part = np.imag(array)
    transformed_array = np.zeros((array.shape[0], array.shape[1]*2, array.shape[2]*2))
    # transformed_array = np.zeros(( array.shape[0]*2, array.shape[1]*2))
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            for k in range(array.shape[2]):
                transformed_array[i, j*2, k*2] = real_part[i, j, k]
                transformed_array[i, j*2, k*2+1] = -imag_part[i, j, k]
                transformed_array[i, j*2+1, k*2] = imag_part[i, j, k]
                transformed_array[i, j*2+1, k*2+1] = real_part[i, j, k]
    # for j in range(array.shape[0]):
    #     for k in range(array.shape[1]):
    #         transformed_array[j*2, k*2] = real_part[ j, k]
    #         transformed_array[j*2, k*2+1] = -imag_part[ j, k]
    #         transformed_array[j*2+1, k*2] = imag_part[ j, k]
    #         transformed_array[ j*2+1, k*2+1] = real_part[ j, k]
    return transformed_array

# 导入数据，合并单散射体数据和双散射体数据
path  = os.getcwd()
data_path = os.path.join(path,'data')
# # snr_list = [0, 3, 6, 10, 15, 18, 20, 24, 27, 30]
# snr_list = [0, 5, 10, 15, 20, 25, 30]
# i_value = [1,2,3,4,5,6,7,8,9,10,11,12]
# filename_x0 = os.path.join(data_path,'X_alpha_0.mat')
# filename_D0 = os.path.join(data_path,'D_matrix_alpha_0.mat')
# D0 = io.loadmat(filename_D0)
# x0 = io.loadmat(filename_x0)
# X0 = x0['x_data']
# D0 = D0['D']
# for i in i_value:
#     filename_x = os.path.join(data_path,'X_alpha_{}.mat'.format(i))
#     filename_D = os.path.join(data_path,'D_matrix_alpha_{}.mat'.format(i))
#     x = io.loadmat(filename_x)
#     D = io.loadmat(filename_D)
#     X = x['x_data']
#     D = D['D']
#     X = np.vstack((X0,X))
#     X = complex_to_column_vector(X)
#     D = np.vstack((D0,D))
#     D = transform_array(D)
#     filename_x_new = os.path.join(data_path,'X_alpha_new_{}.mat'.format(i))
#     filename_D_new = os.path.join(data_path,'D_alpha_new_{}.mat'.format(i)) 
#     io.savemat(filename_D_new, {'D': D})   
#     io.savemat(filename_x_new, {'X': X})  
#     for snr in snr_list:
#         filename_y = os.path.join(data_path,'Y_alpha_{}_snr_{}.mat'.format(i,snr))
#         filename_y0 = os.path.join(data_path,'Y_alpha_0_snr_{}.mat'.format(snr))
#         y0 = io.loadmat(filename_y0)
#         y0 = y0['Y']
#         y = io.loadmat(filename_y)
#         y = y['Y']
#         Y = np.vstack((y0,y))
#         Y = complex_to_column_vector(Y)
#         filename_y_new = os.path.join(data_path,'Y_alpha_new_{}_snr_{}.mat'.format(i,snr))
#         io.savemat(filename_y_new, {'Y': Y})  
D = io.loadmat('/home/amax/Wcx/wangcx/code/AdaTomo/data/change_Amatrix.mat')
D = D['D_matrix'] 
D = transform_array(D)
filename1 = 'D_matrix.mat'
io.savemat(os.path.join(data_path,filename1), {'D_matrix': D})
Y = io.loadmat('/home/amax/Wcx/wangcx/code/AdaTomo/data/change_Y.mat')
Y = Y['Y_data'] 
Y = Y[:,np.newaxis]
Y = complex_to_column_vector(Y)
filename2 = 'Y_data.mat'
io.savemat(os.path.join(data_path,filename2), {'Y_data': Y})
x = io.loadmat('/home/amax/Wcx/wangcx/code/AdaTomo/data/x_data1.mat')
x = x['x_data'] 
x = x[:,np.newaxis]
x = complex_to_column_vector(x)
filename3 = 'x_data.mat'
io.savemat(os.path.join(data_path,filename3), {'x_data': x})


