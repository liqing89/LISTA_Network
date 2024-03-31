import numpy as np
from scipy.linalg import eigvalsh

root = '/Users/liqing/Research/LISTA_Network/AdaTomo/data'

x = np.load(f"{root}/X_train_org.npy") # (100000, 100)
y = np.load(f"{root}/Y_train_org.npy") # (100000, 16)
D = np.load(f"{root}/original_D.npy") # (16, 100)

L = eigvalsh(D.T@D)
L = np.round(max(L))