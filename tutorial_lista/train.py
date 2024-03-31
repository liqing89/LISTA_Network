import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import orth
from LISTA import LISTA,train_lista
from func import shrinkage, ista
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# dimensions of the sparse signal, measurement and sparsity
m, n, k = 1000, 256, 5
# number of test examples
N = 5000

# generate dictionary
Psi = np.eye(m)
Phi = np.random.randn(n, m)
Phi = np.transpose(orth(np.transpose(Phi)))
W_d = np.dot(Phi, Psi)
print(W_d.shape)

# generate sparse signal Z and measurement X
Z = np.zeros((N, m))
X = np.zeros((N, n))
for i in range(N):
    index_k = np.random.choice(a=m, size=k, replace=False, p=None)
    Z[i, index_k] = 5 * np.random.randn(k, 1).reshape([-1,])
    X[i] = np.dot(W_d, Z[i, :])

print(X.shape)
print(X[0].shape)

# computing average reconstruction-SNR
net, err_list = train_lista(X, W_d, 0.1, 2)

# ----------------------------------- Test stage -----------------------------------
# generate sparse signal Z and measurement X
Z = np.zeros((1, m))
X = np.zeros((1, n))
for i in range(1):
    index_k = np.random.choice(a=m, size=k, replace=False, p=None)
    Z[i, index_k] = 5 * np.random.randn(k, 1).reshape([-1,])
    X[i] = np.dot(W_d, Z[i, :])

Z_recon = net(torch.from_numpy(X).float().to(device))
Z_recon = Z_recon.detach().cpu().numpy()
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(X[0])
plt.subplot(2,1,2)
plt.plot(Z[0], label='real')
plt.subplot(2,1,2)
plt.plot(Z_recon[0], '.-', label='LISTA')
plt.show()

# ISTA
Z_recon, recon_errors = ista(np.mat(X).T, np.mat(W_d), 0.1, 2, 1000, 0.00001)
plt.subplot(2, 1, 2)
plt.plot(Z_recon, '--', label='ISTA')
plt.legend()
plt.show()
