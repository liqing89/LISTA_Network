import numpy as np

def shrinkage(x, theta):
    return np.multiply(np.sign(x), np.maximum(np.abs(x) - theta, 0))

def ista(X, W_d, a, L, max_iter, eps):

    eig, eig_vector = np.linalg.eig(W_d.T * W_d)
    assert L > np.max(eig)
    del eig, eig_vector
    
    W_e = W_d.T / L

    recon_errors = []
    Z_old = np.zeros((W_d.shape[1], 1))
    for i in range(max_iter):
        temp = W_d * Z_old - X
        Z_new = shrinkage(Z_old - W_e * temp, a / L)
        if np.sum(np.abs(Z_new - Z_old)) <= eps: break
        Z_old = Z_new
        recon_error = np.linalg.norm(X - W_d * Z_new, 2) ** 2
        recon_errors.append(recon_error)
        
    return Z_new, recon_errors