import numpy as np
from scipy.io import loadmat

def dataLoading(rootPath,datasetName):
    pass

def NMSE(predicted, true):
    mse = np.mean((predicted - true) ** 2)  # 均方差
    true_var = np.var(true)  # 真实值的方差
    return mse / true_var


if __name__ == '__main__':
    saveFolder = 'tutorial_lista/data'
    dataReal = np.load(f"{saveFolder}/X_test/X_test_complex_org_D.npy")
    dataRecon = np.load(f"{saveFolder}/X_recon/X_recon_complex_org_D.npy")
    NMSE_for_X = NMSE(dataRecon,dataReal)


