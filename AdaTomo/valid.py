import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import torch
import numpy as np
from torch.utils.data import DataLoader
import LISTA
import models
import dataset_general
import Change_model
import torch.nn.functional as F
from scipy import io
from scipy.linalg import eigvalsh
import matplotlib.pyplot as plt
import gc
# 清除缓存
gc.collect()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# reserved_size = 10000 * 1024 * 1024  # 10MB
# torch.cuda.memory_reserved(reserved_size)
path  = os.getcwd()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

dataset_loader_Valid = DataLoader(dataset_general.Myvaliddataset(root= 'E:/code/AdaTomo/data'),batch_size=64, shuffle = False )
# 创建与原始模型相同的模型类，并实例化
# 读取模型
D = np.load('E:\code\AdaTomo\data\D.npy') # 观测矩阵
L = eigvalsh(D.T@D)
L = np.round(max(L))
num_samples = len(dataset_general.Myvaliddataset(root= 'E:/code/AdaTomo/data'))

model = LISTA.LISTA_Net( n=60, m=300, T = 10 , lambd=0.5, L=L , D = D )
# model = models.Adaptive_ISTA(n=60, m=300,  D = D,T =10, lambd=0.5)
# model = Change_model.Adaptive_ISTA_AT(n=60, m=300,  D = D,T =10, lambd=0.5)
# model = Change_model.LISTA_Net(n=16, m=1500, LayerNum = 10,lambd = 0.1, L = L,  Dict = D,tao = 0.001)
# 加载模型参数

# 加载.pth文件中的权重
model_class_name = type(model).__name__
# tao_value = round(model.tao,3)
model_filename = f'{model_class_name}_T{model.T}_lambda{model.lambd}.pth'
model_path = os.path.join(BASE_DIR,'model',model_filename)
model.load_state_dict(torch.load(model_path))

# 将模型设置为评估模式
model.eval()
model.to(device)
x_hat_point = []
x_hat_point = torch.tensor(x_hat_point).float().to(device)
test_loss = 0
test_loss_x = 0
epoch = 1
with torch.no_grad ():
    for step, (b_y,b_A,b_x) in enumerate(dataset_loader_Valid):
        b_y = b_y.float().to(device)
        b_A = b_A.float().to(device)
        b_x = b_x.float().to(device)
        if True:
            b_y, b_A, b_x = b_y.cuda(), b_A.cuda(), b_x.cuda()
        x_hat = model(b_y,b_A,epoch )
        x_hat = x_hat.unsqueeze(2)
        x_hat_point = torch.concatenate((x_hat_point , x_hat), axis = 0)
        test_loss += F.mse_loss(x_hat, b_x, reduction="mean").data.item()
        zero = torch.zeros_like(b_x)
        test_loss_x += F.mse_loss(b_x,zero,reduction='mean').data.item()
        torch.cuda.empty_cache()
loss_test = test_loss / len(dataset_loader_Valid)
PSNR = -10*np.log10(test_loss/num_samples)
NMSE = -10*np.log10(test_loss/num_samples/test_loss_x)
x_hat_point = x_hat_point.detach().cpu().numpy()
print(
                " test loss %.8f,NMSE %.8f, PSNR %.8f"
                 % (test_loss, NMSE,PSNR)
            ) 
# 获取当前变量信息生成文件名
file_name = f'{model_class_name}_T{model.T}_lambda{model.lambd}_x_hat_point.mat'
io.savemat(os.path.join(BASE_DIR,'result',file_name),{'x_hat_point':x_hat_point})
