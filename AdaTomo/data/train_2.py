
import numpy as np
import time
import torch.nn.functional as F
import torch
import dataset_general
from scipy.linalg import eigvalsh
from scipy import io
import matplotlib.pyplot as plt
import os
from argparse import ArgumentParser
from torch.utils.data import DataLoader
import model 
import gc
import datetime
# 清除缓存
gc.collect()
path  = os.getcwd()

# 参数设置
parser = ArgumentParser(description='LISTA-FIRST-TRY')
parser.add_argument('--Epoch', type=int, default=30, help='epoch number')
parser.add_argument('--n', type=int, default=60, help='A colum')
parser.add_argument('--m', type=int, default=600, help='A row')
parser.add_argument('--T', type=int, default=10, help='itertive nums')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lambd', type=float, default=0.1, help='regularization parameter:lambda')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.1, help='regularization parameter')
parser.add_argument('--cudaopt', type=bool, default=True , help='if choose cuda')
parser.add_argument('--step_size', type=int, default=10, help='step size')
parser.add_argument('--weight_decay', type=float, default=0.5,help='weigth_decay')
parser.add_argument('--momentum', type=float, default=0.7,help='weigth_decay')
parser.add_argument('--SAVE_MODEL', type=bool, default=False, help='if SAVE_MODEL')
args = parser.parse_args()

Epoch = args.Epoch

try:
    # The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
    torch.backends.cuda.matmul.allow_tf32 = False
    # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
    torch.backends.cudnn.allow_tf32 = False
except:
    pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 加载数据集
dataset_loader_Train = DataLoader(dataset_general.Mytraindataset(root= '/home/amax/Wcx/wangcx/code/AdaTomo/data',test_size=0.3,train=True),batch_size=64, shuffle = True )
dataset_loader_Valid = DataLoader(dataset_general.Mytraindataset(root= '/home/amax/Wcx/wangcx/code/AdaTomo/data',test_size=0.3,train=False),batch_size=128, shuffle = False )
train_batch_size = 64
valid_batch_size = 128
# assert dataset_loader_Train
# assert dataset_loader_Valid 
num_samples = len(dataset_general.Mytraindataset(root= '/home/amax/Wcx/wangcx/code/AdaTomo/data',test_size=0.3,train=True))
# 读取模型
D = np.load('/home/amax/Wcx/wangcx/code/AdaTomo/data/D.npy') # 观测矩阵
L = eigvalsh(D.T@D)
L = np.round(max(L))
# model = model.LISTA( n=60, m=600, lambd=0.5, T = 10 , D = D )
# model = model.Ada_LISTA(n=60, m=600,lambd=0.5 ,T =10,  D = D)
model = model.Ada_AT_LISTA(n=60, m=600,  lambd=1,T =10, D = D)

model.to(device)
# 优化器选择
# optimizer = torch.optim.Adam(
#     model.parameters(), 
#     lr=args.learning_rate)
#         # 学习率调度器调用
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=args.learning_rate,
    momentum=args.momentum,
    weight_decay=args.weight_decay)
# 
# 学习率策略
scheduler = torch.optim.lr_scheduler.StepLR(    
        optimizer, step_size=args.step_size, gamma=args.gamma
        )
loss_train = np.zeros(Epoch)
loss_test = np.zeros(Epoch)
time_test = np.zeros(Epoch)
NMSE = np.zeros(Epoch)
PSNR = np.zeros(Epoch)
t0 = time.perf_counter()
EPOCH_PRINT_NUM = 10 
mse = torch.nn.MSELoss()



# 训练过程
for epoch in range(Epoch):
    start_time = time.time()
    train_loss = 0
    train_loss_normalizer = 0
    for step, (b_y, b_A, b_x) in enumerate(dataset_loader_Train):
        
        b_y = b_y.float().to(device)
        b_A = b_A.float().to(device)
        b_x = b_x.float().to(device)
        if args.cudaopt:
            b_y, b_A, b_x = b_y.cuda(), b_A.cuda(), b_x.cuda()
        x_hat = model(b_y, b_A) 
        # x_hat = x_hat.unsqueeze(2) #LISTA网络需要加上这句话
        # D = b_A.expand(x_hat.shape[0],args.n,args.m)
        # D.to(device)
        # y = torch.bmm(D,x_hat)
        # loss = ((x_hat-b_x)**2).sum()
        loss = ((x_hat-b_x)**2).mean()
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()
        #torch.nn.utils.clip_grad_norm(model.parameters(),1)
        optimizer.step()  # apply gradients
        model.zero_grad()
        train_loss_normalizer += ((b_x**2)).mean().item()
        train_loss += loss.item()
    # PSNR[epoch] = -10*np.log10(train_loss/num_samples)
    loss_train[epoch] = train_loss / train_batch_size
    NMSE[epoch] = 10*np.log10(train_loss/train_loss_normalizer)
    end_time = time.time()
    time_test[epoch] = end_time-start_time
    if scheduler is not None:
            scheduler.step()
    print(
                "Epoch %d, Train loss %.8f,NMSE %.8f,  time %.2f"
                 % (epoch, loss_train[epoch], NMSE[epoch],time_test[epoch])
            ) 
# 画训练损失函数
x_label = np.arange(len(loss_train))
plt.plot(x_label, loss_train, label='train loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Train Loss')
plt.legend()
plt.switch_backend('agg')
plt.savefig("Tain_loss.jpg")

# 保存模型
model_class_name = type(model).__name__
model_filename = f'{model_class_name}_T{model.T}_lambda{model.lambd}.pth'
model_path = os.path.join(BASE_DIR,'model',model_filename)
torch.save(model.state_dict(),model_path)
file_path = os.path.join(BASE_DIR,'result','new_evalution.txt')
current_time = datetime.datetime.now()
# 将时间格式化为字符串
formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
with open(file_path,'a') as file :
     file.write(formatted_time +'\n')
     file.write(f'Model:{model_class_name}\n')
     file.write(f'train_loss:{loss_train}\n')
     file.write(f'NMSE:{NMSE}\n')
     file.write(f'PSNR:{PSNR}\n')
# # 数据集验证过程
# model.to(device)
# x_hat_facade = []
# x_hat_facade = torch.tensor(x_hat_facade).float().to(device)
# model.eval()
# test_loss = 0
# test_loss_x = 0
# num_samples_valid = len(dataset_general.Myvaliddataset(root= '/home/amax/Wcx/wangcx/code/AdaTomo/data'))
# for step, (b_y,b_A,b_x) in enumerate(dataset_loader_Valid):
#     b_y = b_y.float().to(device)
#     b_A = b_A.float().to(device)
#     b_x = b_x.float().to(device)
#     if args.cudaopt:
#         b_y, b_a, b_x = b_y.cuda(), b_A.cuda(), b_x.cuda()
#     x_hat = model(b_y,b_A,epoch )
#     # x_hat = x_hat.unsqueeze(2) 
#     x_hat_facade = torch.concatenate((x_hat_facade,x_hat), axis = 0)
#     test_loss += F.mse_loss(x_hat, b_x, reduction="mean").data.item()
#     zero = torch.zeros_like(b_x)
#     test_loss_x += F.mse_loss(b_x,zero,reduction='mean').data.item()
# loss_test[epoch] = test_loss / len(dataset_loader_Valid)
# time_test[epoch] = time.perf_counter() - t0
# NMSE[epoch] = -10*np.log(test_loss/num_samples_valid/test_loss_x)
# PSNR[epoch] = -10*np.log(test_loss/num_samples_valid)
#     # Print
#     # if epoch % EPOCH_PRINT_NUM == 0:
# print(
#            "Epoch %d, Train loss %.8f,NMSE %.8f, PSNR %.8f, time %.2f"
#                  % (epoch, loss_train[epoch], NMSE[epoch],PSNR[epoch],time_test[epoch])
#             )  
# x_hat_facade = x_hat_facade.detach().cpu().numpy()

# # 获取当前变量信息生成文件名
# # file_name = f'{model_class_name}_T{model.T}_lambda{model.lambd}_tao{model.tao}_x_hat_facade.mat'
# file_name = f'{model_class_name}_T{model.T}_lambda{model.lambd}_x_hat_facade.mat'
# io.savemat(os.path.join(BASE_DIR,'data',file_name),{'x_hat_facade':x_hat_facade})

# # 画出loss损失
# x_label = np.arange(len(loss_test))
# plt.plot(x_label, loss_train, label='test loss')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.title('Test Loss')
# plt.legend()
# plt.switch_backend('agg')
# plt.savefig("Test_loss.jpg")      
