from utils.optimize_matrices import get_matrices
import torch
import torch.nn as nn
import numpy as np
import os
import pandas as pd
from utils.get_data import Synthetic, ComplexVectorDataset
import utils.algorithms as algo_norm
import utils.algorithms_comm as algo_comm
from torch.utils.data import DataLoader
from time import time
import utils.dataset_general
from scipy import io

import utils.conf as conf

device = conf.device

non_learned_algos = [algo_norm.ISTA, algo_norm.FISTA, algo_comm.ISTA, algo_comm.FISTA]


def train_model(m, n, s, k, p, model_fn, noise_fn, epochs, initial_lr, name, model_dir='res/model5/',
                matrix_dir='res/matrices/'):

    model_dir_name = os.path.basename(os.path.normpath(model_dir))
    # 根据 model_dir 的名字选择相应的数据文件夹
    if model_dir_name == 'alpha':
        data_dir = 'E:\code\\na-alista-master\\res\data\\alpha'
    elif model_dir_name == 'phi':
        data_dir = 'E:\code\\na-alista-master\\res\data\\phi'
    elif model_dir_name == 'am':
        data_dir = 'E:\code\\na-alista-master\\res\data\\am'
    elif model_dir_name == 'model':
        data_dir = 'E:\code\\na-alista-master\\res\data\\alpha\model_valid'
    else:
        data_dir = 'E:\code\\na-alista-master\\res\data'

    # 读取数据
    D = np.load('E:\code\\na-alista-master\\res\data\\D.npy')
    phi, W_frob = get_matrices(m, n, matrix_dir=matrix_dir)

    train_dataset = utils.dataset_general.Mytraindataset(root='E:\code\\na-alista-master\\res\data', test_size=0.3, train=False)
    # test_dataset = utils.dataset_general.Mytraindataset(root='/home/amax/Wcx/wangcx/code/na-alista-master/res/data', test_size=0.3, train=False)
    test_dataset = utils.dataset_general.Myvaliddataset(root=data_dir)
    # 创建DataLoader来批量加载数据
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    # 确定模型存储位置
    if model_dir_name =="model":
        file_names = test_dataset.get_file_names()
        # 从文件名中提取特定信息
        extracted_info = []
        for file_name in file_names:
            # 假设文件名格式为 'Y_valid_sidey.npy'，我们想提取 'sidey' 部分
            info = file_name.split('_')[-1]  # 以 '_' 分割文件名，并取最后一个部分
            info = os.path.splitext(info)[0]  # 去掉文件扩展名
            extracted_info.append(info)
            # 将列表转换为字符串
        extracted_info_str = '_'.join(extracted_info)
        model_dir_save = os.path.join(model_dir,extracted_info_str + '/')
    else:
        model_dir_save = model_dir

    if not os.path.exists(model_dir_save + name):
        os.makedirs(model_dir_save + name)

    if os.path.isfile(model_dir_save + name + "/train_log"):
        print("Results for " + name + " are already available. Skipping computation...")
        return None    

    model = model_fn(m, n, s, k, p, D, W_frob=W_frob, ).to(device)

    if type(model) not in non_learned_algos:
        opt = torch.optim.Adam(model.parameters(), lr=initial_lr)
        # scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)

    train_losses = []
    train_dbs = []
    test_losses = []
    test_dbs = []

    if type(model) in non_learned_algos:
        epochs = 1
    for i in range(epochs):
        if type(model) not in non_learned_algos:
            train_loss, train_db = train_one_epoch(model, train_loader, noise_fn, opt)
            # scheduler.step()
        else:
            train_loss = 0
            train_db = 0
        test_loss, test_db,result_paris,X_real = test_one_epoch(model, test_loader, noise_fn)
        # test_loss, test_db,result_paris = test_one_epoch(model, test_loader, noise_fn)


        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_dbs.append(train_db)
        test_dbs.append(test_db)
        
        
        if test_dbs[-1] == min(test_dbs) and type(model) not in non_learned_algos:
            print("saving!")
            model.save(model_dir_save + name + "/checkpoint")
            path = model_dir_save + name + "/result_model.mat"
            io.savemat(path,{'x_result':result_paris})
            path_x = model_dir_save + name + "/x_save.mat"
            io.savemat(path_x,{'x_real':X_real})

        # data.train_data.reset()

        print(i, train_db, test_db)

    print("saving results to " + model_dir_save + name + "/train_log")
    pd.DataFrame(
        {
            "epoch": range(epochs),
            "train_loss": train_losses,
            "test_loss": test_losses,
            "train_dbs": train_dbs,
            "test_dbs": test_dbs,
        }
    ).to_csv(model_dir_save + name + "/train_log")



    
def train_one_epoch(model, loader, noise_fn, opt):
    train_loss = 0
    train_normalizer = 0
    for i, (y, info,X) in enumerate(loader):
        X = X.to(device)
        X = X.squeeze()
        y = y.squeeze()
        info = info.to(device)
        y = y.to(device)
        opt.zero_grad()
        # y = torch.matmul(X, model.phi.T)
        X_hat, gammas, thetas = model(noise_fn(y), info)
        X_hat, gammas, thetas = model(y, info)
        loss = ((X_hat - X) ** 2).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        opt.step()
        train_normalizer += (X ** 2).mean().item()
        train_loss += loss.item()
    return train_loss / len(loader), 10 * np.log10(train_loss / train_normalizer)


def test_one_epoch(model, loader, noise_fn):
    test_loss = 0
    test_normalizer = 0
    result_model = []
    result_model = torch.tensor(result_model).float().to(device)
    x_real = []
    x_real = torch.tensor(x_real).float().to(device)
    with torch.no_grad():
        for i, (y, D, X) in enumerate(loader):
        # for i, (y,D) in enumerate(loader):
            X = X.to(device)
            D = D.to(device)
            X = X.squeeze()
            y = y.squeeze()
            y = y.to(device)
            X_hat, gammas, thetas = model(noise_fn(y), D)
            # X_hat, gammas, thetas = model(y, D)
            # X_hat = X_hat.unsqueeze(2)
            test_loss += ((X_hat - X) ** 2).mean().item()
            test_normalizer += (X ** 2).mean().item()
            # y_hat = torch.matmul(D,X_hat)
            # y_hat = y_hat.squeeze()
            result_model = torch.concatenate((result_model, X_hat), axis = 0)
            x_real = torch.concatenate((x_real, X), axis = 0)
            # test_loss += ((y_hat - y) ** 2).mean().item()
            # test_normalizer += (y ** 2).mean().item()
        X_real = x_real.detach().cpu().numpy()
        result_model = result_model.detach().cpu().numpy()
    return test_loss / len(loader), 10 * np.log10(test_loss / test_normalizer) ,result_model,X_real
    # return test_loss / len(loader), 10 * np.log10(test_loss / test_normalizer) ,result_model

# def evaluate_model(m, n, s, k, p, model_fn, noise_fn, name, model_dir='res/models/'):
#     phi, W_soft_gen, W_frob = get_matrices(m, n)
#     data = Synthetic(m, n, s, s)
#     model = model_fn(m, n, s, k, p, phi, W_soft_gen, W_frob).to(device)
#     model.load(model_dir + name + "/checkpoint")

#     test_loss = []
#     test_normalizer = []
#     sparsities = []
#     t1 = time()
#     with torch.no_grad():
#         for epoch in range(1):
#             for i, (X, info) in enumerate(data.train_loader):
#                 sparsities.extend(list((X != 0).int().sum(dim=1).detach().numpy()))
#                 X = X.to(device)
#                 info = info.to(device)
#                 y = torch.matmul(X, model.phi.T)
#                 X_hat, gammas, thetas = model(noise_fn(y), info)
#                 test_loss.extend(list(((X_hat - X) ** 2).cpu().detach().numpy()))
#                 test_normalizer.extend(list((X ** 2).cpu().detach().numpy()))
#             data.train_data.reset()
#     t2 = time()
#     runtime_evaluation = t2 - t1

#     test_loss = np.array(test_loss)
#     test_normalizer = np.array(test_normalizer)
#     sparsities = np.array(sparsities)

#     keys = []
#     counts = []
#     values = []
#     for s in sorted(np.unique(sparsities)):
#         count = (sparsities == s).mean()
#         if count > 10e-5:
#             keys.append(s)
#             counts.append(count)
#             values.append(
#                 10
#                 * np.log10(
#                     np.sum(test_loss[sparsities == s]) / np.sum(test_normalizer[sparsities == s])
#                 )
#             )

#     return keys, counts, values, 10 * np.log10(np.sum(test_loss) / np.sum(test_normalizer))
