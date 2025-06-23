import os
import random
import numpy as np
import sys
import h5py
import torch
from scipy.io import loadmat
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

def set_all_seed(seed=None, benchmark=True, deterministic=False, use_tf32=True):
    """This is refered to https://github.com/lonePatient/lookahead_pytorch/blob/master/tools.py.
    """
    if seed is not None:
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # some cudnn methods can be random even after fixing the seed
        # unless you tell it to be deterministic
    if deterministic:
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = benchmark
    if use_tf32 and torch.__version__ > "1.8.0":
        # only available for torch > 1.7
        # referred from https://pytorch.org/docs/stable/notes/cuda.html#tf32-on-ampere
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        

def load_data(path):
    data = loadmat(path)
    # print(data.keys())
    H_all = data["H_all"]
    H_est_LS_all = data["H_est_LS_all"]
    
    H_all_tp = H_all.transpose(2,0,1)
    H_est_LS_all_tp = H_est_LS_all.transpose(2,0,1)
    # print("H_all_tp:",H_all_tp.shape)
    # print("H_est_LS_all_tp:",H_est_LS_all_tp.shape)
    
    H_all_tp_sum = H_all_tp.sum(axis=(1,2))
    # print("H_all_tp_sum:",H_all_tp_sum)
    
    H_all_tp = H_all_tp[H_all_tp_sum!=0]
    H_est_LS_all_tp = H_est_LS_all_tp[H_all_tp_sum!=0]
    
    H_all_tp = np.concatenate([H_all_tp.real[:,:,:,np.newaxis],H_all_tp.imag[:,:,:,np.newaxis]],axis=-1)
    H_est_LS_all_tp = np.concatenate([H_est_LS_all_tp.real[:,:,:,np.newaxis],H_est_LS_all_tp.imag[:,:,:,np.newaxis]],axis=-1)
    
    return H_est_LS_all_tp,H_all_tp


# def cal_NMMSE(H_clean, H_denoise):
#     # 计算误差和信号的模平方
#     H_dis = np.abs(H_clean - H_denoise) ** 2
#     H_clean_abssq = np.abs(H_clean) ** 2
#     # print('H dis:',H_dis[0].shape,H_clean_abssq[0],H_clean_abssq[0])

    
#     H_dis_sum = H_dis.mean(axis=( 1, 2))
#     H_clean_abssq_sum = H_clean_abssq.mean(axis=( 1, 2))
    
#     # 避免除零错误
#     nmmse = np.where(H_clean_abssq_sum > 0, H_dis_sum / H_clean_abssq_sum, float('inf'))

#     # 计算平均 NMMSE
#     nmmse_mean = nmmse.mean()

#     # print('nmmse_mean:', nmmse_mean)
    
#     return nmmse_mean, nmmse


def cal_NMSE3(H_noise, H_clean):
    H_dis = (H_noise - H_clean) ** 2
    H_clean_sq = H_clean ** 2

    H_dis_sum = H_dis.sum(axis=( 1, 2, 3))
    H_clean_sq_sum = H_clean_sq.sum(axis=( 1, 2, 3))
    # print("H_dis_sum",H_dis_sum)
    # print("H_clean_sq_sum:",H_clean_sq_sum)
    
    nmmse_batch = H_dis_sum/H_clean_sq_sum
    nmmse_mean = nmmse_batch.mean()
    
    return nmmse_mean


def cal_NMSE4(H_noise, H_clean):
    H_dis = (H_noise - H_clean) ** 2
    H_clean_sq = H_clean ** 2

    H_dis_sum = H_dis.sum(axis=( 1, 2, 3))
    H_clean_sq_sum = H_clean_sq.sum(axis=( 1, 2, 3))
    # print("H_dis_sum",H_dis_sum)
    # print("H_clean_sq_sum:",H_clean_sq_sum)
    
    nmmse_mean = H_dis_sum.mean()/H_clean_sq_sum.mean()
    
    return nmmse_mean



def cal_NMSE_norm(H_noise, H_clean):
    print("H noise :",H_noise.shape)
    # # print("complex dis::",np.linalg.norm(H_noise[:,0,:,:]+1j*H_noise[:,1,:,:] - H_clean[:,0,:,:]+1j*H_clean[:,1,:,:], 'fro')**2/np.linalg.norm(H_clean[:,0,:,:]+1j*H_clean[:,1,:,:])**2)
    # # 差值复数张量 (batch, H, W)
    # diff = (H_noise[:,0,:,:] + 1j*H_noise[:,1,:,:]) - (H_clean[:,0,:,:] + 1j*H_clean[:,1,:,:])

    # # 计算 Frobenius 范数的平方: sum of |diff|^2
    # fro_norm_sq = np.sum(np.abs(diff)**2)

    # # 如果需要 Frobenius 范数本身:
    # fro_norm = np.sqrt(fro_norm_sq)

    # # 同理，真值的范数平方:
    # fro_norm_sq_clean = np.sum(np.abs(H_clean[:,0,:,:] + 1j*H_clean[:,1,:,:])**2)

    # # 计算 NMSE = ||diff||^2 / ||clean||^2
    # nmse = fro_norm_sq / fro_norm_sq_clean

    # print("NMSE =", nmse)
    
    
    H_dis = (H_noise - H_clean) ** 2
    H_clean_sq = H_clean ** 2


    H_dis_sum = H_dis.sum(axis=( 1, 2, 3))
    H_clean_sq_sum = H_clean_sq.sum(axis=( 1, 2, 3))
    print("H_dis_sum",H_dis_sum.shape)
    print("H_clean_sq_sum:",H_clean_sq_sum)

    
    nmmse_batch = H_dis_sum /H_clean_sq_sum
    nmmse_mean = nmmse_batch.mean()
    print("nmmse batch mean:",nmmse_mean)
    print("mse whole mean:",H_dis_sum.mean() /H_clean_sq_sum.mean())
    
    return nmmse_mean


def cal_NMSE_by_matpath(path,name):
    
    data = loadmat(path)
    # print(data.keys())
    H_clean = data["H_all_data"]
    H_noisy = data[name]
    print('H shape:',H_noisy.shape,H_clean.shape,H_clean[:,:,-2])
    x_data = np.concatenate([H_noisy.real[np.newaxis,:,:,:],H_noisy.imag[np.newaxis,:,:,:]],axis=0).transpose(3,0,1,2)
    y_data = np.concatenate([H_clean.real[np.newaxis,:,:,:],H_clean.imag[np.newaxis,:,:,:]],axis=0).transpose(3,0,1,2)
    print('xdata:',x_data.shape,y_data[-2])
    nmse = cal_NMSE3(x_data,y_data)
    return nmse

def cal_NMSE_by_matpath_h5(path,name):
    
    data = h5py.File(path, 'r')
    # print(data.keys())
    H_clean = data["H_all_data"][:]
    H_noisy = data[name][:]
    print('H shape:',H_noisy.shape,H_clean.shape)
    x_data = np.concatenate([H_noisy['real'][:,np.newaxis,:,:],H_noisy['imag'][:,np.newaxis,:,:]],axis=1)#.transpose(3,0,1,2)
    y_data = np.concatenate([H_clean['real'][:,np.newaxis,:,:],H_clean['imag'][:,np.newaxis,:,:]],axis=1)#.transpose(3,0,1,2)
    # print('xdata in cal_NMSE_by_matpath_h5:',x_data.shape)
    # nmse = cal_NMSE3(x_data,y_data)
    nmse = cal_NMSE4(x_data,y_data)
    # nmse = cal_NMSE_norm(x_data,y_data)
    return nmse

# path = '/mnt/parscratch/users/elq20xd/channel_estimation/cc_data/dataset6_withVpinv/valid_Dataset_dB-5_N32_K16_L10_S8_Setup1000_Reliz100.mat'
# nmse = cal_NMSE_by_matpath_h5(path,"H_est_MMSE_all_data")
# print('NMSE:',nmse)




def complex_matrix_multiply(A, B):
    # Real part of the result: (realA * realB - imagA * imagB)
    real_part = torch.einsum('bik,bkj->bij', A[:, 0, :, :], B[:, 0, :, :]) - torch.einsum('bik,bkj->bij', A[:, 1, :, :], B[:, 1, :, :])
    # Imaginary part of the result: (realA * imagB + imagA * realB)
    imag_part = torch.einsum('bik,bkj->bij', A[:, 0, :, :], B[:, 1, :, :]) + torch.einsum('bik,bkj->bij', A[:, 1, :, :], B[:, 0, :, :])

    # Stack real and imaginary parts along the second axis (complex dimension)
    C = torch.stack([real_part, imag_part], dim=1)

    return C


# # Example usage
# A = np.random.rand(5, 2, 3, 4)  # Batch of 5, 3x4 complex matrices
# B = np.random.rand(5, 2, 4, 2)  # Batch of 5, 4x2 complex matrices
# C = complex_matrix_multiply(A, B)
# print("Output shape:", C.shape)
# print("Output sample:", C[0])  # Print the result for the first item in the batch

def get_mat_inv(A):
        # Construct the complex matrices from the real and imaginary parts
    A_complex = A[:, 0] + 1j * A[:, 1]

    # Compute the inverse of each complex matrix in the batch
    A_inv = torch.linalg.inv(A_complex)
    
    A_inv_mat = torch.cat((A_inv.real.unsqueeze(1),A_inv.imag.unsqueeze(1)),dim=1)
    # print("A_inv_mat:",A_inv_mat.shape)
    return A_inv_mat

# a = torch.rand([5,2,3,3])
# b = get_mat_inv(a)
# print('b:',b.shape,b)


# 定义一个类，允许输出到控制台和文件
class Logger(object):
    def __init__(self, file):
        self.terminal = sys.stdout  # 保存原始标准输出
        self.log = open(file, "a")  # 打开日志文件

    def write(self, message):
        self.terminal.write(message)  # 输出到控制台
        self.log.write(message)  # 输出到文件

    def flush(self):
        pass  # Python 需要这个方法

# # 重定向标准输出和错误
# sys.stdout = Logger("console_output.log")
# sys.stderr = sys.stdout

# # 示例输出
# print("这是标准输出")
# print("同时输出到控制台和日志文件")
# raise ValueError("这是一个错误信息")


def kl_rayleigh_loss(pred, target):
    """
    计算 KL 散度损失，使模型的输出接近瑞利分布
    :param pred: 模型输出 (batch_size,2,K,N )
    :param sigma: 目标瑞利分布的尺度参数
    :return: KL Loss
    """
    pred = torch.sqrt(torch.sum(pred**2,dim=1))
    # print('pred shape:',pred.shape)
    pred = torch.clamp(pred, min=1e-6)  # 避免 log(0)
    
    target = torch.sqrt(torch.sum(target**2,dim=1))
    # print('pred shape:',target.shape)
    target = torch.clamp(target, min=1e-6)  # 避免 log(0)
    
    # pred = torch.softmax(pred.view(128,-1),dim=1)
    # target = torch.softmax(target.view(128,-1),dim=1)
    
    pred = pred.view(pred.shape[0],-1)
    target = target.view(target.shape[0],-1)
    print('pred and target shape:',pred.shape,target.shape,target[0].shape)
    # plt.hist(target_softmax[0].view(-1).detach().cpu().numpy(), bins=20, density=True, alpha=0.6, label="Empirical Data")
    plt.hist(target[0].detach().cpu().numpy(), bins=100, density=True, alpha=0.6, label="target Data")
    # plt.hist(pred[0].detach().cpu().numpy(), bins=100, density=True, alpha=0.6, label="pred Data")
    plt.legend()
    plt.savefig('target_sotmax.png')
    # 计算 KL 散度
    kl_loss = F.kl_div(pred,target,reduction='batchmean')
    print('kl loss:',kl_loss)
    
    return kl_loss

# KL Loss 正则项
# class RayleighKLLoss(nn.Module):
#     def __init__(self):
#         super(RayleighKLLoss, self).__init__()

#     def forward(self, y_pred, sigma):
#         y_pred = y_pred.view(y_pred.shape[0], -1)  # 保持 batch 维度，展开剩余维度
#         hist_values, bin_edges = torch.histogram(y_pred, bins=50, density=True)
#         bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # 计算 bin 中心点

#         # 计算理论瑞利分布 PDF
#         rayleigh_pdf = (bin_centers / sigma**2) * torch.exp(-bin_centers**2 / (2 * sigma**2))

#         # 避免 log(0)
#         kl_loss = torch.sum(hist_values * torch.log(hist_values / (rayleigh_pdf + 1e-8)))

#         return kl_loss
    

class RayleighKLLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(RayleighKLLoss, self).__init__()
        self.eps = eps  # 避免 log(0)

    def forward(self, y_pred, sigma):
        device = y_pred.device  # 保持原设备
        y_pred = torch.sqrt(torch.sum(y_pred**2,dim=1))
        # print('pred shape:',pred.shape)
        y_pred = torch.clamp(y_pred, min=1e-6)  # 避免 log(0)
        y_pred = y_pred.view(y_pred.shape[0], -1)
        
        y_pred_np = y_pred.detach().cpu().numpy()  # 转换为 NumPy

        # 计算直方图
        hist_values, bin_edges = np.histogram(y_pred_np, bins=50, density=True)
        hist_values = hist_values / (np.sum(hist_values) + self.eps)  # 归一化
        # print('his value:', hist_values.shape)

        # 转换回 PyTorch，并迁移回原设备
        hist_values = torch.tensor(hist_values, dtype=torch.float32, device=device)
        bin_edges = torch.tensor(bin_edges, dtype=torch.float32, device=device)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # 限制 sigma 范围，防止 sigma 太小或太大
        sigma = torch.clamp(sigma, min=0.1, max=100.0)

        # 计算理论瑞利分布 PDF，并加上 eps 以防止 0
        rayleigh_pdf = (bin_centers / sigma**2) * torch.exp(-bin_centers**2 / (2 * sigma**2))
        rayleigh_pdf = rayleigh_pdf + self.eps  # 避免 log(0)

        # 计算 KL Loss，防止 hist_values 为 0
        kl_loss = torch.sum(hist_values * torch.log((hist_values + self.eps) / rayleigh_pdf))

        return kl_loss
    

class RayleighKLLoss_mat(nn.Module):
    def __init__(self, bins=50, eps=1e-8):
        super(RayleighKLLoss_mat, self).__init__()
        self.bins = bins
        self.eps = eps  # 避免 log(0)

    def forward(self, y_pred, y_true):
        """
        计算 U-Net 输出与真实数据分布之间的 KL 散度
        :param y_pred: U-Net 预测输出 (batch_size, C, H, W) 或 (batch_size, N)
        :param y_true: 真实数据 (batch_size, C, H, W) 或 (batch_size, N)
        :return: KL Loss
        """
        
        y_pred = torch.sqrt(torch.sum(y_pred**2,dim=1))
        # print('pred shape:',pred.shape)
        y_pred = torch.clamp(y_pred, min=1e-6)

        y_true = torch.sqrt(torch.sum(y_true**2,dim=1))
        # print('pred shape:',pred.shape)
        y_true = torch.clamp(y_true, min=1e-6)
    
        device = y_pred.device
        batch_size = y_pred.shape[0]

        # 转换为 NumPy（必须在 CPU 计算直方图）
        y_pred_np = y_pred.view(batch_size, -1).detach().cpu().numpy()
        y_true_np = y_true.view(batch_size, -1).detach().cpu().numpy()

        # 计算 batch 级别的直方图
        hist_pred_list = []
        hist_true_list = []
        bin_edges_list = []
        for i in range(batch_size):
            hist_pred, bin_edges = np.histogram(y_pred_np[i], bins=self.bins, density=True)
            hist_true, _ = np.histogram(y_true_np[i], bins=bin_edges, density=True)  # 真实分布使用相同 bins
            hist_pred_list.append(hist_pred)
            hist_true_list.append(hist_true)
            bin_edges_list.append(bin_edges)

        # 转换为 PyTorch Tensor
        hist_pred = torch.tensor(np.array(hist_pred_list), dtype=torch.float32, device=device)
        hist_true = torch.tensor(np.array(hist_true_list), dtype=torch.float32, device=device)

        # 避免 0 值
        hist_pred = hist_pred + self.eps
        hist_true = hist_true + self.eps

        # 归一化，使得直方图和为 1
        hist_pred /= torch.sum(hist_pred, dim=1, keepdim=True)
        hist_true /= torch.sum(hist_true, dim=1, keepdim=True)

        # 计算 KL 散度 (batch_size,)
        kl_loss = torch.sum(hist_true * torch.log(hist_true / hist_pred), dim=1)

        return kl_loss.mean()  # 取 batch 平均 KL Loss
    

class RayleighKLLoss_mat2(nn.Module):
    def __init__(self, bins=50, eps=1e-8):
        super(RayleighKLLoss_mat2, self).__init__()
        self.bins = bins
        self.eps = eps  # 避免 log(0)

    def forward(self, y_pred, y_true):
        """
        计算 U-Net 输出与真实数据分布之间的 KL 散度
        :param y_pred: U-Net 预测输出 (batch_size, C, H, W) 或 (batch_size, N)
        :param y_true: 真实数据 (batch_size, C, H, W) 或 (batch_size, N)
        :return: KL Loss
        """
        device = y_pred.device
        batch_size = y_pred.shape[0]

        y_pred = torch.sqrt(torch.sum(y_pred**2,dim=1))
        # print('pred shape:',pred.shape)
        y_pred = torch.clamp(y_pred, min=1e-6)

        y_true = torch.sqrt(torch.sum(y_true**2,dim=1))
        # print('pred shape:',pred.shape)
        y_true = torch.clamp(y_true, min=1e-6)

        # 计算 batch 级别的直方图（用 torch.histc 计算）
        hist_pred = torch.zeros((batch_size, self.bins), device=device)
        hist_true = torch.zeros((batch_size, self.bins), device=device)

        min_val = min(y_pred.min(), y_true.min())
        max_val = max(y_pred.max(), y_true.max())

        bin_edges = torch.linspace(min_val, max_val, steps=self.bins + 1, device=device)

        for i in range(batch_size):
            hist_pred[i] = torch.histc(y_pred[i].float(), bins=self.bins, min=min_val.item(), max=max_val.item())
            hist_true[i] = torch.histc(y_true[i].float(), bins=self.bins, min=min_val.item(), max=max_val.item())

        # 归一化
        hist_pred = hist_pred + self.eps
        hist_true = hist_true + self.eps
        hist_pred = hist_pred / hist_pred.sum(dim=1, keepdim=True)
        hist_true = hist_true / hist_true.sum(dim=1, keepdim=True)

        # 计算 KL Loss (batch_size,)
        kl_loss = torch.sum(hist_true * torch.log(hist_true / hist_pred), dim=1)

        return kl_loss.mean()  # 取 batch 平均 KL Loss