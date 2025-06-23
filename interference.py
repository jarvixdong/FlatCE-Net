import os
import argparse
import pprint
import torch
import torch.nn as nn
import time
from torch import optim
from datetime import datetime
import torch.nn.functional as F
from ptflops import get_model_complexity_info
from dataset import *
from utils import *
# from torch.optim.lr_scheduler import StepLR
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import  DataLoader
from utils.tools import *
from net.cdrn import DnCNN_MultiBlock_ds
from net.build_model import DiaUNet1D
from net.basicCNN import SimpleCNN,LeNet


np.set_printoptions(suppress=True)  # 禁用科学计数法
np.set_printoptions(precision=3)  # 设置小数点后 6 位


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_conf(opts=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default="conf/interference.yml")
    parser.add_argument('--log_path', type=str, default="")
    parser.add_argument('--save_path', type=str, default="")

    args = parser.parse_args()

    # args.rank = int(os.environ.get('RANK', 0))
    # args.local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    # args.world_size = int(os.environ.get('WORLD_SIZE', 1))
    # args.host_ip = os.environ.get('MASTER_ADDR', None)
    # args.host_port = os.environ.get('MASTER_PORT', None)
    return args

def build_network(config):
    model_name = config.name
    model_params = config.params

    # 使用反射机制加载模型类
    model_class = globals().get(model_name)
    if model_class is None:
        raise ValueError(f"未找到模型类 '{model_name}'，请检查配置文件和模型定义。")

    # 创建模型实例
    model = model_class(**model_params)
    return model


def main1():

    args = parse_conf()
    cfg = get_config(args.config, args)

    set_all_seed(cfg.get("seed", 10))

    test_dataset = DataGenerator2(path=cfg.dataset.test_path,with_Vpinv=False)
    dataloader = DataLoader(test_dataset,batch_size = cfg.dataloader.batch_size,shuffle=cfg.dataloader.shuffle)
    
    model = build_network(cfg.model).to(device)
    # print('model:',model)
    model,_ = model.load_model(cfg.resume_path,device=device)
    # print('model:',model)
    
    model.eval()
    
    sum_ls_mse = 0.0   # 累计 LS 误差平方和
    sum_mse = 0.0      # 累计网络输出误差平方和
    sum_target = 0.0   # 累计真值平方和

    # start_time = time.time()
    time_lst = []
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            
            # 将输入/目标/等辅助参数放到 GPU（可视实际需求而定）
            inputs = batch[0].cuda(non_blocking=True)
            targets = batch[1].cuda(non_blocking=True)
            # vpinv  = batch[2].to(device)  # 如确有需要可以继续用

            # 模型前向推理
            print("input shape:",inputs.shape)
            B, C, H, W = inputs.shape
            # inputs = inputs.view(B, C, -1)
            start_time = time.time()
            outputs = model(inputs)
            end_time = time.time()
            time_lst.append(end_time-start_time)
            print("Model interference time",end_time-start_time)
            outputs = outputs.view(B, C, H, W)
            print('target:',targets.shape,torch.min(targets),torch.max(targets))
            print("output:",outputs.shape,torch.min(outputs),torch.max(outputs))

            # 直接在 GPU 上计算误差平方
            diff_ls    = (inputs - targets) ** 2
            diff_model = (outputs - targets) ** 2
            targets_sq = targets ** 2

            # 将每个batch的误差、真值平方和累加到Python标量中
            sum_ls_mse += diff_ls.sum().item()
            sum_mse    += diff_model.sum().item()
            sum_target += targets_sq.sum().item()
            # break
            # if idx == 19:
            #     break
    
    avg_nmse = sum_mse / sum_target
    avg_ls_nmse = sum_ls_mse / sum_target
    print("avg_nmse:",avg_nmse)
    print("avg_ls_nmse:",avg_ls_nmse)

def main2():

    args = parse_conf()
    cfg = get_config(args.config, args)

    set_all_seed(cfg.get("seed", 10))

    test_dataset = DataGenerator2(path=cfg.dataset.test_path,with_Vpinv=False)
    dataloader = DataLoader(test_dataset,batch_size = cfg.dataloader.batch_size,shuffle=cfg.dataloader.shuffle)
    
    model = build_network(cfg.model).to(device)
    # print('model:',model)
    model,_ = model.load_model(cfg.resume_path,device=device)
    # print('model:',model)
    
    model.eval()
    
    dl_nmse_sum = 0.0   
    ls_nmse_sum = 0.0     


    # start_time = time.time()
    time_lst = []
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            
            # 将输入/目标/等辅助参数放到 GPU（可视实际需求而定）
            inputs = batch[0].cuda(non_blocking=True)
            targets = batch[1].cuda(non_blocking=True)
            # vpinv  = batch[2].to(device)  # 如确有需要可以继续用

            # 模型前向推理
            print("input shape:",inputs.shape)
            B, C, H, W = inputs.shape
            # inputs = inputs.view(B, C, -1)
            start_time = time.time()
            outputs = model(inputs)
            end_time = time.time()
            time_lst.append(end_time-start_time)
            print("Model interference time",end_time-start_time)
            outputs = outputs.view(B, C, H, W)
            print('target:',targets.shape,torch.min(targets),torch.max(targets))
            print("output:",outputs.shape,torch.min(outputs),torch.max(outputs))

            # 直接在 GPU 上计算误差平方
            diff_ls    = (inputs - targets) ** 2
            diff_model = (outputs - targets) ** 2
            targets_sq = targets ** 2

            print("diff_model and target shape:",diff_model.shape,targets_sq.shape)
            dl_nmse = diff_model.sum(-1).sum(-1).sum(-1)/targets_sq.sum(-1).sum(-1).sum(-1)
            ls_nmse = diff_ls.sum(-1).sum(-1).sum(-1)/targets_sq.sum(-1).sum(-1).sum(-1)
            
            print('dl_nmse:',dl_nmse.shape)
            print("ls_nmse:",ls_nmse.shape)
            dl_nmse_sum += (dl_nmse.sum().item()/20000)
            ls_nmse_sum += (ls_nmse.sum().item()/20000)
            # break

    print('len dotaloader:',len(dataloader))
    avg_nmse = dl_nmse_sum 
    avg_ls_nmse = ls_nmse_sum 
    print("avg_nmse:",avg_nmse)
    print("avg_ls_nmse:",avg_ls_nmse)
    
main1()