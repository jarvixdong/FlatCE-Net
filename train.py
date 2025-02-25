import os
import argparse
import pprint
import torch
import torch.nn as nn
from torch import optim
from datetime import datetime
import torch.nn.functional as F
from dataset import *
from utils import *
# from torch.optim.lr_scheduler import StepLR
import torch.optim.lr_scheduler as lr_scheduler
# from torch.utils.data import TensorDataset, DataLoader
from utils.tools import *
from net.cdrn import DnCNN_MultiBlock_ds
from net.build_model import DiaUNet1D
from net.basicCNN import SimpleCNN,LeNet
# from net import build_network

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_conf(opts=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default="")
    parser.add_argument('--log_path', type=str, default="")
    # parser.add_argument('--local_rank', type=int, default=0)
    # parser.add_argument('--node_rank', type=int, default=0)
    # parser.add_argument('--DEBUG', type=int, default=0)
    # parser.add_argument('--debug_num_samples', type=int, default=10000)
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

def valid(model, dataloader):
    model.eval()

    sum_ls_mse = 0.0   # 累计 LS 误差平方和
    sum_mse = 0.0      # 累计网络输出误差平方和
    sum_target = 0.0   # 累计真值平方和

    with torch.no_grad():
        for idx, batch in enumerate(dataloader.valid_loader):
            # 将输入/目标/等辅助参数放到 GPU（可视实际需求而定）
            inputs = batch[0].cuda(non_blocking=True)
            targets = batch[1].cuda(non_blocking=True)
            # vpinv  = batch[2].cuda(non_blocking=True)  # 如确有需要可以继续用

            # 模型前向推理
            outputs = model(inputs)

            # 直接在 GPU 上计算误差平方
            diff_ls    = (inputs - targets) ** 2
            diff_model = (outputs - targets) ** 2
            targets_sq = targets ** 2

            # 将每个batch的误差、真值平方和累加到Python标量中
            sum_ls_mse += diff_ls.sum().item()
            sum_mse    += diff_model.sum().item()
            sum_target += targets_sq.sum().item()

    # 计算平均 NMSE 和 LS NMSE（全局归一化）
    avg_nmse = sum_mse / sum_target
    avg_ls_nmse = sum_ls_mse / sum_target
    
    return avg_nmse, avg_ls_nmse


    
def train(model, criterion, optimizer, scheduler ,dataloader, epochs=10, lr=1e-4):
    
    # criterion = nn.MSELoss()
    # criterion = nn.L1Loss()
    print('loss function::',criterion)
    # rayleigh_loss = RayleighKLLoss()
    rayleigh_loss = RayleighKLLoss_mat2()
    # criterion = nn.SmoothL1Loss(beta=1.0)
    for epoch in range(epochs):
        model.train()
        sum_mse = 0.0      # 累计网络输出误差平方和
        sum_target = 0.0   # 累计真值平方和
        for idx, batch in enumerate(dataloader.train_loader):
            batch = {'inputs': batch[0].cuda(non_blocking=True),
                    'targets': batch[1].cuda(non_blocking=True),
                    'Vpinv': batch[2].cuda(non_blocking=True)}
            inputdata = batch['inputs']
            # print('input data:',inputdata[0][0])
            target = batch['targets']
            # vpinv = batch['Vpinv']

            output = model(inputdata)
            loss = criterion(output, target)
            

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                # nmse_train = cal_NMSE4(output.detach().cpu().numpy(),target.detach().cpu().numpy())
                diff_model = (output - target) ** 2
                target_sq = target ** 2
                sum_mse    += diff_model.sum().item()
                sum_target += target_sq.sum().item()
        
        train_avg_nmse = sum_mse / sum_target
    
        # break
        nmmse,ls_nmse = valid(model,dataloader)
        # scheduler.step(nmmse) 
        scheduler.step()      
        
        nmmse = float(f"{nmmse:.6f}")  # 保留 6 位小数
        ls_nmse = float(f"{ls_nmse:.6f}")  # 保留 6 位小数
        nmse_train = float(f"{train_avg_nmse:.6f}")  # 保留 6 位小数

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{current_time}] Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}, Train_MMSE: {nmse_train}, NMMSE: {nmmse}, LS_NMSE: {ls_nmse}, Lr: {optimizer.param_groups[0]['lr']}")

def main_MSetup():

    args = parse_conf()
    cfg = get_config(args.config, args)
    print(f"Train.py PID: {os.getpid()}\n")
    
    nmse = cal_NMSE_by_matpath_h5(cfg.dataset.valid_path,"H_est_MMSE_all_data")
    print("NMMSE of valid dataset::",nmse)
    dataloader = BaseBunch.load_data(cfg.dataset,cfg.dataloader)

    # model = DnCNN_MultiBlock_ds(block=3, depth=18, image_channels=2,filters=64, use_bnorm=True).to(device)
    # model = DiaUNet1D(2,2,cfg.model.channel_index,cfg.model.num_layers).to(device)
    
    model = build_network(cfg.model).to(device)
    
    # ---------------------------cal parameter------------------------
    total_params = sum(p.numel() for p in model.parameters())  # 所有参数总数
    model_size_mb = total_params * 4 / 1024**2  # float32 => 4 bytes

    print("config_path:",cfg.config)
    pprint.pprint(cfg)
    # print(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
    # for key, value in cfg.items():
    #     print(f"{key}: {value}")

    print("model::",model)
    print(f"Estimated model size: {model_size_mb:.2f} MB")
    
    optimizer_class = getattr(optim,cfg.trainer.optimizer.name)
    optimizer_params = cfg.trainer.optimizer.params
    optimizer = optimizer_class(model.parameters(),**optimizer_params)
    print('optimizer:',optimizer)

    scheduler_params = cfg.trainer.lr_scheduler.params
    scheduler_class = getattr(lr_scheduler, cfg.trainer.lr_scheduler.name)
    scheduler = scheduler_class(optimizer, **scheduler_params)

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, factor=cfg.trainer.lr_gamma, patience=cfg.trainer.patience,
    #     min_lr=float(cfg.trainer.min_lr), threshold=cfg.trainer.get("threshold", 0.001),
    #     threshold_mode=cfg.trainer.get("threshold_mode", "abs")
    # )
    print('scheduler:',scheduler)
    
    loss_class = getattr(torch.nn, cfg.trainer.loss)
    criterion = loss_class()

    train(model, criterion, optimizer, scheduler ,dataloader, epochs=cfg.trainer.epoch_num)

main_MSetup()
