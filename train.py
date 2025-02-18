import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from datetime import datetime
import numpy as np
import torch.nn.functional as F
from dataset import *
from utils import *
from torch.optim.lr_scheduler import StepLR
# from torch.utils.data import TensorDataset, DataLoader
from utils.tools import *
from net.cdrn import DnCNNMultiBlock,DnCNN_MultiBlock_ds
# from unet_1d import UNet1D,SDUNet1D,AttnSDUNet1D,SE_SDUNet1D,SE_UNet1D,SDUNet1D_3l
from net.build_model import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


    
def train(model, dataloader, epochs=10, lr=1e-4):
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = StepLR(optimizer, step_size=80, gamma=0.1)
    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()
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
        scheduler.step()      
        nmmse,ls_nmse = valid(model,dataloader)
        
        nmmse = float(f"{nmmse:.6f}")  # 保留 6 位小数
        ls_nmse = float(f"{ls_nmse:.6f}")  # 保留 6 位小数
        nmse_train = float(f"{train_avg_nmse:.6f}")  # 保留 6 位小数

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{current_time}] Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}, Train_MMSE: {nmse_train}, NMMSE: {nmmse}, LS_NMSE: {ls_nmse}, Lr: {optimizer.param_groups[0]['lr']}")

def main_MSetup():
    config_path = 'conf/config_multisetup.yml'
    cfg = get_config(config_path)
    # print('cfg:',cfg)

    sys.stdout = Logger(cfg.logger.path)
    sys.stderr = sys.stdout
    
    nmse = cal_NMSE_by_matpath_h5(cfg.dataset.valid_path,"H_est_MMSE_all_data")
    print("NMMSE of valid dataset::",nmse)
    dataloader = BaseBunch.load_data(cfg.dataset,cfg.dataloader)

    model = DnCNN_MultiBlock_ds(block=3, depth=16, image_channels=2, use_bnorm=True).to(device)
    # model = DiaUNet1D(2,2,cfg.model.channel_index,cfg.model.num_layers).to(device)
    
    # ---------------------------cal parameter------------------------
    total_params = sum(p.numel() for p in model.parameters())  # 所有参数总数
    model_size_mb = total_params * 4 / 1024**2  # float32 => 4 bytes

    
    print("config_path:",config_path)
    print('cfg:',cfg)
    print("model::",model)
    print(f"Estimated model size: {model_size_mb:.2f} MB")
    # train(model, dataloader, epochs=400, lr=1e-3)
    train(model, dataloader, epochs=50, lr=1e-3)

main_MSetup()