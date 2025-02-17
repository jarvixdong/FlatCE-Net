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

    total_nmse = 0
    total_ls_nmse = 0
    total_samples = 0

    with torch.no_grad():
        for idx, batch in enumerate(dataloader.valid_loader):
            batch = {'inputs': batch[0].cuda(non_blocking=True),
                    'targets': batch[1].cuda(non_blocking=True),
                    'Vpinv': batch[2].cuda(non_blocking=True)}
            
            inputdata = batch['inputs']
            target = batch['targets']
            # vpinv = batch['Vpinv']

            output = model(inputdata)

            # 计算当前批次的 NMSE 和 LS NMSE
            nmse_batch = cal_NMSE3(output.detach().cpu().numpy(), target.detach().cpu().numpy())
            ls_nmse_batch = cal_NMSE3(inputdata.detach().cpu().numpy(), target.detach().cpu().numpy())

            # 累积 NMSE 和样本数
            batch_size = inputdata.shape[0]
            total_nmse += nmse_batch * batch_size
            total_ls_nmse += ls_nmse_batch * batch_size
            total_samples += batch_size

    # 平均 NMSE 和 LS NMSE
    avg_nmse = total_nmse / total_samples
    avg_ls_nmse = total_ls_nmse / total_samples
    # print('total samples:',total_samples)
    return avg_nmse, avg_ls_nmse


    
def train(model, dataloader, epochs=10, lr=1e-4):
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = StepLR(optimizer, step_size=250, gamma=0.1)
    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()
    print('loss function::',criterion)
    # rayleigh_loss = RayleighKLLoss()
    rayleigh_loss = RayleighKLLoss_mat2()
    # criterion = nn.SmoothL1Loss(beta=1.0)
    for epoch in range(epochs):
        model.train()
        total_l2_loss = 0.0
        total_targets_abs_sq = 0.0
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

            # break
            
            with torch.no_grad():
                nmse_train = cal_NMSE3(output.detach().cpu().numpy(),target.detach().cpu().numpy())
        # break
        scheduler.step()      
        nmmse,ls_nmse = valid(model,dataloader)
        
        nmmse = float(f"{nmmse:.6f}")  # 保留 6 位小数
        ls_nmse = float(f"{ls_nmse:.6f}")  # 保留 6 位小数
        nmse_train = float(f"{nmse_train:.6f}")  # 保留 6 位小数

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

    # model = SDUNet1D_3l(2,2,32).to(device)
    
    model = DnCNN_MultiBlock_ds(block=3, depth=16, image_channels=2, use_bnorm=True).to(device)
    # model = DiaUNet1D(2,2,cfg.model.channel_index,cfg.model.num_layers).to(device)
    total_params = sum(p.numel() for p in model.parameters())  # 所有参数总数
    model_size_mb = total_params * 4 / 1024**2  # float32 => 4 bytes

    
    print("config_path:",config_path)
    print('cfg:',cfg)
    print("model::",model)
    print(f"Estimated model size: {model_size_mb:.2f} MB")
    train(model, dataloader, epochs=50, lr=1e-3)

main_MSetup()