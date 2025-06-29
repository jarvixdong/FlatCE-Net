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
from thop import profile, clever_format


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

# def resume_checkpoint(self, resume_path, is_finetune):
#     model, checkpoint = self.model.load_model(resume_path, device=self.device)
#     if not is_finetune:
#         self.step = checkpoint["step"]
#         self.start_epoch = checkpoint['epoch'] + 1
#         self.optimizer.load_state_dict(checkpoint['optim'])
#         self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
#     self.logger.info('resume params from {}, epoch={}, step={}'.format(
#         resume_path, self.start_epoch, self.step)
#     )
#     return model

def main():

    args = parse_conf()
    cfg = get_config(args.config, args)

    test_dataset = DataGenerator2(path=cfg.dataset.test_path)
    dataloader = DataLoader(test_dataset,batch_size = cfg.dataloader.batch_size,shuffle=cfg.dataloader.shuffle)
    
    model = build_network(cfg.model).to(device)
    # print('model:',model)
    model,_ = model.load_model(cfg.resume_path,device=device)
    # print('model:',model)
    
    model.eval()
    dummy = torch.randn(1, 3, 224, 224)  # batch=1 输入大小
    
    
    sum_ls_mse = 0.0   # 累计 LS 误差平方和
    sum_mse = 0.0      # 累计网络输出误差平方和
    sum_target = 0.0   # 累计真值平方和

    # start_time = time.time()
    time_lst = []
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            if idx == 0:
                continue
            # 将输入/目标/等辅助参数放到 GPU（可视实际需求而定）
            inputs = batch[0].cuda(non_blocking=True)
            targets = batch[1].cuda(non_blocking=True)
            # vpinv  = batch[2].to(device)  # 如确有需要可以继续用

            # 模型大小查看
            # flops, params = profile(model, inputs=(inputs,), verbose=False)
            # flops, params = clever_format([flops, params], "%.3f")
            # print(f"FLOPs: {flops}\nParams: {params}")

            # 模型前向推理
            print("input shape:",inputs.shape)
            B, C, H, W = inputs.shape
            inputs = inputs.view(B, C, -1)
            start_time = time.time()
            outputs = model(inputs)
            end_time = time.time()
            time_lst.append(end_time-start_time)
            print("Model interference time",end_time-start_time)
            outputs = outputs.view(B, C, H, W)









            # # break
            if idx == 100:
                break
    time_lst = time_lst[1:]
    print("whole time:",sum(time_lst),len(time_lst))
    print("mean time:",sum(time_lst)/len(time_lst))

    # # 输入尺寸 (channels, height, width)
    # flops, params = get_model_complexity_info(model, (2, 4,36), as_strings=True)

    # print('FLOPs:', flops)
    # print('Params:', params)
    
    # end_time = time.time()
    # print("Model interference time",end_time-start_time)
    # 计算平均 NMSE 和 LS NMSE（全局归一化）
    # avg_nmse = sum_mse / sum_target
    # avg_ls_nmse = sum_ls_mse / sum_target
    # print("avg_nmse:",avg_nmse)
    # print("avg_ls_nmse:",avg_ls_nmse)

    
    
    
    
    
main()