import torch
import yaml
# from models import DnCNN_MultiBlock_ds, DiaUNet1D

# # 读取配置
# with open("config.yaml", "r") as f:
#     config = yaml.safe_load(f)

# 动态创建模型
def build_model(config):
    model_name = config['model']['name']
    model_params = config['model']['params']

    # 使用反射机制加载模型类
    model_class = globals().get(model_name)
    if model_class is None:
        raise ValueError(f"未找到模型类 '{model_name}'，请检查配置文件和模型定义。")

    # 创建模型实例
    model = model_class(**model_params)
    return model

# # 设备选择
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # 创建模型
# model = build_model(config).to(device)

# # 打印模型信息
# print(f"已加载模型：{model.__class__.__name__}")
# print(model)
