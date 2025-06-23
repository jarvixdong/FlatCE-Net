import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod
import numpy as np

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}M'.format(params / 1e6)


    def load_model(self, check_point_path, device=None):
        checkpoint = torch.load(check_point_path, map_location=torch.device("cpu"))
        if hasattr(self, "loaded_model"):
            own_state = self.loaded_model().state_dict()
        else:
            own_state = self.state_dict()
        for name, param in checkpoint.items():
            if name not in own_state:
                print ('{} not found'.format(name))
                continue
            if param.data.shape != own_state[name].shape:
                print ('{} not match different shape'.format(name))
                continue
            print ('{} loaded'.format(name))
            param = param.data
            own_state[name].copy_(param)
        return self, checkpoint
    
class DnCNN_ResidualBlock(nn.Module):
    def __init__(self, image_channels, filters, depth, use_bnorm):
        super(DnCNN_ResidualBlock, self).__init__()
        layers = []
        for i in range(depth - 1):
            layers.append(nn.Conv2d(image_channels if i == 0 else filters, filters, kernel_size=3, padding=1, bias=False))
            if use_bnorm:
                layers.append(nn.BatchNorm2d(filters))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(filters, image_channels, kernel_size=3, padding=1, bias=False))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.block(x)  # 加法残差连接
    
class LRCNN(BaseModel):
    def __init__(self, block, depth, image_channels, filters=64, use_bnorm=True):
        super(LRCNN, self).__init__()
        self.layers = nn.ModuleList([
            DnCNN_ResidualBlock(image_channels, filters, depth, use_bnorm)
            for _ in range(block)
        ])

    def forward(self, x, Vpinv=None):
        x = x/np.sqrt(0.5)
        for res_block in self.layers:
            x = res_block(x)
        x = x*np.sqrt(0.5)
        return x
    
if __name__ == "__main__":
    # Instantiate the model
    model = LRCNN(block=1, depth=6, image_channels=2, use_bnorm=True)

    # Print model summary
    print(model)
    data = torch.rand(3,2,8,8)
    y = model(data)