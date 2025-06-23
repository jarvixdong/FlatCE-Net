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
    
    
class DnCNNBlock(nn.Module):
    def __init__(self, depth, in_channels, filters=64, use_bnorm=True):
        super(DnCNNBlock, self).__init__()
        
        layers = []
        
        # First layer: Conv2D
        layers.append(nn.Conv2d(in_channels, filters, kernel_size=3, stride=1, padding=1, bias=False))
        if use_bnorm:
            layers.append(nn.BatchNorm2d(filters, eps=1e-4, momentum=0.0))
        layers.append(nn.ReLU(inplace=True))
        
        # Intermediate layers: (depth - 2) layers of Conv2D + BN + ReLU
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1, bias=False))
            if use_bnorm:
                layers.append(nn.BatchNorm2d(filters, eps=1e-4, momentum=0.0))
            layers.append(nn.ReLU(inplace=True))
        
        # Final layer: Conv2D (output channels = in_channels)
        layers.append(nn.Conv2d(filters, in_channels, kernel_size=3, stride=1, padding=1, bias=False))
        
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        residual = self.dncnn(x)  # Predicted noise
        return x - residual  # Input - Noise

class DnCNNMultiBlock(nn.Module):
    def __init__(self, block, depth, image_channels, filters=32, use_bnorm=True):
        super(DnCNNMultiBlock, self).__init__()
        
        self.blocks = nn.ModuleList([
            DnCNNBlock(depth, image_channels, filters, use_bnorm) for _ in range(block)
        ])

    def forward(self, x):
        # print('input shape:',x.shape)
        for block in self.blocks:
            # print('x in block',x.shape)
            x = block(x)
        return x


# Define the DnCNN model in PyTorch
class DnCNN_MultiBlock_ds(BaseModel):
    def __init__(self, block, depth, image_channels, filters=64, use_bnorm=True):
        super(DnCNN_MultiBlock_ds, self).__init__()
        self.block = block
        self.depth = depth
        self.image_channels = image_channels
        self.filters = filters
        self.use_bnorm = use_bnorm

        self.layers = nn.ModuleList()
        for _ in range(block):
            block_layers = []
            for i in range(depth - 1):
                block_layers.append(nn.Conv2d(image_channels if i == 0 else filters, filters, kernel_size=3, padding=1, bias=False))
                if use_bnorm:
                    block_layers.append(nn.BatchNorm2d(filters))
                block_layers.append(nn.ReLU(inplace=True))
            block_layers.append(nn.Conv2d(filters, image_channels, kernel_size=3, padding=1, bias=False))
            self.layers.append(nn.Sequential(*block_layers))

    def forward(self, x,Vpinv=None):
        input_ = x
        for block in self.layers:
            x = block(x)
            x = input_ - x  # Perform subtraction here
            input_ = x  # Update input_ for the next block
        return x
    
if __name__ == "__main__":
    # Instantiate the model
    model = DnCNN_MultiBlock_ds(block=3, depth=16, image_channels=2, use_bnorm=True)

    # Print model summary
    print(model)
    data = torch.rand(3,2,4,32)
    y = model(data)
    print('y shape:',y.shape)