import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, block, depth, image_channels, filters=64, use_bnorm=True):
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

if __name__ == "__main__":
    # Instantiate the model
    model = DnCNNMultiBlock(block=3, depth=16, image_channels=2, use_bnorm=True)

    # Print model summary
    print(model)
    data = torch.rand(3,2,8,8)
    y = model(data)