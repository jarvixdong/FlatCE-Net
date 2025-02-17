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
class DnCNN_MultiBlock_ds(nn.Module):
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

    def forward(self, x):
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
    data = torch.rand(3,2,8,8)
    y = model(data)