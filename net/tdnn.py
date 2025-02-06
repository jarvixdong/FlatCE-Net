import torch
import torch.nn as nn
import torch.nn.functional as F

class TDNN(nn.Module):
    """ 用 TDNN 替换 SD_Module1D """
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=2, padding=2):
        super(TDNN, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, 
                              dilation=dilation, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))