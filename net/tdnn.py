import torch
import torch.nn as nn
import torch.nn.functional as F

# class TDNN(nn.Module):
#     """ 用 TDNN 替换 SD_Module1D """
#     def __init__(self, in_channels, out_channels, kernel_size=3, dilation=2, padding=2):
#         super(TDNN, self).__init__()
#         self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, 
#                               dilation=dilation, padding=padding)
#         self.bn = nn.BatchNorm1d(out_channels)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         return self.relu(self.bn(self.conv(x)))
    

class TDNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TDNN, self).__init__()
        
        # TDNN 结构
        mid_channel = 128
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=mid_channel, kernel_size=5, stride=1, dilation=1,padding=2)
        self.conv2 = nn.Conv1d(in_channels=mid_channel, out_channels=mid_channel, kernel_size=3, stride=1, dilation=2,padding=2)
        self.conv3 = nn.Conv1d(in_channels=mid_channel, out_channels=mid_channel, kernel_size=3, stride=1, dilation=3,padding=3)
        self.conv4 = nn.Conv1d(in_channels=mid_channel, out_channels=mid_channel, kernel_size=1, stride=1, dilation=1)
        self.conv5 = nn.Conv1d(in_channels=mid_channel, out_channels=out_channels, kernel_size=1, stride=1, dilation=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        """
        输入: x -> (batch_size, input_dim, sequence_length)
        输出: (batch_size, output_dim)
        """
        x = self.relu(self.conv1(x))
        # print(x.shape)
        x = self.relu(self.conv2(x))
        # print(x.shape)
        x = self.relu(self.conv3(x))
        # print(x.shape)
        x = self.relu(self.conv4(x))
        # print(x.shape)
        x = self.conv5(x)  # 最后一层不使用 ReLU，保留原始特征
        return x
    
if __name__ == "__main__":
    model = TDNN(in_channels=32, out_channels=32)
    print(model)
    x = torch.randn(8, 32, 36)  # Batch size=8, 1 input channel, sequence length=100
    output = model(x)
    print(output.shape)  # 预期输出: (8, 64, 100)