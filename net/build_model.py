import torch
import torch.nn as nn
import torch.nn.functional as F
from net.basic_unet import DynamicDilatedConv


# 修饰器：允许动态替换 CustomConvBlock
def replace_2nd_module(new_module):
    """ 装饰器用于动态替换 CustomConvBlock """
    def decorator(cls):
        cls.CustomConvBlock = staticmethod(lambda in_ch, out_ch: (new_module or Stdconv1D)(in_ch, out_ch))
        return cls
    return decorator

class Stdconv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(Stdconv1D, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


@replace_2nd_module(DynamicDilatedConv) 
class BasicUnetBlock(nn.Module):
    """ BasicUnetBlock 结合普通 Conv1d 和 可替换的 CustomConvBlock """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(BasicUnetBlock, self).__init__()
        
        self.blocks = nn.ModuleList([
                nn.Sequential(
                    Stdconv1D(in_channels,out_channels,kernel_size,padding)
                ),
                self.CustomConvBlock(out_channels, out_channels) 
            ])

    def forward(self, x):
        for block in self.blocks: 
            x = block(x)
        return x


class DiaUNet1D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64, num_layers=3):
        super(DiaUNet1D, self).__init__()

        self.num_layers = num_layers

        # 编码器（Encoder）和池化层
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()

        prev_channels = in_channels
        for i in range(num_layers):
            next_channels = base_channels * (2 ** i)
            self.encoders.append(BasicUnetBlock(prev_channels, next_channels))
            self.pools.append(nn.MaxPool1d(kernel_size=2, stride=2))
            prev_channels = next_channels

        # Bottleneck 层
        self.bottleneck = BasicUnetBlock(prev_channels, prev_channels * 2)

        # 解码器（Decoder）和反卷积层
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        prev_channels = prev_channels * 2
        for i in reversed(range(num_layers)):
            next_channels = base_channels * (2 ** i)
            self.upconvs.append(nn.ConvTranspose1d(prev_channels, next_channels, kernel_size=2, stride=2))
            self.decoders.append(BasicUnetBlock(prev_channels, next_channels))
            prev_channels = next_channels

        # 输出层
        self.final_conv = nn.Conv1d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, -1)

        # 编码路径（Encoder）
        enc_outputs = []
        for i in range(self.num_layers):
            x = self.encoders[i](x)
            enc_outputs.append(x)  # 存储编码结果用于跳跃连接
            x = self.pools[i](x)

        # Bottleneck
        x = self.bottleneck(x)

        # 解码路径（Decoder）
        for i in range(self.num_layers):
            x = self.upconvs[i](x)
            x = torch.cat([x, enc_outputs[self.num_layers - 1 - i]], dim=1)  # 跳跃连接
            x = self.decoders[i](x)

        # 输出层
        output = self.final_conv(x)
        output = output.view(B, C, H, W)

        return output

    
if __name__ == "__main__":
    model = DiaUNet1D(in_channels=2, out_channels=2,num_layers=4)
    print(model)
    x = torch.randn(8, 2, 16,64)  # Batch size=8, 1 input channel, sequence length=100
    output = model(x)
    print(output.shape)  # 预期输出: (8, 64, 100)