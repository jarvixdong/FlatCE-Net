import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod
from net.basic_unet import DynamicDilatedConv, DynamicDilatedConv_cat
from net.tdnn import TDNN


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
        checkpoint = torch.load(check_point_path, map_location=torch.device("cpu"), weights_only=True)
        
        if hasattr(self, "loaded_model"):
            own_state = self.loaded_model().state_dict()
        else:
            own_state = self.state_dict()
        # for name, param in checkpoint["model"].items():
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

    def forward(self, x,Vpinv=None):
        return self.relu(self.bn(self.conv(x)))


# @replace_2nd_module(DynamicDilatedConv) 
# class DynamicDilatedUnetBlock(nn.Module):
#     """ DynamicDilatedUnetBlock 结合普通 Conv1d 和 可替换的 CustomConvBlock """
    
#     def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
#         super(DynamicDilatedUnetBlock, self).__init__()
        
#         self.blocks = nn.ModuleList([
#                 nn.Sequential(
#                     Stdconv1D(in_channels,out_channels,kernel_size,padding)
#                 ),
#                 self.CustomConvBlock(out_channels, out_channels),
#                 # AttentionBlock(out_channels)
#             ])

#     def forward(self, x, Vpinv=None):
#         for block in self.blocks: 
#             x = block(x, Vpinv)
            
#         return x

@replace_2nd_module(DynamicDilatedConv) 
class DynamicDilatedUnetBlock(nn.Module):
    """ DynamicDilatedUnetBlock 结合普通 Conv1d 和 可替换的 CustomConvBlock """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(DynamicDilatedUnetBlock, self).__init__()
        
        self.blocks = nn.ModuleList(
                [Stdconv1D(in_channels,out_channels,kernel_size,padding),
                self.CustomConvBlock(out_channels, out_channels),
                # AttentionBlock(out_channels)
            ])

    def forward(self, x, Vpinv=None):
        for block in self.blocks: 
            x = block(x, Vpinv)
            
        return x


# @replace_2nd_module(DynamicDilatedConv) 
# @replace_2nd_module(Stdconv1D)
@replace_2nd_module(TDNN)
class StdUnetBlock(nn.Module):
    """ StdUnetBlock 结合普通 Conv1d 和 可替换的 CustomConvBlock """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(StdUnetBlock, self).__init__()
        
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
    
    
class DiaUNet1D(BaseModel):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64, num_layers=3, withVpinv=False):
        super(DiaUNet1D, self).__init__()

        self.num_layers = num_layers
        # self.withVpinv = withVpinv
        # 编码器（Encoder）和池化层
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()

        prev_channels = in_channels
        for i in range(num_layers):
            next_channels = base_channels * (2 ** i)
            self.encoders.append(DynamicDilatedUnetBlock(prev_channels, next_channels))
            self.pools.append(nn.MaxPool1d(kernel_size=2, stride=2))
            # self.pools.append(nn.AvgPool1d(kernel_size=2, stride=2))
            prev_channels = next_channels

        # Bottleneck 层
        self.bottleneck = DynamicDilatedUnetBlock(prev_channels, prev_channels * 2)

        # 解码器（Decoder）和反卷积层
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        prev_channels = prev_channels * 2
        for i in reversed(range(num_layers)):
            next_channels = base_channels * (2 ** i)
            self.upconvs.append(nn.ConvTranspose1d(prev_channels, next_channels, kernel_size=2, stride=2))
            self.decoders.append(DynamicDilatedUnetBlock(prev_channels, next_channels))
            prev_channels = next_channels

        # 输出层
        self.final_conv = nn.Conv1d(base_channels, out_channels, kernel_size=1)
        self.linear = None

    def forward(self, x,Vpinv=None):
        # print("x and Vpinv shape:",x.shape,Vpinv.shape)
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
            # print('x shape:',x.shape)
            x = torch.cat([x, enc_outputs[self.num_layers - 1 - i]], dim=1)  # 跳跃连接
            x = self.decoders[i](x)

        # 输出层
        # print('output shape before final conv:',x.shape)
        output = self.final_conv(x)

        output = output.view(B, C, H, W)

        return output


# class DiaUNet1DwithVpinv(BaseModel):
#     def __init__(self, in_channels=1, out_channels=1, base_channels=64, num_layers=3):
#         super(DiaUNet1DwithVpinv, self).__init__()

#         self.num_layers = num_layers

#         # 编码器（Encoder）和池化层
#         self.encoders = nn.ModuleList()
#         self.pools = nn.ModuleList()

#         prev_channels = in_channels
#         for i in range(num_layers):
#             next_channels = base_channels * (2 ** i)
#             self.encoders.append(DynamicDilatedUnetBlock(prev_channels, next_channels))
#             self.pools.append(nn.MaxPool1d(kernel_size=2, stride=2))
#             # self.pools.append(nn.AvgPool1d(kernel_size=2, stride=2))
#             prev_channels = next_channels

#         # Bottleneck 层
#         self.bottleneck = DynamicDilatedUnetBlock(prev_channels, prev_channels * 2)

#         # 解码器（Decoder）和反卷积层
#         self.upconvs = nn.ModuleList()
#         self.decoders = nn.ModuleList()

#         prev_channels = prev_channels * 2
#         for i in reversed(range(num_layers)):
#             next_channels = base_channels * (2 ** i)
#             self.upconvs.append(nn.ConvTranspose1d(prev_channels, next_channels, kernel_size=2, stride=2))
#             self.decoders.append(DynamicDilatedUnetBlock(prev_channels, next_channels))
#             prev_channels = next_channels

#         # 输出层
#         self.final_conv = nn.Conv1d(base_channels, out_channels, kernel_size=1)

#     def forward(self, x,Vpinv=None):
#         B, C, H, W = x.shape
#         # print("HW shape:",H,W)
#         x = x.view(B, C, -1)

#         # 编码路径（Encoder）
#         enc_outputs = []
#         for i in range(self.num_layers):
#             x = self.encoders[i](x,Vpinv)
#             enc_outputs.append(x)  # 存储编码结果用于跳跃连接
#             x = self.pools[i](x)

#         # Bottleneck
#         x = self.bottleneck(x)
        

#         # 解码路径（Decoder）
#         for i in range(self.num_layers):
#             x = self.upconvs[i](x)
#             x = torch.cat([x, enc_outputs[self.num_layers - 1 - i]], dim=1)  # 跳跃连接
#             x = self.decoders[i](x,Vpinv)

#         # 输出层

#         output = self.final_conv(x)
#         output = output.view(B, C, H, W)

#         return output
    
    
class DiaUNet1DKepSeq(BaseModel):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64, num_layers=3):
        super(DiaUNet1DKepSeq, self).__init__()

        self.num_layers = num_layers

        # 编码器（Encoder）和池化层
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()

        prev_channels = in_channels
        for i in range(num_layers):
            next_channels = base_channels * (2 ** i)
            self.encoders.append(DynamicDilatedUnetBlock(prev_channels, next_channels))
            # self.pools.append(nn.MaxPool1d(kernel_size=2, stride=2))
            self.pools.append(nn.AvgPool1d(kernel_size=2, stride=2))
            prev_channels = next_channels

        # Bottleneck 层
        self.bottleneck = DynamicDilatedUnetBlock(prev_channels+2, prev_channels * 2)

        # 解码器（Decoder）和反卷积层
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        prev_channels = prev_channels * 2
        for i in reversed(range(num_layers)):
            next_channels = base_channels * (2 ** i)
            self.upconvs.append(nn.ConvTranspose1d(prev_channels, next_channels, kernel_size=1, stride=1))
            self.decoders.append(DynamicDilatedUnetBlock(prev_channels, next_channels))
            prev_channels = next_channels

        # 输出层
        self.final_conv = nn.Conv1d(base_channels, out_channels, kernel_size=1)

    def forward(self, x,Vpinv=None):
        B, C, H, W = x.shape
        x = x.view(B, C, -1)
        # print("Vpinv shape:",Vpinv.shape)  #[256, 2, 40, 36]
        sm = Vpinv.shape[2]
        Vpinv_noise = torch.randn([B,C,H,sm]).to(x.device) @ Vpinv
        # Vpinv_noise = torch.randn([B,C,H,sm]).to(x.device)/(torch.sqrt(4*(10**-1.5))) @ Vpinv
        # print("Vpinv_noise",Vpinv_noise.shape)
        Vpinv_noise_flat = Vpinv_noise.view(B,C,-1)
        # x = x-Vpinv_noise_flat

        # 编码路径（Encoder）
        enc_outputs = []
        for i in range(self.num_layers):
            x = self.encoders[i](x)
            # print("encoder:",i,x.shape)
            enc_outputs.append(x)  # 存储编码结果用于跳跃连接
            # x = self.pools[i](x)

        # Bottleneck
        # print('encoder shape:',x.shape)
        x = torch.cat([x,Vpinv_noise_flat],dim=1)
        x = self.bottleneck(x)
        # print('after bottle:',x.shape)

        # 解码路径（Decoder）
        for i in range(self.num_layers):
            x = self.upconvs[i](x)
            x = torch.cat([x, enc_outputs[self.num_layers - 1 - i]], dim=1)  # 跳跃连接
            # print("before encoder:",i,x.shape)
            x = self.decoders[i](x)
            # print("after encoder:",x.shape)

        # 输出层

        output = self.final_conv(x)
        output = output.view(B, C, H, W)

        return output


class AttentionBlock(nn.Module):
    """ QKV Attention 让 U-Net 的每一层与 Vpinv 交互（适用于 1D 数据） """
    def __init__(self, in_channels, v_channels=2):
        """
        in_channels: U-Net 的通道数
        v_channels: Vpinv 的通道数（通常是 2）
        """
        super(AttentionBlock, self).__init__()

        # Query 从 x 计算
        self.query_conv = nn.Conv1d(in_channels, 64, kernel_size=1)

        # Key 从 Vpinv 计算
        self.key_conv = nn.Conv1d(v_channels, 64, kernel_size=1)

        # Value 从 Vpinv 计算，并转换为 in_channels
        self.value_conv = nn.Conv1d(v_channels, in_channels, kernel_size=1)

    def forward(self, x, Vpinv):
        """
        x: (B, in_channels, L) - U-Net 当前层的特征
        Vpinv: (B, v_channels, L) - 先验信息
        """
        # 计算 Q, K, V
        Q = self.query_conv(x)  # (B, 64, L)
        K = self.key_conv(Vpinv)  # (B, 64, L)
        V = self.value_conv(Vpinv)  # (B, in_channels, L)

        # 计算 Attention Score: QK^T
        attn_scores = torch.einsum("bcl,bcl->bl", Q, K)  # (B, L)
        attn_scores = torch.softmax(attn_scores, dim=-1).unsqueeze(1)  # (B, 1, L)

        # 计算 Vpinv' = attention * Vpinv
        Vpinv_adjusted = attn_scores * V  # (B, in_channels, L)

        # 返回 x - Vpinv'，让 U-Net 自适应修正特征
        return x - Vpinv_adjusted
    

class FlatCEkepSeqAttnVpinv(BaseModel):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64, num_layers=3):
        super(FlatCEkepSeqAttnVpinv, self).__init__()

        self.num_layers = num_layers

        # 编码器（Encoder）和池化层
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()

        prev_channels = in_channels
        for i in range(num_layers):
            next_channels = base_channels * (2 ** i)
            self.encoders.append(DynamicDilatedUnetBlock(prev_channels, next_channels))
            self.pools.append(nn.MaxPool1d(kernel_size=2, stride=2))
            # self.pools.append(nn.AvgPool1d(kernel_size=2, stride=2))
            prev_channels = next_channels

        # Bottleneck 层
        self.bottleneck = DynamicDilatedUnetBlock(prev_channels, prev_channels * 2)

        # 解码器（Decoder）和反卷积层
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        prev_channels = prev_channels * 2
        for i in reversed(range(num_layers)):
            next_channels = base_channels * (2 ** i)
            self.upconvs.append(nn.ConvTranspose1d(prev_channels, next_channels, kernel_size=1, stride=1))
            self.decoders.append(DynamicDilatedUnetBlock(prev_channels, next_channels))
            prev_channels = next_channels

        # 输出层
        self.final_conv = nn.Conv1d(base_channels, out_channels, kernel_size=1)
    

    def forward(self, x,Vpinv=None):
        B, C, H, W = x.shape
        x = x.view(B, C, -1)
        # print("Vpinv shape:",Vpinv.shape)  #[256, 2, 40, 36]
        sm = Vpinv.shape[2]
        Vpinv_noise = torch.randn([B,C,H,sm]).to(x.device) @ Vpinv
        # Vpinv_noise = torch.randn([B,C,H,sm]).to(x.device)/(torch.sqrt(4*(10**-1.5))) @ Vpinv
        # print("Vpinv_noise",Vpinv_noise.shape)
        Vpinv_noise_flat = Vpinv_noise.view(B,C,-1)
        # x = x-Vpinv_noise_flat

        # 编码路径（Encoder）
        enc_outputs = []
        for i in range(self.num_layers):
            x = self.encoders[i](x,Vpinv)
            # print("encoder:",i,x.shape)
            enc_outputs.append(x)  # 存储编码结果用于跳跃连接
            # x = self.pools[i](x)

        # Bottleneck
        # print('encoder shape:',x.shape)
        x = torch.cat([x,Vpinv_noise_flat],dim=1)
        x = self.channel_attn(x)
        x = self.bottleneck(x)
        # print('after bottle:',x.shape)

        # 解码路径（Decoder）
        for i in range(self.num_layers):
            x = self.upconvs[i](x)
            x = torch.cat([x, enc_outputs[self.num_layers - 1 - i]], dim=1)  # 跳跃连接
            # print("before encoder:",i,x.shape)
            x = self.decoders[i](x,Vpinv)
            # print("after encoder:",x.shape)

        # 输出层

        output = self.final_conv(x)
        output = output.view(B, C, H, W)

        return output

        
class FlatCEkepSeqSEVpinv(BaseModel):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64, num_layers=3):
        super(DiaUNet1DKepSeq, self).__init__()

        self.num_layers = num_layers

        # 编码器（Encoder）和池化层
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()

        prev_channels = in_channels
        for i in range(num_layers):
            next_channels = base_channels * (2 ** i)
            self.encoders.append(DynamicDilatedUnetBlock(prev_channels, next_channels))
            # self.pools.append(nn.MaxPool1d(kernel_size=2, stride=2))
            self.pools.append(nn.AvgPool1d(kernel_size=2, stride=2))
            prev_channels = next_channels

        # Bottleneck 层
        self.bottleneck = DynamicDilatedUnetBlock(prev_channels, prev_channels * 2)

        # 解码器（Decoder）和反卷积层
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        prev_channels = prev_channels * 2
        for i in reversed(range(num_layers)):
            next_channels = base_channels * (2 ** i)
            self.upconvs.append(nn.ConvTranspose1d(prev_channels, next_channels, kernel_size=1, stride=1))
            self.decoders.append(DynamicDilatedUnetBlock(prev_channels, next_channels))
            prev_channels = next_channels

        # 输出层
        self.final_conv = nn.Conv1d(base_channels, out_channels, kernel_size=1)
        
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局池化
            nn.Conv2d(256 + 2, 64, kernel_size=1),  # 降维
            nn.ReLU(),
            nn.Conv2d(64, 256, kernel_size=1),  # 恢复维度
            nn.Sigmoid()
        )

    def forward(self, x,Vpinv=None):
        B, C, H, W = x.shape
        x = x.view(B, C, -1)
        # print("Vpinv shape:",Vpinv.shape)  #[256, 2, 40, 36]
        sm = Vpinv.shape[2]
        Vpinv_noise = torch.randn([B,C,H,sm]).to(x.device) @ Vpinv
        # Vpinv_noise = torch.randn([B,C,H,sm]).to(x.device)/(torch.sqrt(4*(10**-1.5))) @ Vpinv
        # print("Vpinv_noise",Vpinv_noise.shape)
        Vpinv_noise_flat = Vpinv_noise.view(B,C,-1)
        # x = x-Vpinv_noise_flat

        # 编码路径（Encoder）
        enc_outputs = []
        for i in range(self.num_layers):
            x = self.encoders[i](x)
            # print("encoder:",i,x.shape)
            enc_outputs.append(x)  # 存储编码结果用于跳跃连接
            # x = self.pools[i](x)

        # Bottleneck
        # print('encoder shape:',x.shape)
        x = torch.cat([x,Vpinv_noise_flat],dim=1)
        x = self.channel_attn(x)
        x = self.bottleneck(x)
        # print('after bottle:',x.shape)

        # 解码路径（Decoder）
        for i in range(self.num_layers):
            x = self.upconvs[i](x)
            x = torch.cat([x, enc_outputs[self.num_layers - 1 - i]], dim=1)  # 跳跃连接
            # print("before encoder:",i,x.shape)
            x = self.decoders[i](x)
            # print("after encoder:",x.shape)

        # 输出层

        output = self.final_conv(x)
        output = output.view(B, C, H, W)

        return output

class UEasFeatureUNet1D(BaseModel):
    def __init__(self, in_channels=1, out_channels=1, base_channels=32, num_layers=3):
        super(UEasFeatureUNet1D, self).__init__()

        self.num_layers = num_layers

        # 编码器（Encoder）和池化层
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()

        prev_channels = in_channels
        for i in range(num_layers):
            next_channels = base_channels * (2 ** i)
            self.encoders.append(StdUnetBlock(prev_channels, next_channels))
            self.pools.append(nn.MaxPool1d(kernel_size=2, stride=2))
            prev_channels = next_channels

        # Bottleneck 层
        self.bottleneck = StdUnetBlock(prev_channels, prev_channels * 2)

        # 解码器（Decoder）和反卷积层
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        prev_channels = prev_channels * 2
        for i in reversed(range(num_layers)):
            next_channels = base_channels * (2 ** i)
            if i == num_layers-1:
                self.upconvs.append(nn.ConvTranspose1d(prev_channels, next_channels, kernel_size=2, stride=2, output_padding=1))
            else:
                self.upconvs.append(nn.ConvTranspose1d(prev_channels, next_channels, kernel_size=2, stride=2))
            self.decoders.append(StdUnetBlock(prev_channels, next_channels))
            prev_channels = next_channels

        # 输出层
        
        self.final_conv = nn.Conv1d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        # x = x.view(B, -1, W)
        x = x.permute(0, 2, 1, 3).reshape(B,-1,W)

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
            # print("self.upconv:",self.upconvs[i])
            # print("x shape:",i,x.shape,enc_outputs[self.num_layers - 1 - i].shape)
            x = torch.cat([x, enc_outputs[self.num_layers - 1 - i]], dim=1)  # 跳跃连接
            x = self.decoders[i](x)


        output = self.final_conv(x)
        # output = output.view(B, C, H, W)
        output = output.reshape(B, H, C, W).permute(0, 2, 1, 3)
        
        return output

class UFeatureTDNN(BaseModel):
    def __init__(self, in_channels, out_channels):
        super(UFeatureTDNN,self).__init__()
        
        mid_channel = 128
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=mid_channel, kernel_size=5, stride=1, dilation=1,padding=2)
        self.conv2 = nn.Conv1d(in_channels=mid_channel, out_channels=mid_channel, kernel_size=3, stride=1, dilation=2,padding=2)
        self.conv3 = nn.Conv1d(in_channels=mid_channel, out_channels=mid_channel, kernel_size=3, stride=1, dilation=3,padding=3)
        self.conv4 = nn.Conv1d(in_channels=mid_channel, out_channels=mid_channel, kernel_size=1, stride=1, dilation=1)
        self.conv5 = nn.Conv1d(in_channels=mid_channel, out_channels=out_channels, kernel_size=1, stride=1, dilation=1)

        self.relu = nn.ReLU()

    
    def forward(self,x):
        B, C, H, W = x.shape
        # x = x.view(B, -1, W)
        x = x.permute(0, 2, 1, 3).reshape(B,-1,W)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.conv5(x)
        
        output = x.reshape(B, H, C, W).permute(0, 2, 1, 3)
        
        return output
    
         
if __name__ == "__main__":
    # model = UEasFeatureUNet1D(in_channels=8, out_channels=8,num_layers=3)
    # model = DiaUNet1DKepSeq(2,2)
    # print(model)
    # x = torch.randn(8, 2, 4, 36)  # Batch size=8, 1 input channel, sequence length=100
    # output = model(x)
    # print(output.shape)  # 预期输出: (8, 64, 100)
    
    
    # xx = torch.arange(2*3*4*5)
    # print("xx",xx)
    # xx_4dim = xx.view(3,2,4,5)
    # print(xx_4dim)
    # xx_3dim = xx_4dim.permute(0, 2, 1, 3).reshape(3,-1,5)
    # print(xx_3dim)
    # xx_3dim_return = xx_3dim.reshape(3,4,2,5).permute(0, 2, 1, 3)
    # print(xx_3dim_return)
    
    
    # model = AttentionBlock(32)
    # x = torch.rand([7,32,4*36])
    # Vpinv = torch.rand(7,2,4*36)
    # y = model(x,Vpinv)
    # print('x shape:',x.shape)
    
    
    model = DiaUNet1D(2,2,32,3,True)
    print(model)
    x = torch.randn(8, 2, 4, 36)  # Batch size=8, 1 input channel, sequence length=100
    vpinv = torch.randn(8,2,36,36)
    output = model(x,vpinv)
    print(output.shape)  # 预期输出: (8, 64, 100)