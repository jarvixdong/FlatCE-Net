import torch
import torch.nn as nn
import numpy as np
from abc import abstractmethod


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
    


class FeatureNetwork(nn.Module):
    def __init__(self, in_channels, embed_dim):
        super().__init__()
        self.linear = nn.Linear(in_channels, embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        # print('x shape')
        x = x.view(B, C, H * W)  # (B, HW, C)
        x = self.linear(x)  # (B, HW, embed_dim)
        return x

class MHA_Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ffn(self.norm2(x))
        return x
    

class ResidualBlock1DConv(nn.Module):
    def __init__(self, channels, kernel_size=3, num_layers=2, activation=nn.ReLU):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2))
            layers.append(nn.BatchNorm1d(channels))
            if i < num_layers - 1:
                layers.append(activation())

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.block(x)

class AttentionCE(BaseModel):
    def __init__(self, in_channels=2, feature_dim=64, mha_dim=64, heads=4, height=4, width=32):
        super().__init__()
        self.H, self.W, self.C = height, width, in_channels
        self.feature_net = FeatureNetwork(in_channels,feature_dim)
        self.attn1 = MHA_Block(feature_dim, heads)
        self.attn2 = MHA_Block(feature_dim, heads)
        self.res1 = ResidualBlock1DConv(2,3,6)
        self.attn3 = MHA_Block(feature_dim, heads)
        self.res2 = ResidualBlock1DConv(2,3,6)
        self.attn4 = MHA_Block(feature_dim, heads)
        self.project = nn.Linear(feature_dim, in_channels)

    def forward(self, x,Vpinv=None):
        B, C, H, W = x.shape
        x = self.feature_net(x)             # (B, HW, C)
        # print('after feature net:',x.shape)
        x = self.res1(x)
        x = self.attn1(x)
        x = self.attn2(x)
        x = self.res1(x)
        x = self.attn3(x)
        x = self.res2(x)
        x = self.attn4(x)
        # print('before project:',x.shape)
        x = self.project(x)
        # print('x shape:',x.shape)
        x = x.view(B, C, H, W)
        return x


if __name__ == "__main__":
    # 测试
    model = AttentionCE(in_channels=144, feature_dim=256, mha_dim=16, heads=4, height=4, width=32)
    a = torch.randn(7, 2, 4, 36)
    b = model(a)
    print(model)
    print('输出 shape:', b.shape)  # 应为 [7, 2, 4, 32]