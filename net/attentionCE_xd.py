import torch
import torch.nn as nn


class FeatureNetwork(nn.Module):
    def __init__(self, in_channels, embed_dim):
        super().__init__()
        self.linear = nn.Linear(in_channels, embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)
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

class ResidualBlock1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return x + self.block(x)

class AttentionCE(nn.Module):
    def __init__(self, in_channels=2, feature_dim=64, mha_dim=64, heads=4, height=4, width=32):
        super().__init__()
        self.H, self.W, self.C = height, width, in_channels
        self.feature_net = FeatureNetwork(2,feature_dim)
        self.attn1 = MHA_Block(feature_dim, heads)
        self.attn2 = MHA_Block(feature_dim, heads)
        self.res1 = ResidualBlock1D(feature_dim)
        self.attn3 = MHA_Block(feature_dim, heads)
        self.res2 = ResidualBlock1D(feature_dim)
        self.attn4 = MHA_Block(feature_dim, heads)
        self.project = nn.Linear(feature_dim, in_channels)

    def forward(self, x,Vpinv=None):
        B, C, H, W = x.shape
        x = self.feature_net(x)             # (B, HW, C)
        x = self.res1(x)
        x = self.attn1(x)
        x = self.attn2(x)
        # x = self.res1(x)
        # x = self.attn3(x)
        x = self.res2(x)
        # x = self.attn4(x)
        x = self.project(x)
        x = x.permute(0, 2, 1).contiguous().view(B, C, H, W)
        return x


if __name__ == "__main__":
    # 测试
    model = AttentionCE(in_channels=2, feature_dim=64, mha_dim=16, heads=4, height=4, width=32)
    a = torch.randn(7, 2, 4, 32)
    b = model(a)
    print(model)
    print('输出 shape:', b.shape)  # 应为 [7, 2, 4, 32]