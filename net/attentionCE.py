import torch
import torch.nn as nn
import torch.nn.functional as F
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
    def __init__(self, in_channels, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_dim)

    def forward(self, x):
        # print('x shape:',x.shape)
        B, C, H, W = x.shape
        x = x.view(B, 2 * H, W).permute(0,2,1)
        # print('x flatten shape:',x.shape)
        return self.linear(x)

class MHA_Block(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=7,padding=3),
            nn.GELU(),
            nn.Conv2d(5, 1, kernel_size=7,padding=3)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, N, C)
        x1 = self.norm1(x)
        attn_output, _ = self.mha(x1, x1, x1)
        x = x + attn_output
        x2 = self.norm2(x)
        # print('x2 shape:',x2.shape)
        x2 = x2.unsqueeze(1)
        x2 = self.ffn(x2)
        x2 = x2.squeeze(1)
        # print('x2 out::',x2.shape)
        return x + x2

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels) 
        )

    def forward(self, x):
        # print('x shape in resblock:',x.shape)
        res_x = self.block(x)
        # print('x after resblock:',res_x.shape)
        return x+res_x

class AttentionCE(nn.Module):
    def __init__(self, in_channels=2, feature_dim=128, mha_heads=4):
        super().__init__()
        self.feature_net = FeatureNetwork(in_channels, feature_dim)  # assume H=W=32
        self.itf1 = MHA_Block(feature_dim, mha_heads)
        self.itf2 = MHA_Block(feature_dim, mha_heads)
        self.linear = nn.Linear(feature_dim, feature_dim)
        self.norm = nn.LayerNorm(feature_dim)

        self.decoder = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=3, padding=1),
            ResidualBlock(12),
            nn.Conv2d(12, 1, kernel_size=1)
        )


        self.final = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=3, padding=1),
            nn.Linear(feature_dim,in_channels),
            nn.Conv2d(12, 1, kernel_size=1)
        )
    def forward(self, x,Vpinv=None):
        B, C, H, W = x.shape
        x_flat = self.feature_net(x)
        # print('x flat:',x_flat.shape)
        # x_flat = x_flat.view(B, -1, x_flat.shape[-1])  # (B, N, C)
        
        x = self.itf1(x_flat)
        x = self.itf2(x)
        x = self.linear(x)
        x = self.norm(x)
        # print(x.shape)

        # reshape to 2D map again
        x = x.unsqueeze(1)
        x = self.decoder(x)
        # print('x after decode1:',x.shape)
        
        x = self.itf1(x.squeeze(1))
        
        x = self.decoder(x.unsqueeze(1))
        x = self.itf1(x.squeeze(1))
        # print('x shape before out:',x.shape)
        x = self.final(x.unsqueeze(1))
        x = x.permute(0,1,3,2).view(B,C,H,W)


        return x
    
if __name__ == "__main__":
    model = AttentionCE(in_channels=8,feature_dim=64)
    print('model:',model)
    a = torch.randn(7,2,4,32)
    b = model(a)
    print('b shape:',b.shape)