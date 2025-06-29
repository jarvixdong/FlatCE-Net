import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicDilatedConv(nn.Module):
    def __init__(self, in_channel, out_channel, cycle_length=36):
        """
        :param in_channel: 输入通道数
        :param out_channel: 输出通道数
        :param cycle_length: 64 为周期
        """
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.cycle_length = cycle_length

        self.layers = nn.ModuleList()
        
        # 计算动态膨胀率，使其适应 64 周期
        # self.dia_lst = self.calculate_dilation_rates(cycle_length)
        self.dia_lst = [1, 2, 4, 8, 16, 32]

        # 计算每层的通道数（先减小，然后恢复）
        self.out_channel_lst = [max(1, in_channel // (2 ** (i+1))) for i in range(len(self.dia_lst))]
        self.out_channel_lst[-1] = self.out_channel_lst[-2]  # 最后一层确保通道数恢复

        prev_channel = in_channel
        for dia_i, dilation in enumerate(self.dia_lst):
            new_channel_dim = self.out_channel_lst[dia_i]
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(prev_channel, new_channel_dim, kernel_size=3, padding=dilation, dilation=dilation),
                    nn.BatchNorm1d(new_channel_dim),
                    nn.ReLU(inplace=True)
                )
            )
            prev_channel = new_channel_dim  # 更新输入通道数

        # 1x1 投影层确保最终输出通道数等于输入
        self.projection = nn.Conv1d(sum(self.out_channel_lst), out_channel, kernel_size=1)


    def calculate_dilation_rates(self, cycle_length):
        """ 根据 64 为周期动态计算膨胀率 """
        rates = []
        dilation = 1
        while dilation < cycle_length:
            rates.append(dilation)
            dilation *= 2  # 指数增长，直到 64
        return rates

    def forward(self, x, Vpinv=None):
        outputs = []
        for layer in self.layers:
            x = layer(x)
            # print('x and layer',x.shape)
            outputs.append(x)

        out = torch.cat(outputs, dim=1)  # 在通道维度拼接
        out = self.projection(out)  # 维持最终通道数和输入一致
        return out


class DynamicDilatedConv_cat(nn.Module):
    def __init__(self, in_channel, out_channel, cycle_length=16):
        """
        :param in_channel: 输入通道数
        :param out_channel: 输出通道数
        :param cycle_length: 64 为周期
        """
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.cycle_length = cycle_length

        self.layers = nn.ModuleList()
        
        # 计算动态膨胀率，使其适应 64 周期
        self.dia_lst = self.calculate_dilation_rates(cycle_length)

        # 计算每层的通道数（先减小，然后恢复）
        self.out_channel_lst = [max(1, in_channel // (2 ** (i+1))) for i in range(len(self.dia_lst))]
        self.out_channel_lst[-1] = self.out_channel_lst[-2]  # 最后一层确保通道数恢复

        prev_channel = in_channel
        for dia_i, dilation in enumerate(self.dia_lst):
            new_channel_dim = self.out_channel_lst[dia_i]
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(prev_channel, new_channel_dim, kernel_size=3, padding=dilation, dilation=dilation),
                    nn.BatchNorm1d(new_channel_dim),
                    nn.ReLU(inplace=True)
                )
            )
            prev_channel = new_channel_dim  # 更新输入通道数

        # 1x1 投影层确保最终输出通道数等于输入
        self.projection = nn.Conv1d(sum(self.out_channel_lst), out_channel, kernel_size=1)


    def calculate_dilation_rates(self, cycle_length):
        """ 根据 64 为周期动态计算膨胀率 """
        rates = []
        dilation = 1
        while dilation < cycle_length:
            rates.append(dilation)
            dilation *= 2  # 指数增长，直到 64
        return rates

    def forward(self, x):
        outputs = []
        for layer in self.layers:
            x = layer(x)
            print('x and layer',x.shape)
            outputs.append(x)

        out = torch.cat(outputs, dim=1)  # 在通道维度拼接
        # out = self.projection(out)  # 维持最终通道数和输入一致
        return out

if __name__ == "__main__":
    # Instantiate the model
    model = DynamicDilatedConv(32,32)

    # Print model summary
    print(model)
    data = torch.rand(3,32,64)
    y = model(data)
    print('y shape:',y.shape)
    
