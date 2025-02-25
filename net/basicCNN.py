import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, hidden_channels=64, num_layers=5):
        super(SimpleCNN, self).__init__()

        layers = []

        # 第1层：输入 -> 隐藏通道
        layers.append(nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))

        # 中间层：隐藏通道 -> 隐藏通道
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.BatchNorm2d(hidden_channels, eps=1e-4, momentum=0.0))

        # 最后一层：隐藏通道 -> 输出通道
        layers.append(nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1))

        # 组合成顺序网络
        self.cnn = nn.Sequential(*layers)

    def forward(self, x):
        return self.cnn(x)
    
class LeNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, hidden_dim=512, num_fc_layers=3):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.fc_layers = nn.ModuleList()  # 用于存储多个全连接层
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.num_fc_layers = num_fc_layers

    def forward(self, x):
        inp_shape = x.shape  # (batch_size, channels, height, width)
        batch_size, _, height, width = inp_shape

        # 1. 卷积 + 池化
        x = nn.ReLU()(self.conv1(x))
        x = nn.MaxPool2d(2, ceil_mode=True)(x)
        x = nn.ReLU()(self.conv2(x))
        x = nn.MaxPool2d(2, ceil_mode=True)(x)

        # 2. 展平
        x = x.view(batch_size, -1)

        # 3. 动态定义多层全连接层
        if len(self.fc_layers) == 0:
            device = next(self.parameters()).device
            flattened_dim = x.shape[1]

            # 第一个全连接层
            self.fc_layers.append(nn.Linear(flattened_dim, self.hidden_dim).to(device))
            self.fc_layers.append(nn.ReLU(inplace=True))

            # 中间隐藏层
            for _ in range(self.num_fc_layers - 2):
                self.fc_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim).to(device))
                self.fc_layers.append(nn.ReLU(inplace=True))

            # 最终输出层
            self.fc_layers.append(nn.Linear(self.hidden_dim, self.out_channels * height * width).to(device))

        # 4. 前向传播
        for layer in self.fc_layers:
            x = layer(x)

        # 5. 恢复输出形状
        return x.view(batch_size, self.out_channels, height, width)


if __name__ == "__main__":
    # 示例输入：batch_size=8，输入通道=2，高=64，宽=64
    # x = torch.randn(8, 2, 36, 4)  # (batch_size, channels, height, width)

    # # 实例化模型
    # model = SimpleCNN(in_channels=2, out_channels=2, hidden_channels=64, num_layers=5)
    # output = model(x)

    # print(model)  # 打印模型结构
    # print(f"输入形状: {x.shape}")
    # print(f"输出形状: {output.shape}")


    # 示例
    model = LeNet(in_channels=2, out_channels=2)
    x = torch.randn(8, 2, 36, 4)
    output = model(x)
    print(f"输入形状: {x.shape}, 输出形状: {output.shape}")