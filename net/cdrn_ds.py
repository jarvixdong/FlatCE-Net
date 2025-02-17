import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import time
import os

# Custom NMSE loss function
def NMSE_IRS(y_true, y_pred):
    r_true = y_true[:, :, :, 0]  # shape: (batchsize, 8, 8)
    i_true = y_true[:, :, :, 1]
    r_pred = y_pred[:, :, :, 0]
    i_pred = y_pred[:, :, :, 1]
    mse_r_sum = torch.sum(torch.sum((r_pred - r_true) ** 2, dim=-1), dim=-1)
    r_sum = torch.sum(torch.sum(r_true ** 2, dim=-1), dim=-1)
    mse_i_sum = torch.sum(torch.sum((i_pred - i_true) ** 2, dim=-1), dim=-1)
    i_sum = torch.sum(torch.sum(i_true ** 2, dim=-1), dim=-1)
    num = mse_r_sum + mse_i_sum
    den = r_sum + i_sum
    return num / den

# Custom LossHistory class
class LossHistory:
    def __init__(self):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('NMSE'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_NMSE'))

    def on_epoch_end(self, epoch, logs):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('NMSE'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_NMSE'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('NMSE-loss')
        plt.legend(loc="upper right")
        plt.title('Training and Validation loss')
        plt.savefig('loss_curve.png')
        plt.show()

# Define the DnCNN model in PyTorch
class DnCNN_MultiBlock(nn.Module):
    def __init__(self, block, depth, image_channels, filters=64, use_bnorm=True):
        super(DnCNN_MultiBlock, self).__init__()
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

# Training function
def train_model(model, train, test, epochs, batch_size):
    X_train, Y_train = train
    X_test, Y_test = test

    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    Y_test = torch.tensor(Y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(X_test, Y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    history = LossHistory()

    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = NMSE_IRS(target, output)
            loss.backward()
            optimizer.step()

            history.on_batch_end(batch_idx, {'loss': loss.item(), 'NMSE': loss.item()})

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                val_loss += NMSE_IRS(target, output).item()

        val_loss /= len(test_loader)
        history.on_epoch_end(epoch, {'val_loss': val_loss, 'val_NMSE': val_loss})

    history.loss_plot('epoch')

# Main function
def main():
    batch_size = 64
    epochs = 400

    model = DnCNN_MultiBlock(block=3, depth=16, image_channels=2, use_bnorm=True)
    print(model)

    # Load .mat data
    data_xtr = sio.loadmat('x_train_Rician_CSCG_K10dB_60000_M8_N32_5dB.mat')
    data_ytr = sio.loadmat('y_train_Rician_CSCG_K10dB_60000_M8_N32_5dB.mat')
    data_xtest = sio.loadmat('x_test_Rician_CSCG_K10dB_20000_M8_N32_5dB.mat')
    data_ytest = sio.loadmat('y_test_Rician_CSCG_K10dB_20000_M8_N32_5dB.mat')

    xa_train = data_xtr['x_train']
    ya_train = data_ytr['y_train']
    xa_test = data_xtest['x_test']
    ya_test = data_ytest['y_test']

    train_model(model, (xa_train, ya_train), (xa_test, ya_test), epochs, batch_size)

if __name__ == '__main__':
    main()