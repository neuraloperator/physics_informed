import torch
import torch.nn as nn
from .basics import SpectralConv3d
from torch.utils.checkpoint import checkpoint


class FNet(nn.Module):
    def __init__(self, mode1, mode2, mode3):
        super(FNet, self).__init__()
        self.mode1 = mode1
        self.mode2 = mode2
        self.mode3 = mode3
        self.sp = SpectralConv3d(1, 1, mode1, mode2, mode3)

    def forward(self, x):
        x = x.permute(0, 4, 1, 2, 3)
        x = x[:, -1:, :, :, :]
        y = self.sp(x)
        return y


class GarbNet(nn.Module):
    def __init__(self, width, S, T, mode1, mode2, mode3,
                 fc_dim=128, out_dim=1):
        super(GarbNet, self).__init__()
        self.width = width
        self.S = S
        self.T = T
        self.scale = 1 / (width * T)
        self.v = nn.Parameter(self.scale * torch.rand(1, width, S, S, T))
        self.sp = SpectralConv3d(width, width, mode1, mode2, mode3)
        self.w = nn.Conv1d(width, width, 1)

        self.fc1 = nn.Linear(width, fc_dim)
        self.fc2 = nn.Linear(fc_dim, out_dim)

    def forward(self,):
        x1 = self.sp(self.v)
        x2 = self.w(self.v.view(1, self.width, -1)).view(1, self.width, self.S, self.S, self.T)
        x = x1 + x2
        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        return x


class FNN3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width=16, fc_dim=128, layers=None, in_dim=4, out_dim=1):
        '''
        Args:
            modes1: list of int, first dimension maximal modes for each layer
            modes2: list of int, second dimension maximal modes for each layer
            modes3: list of int, third dimension maximal modes for each layer
            layers: list of int, channels for each layer
            in_dim: int, input dimension
            out_dim: int, output dimension
        '''
        super(FNN3d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        if layers is None:
            self.layers = [width] * 4
        else:
            self.layers = layers
        self.fc0 = nn.Linear(in_dim, layers[0])

        self.sp_convs = nn.ModuleList([SpectralConv3d(
            in_size, out_size, mode1_num, mode2_num, mode3_num)
            for in_size, out_size, mode1_num, mode2_num, mode3_num
            in zip(self.layers, self.layers[1:], self.modes1, self.modes2, self.modes3)])

        self.ws = nn.ModuleList([nn.Conv1d(in_size, out_size, 1)
                                 for in_size, out_size in zip(self.layers, self.layers[1:])])

        self.fc1 = nn.Linear(layers[-1], fc_dim)
        self.fc2 = nn.Linear(fc_dim, out_dim)

    def forward(self, x):
        '''
        Args:
            x: (batchsize, x_grid, y_grid, t_grid, 3)

        Returns:
            u: (batchsize, x_grid, y_grid, t_grid, 1)

        '''
        length = len(self.ws)
        batchsize = x.shape[0]
        size_x, size_y, size_z = x.shape[1], x.shape[2], x.shape[3]

        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            if i == length -1:
                continue
            x1 = speconv(x)
            x2 = w(x.view(batchsize, self.layers[i], -1)).view(batchsize, self.layers[i+1], size_x, size_y, size_z)
            x = x1 + x2
            if i != length - 1:
                x = torch.tanh(x)
        return x


class FNO3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width=16, fc_dim=128, layers=None, in_dim=1, out_dim=1):
        '''
        Args:
            modes1: list of int, first dimension maximal modes for each layer
            modes2: list of int, second dimension maximal modes for each layer
            modes3: list of int, third dimension maximal modes for each layer
            layers: list of int, channels for each layer
            in_dim: int, input dimension
            out_dim: int, output dimension
        '''
        super(FNO3d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        if layers is None:
            self.layers = [width] * 4
        else:
            self.layers = layers
        self.fc0 = nn.Linear(in_dim, layers[0])

        self.sp_convs = nn.ModuleList([SpectralConv3d(
            in_size, out_size, mode1_num, mode2_num, mode3_num)
            for in_size, out_size, mode1_num, mode2_num, mode3_num
            in zip(self.layers, self.layers[1:], self.modes1, self.modes2, self.modes3)])

        self.ws = nn.ModuleList([nn.Conv1d(in_size, out_size, 1)
                                 for in_size, out_size in zip(self.layers, self.layers[1:])])

        self.fc1 = nn.Linear(layers[-1], fc_dim)
        self.fc2 = nn.Linear(fc_dim, out_dim)

    def forward(self, x):
        '''
        Args:
            x: (batchsize, x_grid, y_grid, t_grid, 3)

        Returns:
            u: (batchsize, x_grid, y_grid, t_grid, 1)

        '''
        length = len(self.ws)
        batchsize = x.shape[0]
        size_x, size_y, size_z = x.shape[1], x.shape[2], x.shape[3]

        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x)
            x2 = w(x.view(batchsize, self.layers[i], -1)).view(batchsize, self.layers[i + 1], size_x, size_y,
                                                               size_z)
            x = x1 + x2
            if i != length - 1:
                x = torch.tanh(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        return x


class CIFAR10Model(nn.Module):
    def __init__(self, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.cnn_block_1 = nn.Sequential(*[
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        ])
        self.dropout_1 = nn.Dropout(0.25)
        self.cnn_block_2 = nn.Sequential(*[
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        ])
        self.dropout_2 = nn.Dropout(0.25)
        self.flatten = lambda inp: torch.flatten(inp, 1)
        self.linearize = nn.Sequential(*[
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU()
        ])
        self.dropout_3 = nn.Dropout(0.5)
        self.out = nn.Linear(512, 10)

    def forward(self, X):

        X = checkpoint(self.cnn_block_1, X)
        X = self.dropout_1(X)
        X = checkpoint(self.cnn_block_2, X)
        X = self.dropout_2(X)
        X = self.flatten(X)
        X = self.linearize(X)
        X = self.dropout_3(X)
        X = self.out(X)
        return X