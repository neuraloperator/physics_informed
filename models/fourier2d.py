import torch.nn as nn
import torch.nn.functional as F

from .lowrank2d import LowRank2d
from .basics import SpectralConv2d


class FNN2d(nn.Module):
    def __init__(self, modes1, modes2, width, layers=None, in_dim=3, out_dim=1):
        super(FNN2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        # input channel is 3: (a(x, y), x, y)
        if layers is None:
            self.layers = [width] * 4
        else:
            self.layers = layers
        self.fc0 = nn.Linear(in_dim, layers[0])

        self.sp_convs = nn.ModuleList([SpectralConv2d(
            in_size, out_size, mode1_num, mode2_num)
            for in_size, out_size, mode1_num, mode2_num
            in zip(self.layers, self.layers[1:], self.modes1, self.modes2)])

        self.ws = nn.ModuleList([nn.Conv1d(in_size, out_size, 1)
                                 for in_size, out_size in zip(self.layers, self.layers[1:])])

        self.fc1 = nn.Linear(layers[-1], 128)
        self.fc2 = nn.Linear(128, out_dim)

    def forward(self, x):
        '''
        Args:
            - x : (batch size, x_grid, y_grid, 2)
        Returns:
            - x: (batch size, x_grid, y_grid, 1)
        '''
        length = len(self.ws)
        batchsize = x.shape[0]
        size_x, size_y = x.shape[1], x.shape[2]

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x)
            x2 = w(x.view(batchsize, self.layers[i], -1)).view(batchsize, self.layers[i+1], size_x, size_y)
            x = x1 + x2
            if i != length - 1:
                x = F.elu(x)
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.elu(x)
        x = self.fc2(x)
        return x


class PINO2d(nn.Module):
    def __init__(self, modes1, modes2, width, layers=None, in_dim=3, out_dim=1):
        '''
        Args:
            modes1: number of modes to keep
            modes2: number of modes to keep
            width: width of features
            layers: list of integers
            in_dim: input dimensionality, default: a(x), x, t
            out_dim: output dimensionality, default: u(x,t)
        '''
        super(PINO2d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        if layers is None:
            self.layers = [width] * 4
        else:
            self.layers = layers
        self.fc0 = nn.Linear(in_dim, layers[0])

        self.sp_convs = nn.ModuleList([SpectralConv2d(
            in_size, out_size, mode1_num, mode2_num)
            for in_size, out_size, mode1_num, mode2_num
            in zip(self.layers, self.layers[1:], self.modes1, self.modes2)])

        self.ws = nn.ModuleList([nn.Conv1d(in_size, out_size, 1)
                                 for in_size, out_size in zip(self.layers[:-1], self.layers[1:-1])])
        self.ws.append(LowRank2d(self.layers[-2], self.layers[-1]))
        self.fc1 = nn.Linear(layers[-1], layers[-1] * 4)
        self.fc2 = nn.Linear(layers[-1] * 4, out_dim)

    def forward(self, x, y=None):
        batchsize = x.shape[0]
        size_x, size_y = x.shape[1], x.shape[2]

        length = len(self.ws)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            if i != length - 1:
                x1 = speconv(x)
                x2 = w(x.view(batchsize, self.layers[i], -1))\
                    .view(batchsize, self.layers[i+1], size_x, size_y)
                x = x1 + x2
                x = F.selu(x)
            else:
                x1 = speconv(x, y).reshape(batchsize, self.layers[-1], -1)
                x2 = w(x, y).reshape(batchsize, self.layers[-1], -1)
                x = x1 + x2
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.selu(x)
        x = self.fc2(x)
        return x
