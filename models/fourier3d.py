import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from .basics import FourierBlock


class FNN3d(nn.Module):
    def __init__(self, modes1, modes2, modes3,
                 width=16, fc_dim=128,
                 layers=None,
                 in_dim=4, out_dim=1,
                 activation='tanh',
                 use_checkpoint=False):
        '''
        Args:
            modes1: list of int, first dimension maximal modes for each layer
            modes2: list of int, second dimension maximal modes for each layer
            modes3: list of int, third dimension maximal modes for each layer
            layers: list of int, channels for each layer
            in_dim: int, input dimension
            out_dim: int, output dimension
            activation: str or list, name of activation functions, options: tanh, gelu, none;
                        If none, no activation function will be applied.
            use_checkpoint: If True, use gradient checkpointing to trade compute for memory
        '''
        super(FNN3d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        if isinstance(activation, str):
            self.activation = [activation] * len(self.modes1)
        elif isinstance(activation, list):
            self.activation = activation
        else:
            raise ValueError('activations are not supported')

        self.use_checkpoint = use_checkpoint

        if layers is None:
            self.layers = [width] * 4
        else:
            self.layers = layers
        self.fc0 = nn.Linear(in_dim, layers[0])

        self.blocks = nn.ModuleList(
            [
                FourierBlock(in_size, out_size, mode1_num, mode2_num, mode3_num, activation=act)
                for in_size, out_size, mode1_num, mode2_num, mode3_num, act
                in zip(self.layers, self.layers[1:], self.modes1, self.modes2, self.modes3, self.activation)
            ]
        )

        self.project = nn.Sequential(
            nn.Linear(layers[-1], fc_dim),
            nn.Tanh(),
            nn.Linear(fc_dim, out_dim)
        )

    def forward(self, x):
        '''
        Args:
            x: (batchsize, x_grid, y_grid, t_grid, 3)

        Returns:
            u: (batchsize, x_grid, y_grid, t_grid, 1)

        '''
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        # if self.use_checkpoint:
        #     x = checkpoint.checkpoint(self.blocks, x)
        # else:
        #     x = self.blocks(x)
        for block in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(block, x)
            else:
                x = block(x)
        x = x.permute(0, 2, 3, 4, 1)
        if self.use_checkpoint:
            x=  checkpoint.checkpoint(self.project, x)
        else:
            x = self.project(x)
        return x