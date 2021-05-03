from .FCN import DenseNet

import numpy as np
import torch
import torch.nn as nn


class LowRank2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LowRank2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.phi = DenseNet([2, 64, 128, in_channels*out_channels], torch.nn.ReLU)
        self.psi = DenseNet([2, 64, 128, in_channels*out_channels], torch.nn.ReLU)

    def get_grid(self, S1, S2, batchsize, device):
        gridx = torch.tensor(np.linspace(0, 1, S1+1)[:-1], dtype=torch.float)
        gridx = gridx.reshape(1, S1, 1).repeat([batchsize, 1, S2])
        gridy = torch.tensor(np.linspace(0, 1, S2+1)[:-1], dtype=torch.float)
        gridy = gridy.reshape(1, 1, S2).repeat([batchsize, S1, 1])
        return torch.stack((gridx, gridy), dim=-1).to(device)

    def forward(self, x, gridy=None):
        # x (batch, channel, x, y)
        # y (batch, Ny, 2)
        batchsize, size1, size2 = x.shape[0], x.shape[2], x.shape[3]

        gridx = self.get_grid(S1=size1, S2=size2, batchsize=1, device=x.device).reshape(size1 * size2, 2)
        if gridy is None:
            gridy = self.get_grid(S1=size1, S2=size2, batchsize=batchsize, device=x.device).reshape(batchsize, size1 * size2, 2)
        Nx = size1 * size2
        Ny = gridy.shape[1]

        phi_eval = self.phi(gridx).reshape(Nx, self.out_channels, self.in_channels)
        psi_eval = self.psi(gridy).reshape(batchsize, Ny, self.out_channels, self.in_channels)
        x = x.reshape(batchsize, self.in_channels, Nx)

        x = torch.einsum('noi,bin,bmoi->bom', phi_eval, x, psi_eval) / Nx
        return x