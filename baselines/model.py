import torch
import torch.nn as nn
from models.FCN import DenseNet


class DeepONet(nn.Module):
    def __init__(self, branch_layer, trunk_layer):
        super(DeepONet, self).__init__()
        self.branch = DenseNet(branch_layer, nn.ReLU)
        self.trunk = DenseNet(trunk_layer, nn.ReLU)

    def forward(self, u0, grid):
        a = self.branch(u0)
        b = self.trunk(grid)
        batchsize = a.shape[0]
        dim = a.shape[1]
        return torch.bmm(a.view(batchsize, 1, dim), b.view(batchsize, dim, 1))


class DeepONetCP(nn.Module):
    def __init__(self, branch_layer, trunk_layer):
        super(DeepONetCP, self).__init__()
        self.branch = DenseNet(branch_layer, nn.ReLU)
        self.trunk = DenseNet(trunk_layer, nn.ReLU)

    def forward(self, u0, grid):
        a = self.branch(u0)
        # batchsize x width
        b = self.trunk(grid)
        # N x width
        return torch.einsum('bi,ni->bn', a, b)
