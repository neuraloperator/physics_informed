import torch
import torch.nn as nn
from models.FCN import DenseNet
from typing import List
from .utils import weighted_mse


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


class SAWeight(nn.Module):
    def __init__(self, out_dim, num_init: List, num_bd: List, num_collo: List):
        super(SAWeight, self).__init__()
        self.init_param = nn.ParameterList(
            [nn.Parameter(100 * torch.rand(num, out_dim)) for num in num_init]
        )

        self.bd_param = nn.ParameterList(
            [nn.Parameter(torch.rand(num, out_dim)) for num in num_bd]
        )

        self.collo_param = nn.ParameterList(
            [nn.Parameter(torch.rand(num, out_dim)) for num in num_collo]
        )

    def forward(self, init_cond: List, bd_cond: List, residual: List):
        total_loss = 0.0
        for param, init_loss in zip(self.init_param, init_cond):
            total_loss += weighted_mse(init_loss, 0, param)

        for param, bd in zip(self.bd_param, bd_cond):
            total_loss += weighted_mse(bd, 0, param)

        for param, res in zip(self.collo_param, residual):
            total_loss += weighted_mse(res, 0, param)
        return total_loss