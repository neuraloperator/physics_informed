import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import matplotlib.pyplot as plt

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
from utilities3 import *

torch.manual_seed(0)
np.random.seed(0)

import torch.autograd.functional as AF



def compl_mul1d(a, b):
    # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
    op = partial(torch.einsum, "bix,iox->box")
    return op(a, b)
    # return torch.stack([
    #     op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
    #     op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
    # ], dim=-1)

class SpectralConv(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1


        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    def forward(self, x, y=None):

        size = x.size(-1)

        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[2,])


        #Return to physical space
        if y == None:
            # Multiply relevant Fourier modes
            out_ft = torch.zeros(batchsize, self.out_channels, size // 2 + 1, device=x.device, dtype=torch.cfloat)
            out_ft[:, :, :self.modes1] = \
                compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)
            x = torch.fft.irfftn(out_ft, s=(size, ), dim=[2,])
        else:
            factor = compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)
            x = self.ifft1d(y, factor, size, self.modes1)
        return x

    def ifft1d(self, y, coeff, N, k_max):
        device = y.device
        # y (n, ) locations in [0,1]
        # coeff (batch, channels, kmax)

        # wavenumber (2kmax-1,)
        wavenumber =  torch.cat((torch.arange(start=0, end=k_max, step=1), \
                            torch.arange(start=-(k_max-1), end=0, step=1)), 0).to(device)

        # K (n, 2kmax-1)
        K = torch.outer(y, wavenumber)

        # basis (n, 2kmax-1)
        basis = torch.exp( 1j * 2* np.pi * K).to(device)

        # coeff (batch, channels, kmax) -> (batch, channels, 2kmax-1)
        coeff = torch.cat([coeff, coeff[..., 1:].flip(-1).conj()], dim=-1)

        # Y (batch, channels, n)
        Y = torch.einsum("bck,nk->bcn", coeff, basis) / N
        Y = Y.real
        return Y



class SimpleBlock(nn.Module):
    def __init__(self, in_dim, out_dim, modes1, width):
        super(SimpleBlock, self).__init__()

        self.modes1 = modes1

        self.width = width
        self.grid_dim = 0

        self.fc0 = nn.Linear(in_dim, width)

        self.conv0 = SpectralConv(width, width, self.modes1)
        self.w0 = nn.Conv1d(width, width, 1)


        self.fc1 = nn.Linear(width, width*2)
        self.fc3 = nn.Linear(width*2, out_dim)

    def forward(self, x, y=None):

        batchsize = x.shape[0]
        size = x.shape[1]

        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        if y == None:
            x1 = self.conv0(x)
            x2 = self.w0(x.view(batchsize, self.width, size)).view(batchsize, self.width, size)
            x = x1 + x2
            x = F.relu(x)
        else:
            y = y.reshape(-1)
            y1 = self.conv0(x, y)
            y2 = self.w0(self.fc0(y.reshape(batchsize,size, 1)).permute(0, 2, 1)).view(batchsize, self.width, size)
            # y2 = F.interpolate(x2, size=y.shape[0], mode='linear')

            # y = y.reshape(1, size, 1, 1)
            # y = torch.cat([y, torch.zeros(y.shape, device=y.device)], dim=-1)
            # y2 = F.grid_sample(x2.reshape(batchsize, self.width, size, 1), y, mode='bilinear').view(batchsize, self.width, size)
            y = y1 + y2
            x = F.relu(y)

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.selu(x)
        x = self.fc3(x)
        return x


class Net(nn.Module):
    def __init__(self, in_dim=1, out_dim=1, modes=5, width=64):
        super(Net, self).__init__()
        self.conv1 = SimpleBlock(in_dim, out_dim, modes, width)

    def forward(self, x, y=None):
        x = self.conv1(x.reshape(1,-1,1), y)
        return x.squeeze()

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c


model = Net()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

model.cuda()

# s = 1024
# dx = 1 / s
# xs = torch.linspace(0,1,s+1)[:-1]
#
# fs = torch.sin(2*np.pi*xs)
# dfs = 2*np.pi*torch.cos(2*np.pi*xs)
# ddfs = -4*np.pi**2*torch.sin(2*np.pi*xs)
#
# # fs = xs**2
# # dfs = 2*xs
# # ddfs = 2 * torch.ones(xs.shape)
#
# xs = xs.cuda()
# fs = fs.cuda()
# dfs = dfs.cuda()
# ddfs = ddfs.cuda()
# t1 = default_timer()

nt = 101
ts = torch.linspace(0,1,nt)
ft = torch.sin(4*ts)
dft = 4*torch.cos(4*ts)
dt = 1/nt

nx = 256
xs = torch.linspace(0,1,nx)
fx = torch.cos(4*xs)
dfx = -4*torch.sin(4*xs)
dfxx = -16*torch.cos(4*xs)
dx = 1/nx

u = torch.outer(ft, fx)
def Ufun(x):
    return torch.sin(4*x[0]) * torch.cos(4*x[1])
print(u.shape)
plt.imshow(u)
plt.show()



# Finite difference
# out = model(xs)
# grad3 = (out[2:] - out[:-2]) / (2*dx)
# loss_f3 = F.mse_loss(grad3, dfx[1:-1], reduction='mean')
#
# # autograd
# ys = xs.clone()
# ys.requires_grad = True
# out = model(xs,ys)
# grad4 = torch.autograd.grad(out.sum(), ys, create_graph=True)[0]
# loss_f = F.mse_loss(grad4, dfx, reduction='mean')

#
#
# #
# ###### FDM Burgers
# u = u.reshape(1, nt, nx)
# D = 1
# dt = D / (nt - 1)
# dx = D / (nx)
#
# # ux: (batch, size-2, size-2)
# ut = (u[:, 2:, 1:-1] - u[:, :-2, 1:-1]) / (2 * dt)
# ux = (u[:, 1:-1, 2:] - u[:, 1:-1, :-2]) / (2 * dx)
# uxx = (u[:, 1:-1, 2:] - 2 * u[:, 1:-1, 1:-1] + u[:, 1:-1, :-2]) / (dx ** 2)
# u = u[:, 1:-1, 1:-1]
#
# fut = torch.outer(dft, fx)
# fux = torch.outer(ft, dfx)
# fuxx = torch.outer(ft, dfxx)
# print(F.mse_loss(ut[0] , fut[1:-1, 1:-1]))
# print(F.mse_loss(ux[0] , fux[1:-1, 1:-1]))
# print(F.mse_loss(uxx[0] , fuxx[1:-1, 1:-1]))


# dataloader = MatReader('data/piececonst_r241_N1024_smooth1.mat')
# x_data = dataloader.read_field('coeff')[:,:,:]
# y_data = dataloader.read_field('sol')[:,:,:]
# plt.imshow(x_data[0])
# plt.show()
# plt.imshow(y_data[0])
# plt.show()
# def FDM_Darcy(u, a, D=1):
#     batchsize = u.size(0)
#     size = u.size(1)
#     u = u.reshape(batchsize, size, size)
#     a = a.reshape(batchsize, size, size)
#     dx = D / (size - 1)
#     dy = dx
#
#     # ux: (batch, size-2, size-2)
#     ux = (u[:, 2:, 1:-1] - u[:, :-2, 1:-1]) / (2 * dx)
#     uy = (u[:, 1:-1, 2:] - u[:, 1:-1, :-2]) / (2 * dy)
#     a = a[:, 1:-1, 1:-1]
#     aux = a * ux
#     auy = a * uy
#     auxx = (aux[:, 2:, 1:-1] - aux[:, :-2, 1:-1]) / (2 * dx)
#     auyy = (auy[:, 1:-1, 2:] - auy[:, 1:-1, :-2]) / (2 * dy)
#
#     Du = - (auxx + auyy)
#     return Du
#
# def PINO_loss(u):
#     batchsize = u.size(0)
#     nt = u.size(1)
#     nx = u.size(2)
#
#     u = u.reshape(batchsize, nt, nx)
#     lploss = LpLoss(size_average=True)
#
#     index_t = torch.cat([torch.tensor(range(0, nt)), (nt - 1) * torch.ones(nx - 1), torch.tensor(range(nt-1, 1, -1)),
#                          torch.zeros(nx - 1)], dim=0).long()
#     index_x = torch.cat([(nx - 1) * torch.ones(nt - 1), torch.tensor(range(nx-1, 1, -1)), torch.zeros(nt - 1),
#                          torch.tensor(range(0, nx))], dim=0).long()
#     boundary_u = u[:, index_t, index_x]
#     truth_u = torch.zeros(boundary_u.shape, device=u.device)
#     loss_u = lploss.abs(boundary_u, truth_u)
#
#     Du = FDM_Darcy(u, x_data)
#     f = torch.ones(Du.shape, device=u.device)
#     loss_f = F.mse_loss(Du, f)
#     print(torch.mean(Du), torch.max(Du), torch.min(Du))
#     # x = (Du>100).nonzero(as_tuple=True)
#     # print((Du>100).nonzero(as_tuple=True))
#     plt.imshow(Du[0])
#     plt.colorbar()
#     plt.show()
#
#     return loss_f
#
# print(PINO_loss(y_data))
#
# myloss = LpLoss()
# for i in range(1000):
#     model.train()
#     optimizer.zero_grad()
#
#     # loss = F.mse_loss(out, fs, reduction='mean')
#
#     # diagonal of Jacobian
#     # grad = AF.jacobian(model, xs, create_graph=True, strict=False, vectorize=False)
#     # grad1 = torch.diag(grad)
#     # loss_f = F.mse_loss(grad1, dfs, reduction='mean')
#
#     # autograd sum
#     # xs.requires_grad = True
#     # grad2 = torch.autograd.grad(torch.sum(model(xs)), xs, create_graph=True)[0]
#     # loss_f = F.mse_loss(grad2, dfs, reduction='mean')
#
#     # Finite difference
#     out3 = model(xs)
#     grad3 = (out3[2:] - out3[:-2]) / (2*dx)
#     ggrad3 = (out3[2:] + out3[:-2] - 2*out3[1:-1]) / (dx**2)
#     loss_f3 = myloss(ggrad3.view(1,-1), ddfs.view(1,-1)[:,1:-1])
#
#     # autograd
#     ys = xs.clone()
#     ys.requires_grad = True
#     out = model(xs,ys)
#     grad4 = torch.autograd.grad(out.sum(), ys, create_graph=True)[0]
#     ggrad4 = torch.autograd.grad(grad4.sum(), ys, create_graph=True)[0]
#     loss_f = myloss(ggrad4.view(1,-1), ddfs.view(1,-1))
#
#     loss_u = myloss(out.view(1,-1), fs.view(1,-1))
#     loss = loss_u + loss_f*0
#
#     print(i, loss_u.item(), loss_f.item(), loss_f3.item())
#     loss.backward()
#     optimizer.step()
#
#     with torch.no_grad():
#         if i%900 == 0:
#             plt.plot(xs.cpu().numpy(), out3.cpu().numpy(), label='model')
#             plt.plot(xs.cpu().numpy(), out.cpu().numpy(), label='model')
#             plt.plot(xs.cpu().numpy(), fs.cpu().numpy(), label='truth')
#             leg = plt.legend(loc='best')
#             leg.get_frame().set_alpha(0.5)
#             plt.show()
#
#             plt.plot(xs.cpu().numpy(), dfs.cpu().numpy(), label='truth')
#             plt.plot(xs.cpu().numpy(), grad4.cpu().numpy(), label='autograd')
#             plt.plot(xs[1:-1].cpu().numpy(), grad3.cpu().numpy(), label='FDM')
#             leg = plt.legend(loc='best')
#             leg.get_frame().set_alpha(0.5)
#             plt.show()
#
#             plt.plot(xs.cpu().numpy(), ddfs.cpu().numpy(), label='truth')
#             plt.plot(xs.cpu().numpy(), ggrad4.cpu().numpy(), label='autograd')
#             plt.plot(xs[1:-1].cpu().numpy(), ggrad3.cpu().numpy(), label='FDM')
#             leg = plt.legend(loc='best')
#             leg.get_frame().set_alpha(0.5)
#             plt.show()
#
#
#
#
# out = model(xs)
# # out2 = model(xs,xs)
#
#
# t2 = default_timer()
# print(t2-t1, F.mse_loss(out, fs, reduction='mean'))
#
# xs = xs.cpu().detach().numpy()
# plt.plot(xs, fs.cpu())
# plt.plot(xs, out.cpu().detach().numpy())
#
# # plt.plot(xs, grad1.cpu().detach().numpy())
# # plt.plot(xs, grad2.cpu().detach().numpy())
# plt.show()
