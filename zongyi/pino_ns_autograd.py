import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from utilities4 import *

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
import scipy.io

torch.manual_seed(0)
np.random.seed(0)

activation = F.relu


def compl_mul3d(a, b):
    # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
    return torch.einsum("bixyz,ioxyz->boxyz", a, b)

    # return torch.stack([
    #     op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
    #     op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
    # ], dim=-1)

class LowRank3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LowRank3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.phi = DenseNet([3, 64, 128, in_channels*out_channels], torch.nn.ELU)
        self.psi = DenseNet([3, 64, 128, in_channels*out_channels], torch.nn.ELU)

    def get_grid(self, S1, S2, S3, batchsize, device):
        gridx = torch.tensor(np.linspace(0, 1, S1+1)[:-1], dtype=torch.float)
        gridx = gridx.reshape(1, S1, 1, 1).repeat([batchsize, 1, S2, S3])
        gridy = torch.tensor(np.linspace(0, 1, S2+1)[:-1], dtype=torch.float)
        gridy = gridy.reshape(1, 1, S2, 1).repeat([batchsize, S1, 1, S3])
        gridt = torch.tensor(np.linspace(0, 1, S3), dtype=torch.float)
        gridt = gridt.reshape(1, 1, 1, S3).repeat([batchsize, S1, S2, 1])
        return torch.stack((gridx, gridy, gridt), dim=-1).to(device)


    def forward(self, x, gridy=None):
        # x (batch, channel, x, y, t)
        # y (batch, Ny, 3)
        batchsize, S1, S2, S3 = x.shape[0], x.shape[2], x.shape[3], x.shape[4]

        gridx = self.get_grid(S1=S1, S2=S2, S3=S3, batchsize=1, device=x.device).reshape(S1 * S2 * S3, 3)
        if gridy==None:
            gridy = self.get_grid(S1=S1, S2=S2, S3=S3, batchsize=batchsize, device=x.device).reshape(batchsize, S1 * S2 * S3, 3)
        Nx = S1 * S2 * S3
        Ny = gridy.shape[1]

        phi_eval = self.phi(gridx).reshape(Nx, self.out_channels, self.in_channels)
        psi_eval = self.psi(gridy).reshape(batchsize, Ny, self.out_channels, self.in_channels)
        x = x.reshape(batchsize, self.in_channels, Nx)

        x = torch.einsum('noi,bin,bmoi->bom', phi_eval, x, psi_eval) / Nx
        return x

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    def forward(self, x, gridy=None):
        batchsize, S1, S2, S3 = x.shape[0], x.shape[2], x.shape[3], x.shape[4]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[2,3,4], norm="ortho")
        if gridy==None:
            # Multiply relevant Fourier modes
            out_ft = torch.zeros(batchsize, self.out_channels, x.size(2), x.size(3), x.size(4) // 2 + 1,
                                 device=x.device, dtype=torch.cfloat)
            out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
                compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
            out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
                compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
            out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
                compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
            out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
                compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

            # Return to physical space
            x = torch.fft.irfftn(out_ft, s=(x.size(2), x.size(3), x.size(4)), dim=[2, 3, 4], norm="ortho")
        else:
            out_ft = torch.zeros(batchsize, self.out_channels, 2*self.modes1, 2*self.modes2, 2*self.modes3,
                                 device=x.device, dtype=torch.cfloat)
            out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
                compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
            out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
                compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
            out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
                compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
            out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
                compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)
            x = self.ifft3d(gridy, out_ft, [self.modes1,self.modes2,self.modes3]) / np.sqrt(S1 * S2 * S3)
        return x

    def ifft3d(self, gridy, coeff, ks):
        [k1,k2,k3] = ks

        # y (batch, N, 3) locations in [0,2pi]*[0,2pi]*[0,1]
        # coeff (batch, channels, kmax, kmax)

        batchsize = gridy.shape[0]
        N = gridy.shape[1]
        device = gridy.device
        m1 = 2 * k1
        m2 = 2 * k2
        m3 = 2 * k3

        # wavenumber (m1, m2, m3)
        k_x1 =  torch.cat((torch.arange(start=0, end=k1, step=1), \
                            torch.arange(start=-k1, end=0, step=1)), 0).reshape(m1,1,1).repeat(1,m2,m3).to(device)
        k_x2 =  torch.cat((torch.arange(start=0, end=k2, step=1), \
                            torch.arange(start=-k2, end=0, step=1)), 0).reshape(1,m2,1).repeat(m1,1,m3).to(device)
        k_x3 =  torch.cat((torch.arange(start=0, end=k3, step=1), \
                            torch.arange(start=-k3, end=0, step=1)), 0).reshape(1,1,m3).repeat(m1,m2,1).to(device)

        # K = <y, k_x>,  (batch, N, m1, m2, m3)
        K1 = torch.outer(gridy[:,:,0].view(-1), k_x1.view(-1)).reshape(batchsize, N, m1, m2, m3)
        K2 = torch.outer(gridy[:,:,1].view(-1), k_x2.view(-1)).reshape(batchsize, N, m1, m2, m3)
        K3 = torch.outer(gridy[:,:,2].view(-1), k_x3.view(-1)).reshape(batchsize, N, m1, m2, m3)
        K = 1j*K1 + 1j*K2 + 2j*np.pi*K3

        # basis (N, m1, m2)
        basis = torch.exp(K).to(device)

        # coeff (batch, channels, m1, m2, m3)
        coeff[..., -k3+1:] = coeff[..., 1:k3].flip(-1, -2, -3).conj()

        # Y (batch, channels, N)
        Y = torch.einsum("bcxyz,bnxyz->bcn", coeff, basis)
        Y = Y.real
        return Y

class SimpleBlock3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width, in_dim=4, out_dim=3):
        super(SimpleBlock3d, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        self.width = width
        self.fc0 = nn.Linear(in_dim, self.width)

        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.k3 = LowRank3d(self.width, self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_dim)

    def forward(self, x, y=None):
        # x (batch, x, y, t, channel)
        # y (n, 3)

        batchsize = x.shape[0]
        size_x, size_y, size_z = x.shape[1], x.shape[2], x.shape[3]


        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)

        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x = x1 + x2
        x = F.elu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x = x1 + x2
        x = F.elu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x = x1 + x2
        x = F.elu(x)

        if y==None:
            x1 = self.conv3(x).view(batchsize, self.width, size_x, size_y, size_z)
            x2 = self.k3(x).view(batchsize, self.width, size_x, size_y, size_z)
            x = x1 + x2
            x = x.permute(0, 2, 3, 4, 1)
        else:
            x1 = self.conv3(x, y).view(batchsize, self.width, -1)
            x2 = self.k3(x, y).view(batchsize, self.width, -1)
            x = x1 + x2
            x = x.permute(0, 2, 1)

        x = self.fc1(x)
        x = F.tanh(x)
        x = self.fc2(x)
        return x


class Net3d(nn.Module):
    def __init__(self, modes, width):
        super(Net3d, self).__init__()
        self.conv1 = SimpleBlock3d(modes, modes, modes, width)

    def forward(self, x, y=None):
        x = self.conv1(x, y)
        return x

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c


#####################################################################################
# set the parameters
#####################################################################################

Ntrain = 999
Ntest = 1
ntrain = Ntrain
ntest = Ntest

modes = 12
width = 24

batch_size = 1
batch_size2 = batch_size

epochs = 5000
learning_rate = 0.01
scheduler_step = 500
scheduler_gamma = 0.5

runtime = np.zeros(2, )
t1 = default_timer()

sub = 1
S = 64 // sub
T_in = 1
sub_t = 1
T = 64 // sub_t + 1

print(S, T)
print(epochs, learning_rate, scheduler_step, scheduler_gamma)

HOME_PATH = '../'
path = 'pino_ns_autograd40_N' + str(ntrain) + '_ep' + str(epochs) + '_w' + str(width) + '_s' + str(S) + '_t' + str(T)
path_model = HOME_PATH + 'model/' + path
path_train_err = HOME_PATH + 'results/' + path + 'train.txt'
path_test_err = HOME_PATH + 'results/' + path + 'test.txt'
path_image = HOME_PATH + 'image/' + path

#####################################################################################
# load data
#####################################################################################

data = np.load(HOME_PATH + 'data/NS_fine_Re40_s64_T1000.npy')
print(data.shape)
data = torch.tensor(data, dtype=torch.float)[..., ::sub, ::sub]
print(data.shape)

N = 1000
data2 = torch.zeros(N, S, S, T)
for i in range(N):
    data2[i] = data[i * 64:(i + 1) * 64 + 1:sub_t, :, :].permute(1, 2, 0)
data = data2
train_a = data[:Ntrain, :, :, 0].reshape(ntrain, S, S)
train_u = data[:Ntrain].reshape(ntrain, S, S, T)
print(torch.mean(abs(data[..., 0] - data[..., -1])))

test_a = data[-1, :, :, 0].reshape(ntest, S, S)
test_u = data[-1].reshape(ntest, S, S, T)

print(torch.mean(torch.abs(train_a)), torch.mean(torch.abs(train_u)))

print(train_u.shape)
print(test_u.shape)

train_a = train_a.reshape(ntrain, S, S)
test_a = test_a.reshape(ntest, S, S)

t2 = default_timer()

print('preprocessing finished, time used:', t2 - t1)
device = torch.device('cuda')

#####################################################################################
# PDE loss
#####################################################################################

def forcing(sample):
    return -4 * (torch.cos(4 * (sample[:,1])))

def Autograd_NS_vorticity(vx, vy, w, grid, nu=1 / 40):
    from torch.autograd import grad

    x,y,t = grid

    wx = grad(w.sum(), x, create_graph=True)[0]
    wy = grad(w.sum(), y, create_graph=True)[0]
    wt = grad(w.sum(), t, create_graph=True)[0]

    wxx = grad(wx.sum(), x, create_graph=True)[0]
    wyy = grad(wy.sum(), y, create_graph=True)[0]

    LHS = wt + vx * wx + vy * wy - nu * (wxx+wyy) # - forcing

    vx_y = grad(vx.sum(), y, create_graph=True)[0]
    vy_x = grad(vy.sum(), x, create_graph=True)[0]
    vx_x = grad(vx.sum(), x, create_graph=True)[0]
    vy_y = grad(vy.sum(), y, create_graph=True)[0]
    W = vy_x - vx_y

    return LHS, W, vx_x, vy_y


def PINN_loss(u, truth, N_f, N_ic, N_bc, input, grid, index, ep):
    u = u.squeeze(0)
    input = input.squeeze(0)
    N = u.size(0)
    assert N == N_f + N_ic + N_bc

    vx = u[:, 0]
    vy = u[:, 1]
    w = u[:, 2]

    truthu = w_to_u(truth)

    # lploss = LpLoss(size_average=True)
    myloss = F.mse_loss

    ### IC loss on vorticity
    index_ic = index[:N_ic]

    out_ic_w = w[:N_ic]
    truth_ic_w = truth[0, index_ic[:,0], index_ic[:,1], 0]
    loss_ic_w = myloss(out_ic_w.view(1,-1), truth_ic_w.view(1,-1))

    ### IC loss on velocity
    out_ic_u = u[:N_ic, [0,1]]
    truth_ic_u = truthu[:, index_ic[:, 0], index_ic[:, 1], 0]
    loss_ic_u = myloss(out_ic_u.view(1,-1), truth_ic_u.view(1,-1))

    loss_ic = loss_ic_w + loss_ic_u

    # ### BC loss on vorticity
    # index_bc = index[N_ic:N_ic+N_bc]
    #
    # out_bc_w = w[N_ic:N_ic+N_bc]
    # truth_bc_w = truth[0, index_bc[:, 0], index_bc[:, 1], index_bc[:, 2]]
    # loss_bc_w = myloss(out_bc_w.view(1, -1), truth_bc_w.view(1, -1))
    #
    # ### BC loss on velocity
    # out_bc_u = u[N_ic:N_ic+N_bc, [0,1]]
    # truth_bc_u = truthu[:, index_bc[:, 0], index_bc[:, 1], index_bc[:, 2]]
    # loss_bc_u = myloss(out_bc_u.view(1,-1), truth_bc_u.view(1,-1))
    #
    # loss_bc = loss_bc_w + loss_bc_u
    loss_bc=0

    ### Residual loss
    LHS, W, vx_x, vy_y = Autograd_NS_vorticity(vx, vy, w, grid)
    f = forcing(input)
    loss_f1 = myloss(LHS.view(1,-1), f.view(1,-1))
    loss_f2 = myloss(W.view(1,-1), w.view(1,-1))
    loss_f3 = myloss(vx_x.view(1,-1), -vy_y.view(1,-1))
    loss_f = loss_f1+loss_f2+loss_f3

    return loss_ic, loss_bc, loss_f,  loss_f1, loss_f2, loss_f3


myloss = LpLoss(size_average=True)
error = np.zeros((epochs, 4))
# x_normalizer.cuda()
# y_normalizer.cuda()

model = Net3d(modes, width).cuda()
num_param = model.count_params()
print(num_param)

#####################################################################################
# sample
#####################################################################################
length = torch.tensor([S/(2*np.pi), S/(2*np.pi), T - 1]).reshape(1, 3)

gridx = torch.tensor(np.linspace(0, S - 1, S), dtype=torch.long)
gridx = gridx.reshape(S, 1, 1)
gridy = torch.tensor(np.linspace(0, S - 1, S), dtype=torch.long)
gridy = gridy.reshape(1, S, 1)
gridt = torch.tensor(np.linspace(0, T - 1, T), dtype=torch.long)
gridt = gridt.reshape(1, 1, T)

grid0t = torch.zeros((S, S, 1), dtype=torch.long)
grid0x = torch.zeros((1, S, T), dtype=torch.long)
grid0y = torch.zeros((S, 1, T), dtype=torch.long)
grid1x = torch.ones((1, S, T), dtype=torch.long) * (S-1)
grid1y = torch.ones((S, 1, T), dtype=torch.long) * (S-1)

# (S,S,1)
grid_ic = torch.stack([gridx.repeat(1, S, 1), gridy.repeat(S, 1, 1), grid0t], dim=-1).reshape(S * S, 3)

# (S,1,T)
grid_bc_top = torch.stack([gridx.repeat(1, 1, T), grid1y, gridt.repeat(S, 1, 1)], dim=-1).reshape(S * T, 3)
grid_bc_bottom = torch.stack([gridx.repeat(1, 1, T), grid0y, gridt.repeat(S, 1, 1)], dim=-1).reshape(S * T, 3)
grid_bc_left = torch.stack([grid0x, gridy.repeat(1, 1, T), gridt.repeat(1, S, 1)], dim=-1).reshape(S * T, 3)
grid_bc_right = torch.stack([grid1x, gridy.repeat(1, 1, T), gridt.repeat(1, S, 1)], dim=-1).reshape(S * T, 3)
grid_bc = torch.cat([grid_bc_top, grid_bc_bottom, grid_bc_left, grid_bc_right], dim=0)

# (S, S, T)
gridx = torch.tensor(np.linspace(0, 1, S+1)[:-1], dtype=torch.float)
gridx = gridx.reshape(1, S, 1, 1, 1).repeat([1, 1, S, T, 1])
gridy = torch.tensor(np.linspace(0, 1, S+1)[:-1], dtype=torch.float)
gridy = gridy.reshape(1, 1, S, 1, 1).repeat([1, S, 1, T, 1])
gridt = torch.tensor(np.linspace(0, 1, T), dtype=torch.float)
gridt = gridt.reshape(1, 1, 1, T, 1).repeat([1, S, S, 1, 1])

grid_full = torch.cat((gridx, gridy, gridt, test_a.reshape(1, S, S, 1, 1).repeat(1,1,1,T,1)), dim=-1).cuda()


def get_sample(N_f, N_bc, N_ic):
    # sample IC
    perm = torch.randperm(S*S)[:N_ic]
    index_ic = grid_ic[perm]
    sample_ic = index_ic / length

    # sample BC
    perm = torch.randperm(S * T * 4)[:N_bc]
    index_bc = grid_bc[perm]
    sample_bc = index_bc / length

    # sample f
    sample_i_t = torch.rand(size=(N_f, 1))
    # sample_i_t = torch.rand(size=(N_f, 1)) ** 2
    # sample_i_t = -torch.cos(torch.rand(size=(N_f, 1))*np.pi/2) + 1
    sample_i_x = torch.rand(size=(N_f, 1))
    sample_i_y = torch.rand(size=(N_f, 1))
    sample_f = torch.cat([sample_i_x, sample_i_y, sample_i_t], dim=1)

    sample = torch.cat([sample_ic, sample_bc, sample_f], dim=0).reshape(N_ic + N_bc + N_f, 3).cuda()
    index = torch.cat([index_ic, index_bc], dim=0).reshape(N_ic + N_bc, 3).long().cuda()

    x = sample[:, 0]
    y = sample[:, 1]
    t = sample[:, 2]
    x.requires_grad = True
    y.requires_grad = True
    t.requires_grad = True
    grid = (x,y,t)
    return index, grid


#####################################################################################
# Fine-tune (solving)
#####################################################################################

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

u0 = test_a.cuda()
truth = test_u.cuda()
N_f = 4000
N_bc = 0
N_ic = 1000
N = N_f + N_bc + N_ic

test_pino = 0.0
test_l2 = 0.0
test_f = 0.0
test_f1 = 0.0
test_f2 = 0.0

for ep in range(epochs):
    model.train()
    t1 = default_timer()

    optimizer.zero_grad()

    index, grid = get_sample(N_f=N_f, N_bc=N_bc, N_ic=N_ic)
    input = torch.stack(grid, dim=-1).unsqueeze(dim=0)

    # x_in = F.pad(x, (0, 0, 0, T_pad), "constant", 0)
    # out, DX = model(x_in)
    # out = out.view(batch_size, S, S, T + T_pad)
    # # out = out[..., :-T_pad]
    # x = x[:, :, :, 0, -1]

    out = model(grid_full, input)
    # out = model(input).reshape(N,3)

    loss_ic, loss_bc, loss_f,  loss_f1, loss_f2, loss_f3 = PINN_loss(out, truth, N_f, N_ic, N_bc, input, grid, index, ep)
    pino_loss = (loss_ic * 5 + loss_bc * 0  + loss_f * 1)

    pino_loss.backward()

    optimizer.step()
    test_l2 = 0
    test_pino = pino_loss.item()
    loss_ic = loss_ic.item()
    # test_bc = loss_bc.item()
    test_f = loss_f.item()

    scheduler.step()

    if ep % 100 == 1:
        out = model(grid_full)[..., 2].reshape(S,S,T)
        test_l2 = myloss(out.view(batch_size, S, S, T), truth.view(batch_size, S, S, T)).item()
        print(test_l2)

    if ep % 1000 == 1:
        y = truth[0, :, :, :].cpu().numpy()
        out = out.detach().cpu().numpy()

        fig, ax = plt.subplots(4, 5)
        ax[0,0].imshow(y[..., 0])
        ax[0,1].imshow(y[..., 16])
        ax[0,2].imshow(y[..., 32])
        ax[0,3].imshow(y[..., 48])
        ax[0,4].imshow(y[..., 64])

        ax[1,0].imshow(out[..., 0])
        ax[1,1].imshow(out[..., 16])
        ax[1,2].imshow(out[..., 32])
        ax[1,3].imshow(out[..., 48])
        ax[1,4].imshow(out[..., 64])

        ax[2,0].imshow(y[..., 0, :, :])
        ax[2,1].imshow(y[..., :, 0, :])
        ax[2,2].imshow(y[..., -1, :, :])
        ax[2,3].imshow(y[..., :, -1, :])

        ax[3,0].imshow(out[..., 0, :, :])
        ax[3,1].imshow(out[..., :, 0, :])
        ax[3,2].imshow(out[..., -1, :, :])
        ax[3,3].imshow(out[..., :, -1, :])
        plt.show()

    t2 = default_timer()
    print(ep, t2 - t1, test_pino, loss_ic, test_f)
    print(loss_f1.item(), loss_f2.item(), loss_f3.item())

torch.save(model, path_model + '_finetune')
