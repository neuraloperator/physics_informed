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

class PINN(nn.Module):
    def __init__(self, width):
        super(PINN, self).__init__()
        # layer definitions
        self.FC1 = nn.Linear(3, width)
        self.FC2 = nn.Linear(width, width)
        self.FC3 = nn.Linear(width, width)
        self.FC4 = nn.Linear(width, width)
        self.FC5 = nn.Linear(width, width)
        self.FC6 = nn.Linear(width, width)
        self.FC7 = nn.Linear(width, 3)

    def forward(self, x):
        x = self.FC1(x)
        x = F.tanh(x)
        x = self.FC2(x)
        x = F.tanh(x)
        x = self.FC3(x)
        x = F.tanh(x)
        x = self.FC4(x)
        x = F.tanh(x)
        x = self.FC5(x)
        x = F.tanh(x)
        x = self.FC6(x)
        x = F.tanh(x)
        x = self.FC7(x)
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

width = 100

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
path = 'pinn_ns_autograd40_N' + str(ntrain) + '_ep' + str(epochs) + '_w' + str(width) + '_s' + str(S) + '_t' + str(T)
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

    return LHS, W, wx, wy, vx_x, vx_y, vy_x, vy_y


def PINN_loss(u, truth, N_f, N_ic, N_bc, input, grid, index, ep):
    N = u.size(0)
    assert N == N_f + N_ic + N_bc

    vx = u[:, 0]
    vy = u[:, 1]
    w = u[:, 2]

    truthu = w_to_u(truth)

    # lploss = LpLoss(size_average=True)
    myloss = F.mse_loss

    ### Residual loss
    LHS, W, wx, wy, vx_x, vx_y, vy_x, vy_y = Autograd_NS_vorticity(vx, vy, w, grid)
    f = forcing(input)
    loss_f1 = myloss(LHS.view(1,-1), f.view(1,-1))
    loss_f2 = myloss(W.view(1,-1), w.view(1,-1))
    loss_f3 = myloss(vx_x.view(1,-1), -vy_y.view(1,-1))
    loss_f = loss_f1+loss_f2+loss_f3

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

    ### BC loss on vorticity
    # index_bc1 = index[N_ic:N_ic+N_bc//2]
    # index_bc2 = index[N_ic+N_bc//2:N_ic+N_bc]
    out_bc_w = w[N_ic:N_ic+N_bc]
    loss_bc_w = myloss(out_bc_w[:N_bc//2].view(1, -1), out_bc_w[N_bc//2:].view(1, -1))
    dw = torch.stack([wx,wy],dim=-1)
    loss_bc_dw = myloss(dw[N_ic:N_ic+N_bc//2].view(1, -1), dw[N_ic+N_bc//2:N_ic+N_bc].view(1, -1))

    ### BC loss on velocity
    out_bc_u = u[N_ic:N_ic+N_bc, [0,1]]
    loss_bc_u = myloss(out_bc_u[:N_bc // 2].view(1, -1), out_bc_u[N_bc // 2:].view(1, -1))
    du = torch.stack([vx_x, vx_y, vy_x, vy_y],dim=-1)
    loss_bc_du = myloss(du[N_ic:N_ic+N_bc//2].view(1, -1), du[N_ic+N_bc//2:N_ic+N_bc].view(1, -1))
    loss_bc = loss_bc_w + loss_bc_u + loss_bc_dw + loss_bc_du
    # loss_bc = loss_bc_w



    return loss_ic, loss_bc, loss_f,  loss_f1, loss_f2, loss_f3


myloss = LpLoss(size_average=True)
error = np.zeros((epochs, 4))
# x_normalizer.cuda()
# y_normalizer.cuda()

model = PINN(width).cuda()
num_param = model.count_params()
print(num_param)

#####################################################################################
# sample
#####################################################################################
length = torch.tensor([S/(2*np.pi), S/(2*np.pi), T - 1]).reshape(1, 3) # no end point in x,y
length_bc = torch.tensor([(S-1)/(2*np.pi), (S-1)/(2*np.pi), T - 1]).reshape(1, 3) # with end point in x,y

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
grid_bc = torch.cat([grid_bc_top, grid_bc_right, grid_bc_bottom, grid_bc_left], dim=0)

# (S, S, T)
grid_full = torch.stack([gridx.repeat(1,S,T), gridy.repeat(S,1,T), gridt.repeat(S,S,1)], dim=-1).reshape(-1, 3)/ length
grid_full = grid_full.cuda()

def get_sample(N_f, N_bc, N_ic):
    # sample IC
    perm = torch.randperm(S*S)[:N_ic]
    index_ic = grid_ic[perm]
    sample_ic = index_ic / length

    # sample BC
    perm = torch.randperm(S * T * 2)[:N_bc//2]
    index_bc1 = grid_bc[perm]
    index_bc2 = grid_bc[perm + S*T*2]
    index_bc = torch.cat([index_bc1, index_bc2], dim=0)
    sample_bc = index_bc / length_bc

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
N_bc = 1000
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
    input = torch.stack(grid, dim=-1)

    out = model(input).reshape(N,3)

    loss_ic, loss_bc, loss_f,  loss_f1, loss_f2, loss_f3 = PINN_loss(out, truth, N_f, N_ic, N_bc, input, grid, index, ep)
    pino_loss = (loss_ic * 5 + loss_bc * 0.2  + loss_f)

    pino_loss.backward()

    optimizer.step()
    test_l2 = 0
    test_pino = pino_loss.item()
    loss_ic = loss_ic.item()
    test_bc = loss_bc.item()
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
    print(ep, t2 - t1, test_pino, loss_ic, test_bc, test_f)
    print(loss_f1.item(), loss_f2.item(), loss_f3.item())

torch.save(model, path_model + '_finetune')

# pred = torch.zeros(test_u.shape)
# index = 0
# test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=1, shuffle=False)
# with torch.no_grad():
#     for x, y in test_loader:
#         test_l2 = 0
#         x, y = x.cuda(), y.cuda()
#
#         out = model(x)
#         out = y_normalizer.decode(out)
#         pred[index] = out
#
#         test_l2 += myloss(out.view(1, -1), y.view(1, -1)).item()
#         print(index, test_l2)
#         index = index + 1
#
# scipy.io.savemat('pred/'+path+'.mat', mdict={'pred': pred.cpu().numpy()})


