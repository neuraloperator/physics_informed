import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from utilities3 import *

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
import scipy.io

torch.manual_seed(0)
np.random.seed(0)

activation = F.relu

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
        self.FC7 = nn.Linear(width, 1)

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


sub_train = 1
S_train = 64 // sub_train
sub_test = 2
S_test = 256 // sub_test
T_interval = 1.0
sub_t = 2
T = int(128*T_interval)//sub_t +1
T_pad = 10
S = S_test
print(S, T)
print(epochs, learning_rate, scheduler_step, scheduler_gamma)

HOME_PATH = '../'
path = 'pinn_ns_fdm500_N'+str(ntrain)+'_ep' + str(epochs) + '_w' + str(width) + '_s' + str(S) + '_t' + str(T)
path_model = HOME_PATH+'model/'+path
path_train_err = HOME_PATH+'results/'+path+'train.txt'
path_test_err = HOME_PATH+'results/'+path+'test.txt'
path_image = HOME_PATH+'image/'+path


#####################################################################################
# load data
#####################################################################################



if T_interval == 0.5:
    data = np.load(HOME_PATH+'data/NS_fine_Re500_s2048_T100.npy')
    print(data.shape)
    # data = torch.tensor(data, dtype=torch.float)[:,:T:sub_t,::sub_test,::sub_test].permute(0,2,3,1)
    data = torch.tensor(data, dtype=torch.float)[:, :T, ::sub_test, ::sub_test].permute(0, 2, 3, 1)
    print(data.shape)
    test_a = data[-Ntest:, :, :, 0].reshape(ntest, S_test, S_test)
    test_u = data[-Ntest:].reshape(ntest, S_test, S_test, T)

if T_interval==1.0:
    data = np.load(HOME_PATH+'data/NS_fine_Re500_s2048_T100.npy')
    data = torch.tensor(data, dtype=torch.float)[:,::sub_t,::sub_test,::sub_test].permute(0,2,3,1)
    print(data.shape)
    test_a = data[-Ntest:, :, :, 0].reshape(ntest, S_test, S_test)
    test_u = data[-Ntest:].reshape(ntest, S_test, S_test, T)


test_a = test_a.reshape(ntest, S, S)

gridx = torch.tensor(np.linspace(0, 1, S+1)[:-1], dtype=torch.float)
gridx = gridx.reshape(1, S, 1, 1, 1).repeat([1, 1, S, T, 1])
gridy = torch.tensor(np.linspace(0, 1, S+1)[:-1], dtype=torch.float)
gridy = gridy.reshape(1, 1, S, 1, 1).repeat([1, S, 1, T, 1])
gridt = torch.tensor(np.linspace(0, 1, T), dtype=torch.float)
gridt = gridt.reshape(1, 1, 1, T, 1).repeat([1, S, S, 1, 1])

train_grid = torch.cat((gridx.repeat([ntrain,1,1,1,1]), gridy.repeat([ntrain,1,1,1,1]),
                       gridt.repeat([ntrain,1,1,1,1])), dim=-1)
test_grid = torch.cat((gridx.repeat([ntest,1,1,1,1]), gridy.repeat([ntest,1,1,1,1]),
                       gridt.repeat([ntest,1,1,1,1])), dim=-1)

test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_grid, test_a, test_u), batch_size=batch_size, shuffle=True)

t2 = default_timer()

print('preprocessing finished, time used:', t2-t1)
device = torch.device('cuda')

#####################################################################################
# PDE loss
#####################################################################################

x1 = torch.tensor(np.linspace(0, 2*np.pi, S+1)[:-1], dtype=torch.float).reshape(S, 1).repeat(1, S)
x2 = torch.tensor(np.linspace(0, 2*np.pi, S+1)[:-1], dtype=torch.float).reshape(1, S).repeat(S, 1)
forcing = -4 * (torch.cos(4*(x2))).reshape(1,S,S,1).cuda()

def FDM_NS_vorticity(w, v=1/500):
    batchsize = w.size(0)
    nx = w.size(1)
    ny = w.size(2)
    nt = w.size(3)
    device = w.device
    w = w.reshape(batchsize, nx, ny, nt)

    w_h = torch.fft.fft2(w, dim=[1, 2])
    # Wavenumbers in y-direction
    k_max = nx//2
    N = nx
    k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).reshape(N, 1).repeat(1, N).reshape(1,N,N,1)
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).reshape(1, N).repeat(N, 1).reshape(1,N,N,1)
    # Negative Laplacian in Fourier space
    lap = (k_x ** 2 + k_y ** 2)
    lap[0, 0, 0, 0] = 1.0
    f_h = w_h / lap

    ux_h = 1j * k_y * f_h
    uy_h = -1j * k_x * f_h
    wx_h = 1j * k_x * w_h
    wy_h = 1j * k_y * w_h
    wlap_h = -lap * w_h

    ux = torch.fft.irfft2(ux_h[:, :, :k_max + 1], dim=[1, 2])
    uy = torch.fft.irfft2(uy_h[:, :, :k_max + 1], dim=[1, 2])
    wx = torch.fft.irfft2(wx_h[:, :, :k_max+1], dim=[1,2])
    wy = torch.fft.irfft2(wy_h[:, :, :k_max+1], dim=[1,2])
    wlap = torch.fft.irfft2(wlap_h[:, :, :k_max+1], dim=[1,2])

    dt = 1/(nt-1)
    wt = (w[:, :, :, 2:] - w[:, :, :, :-2]) / (2 * dt)

    Du1 = wt + (ux*wx + uy*wy - v*wlap)[...,1:-1] #- forcing
    return Du1


def PINO_loss(u, u0):
    batchsize = u.size(0)
    nx = u.size(1)
    ny = u.size(2)
    nt = u.size(3)

    u = u.reshape(batchsize, nx, ny, nt)
    lploss = LpLoss(size_average=True)

    u_in = u[:, :, :, 0]
    loss_ic = lploss(u_in, u0)

    Du = FDM_NS_vorticity(u)
    # f = torch.zeros(Du.shape, device=u.device)
    f = forcing.repeat(batch_size, 1, 1, nt-2)
    loss_f = lploss(Du, f)
    # f2 = torch.zeros(Du2.shape, device=u.device)
    # loss_f2 = F.mse_loss(Du2, f2)

    return loss_ic, loss_f

myloss = LpLoss(size_average=True)
error = np.zeros((epochs, 4))
# x_normalizer.cuda()
# y_normalizer.cuda()

model = PINN(width).cuda()
num_param = model.count_params()
print(num_param)


#####################################################################################
# Fine-tune (solving)
#####################################################################################

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

for ep in range(epochs):
    model.train()
    t1 = default_timer()
    test_pino = 0.0
    test_l2 = 0.0
    test_f = 0.0

    for grid, x, y in test_loader:
        grid, x, y = grid.cuda(), x.cuda(), y.cuda()

        optimizer.zero_grad()

        out = model(grid).view(batch_size, S, S, T)

        loss = myloss(out.view(batch_size, S, S, T), y.view(batch_size, S, S, T))
        loss_ic, loss_f = PINO_loss(out.view(batch_size, S, S, T), x)
        pino_loss = (loss_ic*10 + loss_f)

        pino_loss.backward()

        optimizer.step()
        test_l2 += loss.item()
        test_pino += pino_loss.item()
        test_f += loss_f.item()
        # train_f2 += loss_f2.item()

    scheduler.step()

    if ep % 1000 == 1:
        y = y[0,:,:,:].cpu().numpy()
        out = out[0,:,:,:].detach().cpu().numpy()

        fig, ax = plt.subplots(2, 4)
        ax[0,0].imshow(y[..., 16])
        ax[0,1].imshow(y[..., 32])
        ax[0,2].imshow(y[..., 48])
        ax[0,3].imshow(y[..., 64])
        print(np.mean(np.abs(y[..., 16]-y[..., 64])))

        ax[1,0].imshow(out[..., 16])
        ax[1,1].imshow(out[..., 32])
        ax[1,2].imshow(out[..., 48])
        ax[1,3].imshow(out[..., 64])
        print(np.mean(np.abs(out[..., 16]-out[..., 64])))
        plt.show()

    test_l2 /= len(test_loader)
    test_f /= len(test_loader)
    test_pino /= len(test_loader)

    t2 = default_timer()
    print(ep, t2-t1, test_pino, test_f, test_l2)

torch.save(model, path_model+ '_finetune')


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


