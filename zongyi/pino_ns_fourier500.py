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


def compl_mul3d(a, b):
    # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
    op = partial(torch.einsum, "bixyz,ioxyz->boxyz")
    return op(a, b)
    # op = partial(torch.einsum, "bixyz,ioxyz->boxyz")
    # return torch.stack([
    #     op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
    #     op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
    # ], dim=-1)

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

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[2,3,4], norm="ortho")

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(2), x.size(3), x.size(4)//2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(2), x.size(3), x.size(4)), dim=[2,3,4], norm="ortho")
        return x

class SimpleBlock2d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width):
        super(SimpleBlock2d, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.fc0 = nn.Linear(4, self.width)

        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
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

        x1 = self.conv3(x)
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x = x1 + x2


        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)
        x = F.elu(x)
        x = self.fc2(x)
        return x

class Net2d(nn.Module):
    def __init__(self, modes, width):
        super(Net2d, self).__init__()

        self.conv1 = SimpleBlock2d(modes, modes, modes, width)


    def forward(self, x):
        x = self.conv1(x)
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

modes = 16
width = 32

batch_size = 1
batch_size2 = batch_size


epochs = 2000
learning_rate = 0.0025
scheduler_step = 400
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
T_pad = 11
S = S_test
print(S, T)
print(epochs, learning_rate, scheduler_step, scheduler_gamma)

HOME_PATH = '../'
path = 'pino_ns_fourier500_N'+str(ntrain)+'_ep' + str(epochs) + '_m' + str(modes) + '_w' + str(width) + '_s' + str(S) + '_t' + str(T)
path_model = HOME_PATH+'model/'+path
path_train_err = HOME_PATH+'results/'+path+'train.txt'
path_test_err = HOME_PATH+'results/'+path+'test.txt'
path_image = HOME_PATH+'image/'+path


#####################################################################################
# load data
#####################################################################################

data = np.load(HOME_PATH+'data/NS_fine_Re500_S512_s64_T500_t128.npy')
print(data.shape)
data = torch.tensor(data, dtype=torch.float)
N = Ntrain
data2 = torch.zeros(N,S_train,S_train,T)

if T_interval == 0.5:
    for i in range(500):
        data2[2 * i] = data[i, :T, ::sub_train, ::sub_train].permute(1, 2, 0)
        data2[2*i+1] = data[i, T-1:, ::sub_train, ::sub_train].permute(1, 2, 0)
    data = data2
    train_a = data[:Ntrain, :, :, 0].reshape(ntrain, S_train, S_train)
    train_u = data[:Ntrain].reshape(ntrain, S_train, S_train, T)
    print(torch.mean(torch.abs(train_a)), torch.mean(torch.abs(train_u)))

    data = np.load(HOME_PATH+'data/NS_fine_Re500_s2048_T100.npy')
    print(data.shape)
    # data = torch.tensor(data, dtype=torch.float)[:,:T:sub_t,::sub_test,::sub_test].permute(0,2,3,1)
    data = torch.tensor(data, dtype=torch.float)[:, :T, ::sub_test, ::sub_test].permute(0, 2, 3, 1)
    print(data.shape)
    test_a = data[-Ntest:, :, :, 0].reshape(ntest, S_test, S_test)
    test_u = data[-Ntest:].reshape(ntest, S_test, S_test, T)

if T_interval==1.0:
    for i in range(500):
        data2[i] = data[i, ::sub_t, ::sub_train, ::sub_train].permute(1, 2, 0)
        # data2[2*i+1] = data[i, 64:, ::sub_train, ::sub_train].permute(1, 2, 0)
    data = data2
    print(data.shape)
    train_a = data[:Ntrain, :, :, 0].reshape(ntrain, S_train, S_train)
    train_u = data[:Ntrain].reshape(ntrain, S_train, S_train, T)
    print(torch.mean(torch.abs(train_a)), torch.mean(torch.abs(train_u)))

    data = np.load(HOME_PATH+'data/NS_fine_Re500_s2048_T100.npy')
    data = torch.tensor(data, dtype=torch.float)[:,::sub_t,::sub_test,::sub_test].permute(0,2,3,1)
    print(data.shape)
    test_a = data[-Ntest:, :, :, 0].reshape(ntest, S_test, S_test)
    test_u = data[-Ntest:].reshape(ntest, S_test, S_test, T)

print(torch.mean(torch.abs(test_a)), torch.mean(torch.abs(test_u)))


train_a = train_a.reshape(ntrain, S_train, S_train, 1, 1).repeat([1,1,1,T,1])
test_a = test_a.reshape(ntest, S_test, S_test, 1, 1).repeat([1,1,1,T,1])

def pad_grid(data, S, T):
    gridx = torch.tensor(np.linspace(0, 1, S+1)[:-1], dtype=torch.float)
    gridx = gridx.reshape(1, S, 1, 1, 1).repeat([1, 1, S, T, 1])
    gridy = torch.tensor(np.linspace(0, 1, S+1)[:-1], dtype=torch.float)
    gridy = gridy.reshape(1, 1, S, 1, 1).repeat([1, S, 1, T, 1])
    gridt = torch.tensor(np.linspace(0, 1, T), dtype=torch.float)
    gridt = gridt.reshape(1, 1, 1, T, 1).repeat([1, S, S, 1, 1])
    n = data.shape[0]
    return torch.cat((gridx.repeat([n,1,1,1,1]), gridy.repeat([n,1,1,1,1]),
                       gridt.repeat([n,1,1,1,1]), data), dim=-1)

train_a = pad_grid(train_a, S_train, T)
test_a = pad_grid(test_a, S_test, T)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)

t2 = default_timer()

print('preprocessing finished, time used:', t2-t1)
device = torch.device('cuda')

#####################################################################################
# PDE loss
#####################################################################################

x1 = torch.tensor(np.linspace(0, 2*np.pi, S+1)[:-1], dtype=torch.float).reshape(S, 1).repeat(1, S)
x2 = torch.tensor(np.linspace(0, 2*np.pi, S+1)[:-1], dtype=torch.float).reshape(1, S).repeat(S, 1)
forcing = -4 * (torch.cos(4*(x2))).reshape(1,S,S,1).cuda()

pi = np.pi
def Fourier_NS_vorticity(w, Lt=1*(T-1+T_pad)/(T-1), nu=1/500):
    batchsize = w.size(0)
    nx = w.size(-3)
    ny = w.size(-2)
    nt = w.size(-1)

    device = w.device
    w = w.reshape(batchsize, nx, ny, nt)

    w_h = torch.fft.fftn(w, dim=[-3, -2, -1])
    # Wavenumbers in y-direction


    k_x = torch.cat((torch.arange(start=0, end=nx//2, step=1, device=device),
                     torch.arange(start=-nx//2, end=0, step=1, device=device)), 0).reshape(nx, 1, 1).repeat(1, ny, nt).reshape(1,nx,ny,nt)
    k_y = torch.cat((torch.arange(start=0, end=ny//2, step=1, device=device),
                     torch.arange(start=-ny//2, end=0, step=1, device=device)), 0).reshape(1, ny, 1).repeat(nx, 1, nt).reshape(1,nx,ny,nt)
    k_t = torch.cat((torch.arange(start=0, end=nt//2, step=1, device=device),
                     torch.arange(start=-nt//2, end=0, step=1, device=device)), 0).reshape(1, 1, nt).repeat(nx, ny, 1).reshape(1,nx,ny,nt)
    # Negative Laplacian in Fourier space

    # Negative Laplacian in Fourier space
    lap = (k_x ** 2 + k_y ** 2)
    lap[0,0,0,:] = 1.0
    f_h = w_h / lap

    ux_h = 1j * k_y * f_h
    uy_h = -1j * k_x * f_h

    wx_h = 1j * k_x * w_h
    wy_h = 1j * k_y * w_h
    wt_h = 1j * k_t * w_h * (2*pi/Lt)
    wlap_h = -lap * w_h

    ux = torch.fft.irfftn(ux_h[..., :nt//2+1], dim=[-3,-2,-1])[...,:T]
    uy = torch.fft.irfftn(uy_h[..., :nt//2+1], dim=[-3,-2,-1])[...,:T]
    wx = torch.fft.irfftn(wx_h[..., :nt//2+1], dim=[-3,-2,-1])[...,:T]
    wy = torch.fft.irfftn(wy_h[..., :nt//2+1], dim=[-3,-2,-1])[...,:T]
    wt = torch.fft.irfftn(wt_h[..., :nt//2+1], dim=[-3,-2,-1])[...,:T]
    wlap = torch.fft.irfftn(wlap_h[..., :nt//2+1], dim=[-3,-2,-1])[...,:T]

    Du = wt + (ux*wx + uy*wy - nu*wlap)  #- forcing
    return Du


def PINO_loss(u, u0):
    batchsize = u.size(0)
    nx = u.size(1)
    ny = u.size(2)
    nt = u.size(3)

    u = u.reshape(batchsize, nx, ny, nt)
    lploss = LpLoss(size_average=True)

    u_in = u[:, :, :, 0]
    loss_ic = lploss(u_in, u0)

    Du = Fourier_NS_vorticity(u)
    # f = torch.zeros(Du.shape, device=u.device)
    f = forcing.repeat(batch_size, 1, 1, T)
    loss_f = lploss(Du, f)
    # f2 = torch.zeros(Du2.shape, device=u.device)
    # loss_f2 = F.mse_loss(Du2, f2)

    return loss_ic, loss_f

myloss = LpLoss(size_average=True)
error = np.zeros((epochs, 4))
# x_normalizer.cuda()
# y_normalizer.cuda()

model = Net2d(modes, width).cuda()
num_param = model.count_params()
print(num_param)

#####################################################################################
# pre-train
#####################################################################################

# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
#
# for ep in range(100):
#     model.train()
#     t1 = default_timer()
#     train_pino = 0.0
#     train_l2 = 0.0
#     train_f = 0.0
#     # train_f2 = 0.0
#
#     # train with ground truth
#     # N = 10
#     # ux, uy = train_a[:N].cuda(), train_u[:N].cuda()
#
#     for x, y in train_loader:
#         x, y = x.cuda(), y.cuda()
#
#         optimizer.zero_grad()
#
#         out = model(x).reshape(batch_size, S, S, T)
#         x = x[:, :, :, 0, -1]
#
#
#         loss = myloss(out.view(batch_size, S, S, T), y.view(batch_size, S, S, T))
#         loss_ic, loss_f = PINO_loss(out.view(batch_size, S, S, T), x)
#         pino_loss = (loss_ic + loss_f)*0.5 + loss
#
#         pino_loss.backward()
#
#         optimizer.step()
#         train_l2 += loss.item()
#         train_pino += pino_loss.item()
#         train_f += loss_f.item()
#         # train_f2 += loss_f2.item()
#
#     test_pino = 0.0
#     test_l2 = 0.0
#     with torch.no_grad():
#         for x, y in test_loader:
#             x, y = x.cuda(), y.cuda()
#
#             out = model(x).reshape(batch_size,S,S,T)
#             x = x[:, :, :, 0, -1]
#
#             loss = myloss(out.view(batch_size, S, S, T), y.view(batch_size, S, S, T))
#             loss_ic, loss_f = PINO_loss(out.view(batch_size, S, S, T), x)
#             pino_loss = (loss_ic + loss_f)*1
#
#             test_l2 += loss.item()
#             test_pino += pino_loss.item()
#
#     scheduler.step()
#     train_l2 /= len(train_loader)
#     train_f /= len(train_loader)
#     train_pino /= len(train_loader)
#     test_l2 /= len(test_loader)
#     test_pino /= len(test_loader)
#     t2 = default_timer()
#     print(ep, t2-t1, train_pino, train_f, train_l2)
#     print(ep, test_pino, test_l2)

# torch.save(model, path_model + '_pretrain5')
# model = torch.load('model/pino_fdm_ns40_N999_ep5000_m12_w32_s64_t65_pretrain5').cuda()

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

    for x, y in test_loader:
        x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()

        x_in = F.pad(x, (0,0,0,T_pad), "constant", 0)
        out = model(x_in).reshape(batch_size,S,S,T+T_pad)
        # out = out[..., :-T_pad]
        x = x[:, :, :, 0, -1]

        loss = myloss(out[..., :T].view(batch_size, S, S, T), y.view(batch_size, S, S, T))
        loss_ic, loss_f = PINO_loss(out.view(batch_size, S, S, T+T_pad), x)
        pino_loss = (loss_ic*5 + loss_f)

        pino_loss.backward()

        optimizer.step()
        test_l2 += loss.item()
        test_pino += pino_loss.item()
        test_f += loss_f.item()
        # train_f2 += loss_f2.item()

    scheduler.step()

    if ep % 1000 == 1:
        y = y[0,:,:,:].cpu().numpy()
        out = out[0,:,:,:T].detach().cpu().numpy()

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


