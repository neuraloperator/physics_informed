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
from torch.autograd import grad
from torch.autograd.functional import hessian

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

class Net2dA(nn.Module):
    def __init__(self, modes, width):
        super(Net2dA, self).__init__()

        self.modes1 = modes
        self.modes2 = modes
        self.modes3 = modes
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

        return x

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c

class Net2dB(nn.Module):
    def __init__(self, modes, width):
        super(Net2dB, self).__init__()
        self.modes1 = modes
        self.modes2 = modes
        self.modes3 = modes
        self.width = width

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x, is_sum=True):
        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)
        x = F.elu(x)
        x = self.fc2(x)
        # x = x**1

        if is_sum:
            return x.sum()

        return x

#####################################################################################
# set the parameters
#####################################################################################

Ntrain = 999
Ntest = 1
ntrain = Ntrain
ntest = Ntest

modes = 20
width = 32

batch_size = 1
batch_size2 = batch_size


epochs = 2000
learning_rate = 0.01
scheduler_step = 200
scheduler_gamma = 0.5


runtime = np.zeros(2, )
t1 = default_timer()

sub = 1
S = 64 // sub
T_in = 1
sub_t = 1
T = 64//sub_t +1
T_pad = 11

print(S, T)
print(epochs, learning_rate, scheduler_step, scheduler_gamma)

HOME_PATH = '../'
path = 'pino_fdm_ns40_N'+str(ntrain)+'_ep' + str(epochs) + '_m' + str(modes) + '_w' + str(width) + '_s' + str(S) + '_t' + str(T)
path_model = HOME_PATH+'model/'+path
path_train_err = HOME_PATH+'results/'+path+'train.txt'
path_test_err = HOME_PATH+'results/'+path+'test.txt'
path_image = HOME_PATH+'image/'+path


#####################################################################################
# load data
#####################################################################################

data = np.load(HOME_PATH+'data/NS_fine_Re40_s64_T1000.npy')
print(data.shape)
data = torch.tensor(data, dtype=torch.float)[..., ::sub,::sub]
print(data.shape)
print(torch.mean(abs(data[64:] - data[:-64])))

N = 1000
data2 = torch.zeros(N,S,S,T)
for i in range(N):
    data2[i] = data[i*64:(i+1)*64+1:sub_t,:,:].permute(1,2,0)
data = data2
train_a = data[:Ntrain, :, :, 0].reshape(ntrain, S, S)
train_u = data[:Ntrain].reshape(ntrain, S, S, T)
print(torch.mean(abs(data[...,0] - data[..., -1])))


test_a = data[-1, :, :, 0].reshape(ntest, S, S)
test_u = data[-1].reshape(ntest, S, S, T)

print(torch.mean(torch.abs(train_a)), torch.mean(torch.abs(train_u)))

print(train_u.shape)
print(test_u.shape)



train_a = train_a.reshape(ntrain, S, S, 1, 1).repeat([1,1,1,T,1])
test_a = test_a.reshape(ntest, S, S, 1, 1).repeat([1,1,1,T,1])

gridx = torch.tensor(np.linspace(0, 1, S+1)[:-1], dtype=torch.float)
gridx = gridx.reshape(1, S, 1, 1, 1).repeat([1, 1, S, T, 1])
gridy = torch.tensor(np.linspace(0, 1, S+1)[:-1], dtype=torch.float)
gridy = gridy.reshape(1, 1, S, 1, 1).repeat([1, S, 1, T, 1])
gridt = torch.tensor(np.linspace(0, 1, T), dtype=torch.float)
gridt = gridt.reshape(1, 1, 1, T, 1).repeat([1, S, S, 1, 1])

train_a = torch.cat((gridx.repeat([ntrain,1,1,1,1]), gridy.repeat([ntrain,1,1,1,1]),
                       gridt.repeat([ntrain,1,1,1,1]), train_a), dim=-1)
test_a = torch.cat((gridx.repeat([ntest,1,1,1,1]), gridy.repeat([ntest,1,1,1,1]),
                       gridt.repeat([ntest,1,1,1,1]), test_a), dim=-1)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=True)

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
def Ajdoint1(w, Lx=2*pi, Ly=2*pi, Lt=1*(T+T_pad)/(T-1)):
    batchsize = w.size(0)
    channel = w.size(1)
    nx = w.size(-3)
    ny = w.size(-2)
    nt = w.size(-1)

    device = w.device
    w = w.reshape(batchsize, channel, nx, ny, nt)

    w_h = torch.fft.fftn(w, dim=[-3, -2, -1])
    # Wavenumbers in y-direction


    k_x = torch.cat((torch.arange(start=0, end=nx//2, step=1, device=device),
                     torch.arange(start=-nx//2, end=0, step=1, device=device)), 0).reshape(nx, 1, 1).repeat(1, ny, nt).reshape(1,1,nx,ny,nt)
    k_y = torch.cat((torch.arange(start=0, end=ny//2, step=1, device=device),
                     torch.arange(start=-ny//2, end=0, step=1, device=device)), 0).reshape(1, ny, 1).repeat(nx, 1, nt).reshape(1,1,nx,ny,nt)
    k_t = torch.cat((torch.arange(start=0, end=nt//2, step=1, device=device),
                     torch.arange(start=-nt//2, end=0, step=1, device=device)), 0).reshape(1, 1, nt).repeat(nx, ny, 1).reshape(1,1,nx,ny,nt)
    # Negative Laplacian in Fourier space

    wx_h = 1j * k_x * w_h * (2*pi/Lx)
    wy_h = 1j * k_y * w_h * (2*pi/Ly)
    wt_h = 1j * k_t * w_h * (2*pi/Lt)
    wxx_h = 1j * k_x * wx_h * (2*pi/Lx)
    wyy_h = 1j * k_y * wy_h * (2*pi/Ly)

    wx = torch.fft.irfftn(wx_h[..., :nt//2+1], dim=[-3,-2,-1])[...,:T]
    wy = torch.fft.irfftn(wy_h[..., :nt//2+1], dim=[-3,-2,-1])[...,:T]
    wt = torch.fft.irfftn(wt_h[..., :nt//2+1], dim=[-3,-2,-1])[...,:T]
    wxx = torch.fft.irfftn(wxx_h[..., :nt//2+1], dim=[-3,-2,-1])[...,:T]
    wyy = torch.fft.irfftn(wyy_h[..., :nt//2+1], dim=[-3,-2,-1])[...,:T]
    return (wx,wy,wt,wxx,wyy)

def Ajdoint2(Dw, dQ, d2Q):
    dQ = dQ[...,:T]
    d2Q = d2Q[...,:T]
    wx, wy, wt, wxx, wyy = Dw

    wx = torch.sum(wx * dQ, dim=1)
    wy = torch.sum(wy * dQ, dim=1)
    wt = torch.sum(wt * dQ, dim=1)

    wxx = torch.sum(wx**2 * d2Q + wxx * dQ, dim=1)
    wyy = torch.sum(wy**2 * d2Q + wyy * dQ, dim=1)
    return (wx,wy,wt,wxx,wyy)

def FDM_NS_vorticity(w, Dw, v=1/40):
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
    wxx_h = 1j * k_x * wx_h
    wlap_h = -lap * w_h

    ux = torch.fft.irfft2(ux_h[:, :, :k_max + 1], dim=[1, 2])
    uy = torch.fft.irfft2(uy_h[:, :, :k_max + 1], dim=[1, 2])
    wx2 = torch.fft.irfft2(wx_h[:, :, :k_max+1], dim=[1,2])
    wy2 = torch.fft.irfft2(wy_h[:, :, :k_max+1], dim=[1,2])
    wlap2 = torch.fft.irfft2(wlap_h[:, :, :k_max+1], dim=[1,2])
    wxx2 = torch.fft.irfft2(wxx_h[:, :, :k_max + 1], dim=[1, 2])

    dt = 1/(nt-1)
    wt2 = (w[:, :, :, 2:] - w[:, :, :, :-2]) / (2 * dt)
    # Du = wt + (ux*wx + uy*wy - v*wlap)[...,1:-1] #- forcing

    wx, wy, wt, wxx, wyy = Dw
    wlap = wxx + wyy
    Du = wt + (ux*wx + uy*wy - v*wlap2) #- forcing

    # print(F.mse_loss(wx,wx2))
    # print(F.mse_loss(wy,wy2))
    # print(F.mse_loss(wxx, wxx2))
    # print(F.mse_loss(wlap,wlap2))
    # print(F.mse_loss(wt[...,1:-1],wt2))

    return Du


def PINO_loss(u, u0, Dw):
    batchsize = u.size(0)
    nx = u.size(1)
    ny = u.size(2)
    nt = u.size(3)

    u = u.reshape(batchsize, nx, ny, nt)
    lploss = LpLoss(size_average=True)

    u_in = u[:, :, :, 0]
    loss_ic = lploss(u_in, u0)

    Du = FDM_NS_vorticity(u, Dw)
    # f = torch.zeros(Du.shape, device=u.device)
    f = forcing.repeat(batch_size, 1, 1, nt)
    loss_f = lploss(Du, f)
    # f2 = torch.zeros(Du2.shape, device=u.device)
    # loss_f2 = F.mse_loss(Du2, f2)

    return loss_ic, loss_f

myloss = LpLoss(size_average=True)
error = np.zeros((epochs, 4))
# x_normalizer.cuda()
# y_normalizer.cuda()

modelA = Net2dA(modes, width).cuda()
modelB = Net2dB(modes, width).cuda()
num_param = modelA.count_params()
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
params = list(modelA.parameters()) + list(modelB.parameters())
optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

for ep in range(epochs):
    modelA.train()
    modelB.train()
    t1 = default_timer()
    test_pino = 0.0
    test_l2 = 0.0
    test_f = 0.0

    for x, y in test_loader:
        x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()

        x_in = F.pad(x, (0,0,0,T_pad), "constant", 0)
        X = modelA(x_in)
        out = modelB(X, is_sum=False)
        out = out.reshape(batch_size,S,S,T+T_pad)

        dQ = grad(out.sum(), X, create_graph=True)[0]
        d2Q = hessian(modelB, X, create_graph=True)[0]
        print(dQ.shape, d2Q.shape)

        Dw = Ajdoint1(X)
        Dw = Ajdoint2(Dw, dQ, d2Q)

        out = out[..., :-T_pad]
        x = x[:, :, :, 0, -1]

        loss = myloss(out.view(batch_size, S, S, T), y.view(batch_size, S, S, T))
        loss_ic, loss_f = PINO_loss(out.view(batch_size, S, S, T), x, Dw)
        pino_loss = (loss_ic + loss_f)

        pino_loss.backward()

        optimizer.step()
        test_l2 += loss.item()
        test_pino += pino_loss.item()
        test_f += loss_f.item()
        # train_f2 += loss_f2.item()

    scheduler.step()

    # if ep % 1000 == 1:
    #     y = y[0,:,:,:].cpu().numpy()
    #     out = out[0,:,:,:].detach().cpu().numpy()
    #
    #     fig, ax = plt.subplots(2, 4)
    #     ax[0,0].imshow(y[..., 16])
    #     ax[0,1].imshow(y[..., 32])
    #     ax[0,2].imshow(y[..., 48])
    #     ax[0,3].imshow(y[..., 64])
    #     print(np.mean(np.abs(y[..., 16]-y[..., 64])))
    #
    #     ax[1,0].imshow(out[..., 16])
    #     ax[1,1].imshow(out[..., 32])
    #     ax[1,2].imshow(out[..., 48])
    #     ax[1,3].imshow(out[..., 64])
    #     print(np.mean(np.abs(out[..., 16]-out[..., 64])))
    #     plt.show()

    test_l2 /= len(test_loader)
    test_f /= len(test_loader)
    test_pino /= len(test_loader)

    t2 = default_timer()
    print(ep, t2-t1, test_pino, test_f, test_l2)

# torch.save(modelA, path_model+ '_finetuneA')
# torch.save(modelB, path_model+ '_finetuneB')


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


