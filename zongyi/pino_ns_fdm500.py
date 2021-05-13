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
        x_ft = torch.fft.rfftn(x, dim=[2,3,4])

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
        x = torch.fft.irfftn(out_ft, s=(x.size(2), x.size(3), x.size(4)), dim=[2,3,4])
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
        self.bn0 = torch.nn.BatchNorm3d(self.width)
        self.bn1 = torch.nn.BatchNorm3d(self.width)
        self.bn2 = torch.nn.BatchNorm3d(self.width)
        self.bn3 = torch.nn.BatchNorm3d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y, size_z = x.shape[1], x.shape[2], x.shape[3]

        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)

        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        # x = self.bn0(x1 + x2)
        x = x1 + x2
        x = F.elu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        # x = self.bn1(x1 + x2)
        x = x1 + x2
        x = F.elu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        # x = self.bn2(x1 + x2)
        x = x1 + x2
        x = F.elu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        # x = self.bn3(x1 + x2)
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


Ntrain = 80
Ntest = 1
ntrain = Ntrain
ntest = Ntest

modes = 12
width = 32

batch_size = 1
batch_size2 = batch_size


epochs = 5000
learning_rate = 0.0025
scheduler_step = 500
scheduler_gamma = 0.5



runtime = np.zeros(2, )
t1 = default_timer()


sub = 4
S = 256 // sub
T_in = 1
sub_t = 1
T = 64//sub_t +1

print(S, T)
print(epochs, learning_rate, scheduler_step, scheduler_gamma)

path = 'pino_fdm_ns500_N'+str(ntrain)+'_ep' + str(epochs) + '_m' + str(modes) + '_w' + str(width) + '_s' + str(S) + '_t' + str(T)
path_model = 'model/'+path
path_train_err = 'results/'+path+'train.txt'
path_test_err = 'results/'+path+'test.txt'
path_image = 'image/'+path

data = np.load('data/NS_fine_Re500_s256.npy')
print(data.shape)
data = torch.tensor(data, dtype=torch.float)[..., ::sub,::sub]
print(data.shape)

N = 100
data2 = torch.zeros(N,S,S,T)
for i in range(N):
    data2[i] = data[i*64:(i+1)*64+1:sub_t,:,:].permute(1,2,0)
data = data2
train_a = data[:Ntrain, :, :, 0].reshape(ntrain, S, S)
train_u = data[:Ntrain].reshape(ntrain, S, S, T)
test_a = data[-Ntest:, :, :, 0].reshape(ntest, S, S)
test_u = data[-Ntest:].reshape(ntest, S, S, T)

print(torch.mean(torch.abs(train_a)), torch.mean(torch.abs(train_u)))

print(train_u.shape)
print(test_u.shape)


# x_normalizer = UnitGaussianNormalizer(train_a)
# train_a = x_normalizer.encode(train_a)
# test_a = x_normalizer.encode(test_a)
# y_normalizer = UnitGaussianNormalizer(train_u)
# train_u = y_normalizer.encode(train_u)

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

    Du1 = wt + (ux*wx + uy*wy - v*wlap)[...,1:-1] - forcing
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
    f = torch.zeros(Du.shape, device=u.device)
    loss_f = F.mse_loss(Du, f)
    # f2 = torch.zeros(Du2.shape, device=u.device)
    # loss_f2 = F.mse_loss(Du2, f2)

    return loss_ic, loss_f

myloss = LpLoss(size_average=True)
error = np.zeros((epochs, 4))
# x_normalizer.cuda()
# y_normalizer.cuda()

model = Net2d(modes, width).cuda()
# model = torch.load('model/ns_fourier_V100_N1000_ep100_m8_w20')
num_param = model.count_params()
print(num_param)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)

for ep in range(100):
    model.train()
    t1 = default_timer()
    train_pino = 0.0
    train_l2 = 0.0
    train_f = 0.0
    # train_f2 = 0.0

    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()

        out = model(x).reshape(batch_size, S, S, T)
        x = x[:, :, :, 0, -1]

        loss = myloss(out.view(batch_size, S, S, T), y.view(batch_size, S, S, T))
        loss_ic, loss_f = PINO_loss(out.view(batch_size, S, S, T), x)
        pino_loss = (loss_ic + loss_f) * 1

        loss.backward()

        optimizer.step()
        train_l2 += loss.item()
        train_pino += pino_loss.item()
        train_f += loss_f.item()
        # train_f2 += loss_f2.item()

    scheduler.step()
    train_l2 /= len(train_loader)
    train_f /= len(train_loader)
    # train_f2 /= len(train_loader)
    train_pino /= len(train_loader)
    t2 = default_timer()
    print(ep, t2-t1, train_pino, train_f, train_l2)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

for ep in range(epochs):
    model.train()
    t1 = default_timer()
    test_pino = 0.0
    test_l2 = 0.0
    test_f = 0.0
    # train_f2 = 0.0

    for x, y in test_loader:
        x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()

        out = model(x).reshape(batch_size,S,S,T)
        x = x[:, :, :, 0, -1]
        # x = x_normalizer.decode(x[:, :, :, 0, -1])
        # out = y_normalizer.decode(out)
        # y = y_normalizer.decode(y)

        loss = myloss(out.view(batch_size, S, S, T), y.view(batch_size, S, S, T))
        loss_ic, loss_f = PINO_loss(out.view(batch_size, S, S, T), x)
        pino_loss = (loss_ic + loss_f)*1

        pino_loss.backward()

        optimizer.step()
        test_l2 += loss.item()
        test_pino += pino_loss.item()
        test_f += loss_f.item()
        # train_f2 += loss_f2.item()

    scheduler.step()

    # if ep % scheduler_step == scheduler_step-1:
        # plt.imshow(y[0,:,:,-1].cpu().numpy())
        # plt.show()
        # plt.imshow(out[0,:,:,-1].cpu().numpy())
        # plt.show()

    test_l2 /= len(test_loader)
    test_f /= len(test_loader)
    test_pino /= len(test_loader)

    t2 = default_timer()
    print(ep, t2-t1, test_pino, test_f, test_l2)

torch.save(model, path_model)


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


