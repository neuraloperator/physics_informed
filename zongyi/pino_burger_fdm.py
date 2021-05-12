
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



def compl_mul2d(a, b):
    # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
    return torch.einsum("bixy,ioxy->boxy", a, b)

    # return torch.stack([
    #     op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
    #     op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
    # ], dim=-1)

class LowRank2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LowRank2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.phi = DenseNet([2, 64, 128, in_channels*out_channels], torch.nn.ReLU)
        self.psi = DenseNet([2, 64, 128, in_channels*out_channels], torch.nn.ReLU)

    def get_grid(self, S1, S2, batchsize, device):
        gridx = torch.tensor(np.linspace(0, 1, S1), dtype=torch.float)
        gridx = gridx.reshape(1, S1, 1).repeat([batchsize, 1, S2])
        gridy = torch.tensor(np.linspace(0, 1, S2), dtype=torch.float)
        gridy = gridy.reshape(1, 1, S2).repeat([batchsize, S1, 1])
        return torch.stack((gridx, gridy), dim=-1).to(device)


    def forward(self, x, gridy=None):
        # x (batch, channel, x, y)
        # y (Ny, 2)
        batchsize, size1, size2 = x.shape[0], x.shape[2], x.shape[3]

        gridx = self.get_grid(S1=size1, S2=size2, batchsize=1, device=x.device).reshape(size1 * size2, 2)
        if gridy==None:
            gridy = self.get_grid(S1=size1, S2=size2, batchsize=1, device=x.device).reshape(size1 * size2, 2)
        Nx = size1 * size2
        Ny = gridy.shape[0]

        phi_eval = self.phi(gridx).reshape(Nx, self.out_channels, self.in_channels)
        psi_eval = self.psi(gridy).reshape(Ny, self.out_channels, self.in_channels)
        x = x.reshape(batchsize, self.in_channels, Nx)

        x = torch.einsum('noi,bin,moi->bom', phi_eval, x, psi_eval) / Nx
        return x

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def forward(self, x, size=None):

        if size==None:
            size = (x.size(2), x.size(3))

        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[2,3])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, size[0], size[1]//2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(size[0], size[1]), dim=[2,3])

        return x

class SimpleBlock2d(nn.Module):
    def __init__(self, modes1, modes2, width, in_dim=3, out_dim=1):
        super(SimpleBlock2d, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2

        self.width_list = [width*2//4, width*3//4, width*4//4, width*4//4, width*5//4]

        self.fc0 = nn.Linear(in_dim, self.width_list[0])

        self.conv0 = SpectralConv2d(self.width_list[0], self.width_list[1], self.modes1*4//4, self.modes2*4//4)
        self.conv1 = SpectralConv2d(self.width_list[1], self.width_list[2], self.modes1*3//4, self.modes2*3//4)
        self.conv2 = SpectralConv2d(self.width_list[2], self.width_list[3], self.modes1*2//4, self.modes2*2//4)
        self.conv3 = SpectralConv2d(self.width_list[3], self.width_list[4], self.modes1*1//4, self.modes2*1//4)
        self.w0 = nn.Conv1d(self.width_list[0], self.width_list[1], 1)
        self.w1 = nn.Conv1d(self.width_list[1], self.width_list[2], 1)
        self.w2 = nn.Conv1d(self.width_list[2], self.width_list[3], 1)
        self.w3 = nn.Conv1d(self.width_list[3], self.width_list[4], 1)
        self.k3 = LowRank2d(self.width_list[3], self.width_list[4])

        self.fc1 = nn.Linear(self.width_list[4], self.width_list[4]*2)
        # self.fc2 = nn.Linear(self.width_list[4]*2, self.width_list[4]*2)
        self.fc3 = nn.Linear(self.width_list[4]*2, out_dim)

    def forward(self, x, sub=1):

        batchsize = x.shape[0]
        size_x, size_y = x.shape[1], x.shape[2]
        size = (size_x*sub, size_y*sub)

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width_list[0], size_x*size_y)).view(batchsize, self.width_list[1], size_x, size_y)
        # x2 = F.interpolate(x2, size=size_list[1], mode='bilinear')
        x = x1 + x2
        x = F.elu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width_list[1], size_x*size_y)).view(batchsize, self.width_list[2], size_x, size_y)
        # x2 = F.interpolate(x2, size=size_list[2], mode='bilinear')
        x = x1 + x2
        x = F.elu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width_list[2], size_x*size_y)).view(batchsize, self.width_list[3], size_x, size_y)
        # x2 = F.interpolate(x2, size=size_list[3], mode='bilinear')
        x = x1 + x2
        x = F.elu(x)

        x1 = self.conv3(x, size)
        x2 = self.w3(x.view(batchsize, self.width_list[3], size_x*size_y)).view(batchsize, self.width_list[4], size_x, size_y)
        # x2 = self.k3(x).reshape(batchsize, self.width_list[4], size_x, size_y)
        # x2 = F.interpolate(x2, size=size, mode='bilinear')
        x = x1 + x2

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.elu(x)
        x = self.fc3(x)
        return x

    def get_grid(self, S, batchsize, device):
        gridx = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
        gridx = gridx.reshape(1, 1, S, 1).repeat([batchsize, 1, 1, S])
        gridy = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
        gridy = gridy.reshape(1, 1, 1, S).repeat([batchsize, 1, S, 1])

        return torch.cat((gridx, gridy), dim=1).to(device)

class Net2d(nn.Module):
    def __init__(self, modes, width):
        super(Net2d, self).__init__()

        self.conv1 = SimpleBlock2d(modes, modes,  width)


    def forward(self, x, sub=1):
        x = self.conv1(x, sub)
        return x


    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c


ntrain = 1
ntest = 1

sub = 1 #subsampling rate
h = 128 // sub
s = h
sub_t = 1
T = 101 // sub_t

batch_size = 1
learning_rate = 0.001

epochs = 5000
step_size = 500
gamma = 0.5

modes = 20
width = 64

dataloader = MatReader('data/burgers_pino.mat')
x_data = dataloader.read_field('input')[:,::sub]
y_data = dataloader.read_field('output')[:,::sub_t,::sub]

print(y_data.shape, torch.mean(torch.abs(y_data)))
index = 1009
x_train = x_data[index:index+1]
y_train = y_data[index:index+1]
x_test = x_data[index:index+1]
y_test = y_data[index:index+1]
# x_test = x_data[:ntest]
# y_test = y_data[:ntest]

# x_normalizer = UnitGaussianNormalizer(x_train)
# x_train = x_normalizer.encode(x_train)
# x_test = x_normalizer.encode(x_test)
# y_normalizer = UnitGaussianNormalizer(y_train)
# y_train = y_normalizer.encode(y_train)

path = 'PINO_FDM_burgers_N'+str(ntrain)+'_ep' + str(epochs) + '_m' + str(modes) + '_w' + str(width)
path_model = 'model/'+path
path_pred = 'pred/'+path+'.mat'
path_train_err = 'results/'+path+'train.txt'
path_test_err = 'results/'+path+'test.txt'
path_image = 'image/'+path

gridt = torch.tensor(np.linspace(0,1,T), dtype=torch.float)
gridt = gridt.reshape(1, T, 1)
gridx = torch.tensor(np.linspace(0, 1, s+1)[:-1], dtype=torch.float)
gridx = gridx.reshape(1, 1, s)

x_train = x_train.reshape(ntrain, 1, s).repeat([1,T,1])
x_test = x_test.reshape(ntest, 1, s).repeat([1,T,1])

x_train = torch.stack([x_train, gridx.repeat([ntrain, T, 1]), gridt.repeat([ntrain, 1, s])], dim=3)
x_test = torch.stack([x_test, gridx.repeat([ntest, T, 1]), gridt.repeat([ntest, 1, s])], dim=3)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

model = Net2d(modes, width).cuda()
num_param = model.count_params()
print(num_param)


optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

myloss = LpLoss(size_average=True)


def FDM_Burgers(u, D=1, v=1/100):
    batchsize = u.size(0)
    nt = u.size(1)
    nx = u.size(2)

    u = u.reshape(batchsize, nt, nx)
    dt = D / (nt-1)
    dx = D / (nx)

    # ux: (batch, size-2, size-2)
    # ut = (u[:, 2:, 1:-1] - u[:, :-2, 1:-1]) / (2 * dt)
    # ux = (u[:, 1:-1, 2:] - u[:, 1:-1, :-2]) / (2 * dx)
    # uxx = (u[:, 1:-1, 2:] - 2*u[:, 1:-1, 1:-1] + u[:, 1:-1, :-2]) / (dx**2)
    #
    # u = u[:, 1:-1, 1:-1]
    # Du = ut + ux*u - v*uxx

    u_h = torch.fft.fft(u, dim=2)
    # Wavenumbers in y-direction
    k_max = nx//2
    k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).reshape(1,1,nx)
    ux_h = 2j *np.pi*k_x*u_h
    uxx_h = 2j *np.pi*k_x*ux_h
    ux = torch.fft.irfft(ux_h[:, :, :k_max+1], dim=2, n=nx)
    uxx = torch.fft.irfft(uxx_h[:, :, :k_max+1], dim=2, n=nx)
    ut = (u[:, 2:, :] - u[:, :-2, :]) / (2 * dt)
    Du = ut + (ux*u - v*uxx)[:,1:-1,:]
    return Du


def PINO_loss(u, u0):
    batchsize = u.size(0)
    nt = u.size(1)
    nx = u.size(2)

    u = u.reshape(batchsize, nt, nx)
    # lploss = LpLoss(size_average=True)

    index_t = torch.zeros(nx,).long()
    index_x = torch.tensor(range(nx)).long()
    boundary_u = u[:, index_t, index_x]
    loss_ic = F.mse_loss(boundary_u, u0)

    Du = FDM_Burgers(u)[:,:,:]
    f = torch.zeros(Du.shape, device=u.device)
    loss_f = F.mse_loss(Du, f)

    # loss_bc0 = F.mse_loss(u[:, :, 0], u[:, :, -1])
    # loss_bc1 = F.mse_loss((u[:, :, 1] - u[:, :, -1])/(2/(nx)), (u[:, :, 0] - u[:, :, -2])/(2/(nx)))
    return loss_ic, loss_f, 0


error = np.zeros((epochs, 4))
# x_normalizer.cuda()
# y_normalizer.cuda()
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_pino = 0.0
    train_l2 = 0.0
    train_f = 0.0

    # train with ground truth
    N = 10
    ux, uy = x_train[:N].cuda(), y_train[:N].cuda()

    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()


        out = model(x)

        # out = y_normalizer.decode(out.view(batch_size, T, s))
        # y = y_normalizer.decode(y)
        # u0 = x_normalizer.decode(x[:, 0, :, 0])
        u0 = x[:, 0, :, 0]

        loss = myloss(out.view(batch_size, T, s), y.view(batch_size, T, s))

        loss_ic, loss_f, loss_bc = PINO_loss(out.view(batch_size,T,s), u0)
        # pino_loss = loss_u + loss_ic + loss_f + loss_bc
        pino_loss = (20*loss_ic + loss_f + loss_bc)*100

        pino_loss.backward()

        optimizer.step()
        train_l2 += loss.item()
        train_pino += pino_loss.item()
        train_f += loss_f.item()

    scheduler.step()

    model.eval()
    test_l2 = 0.0
    test_pino = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()

            out = model(x)
            # out = y_normalizer.decode(out.view(batch_size, T, s))
            # u0 = x_normalizer.decode(x[:, 0, :, 0])
            u0 = x[:, 0, :, 0]

            test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()
            test_pino += sum(PINO_loss(out, u0)).item()

    if ep % step_size == 0:
        plt.imshow(y[0,:,:].cpu().numpy())
        plt.show()
        plt.imshow(out[0,:,:,0].cpu().numpy())
        plt.show()

    train_l2 /= len(train_loader)
    test_l2 /= len(test_loader)
    train_pino /= len(train_loader)
    test_pino /= len(test_loader)
    train_f /= len(train_loader)

    error[ep] = [train_pino, train_l2, test_pino, test_l2]

    t2 = default_timer()
    print(ep, t2-t1, train_pino, train_l2, train_f, test_pino, test_l2)

torch.save(model, path_model)
#
# pred = torch.zeros(y_test.shape)
# index = 0
# test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1, shuffle=False)
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
# scipy.io.savemat(path_pred, mdict={'pred': pred.cpu().numpy(), 'error': error})
