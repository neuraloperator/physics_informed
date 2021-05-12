
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
            size = x.size(-1)

        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[2,3])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, size, size//2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)


        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(size, size), dim=[2,3])
        return x

class SimpleBlock2d(nn.Module):
    def __init__(self, modes1, modes2, width, in_dim=3, out_dim=1):
        super(SimpleBlock2d, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2

        self.width_list = [width*2//4, width*3//4, width*4//4, width*4//4, width*5//4]
        self.grid_dim = 2

        self.fc0 = nn.Linear(in_dim+2, self.width_list[0])

        self.conv0 = SpectralConv2d(self.width_list[0]+self.grid_dim, self.width_list[1], self.modes1*4//4, self.modes2*4//4)
        self.conv1 = SpectralConv2d(self.width_list[1]+self.grid_dim, self.width_list[2], self.modes1*3//4, self.modes2*3//4)
        self.conv2 = SpectralConv2d(self.width_list[2]+self.grid_dim, self.width_list[3], self.modes1*2//4, self.modes2*2//4)
        self.conv3 = SpectralConv2d(self.width_list[3]+self.grid_dim, self.width_list[4], self.modes1*1//4, self.modes2*1//4)
        self.w0 = nn.Conv1d(self.width_list[0]+self.grid_dim, self.width_list[1], 1)
        self.w1 = nn.Conv1d(self.width_list[1]+self.grid_dim, self.width_list[2], 1)
        self.w2 = nn.Conv1d(self.width_list[2]+self.grid_dim, self.width_list[3], 1)
        self.w3 = nn.Conv1d(self.width_list[3]+self.grid_dim, self.width_list[4], 1)

        self.fc1 = nn.Linear(self.width_list[4], self.width_list[4]*2)
        self.fc3 = nn.Linear(self.width_list[4]*2, out_dim)

    def forward(self, x):

        batchsize = x.shape[0]
        size_x, size_y = x.shape[1], x.shape[2]
        grid = self.get_grid(size_x, batchsize, x.device)
        size_list = [size_x,] * 5

        x = torch.cat((x, grid.permute(0, 2, 3, 1).repeat([1,1,1,1])), dim=-1)

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x = torch.cat((x, grid), dim=1)
        x1 = self.conv0(x, size_list[1])
        x2 = self.w0(x.view(batchsize, self.width_list[0]+self.grid_dim, size_list[0]**2)).view(batchsize, self.width_list[1], size_list[0], size_list[0])
        # x2 = F.interpolate(x2, size=size_list[1], mode='bilinear')
        x = x1 + x2
        x = F.elu(x)

        x = torch.cat((x, self.get_grid(size_list[1], batchsize, x.device)), dim=1)
        x1 = self.conv1(x, size_list[2])
        x2 = self.w1(x.view(batchsize, self.width_list[1]+self.grid_dim, size_list[1]**2)).view(batchsize, self.width_list[2], size_list[1], size_list[1])
        # x2 = F.interpolate(x2, size=size_list[2], mode='bilinear')
        x = x1 + x2
        x = F.elu(x)

        x = torch.cat((x, self.get_grid(size_list[2], batchsize, x.device)), dim=1)
        x1 = self.conv2(x, size_list[3])
        x2 = self.w2(x.view(batchsize, self.width_list[2]+self.grid_dim, size_list[2]**2)).view(batchsize, self.width_list[3], size_list[2], size_list[2])
        # x2 = F.interpolate(x2, size=size_list[3], mode='bilinear')
        x = x1 + x2
        x = F.elu(x)

        x = torch.cat((x, self.get_grid(size_list[3], batchsize, x.device)), dim=1)
        x1 = self.conv3(x, size_list[4])
        x2 = self.w3(x.view(batchsize, self.width_list[3]+self.grid_dim, size_list[3]**2)).view(batchsize, self.width_list[4], size_list[3], size_list[3])
        # x2 = F.interpolate(x2, size=size_list[4], mode='bilinear')
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

    def forward(self, x):
        x = self.conv1(x)
        return x

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c


# TRAIN_PATH = 'data/darcy_s61_N1200.mat'
# TEST_PATH = 'data/darcy_s61_N1200.mat'
TRAIN_PATH = 'data/lognormal_N1024_s61.mat'
TEST_PATH = 'data/lognormal_N1024_s61.mat'
# TRAIN_PATH = 'data/piececonst_r241_N1024_smooth1.mat'
# TEST_PATH = 'data/piececonst_r241_N1024_smooth2.mat'

ntrain = 1
ntest = 1

batch_size = 1
learning_rate = 0.001

epochs = 5000
step_size = 500
gamma = 0.5

modes = 20
width = 64

r = 1
h = int(((61 - 1)/r) + 1)
s = h

print(s)

path = 'PINO_FDM_darcy_N'+str(ntrain)+'_ep' + str(epochs) + '_m' + str(modes) + '_w' + str(width)
path_model = 'model/'+path
path_pred = 'pred/'+path+'.mat'
path_train_err = 'results/'+path+'train.txt'
path_test_err = 'results/'+path+'test.txt'
path_image = 'image/'+path

reader = MatReader(TRAIN_PATH)
x_train = reader.read_field('input')[:ntrain,::r,::r][:,:s,:s]
y_train = reader.read_field('output')[:ntrain,::r,::r][:,:s,:s]

reader.load_file(TEST_PATH)
# x_test = reader.read_field('coeff')[-ntest:,::r,::r][:,:s,:s]
# y_test = reader.read_field('sol')[-ntest:,::r,::r][:,:s,:s]
x_test = reader.read_field('input')[:ntrain,::r,::r][:,:s,:s]
y_test = reader.read_field('output')[:ntrain,::r,::r][:,:s,:s]

print(torch.mean(x_train), torch.mean(y_train))

# x_normalizer = UnitGaussianNormalizer(x_train)
# x_train = x_normalizer.encode(x_train)
# x_test = x_normalizer.encode(x_test)
#
# y_normalizer = UnitGaussianNormalizer(y_train)
# y_train = y_normalizer.encode(y_train)

grids = []
grids.append(np.linspace(0, 1, s))
grids.append(np.linspace(0, 1, s))
grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T
grid = grid.reshape(1,s,s,2)
grid = torch.tensor(grid, dtype=torch.float)
x_train = torch.cat([x_train.reshape(ntrain,s,s,1), grid.repeat(ntrain,1,1,1)], dim=3)
x_test = torch.cat([x_test.reshape(ntest,s,s,1), grid.repeat(ntest,1,1,1)], dim=3)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

model = Net2d(modes, width).cuda()
num_param = model.count_params()
print(num_param)


optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=step_size)

myloss = LpLoss(size_average=False)


def FDM_Darcy(u, a, D=1, f=1):
    batchsize = u.size(0)
    size = u.size(1)
    u = u.reshape(batchsize, size, size)
    a = a.reshape(batchsize, size, size)
    dx = D / (size - 1)
    dy = dx

    # ux: (batch, size-2, size-2)
    ux = (u[:, 2:, 1:-1] - u[:, :-2, 1:-1]) / (2 * dx)
    uy = (u[:, 1:-1, 2:] - u[:, 1:-1, :-2]) / (2 * dy)

    ax = (a[:, 2:, 1:-1] - a[:, :-2, 1:-1]) / (2 * dx)
    ay = (a[:, 1:-1, 2:] - a[:, 1:-1, :-2]) / (2 * dy)
    uxx = (u[:, 2:, 1:-1] -2*u[:,1:-1,1:-1] +u[:, :-2, 1:-1]) / (dx**2)
    uyy = (u[:, 1:-1, 2:] -2*u[:,1:-1,1:-1] +u[:, 1:-1, :-2]) / (dy**2)

    a = a[:, 1:-1, 1:-1]
    u = u[:, 1:-1, 1:-1]
    # Du = -(ax*ux + ay*uy + a*uxx + a*uyy)

    # inner1 = torch.mean(a*(ux**2 + uy**2), dim=[1,2])
    # inner2 = torch.mean(f*u, dim=[1,2])
    # return 0.5*inner1 - inner2

    aux = a * ux
    auy = a * uy
    auxx = (aux[:, 2:, 1:-1] - aux[:, :-2, 1:-1]) / (2 * dx)
    auyy = (auy[:, 1:-1, 2:] - auy[:, 1:-1, :-2]) / (2 * dy)
    Du = - (auxx + auyy)

    return Du


def PINO_loss(u, a):
    batchsize = u.size(0)
    size = u.size(1)
    u = u.reshape(batchsize, size, size)
    a = a.reshape(batchsize, size, size)
    lploss = LpLoss(size_average=True)

    index_x = torch.cat([torch.tensor(range(0, size)), (size - 1) * torch.ones(size), torch.tensor(range(size-1, 1, -1)),
                         torch.zeros(size)], dim=0).long()
    index_y = torch.cat([(size - 1) * torch.ones(size), torch.tensor(range(size-1, 1, -1)), torch.zeros(size),
                         torch.tensor(range(0, size))], dim=0).long()

    boundary_u = u[:, index_x, index_y]
    truth_u = torch.zeros(boundary_u.shape, device=u.device)
    loss_u = lploss.abs(boundary_u, truth_u)

    Du = FDM_Darcy(u, a)
    f = torch.ones(Du.shape, device=u.device)
    loss_f = lploss.rel(Du, f)

    # im = (Du-f)[0].detach().cpu().numpy()
    # plt.imshow(im)
    # plt.show()

    # loss_f = FDM_Darcy(u, a)
    # loss_f = torch.mean(loss_f)
    return loss_f, loss_u

error = np.zeros((epochs, 4))
# x_normalizer.cuda()
# y_normalizer.cuda()
grid = grid.cuda()
mollifier = torch.sin(np.pi*grid[...,0]) * torch.sin(np.pi*grid[...,1]) * 0.001
plt.imshow(mollifier[0, :, :].cpu().numpy())
plt.show()
print(mollifier.shape)
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_pino = 0.0
    train_l2 = 0.0
    train_loss = 0
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()


        out = model(x).reshape(batch_size, s, s)
        out = out * mollifier
        # out = y_normalizer.decode(out)
        # y = y_normalizer.decode(y)

        loss = myloss(out.view(batch_size,-1), y.view(batch_size,-1))
        # a = x_normalizer.decode(x[..., 0])
        a = x[..., 0]
        beta = 0
        loss_f, loss_u = PINO_loss(out, a)
        pino_loss = loss_f + beta*loss_u
        pino_loss.backward()

        optimizer.step()
        train_l2 += loss.item()
        train_pino += pino_loss.item()
        train_loss += torch.tensor([loss_u, loss_f])

    scheduler.step()

    model.eval()
    test_l2 = 0.0
    test_pino = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()

            out = model(x).reshape(batch_size, s, s)
            out = out * mollifier
            # out = y_normalizer.decode(out)

            test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()
            # a = x_normalizer.decode(x[..., 0])
            a = x[..., 0]
            loss_f, loss_u = PINO_loss(out, a)
            test_pino += loss_f.item() + beta*loss_u.item()

    train_l2 /= ntrain
    test_l2 /= ntest
    train_pino /= len(train_loader)
    test_pino /= len(test_loader)
    train_loss /= len(train_loader)

    # if ep % step_size == step_size-1:
    #     plt.imshow(y[0,:,:].cpu().numpy())
    #     plt.show()
    #     plt.imshow(out[0,:,:].cpu().numpy())
    #     plt.show()

    error[ep] = [train_pino, train_l2, test_pino, test_l2]

    t2 = default_timer()
    print(ep, t2-t1, train_pino, train_l2, test_pino, test_l2)
    print(train_loss)

# torch.save(model, path_model)

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
