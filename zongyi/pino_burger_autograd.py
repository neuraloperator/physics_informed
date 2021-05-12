
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
from utilities4 import *

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
        gridx = torch.tensor(np.linspace(0, 1, S1+1)[:-1], dtype=torch.float)
        gridx = gridx.reshape(1, S1, 1).repeat([batchsize, 1, S2])
        gridy = torch.tensor(np.linspace(0, 1, S2+1)[:-1], dtype=torch.float)
        gridy = gridy.reshape(1, 1, S2).repeat([batchsize, S1, 1])
        return torch.stack((gridx, gridy), dim=-1).to(device)


    def forward(self, x, gridy=None):
        # x (batch, channel, x, y)
        # y (batch, Ny, 2)
        batchsize, size1, size2 = x.shape[0], x.shape[2], x.shape[3]

        gridx = self.get_grid(S1=size1, S2=size2, batchsize=1, device=x.device).reshape(size1 * size2, 2)
        if gridy==None:
            gridy = self.get_grid(S1=size1, S2=size2, batchsize=batchsize, device=x.device).reshape(batchsize, size1 * size2, 2)
        Nx = size1 * size2
        Ny = gridy.shape[1]

        phi_eval = self.phi(gridx).reshape(Nx, self.out_channels, self.in_channels)
        psi_eval = self.psi(gridy).reshape(batchsize, Ny, self.out_channels, self.in_channels)
        x = x.reshape(batchsize, self.in_channels, Nx)

        x = torch.einsum('noi,bin,bmoi->bom', phi_eval, x, psi_eval) / Nx
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

    def forward(self, x, gridy=None):
        batchsize = x.shape[0]
        size1 = x.shape[-2]
        size2 = x.shape[-1]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[2,3])

        if gridy==None:
            # Multiply relevant Fourier modes
            out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, device=x.device, dtype=torch.cfloat)
            out_ft[:, :, :self.modes1, :self.modes2] = \
                compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
            out_ft[:, :, -self.modes1:, :self.modes2] = \
                compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

            # Return to physical space
            x = torch.fft.irfftn(out_ft, s=(x.size(-2), x.size(-1)), dim=[2, 3])
        else:
            factor1 = compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
            factor2 = compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
            x = self.ifft2d(gridy, factor1, factor2, self.modes1, self.modes2) / (size1 * size2)
        return x

    def ifft2d(self, gridy, coeff1, coeff2, k1, k2):

        # y (batch, N, 2) locations in [0,1]*[0,1]
        # coeff (batch, channels, kmax, kmax)

        batchsize = gridy.shape[0]
        N = gridy.shape[1]
        device = gridy.device
        m1 = 2 * k1
        m2 = 2 * k2 - 1

        # wavenumber (m1, m2)
        k_x1 =  torch.cat((torch.arange(start=0, end=k1, step=1), \
                            torch.arange(start=-(k1), end=0, step=1)), 0).reshape(m1,1).repeat(1,m2).to(device)
        k_x2 =  torch.cat((torch.arange(start=0, end=k2, step=1), \
                            torch.arange(start=-(k2-1), end=0, step=1)), 0).reshape(1,m2).repeat(m1,1).to(device)

        # K = <y, k_x>,  (batch, N, m1, m2)
        K1 = torch.outer(gridy[:,:,0].view(-1), k_x1.view(-1)).reshape(batchsize, N, m1, m2)
        K2 = torch.outer(gridy[:,:,1].view(-1), k_x2.view(-1)).reshape(batchsize, N, m1, m2)
        K = K1 + K2

        # basis (N, m1, m2)
        basis = torch.exp( 1j * 2* np.pi * K).to(device)

        # coeff (batch, channels, m1, m2)
        coeff3 = coeff1[:,:,1:,1:].flip(-1, -2).conj()
        coeff4 = torch.cat([coeff1[:,:,0:1,1:].flip(-1).conj(), coeff2[:,:,:,1:].flip(-1, -2).conj()], dim=-2)
        coeff12 = torch.cat([coeff1, coeff2], dim=-2)
        coeff43 = torch.cat([coeff4, coeff3], dim=-2)
        coeff = torch.cat([coeff12, coeff43], dim=-1)

        # Y (batch, channels, N)
        Y = torch.einsum("bcxy,bnxy->bcn", coeff, basis)
        Y = Y.real
        return Y

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
        self.fc3 = nn.Linear(self.width_list[4]*2, out_dim)

    def forward(self, x, y=None):
        # x (batch, x, y, channel)
        # y (n, 2)

        batchsize = x.shape[0]
        size_x, size_y = x.shape[1], x.shape[2]

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width_list[0], size_x*size_y)).view(batchsize, self.width_list[1], size_x, size_y)
        x = x1 + x2
        x = F.selu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width_list[1], size_x*size_y)).view(batchsize, self.width_list[2], size_x, size_y)
        x = x1 + x2
        x = F.selu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width_list[2], size_x*size_y)).view(batchsize, self.width_list[3], size_x, size_y)
        x = x1 + x2
        x = F.selu(x)

        x1 = self.conv3(x, y).reshape(batchsize, self.width_list[4], -1)
        x2 = self.k3(x, y).reshape(batchsize, self.width_list[4], -1)
        x = x1 + x2

        x = x.permute(0, 2, 1)

        x = self.fc1(x)
        x = F.selu(x)
        x = self.fc3(x)
        return x


class Net2d(nn.Module):
    def __init__(self, modes, width):
        super(Net2d, self).__init__()
        self.conv1 = SimpleBlock2d(modes, modes,  width)

    def forward(self, x, y=None):
        x = self.conv1(x, y)
        return x

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c


ntrain = 1000
ntest = 100

sub = 2 #subsampling rate
h = 128 // sub
s = h
sub_t = 2
T = 100 // sub_t +1

batch_size = 20
learning_rate = 0.001

epochs = 500
step_size = 50
gamma = 0.5

modes = 20
width = 64

dataloader = MatReader('data/burgers_pino.mat')
x_data = dataloader.read_field('input')[:,::sub]
y_data = dataloader.read_field('output')[:,::sub_t,::sub]
print(x_data.shape)

x_train = x_data[:ntrain]
y_train = y_data[:ntrain]
x_test = x_data[-ntest:]
y_test = y_data[-ntest:]

# x_normalizer = UnitGaussianNormalizer(x_train)
# x_train = x_normalizer.encode(x_train)
# x_test = x_normalizer.encode(x_test)
#
# y_normalizer = UnitGaussianNormalizer(y_train)
# y_train = y_normalizer.encode(y_train)

path = 'PINO_autograd_burgers_N'+str(ntrain)+'_ep' + str(epochs) + '_m' + str(modes) + '_w' + str(width)
path_model = 'model/'+path
path_pred = 'pred/'+path+'.mat'
path_train_err = 'results/'+path+'train.txt'
path_test_err = 'results/'+path+'test.txt'
path_image = 'image/'+path

gridt = torch.tensor(np.linspace(0, 1, T), dtype=torch.float)
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
# myloss = HpLoss(size_average=False, k=2, group=True)


def Autograd_Burgers(u, grid, v=1/100):
    from torch.autograd import grad
    gridt, gridx = grid

    ut = grad(u.sum(), gridt, create_graph=True)[0]
    ux = grad(u.sum(), gridx, create_graph=True)[0]
    uxx = grad(ux.sum(), gridx, create_graph=True)[0]
    # print(ut.device, ux.device, uxx.device, u.device)

    Du = ut + ux*u - v*uxx
    return Du, ux, uxx, ut

def FDM_Burgers(u, D=1, v=1/100):
    batchsize = u.size(0)
    nt = u.size(1)
    nx = u.size(2)

    u = u.reshape(batchsize, nt, nx)
    dt = D / (nt-1)
    dx = D / (nx)

    # ux: (batch, size-2, size-2)
    ut = (u[:, 2:, 1:-1] - u[:, :-2, 1:-1]) / (2 * dt)
    ux = (u[:, 1:-1, 2:] - u[:, 1:-1, :-2]) / (2 * dx)
    uxx = (u[:, 1:-1, 2:] - 2*u[:, 1:-1, 1:-1] + u[:, 1:-1, :-2]) / (dx**2)

    u = u[:, 1:-1, 1:-1]
    Du = ut + ux*u - v*uxx
    return Du, ux, uxx, ut

def PINO_loss(u, u0, grid, index_ic=None, p=None, q=None):
    batchsize = u.size(0)
    # lploss = LpLoss(size_average=True)

    Du, ux, uxx, ut = Autograd_Burgers(u, grid)

    if index_ic==None:
        # u in on a uniform grid
        nt = u.size(1)
        nx = u.size(2)
        u = u.reshape(batchsize, nt, nx)

        index_t = torch.zeros(nx,).long()
        index_x = torch.tensor(range(nx)).long()
        boundary_u = u[:, index_t, index_x]

        loss_bc0 = F.mse_loss(u[:, :, 0], u[:, :, -1])
        loss_bc1 = F.mse_loss(ux[:, :, 0], ux[:, :, -1])
    else:
        # u is randomly sampled, 0:p are BC, p:2p are ic, 2p:2p+q are interior
        boundary_u = u[:, :p]
        batch_index = torch.tensor(range(batchsize)).reshape(batchsize, 1).repeat(1, p)
        u0 = u0[batch_index, index_ic]

        loss_bc0 = F.mse_loss(u[:, p:p+p//2], u[:, p+p//2:2*p])
        loss_bc1 = F.mse_loss(ux[:, p:p+p//2], ux[:, p+p//2:2*p])

    loss_ic = F.mse_loss(boundary_u, u0)
    f = torch.zeros(Du.shape, device=u.device)
    loss_f = F.mse_loss(Du, f)

    # errors = torch.tensor([loss_u, loss_bc0, loss_f, loss_bc1])
    return loss_ic, loss_f, 0


error = np.zeros((epochs, 4))
def get_grid(N, T, s):
    gridt = torch.tensor(np.linspace(0, 1, T), dtype=torch.float).reshape(1, T, 1).repeat(N, 1, s).cuda()
    gridt.requires_grad = True
    gridx = torch.tensor(np.linspace(0, 1, s+1)[:-1], dtype=torch.float).reshape(1, 1, s).repeat(N, T, 1).cuda()
    gridx.requires_grad = True
    grid = torch.stack([gridt, gridx], dim=-1).reshape(N, T*s, 2)
    return grid, gridt, gridx

def get_sample(N, T, s, p, q):
    # sample p nodes from Initial Condition, p nodes from Boundary Condition, q nodes from Interior

    # sample IC
    index_ic = torch.randint(s, size=(N,p))
    sample_ic_t = torch.zeros(N,p)
    sample_ic_x = index_ic/s

    # sample BC
    sample_bc = torch.rand(size=(N,p//2))
    sample_bc_t =  torch.cat([sample_bc, sample_bc],dim=1)
    sample_bc_x = torch.cat([torch.zeros(N,p//2), torch.ones(N,p//2)],dim=1)

    # sample I
    # sample_i_t = torch.rand(size=(N,q))
    # sample_i_t = torch.rand(size=(N,q))**2
    sample_i_t = -torch.cos(torch.rand(size=(N, q))*np.pi/2) + 1
    sample_i_x = torch.rand(size=(N,q))

    sample_t = torch.cat([sample_ic_t, sample_bc_t, sample_i_t], dim=1).cuda()
    sample_t.requires_grad = True
    sample_x = torch.cat([sample_ic_x, sample_bc_x, sample_i_x], dim=1).cuda()
    sample_x.requires_grad = True
    sample = torch.stack([sample_t, sample_x], dim=-1).reshape(N, (p+p+q), 2)
    return sample, sample_t, sample_x, index_ic.long()

# x_normalizer.cuda()
# y_normalizer.cuda()
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_pino = 0.0
    train_l2 = 0.0
    train_loss = torch.zeros(4)

    # train with ground truth
    N = 10
    ux, uy = x_train[:N].cuda(), y_train[:N].cuda()
    # optimizer.zero_grad()
    # out = model(ux)
    # loss_u = myloss(out.view(N,T,s), uy.view(N,T,s))
    # loss_u.backward()
    # optimizer.step()
    # print(loss_u.item())

    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()
        grid, gridt, gridx = get_grid(batch_size, T, s)
        # p = 100
        # q = 400
        # P = p+p+q
        # sample, sample_t, sample_x, index_ic =get_sample(batch_size, T, s, p=p, q=q)


        optimizer.zero_grad()
        out = model(x, grid)
        # out = model(x, sample)

        # from torch.autograd import grad
        # ut = grad(out.sum(), grid, create_graph=True, allow_unused=True)[0]
        # print(ut.shape)

        # out1 = model(x, grid)
        # print(F.mse_loss(out.view(batch_size,-1), out1.view(batch_size,-1)).item())

        # out = y_normalizer.decode(out)
        # y = y_normalizer.decode(y)

        uout = model(ux)
        loss_u = myloss(uout.view(N, T, s), uy.view(N, T, s))

        loss_ic, loss_f, loss_bc = PINO_loss(out.view(batch_size, T, s), x[:, 0, :, 0], (gridt, gridx))
        # loss_ic, loss_f, loss_bc = PINO_loss(out.view(batch_size, P), x[:, 0, :, 0], (sample_t, sample_x), index_ic, p, q)

        pino_loss = (20*loss_ic + loss_f + loss_bc)*100


        pino_loss.backward()
        train_loss += torch.tensor([loss_u, loss_ic, loss_f, loss_bc])

        optimizer.step()
        # loss = myloss(out.view(batch_size,T,s), y.view(batch_size,T,s))
        train_l2 += 0
        train_pino += pino_loss.item()


    scheduler.step()

    model.eval()
    test_l2 = 0.0
    test_pino = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()

            out = model(x)
            # out = y_normalizer.decode(out)

            test_l2 += myloss(out.view(batch_size,T,s), y.view(batch_size,T,s)).item()
            # test_pino += PINO_loss(out.view(batch_size,T,s), x[:, 0, :, 0], (gridt, gridx)).item()

    # if ep % 100 == 0:
    #     plt.imshow(y[0,:,:].cpu().numpy())
    #     plt.show()
    #     plt.imshow(out.reshape(batch_size,T,s)[0,:,:].cpu().numpy())
    #     plt.show()

    train_l2 /= len(train_loader)
    test_l2 /= len(test_loader)
    train_pino /= len(train_loader)
    test_pino /= len(test_loader)
    train_loss /= len(train_loader)

    error[ep] = [train_pino, train_l2, test_pino, test_l2]

    t2 = default_timer()
    print(ep, t2-t1, train_pino, train_l2, test_pino, test_l2)
    print(ep, train_loss)

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
