

from timeit import default_timer

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam

from train_utils.datasets import MatReader
from train_utils.losses import LpLoss
from train_utils.utils import count_params

torch.manual_seed(0)
np.random.seed(0)


################################################################
# fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 9  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(3, 128)  # input channel is 3: (a(x, y), x, y)
        self.fc1 = nn.Linear(128, self.width)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc2 = nn.Linear(self.width, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):

        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = F.gelu(x)
        x = self.fc1(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0, self.padding, 0, self.padding])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)
        x = self.fc2(x)
        x = F.gelu(x)
        x = self.fc3(x)

        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

pretrain = False
finetune = not pretrain

TRAIN_PATH = '../data/darcy_s61_N1200.mat'
TEST_PATH = '../data/darcy_s61_N1200.mat'
# TRAIN_PATH = '../data/lognormal_N1024_s61.mat'
# TEST_PATH = '../data/lognormal_N1024_s61.mat'
# TRAIN_PATH = '../data/piececonst_r241_N1024_smooth1.mat'
# TEST_PATH = '../data/piececonst_r241_N1024_smooth2.mat'

ntrain = 1000
ntest = 1

batch_size = 1
learning_rate = 0.001

epochs = 500
step_size = 100
gamma = 0.5

modes = 12
width = 32

r = 1
h = int(((61 - 1)/r) + 1)
s = h

print(s)

path = 'PINO_FDM_darcy_N'+str(ntrain)+'_ep' + str(epochs) + '_m' + str(modes) + '_w' + str(width)
path_model = '../model/'+path
path_pred = '../pred/'+path+'.mat'

reader = MatReader(TRAIN_PATH)
# x_train = reader.read_field('coeff')[:ntrain,::r,::r][:,:s,:s]
# y_train = reader.read_field('sol')[:ntrain,::r,::r][:,:s,:s]
x_train = reader.read_field('input')[:ntrain,::r,::r][:,:s,:s]
y_train = reader.read_field('output')[:ntrain,::r,::r][:,:s,:s]

reader.load_file(TEST_PATH)
# x_test = reader.read_field('coeff')[-ntest:,::r,::r][:,:s,:s]
# y_test = reader.read_field('sol')[-ntest:,::r,::r][:,:s,:s]
a = 1
x_test = reader.read_field('input')[-ntest-a:-a,::r,::r][:,:s,:s]
y_test = reader.read_field('output')[-ntest-a:-a,::r,::r][:,:s,:s]


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
    loss_bd = lploss.abs(boundary_u, truth_u)

    Du = FDM_Darcy(u, a)
    f = torch.ones(Du.shape, device=u.device)
    loss_f = lploss(Du, f)


    # im = (Du-f)[0].detach().cpu().numpy()
    # plt.imshow(im)
    # plt.show()

    # loss_f = FDM_Darcy(u, a)
    # loss_f = torch.mean(loss_f)
    return loss_f, loss_bd

error = np.zeros((epochs, 4))
# x_normalizer.cuda()
# y_normalizer.cuda()
grid = grid.cuda()
mollifier = torch.sin(np.pi*grid[...,0]) * torch.sin(np.pi*grid[...,1]) * 0.001

print(mollifier.shape)
if pretrain:
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size,
                                              shuffle=False)

    model = FNO2d(modes, modes, width).cuda()
    num_param = count_params(model)
    print(num_param)
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_pino = 0.0
        train_l2 = 0.0
        train_loss = 0
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()

            optimizer.zero_grad()
            out = model(x.reshape(batch_size, s, s, 1)).reshape(batch_size, s, s)
            out = out * mollifier

            loss_data = myloss(out.view(batch_size,-1), y.view(batch_size,-1))
            loss_f, loss_bd = PINO_loss(out, x)
            pino_loss = loss_f
            pino_loss.backward()

            optimizer.step()
            train_l2 += loss_data.item()
            train_pino += pino_loss.item()
            train_loss += torch.tensor([loss_bd, loss_f])

        scheduler.step()

        model.eval()
        test_l2 = 0.0
        test_pino = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()

                out = model(x.reshape(batch_size, s, s, 1)).reshape(batch_size, s, s)
                out = out * mollifier

                test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()
                loss_f, loss_bd = PINO_loss(out, x)
                test_pino += loss_f.item() + loss_bd.item()

        train_l2 /= ntrain
        test_l2 /= ntest
        train_pino /= ntrain
        test_pino /= ntest
        train_loss /= ntrain

        error[ep] = [train_pino, train_l2, test_pino, test_l2]

        t2 = default_timer()
        print(ep, t2-t1, train_pino, train_l2, test_pino, test_l2)
        print(train_loss)

    # torch.save(model, '../model/IP-dracy-forward')

def darcy_mask1(x):
    return 1 / (1 + torch.exp(-x)) * 9 + 3

def darcy_mask2(x):
    x = 1 / (1 + torch.exp(-x))
    x[x>0.5] = 1
    x[x<=0.5] = 0
    # x = torch.tensor(x>0.5, dtype=torch.float)
    return  x * 9 + 3

def total_variance(x):
    return torch.mean(torch.abs(x[...,:-1] - x[...,1:])) + torch.mean(torch.abs(x[...,:-1,:] - x[...,1:,:]))


if finetune:
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size,
                                              shuffle=False)

    model = torch.load('../model/IP-dracy-forward').cuda()
    num_param = count_params(model)
    print(num_param)
    xout = torch.rand([1,s,s,1], requires_grad=True, device="cuda")

    optimizer = Adam([xout], lr=0.1, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=step_size)

    for ep in range(10000):
        model.train()
        t1 = default_timer()

        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()

            optimizer.zero_grad()
            out_masked = darcy_mask1(xout)

            yout = model(out_masked.reshape(batch_size, s, s, 1)).reshape(batch_size, s, s)
            yout = yout * mollifier
            loss_data = myloss(yout.view(batch_size, -1), y.view(batch_size, -1))
            loss_f, loss_bd = PINO_loss(y, out_masked)
            loss_TV = total_variance(xout)
            pino_loss = 0.2 * loss_f + loss_data + 0.05 * loss_TV
            # pino_loss = 0. * loss_f + loss_data + 0.05 * loss_TV
            pino_loss.backward()
            optimizer.step()
            scheduler.step()

            out_masked2 = darcy_mask2(xout)
            yout2 = model(out_masked2.reshape(batch_size, s, s, 1)).reshape(batch_size, s, s)
            yout2 = yout2 * mollifier
            testx_l2 = myloss(out_masked.view(batch_size, -1), x.view(batch_size, -1)).item()
            testy_l2 = myloss(yout.view(batch_size, -1), y.view(batch_size, -1)).item()



        t2 = default_timer()
        print(ep, t2 - t1, loss_data.item(), loss_f.item(), testx_l2, testy_l2)

        if ep % 2000 == 1:
            # fig, axs = plt.subplots(2, 3, figsize=(8, 8))
            # axs[0,0].imshow(x.reshape(s,s).detach().cpu().numpy())
            # axs[0,1].imshow(out_masked.reshape(s,s).detach().cpu().numpy())
            # axs[0,2].imshow(out_masked2.reshape(s,s).detach().cpu().numpy())
            # axs[1,0].imshow(y.reshape(s,s).detach().cpu().numpy())
            # axs[1,1].imshow(yout.reshape(s,s).detach().cpu().numpy())
            # axs[1,2].imshow(yout2.reshape(s,s).detach().cpu().numpy())
            # plt.show()
            name_tag = 'PINO-'
            plt.imshow(x.reshape(s,s).detach().cpu().numpy())
            plt.savefig(name_tag+'true-input.pdf',bbox_inches='tight')
            plt.imshow(out_masked.reshape(s,s).detach().cpu().numpy())
            plt.savefig(name_tag+'raw-input.pdf',bbox_inches='tight')
            plt.imshow(out_masked2.reshape(s,s).detach().cpu().numpy())
            plt.savefig(name_tag+'clip-input.pdf',bbox_inches='tight')

            plt.imshow(y.reshape(s,s).detach().cpu().numpy())
            plt.savefig(name_tag+'true-output.pdf',bbox_inches='tight')
            plt.imshow(yout.reshape(s,s).detach().cpu().numpy())
            plt.savefig(name_tag+'raw-output.pdf',bbox_inches='tight')
            plt.imshow(yout.reshape(s,s).detach().cpu().numpy())
            plt.savefig(name_tag+'clip-output.pdf',bbox_inches='tight')

            # scipy.io.savemat('../pred/IP-darcy-forward.mat', mdict={'input_truth': x.reshape(s,s).detach().cpu().numpy(),
            #                                    'input_pred': out_masked.reshape(s,s).detach().cpu().numpy(),
            #                                     'output_truth': y.reshape(s,s).detach().cpu().numpy(),
            #                                     'output_pred': yout.reshape(s,s).detach().cpu().numpy()})

