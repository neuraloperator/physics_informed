"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 3D problem such as the Navier-Stokes equation discussed in Section 5.3 in the [paper](https://arxiv.org/pdf/2010.08895.pdf),
which takes the 2D spatial + 1D temporal equation directly as a 3D problem
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt


from timeit import default_timer
from torch.optim import Adam
from train_utils.datasets import MatReader
from train_utils.losses import LpLoss
from train_utils.utils import count_params

torch.manual_seed(0)
np.random.seed(0)


################################################################
# 3d fourier layers
################################################################

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1) // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x


class FNO3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width, padding):
        super(FNO3d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t). It's a constant function in time, except for the last index.
        input shape: (batchsize, x=64, y=64, t=40, c=13)
        output: the solution of the next 40 timesteps
        output shape: (batchsize, x=64, y=64, t=40, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.padding = padding  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(5, 32)
        self.fc1 = nn.Linear(32, self.width)
        # input channel is 12: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)

        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)

        self.fc2 = nn.Linear(self.width, 128)
        self.fc3 = nn.Linear(128, 3)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = F.tanh(x)
        x = self.fc1(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = F.pad(x, [0, self.padding, 0, self.padding, 0, self.padding])  # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.tanh(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.tanh(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.tanh(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[:, :, :-self.padding, :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 4, 1)  # pad the domain if input is non-periodic
        x = self.fc2(x)
        x = F.tanh(x)
        x = self.fc3(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)


################################################################
# configs
################################################################



# PATH = '../data/cavity.mat'
PATH = '../data/lid-cavity.mat'
ntest = 1

modes = 8
width = 32

batch_size = 1

path = 'cavity'
path_model = 'model/' + path
path_train_err = 'results/' + path + 'train.txt'
path_test_err = 'results/' + path + 'test.txt'
path_image = 'image/' + path



sub_s = 4
sub_t = 20
S = 256 // sub_s
T_in = 1000 # 1000*0.005 = 5s
T = 50 # 1000 + 50*20*0.005 = 10s
padding = 14

################################################################
# load data
################################################################

# 15s, 3000 frames
reader = MatReader(PATH)
data_u = reader.read_field('u')[T_in:T_in+T*sub_t:sub_t, ::sub_s, ::sub_s].permute(1,2,0)
data_v = reader.read_field('v')[T_in:T_in+T*sub_t:sub_t, ::sub_s, ::sub_s].permute(1,2,0)

data_output = torch.stack([data_u, data_v],dim=-1).reshape(batch_size,S,S,T,2)
data_input = data_output[:,:,:,:1,:].repeat(1,1,1,T,1).reshape(batch_size,S,S,T,2)

print(data_output.shape)


device = torch.device('cuda')

def PINO_loss_Fourier_f(out, Re=500):
    pi = np.pi
    Lx = 1*(S + padding-1)/S
    Ly = 1*(S + padding-1)/S
    Lt = (0.005*sub_t*T) *(T + padding)/T

    nx = out.size(1)
    ny = out.size(2)
    nt = out.size(3)
    device = out.device

    # Wavenumbers in y-direction
    k_x = torch.cat((torch.arange(start=0, end=nx//2, step=1, device=device),
                     torch.arange(start=-nx//2, end=0, step=1, device=device)), 0).reshape(nx, 1, 1).repeat(1, ny, nt).reshape(1,nx,ny,nt,1)
    k_y = torch.cat((torch.arange(start=0, end=ny//2, step=1, device=device),
                     torch.arange(start=-ny//2, end=0, step=1, device=device)), 0).reshape(1, ny, 1).repeat(nx, 1, nt).reshape(1,nx,ny,nt,1)
    k_t = torch.cat((torch.arange(start=0, end=nt//2, step=1, device=device),
                     torch.arange(start=-nt//2, end=0, step=1, device=device)), 0).reshape(1, 1, nt).repeat(nx, ny, 1).reshape(1,nx,ny,nt,1)

    out_h = torch.fft.fftn(out, dim=[1, 2, 3])
    outx_h = 1j * k_x * out_h * (2 * pi / Lx)
    outy_h = 1j * k_y * out_h * (2 * pi / Ly)
    outt_h = 1j * k_t * out_h * (2 * pi / Lt)
    outxx_h = 1j * k_x * outx_h * (2 * pi / Lx)
    outyy_h = 1j * k_y * outy_h * (2 * pi / Ly)

    outx = torch.fft.irfftn(outx_h[:, :, :, :nt//2+1, :], dim=[1,2,3])[:,:S,:S,:T]
    outy = torch.fft.irfftn(outy_h[:, :, :, :nt//2+1, :], dim=[1,2,3])[:,:S,:S,:T]
    outt = torch.fft.irfftn(outt_h[:, :, :, :nt//2+1, :], dim=[1,2,3])[:,:S,:S,:T]
    outxx = torch.fft.irfftn(outxx_h[:, :, :, :nt//2+1, :], dim=[1,2,3])[:,:S,:S,:T]
    outyy = torch.fft.irfftn(outyy_h[:, :, :, :nt//2+1, :], dim=[1,2,3])[:,:S,:S,:T]
    out = out[:,:S,:S,:T]


    E1 = outt[..., 0] + out[..., 0]*outx[..., 0] + out[..., 1]*outy[..., 0] + outx[..., 2] - 1/Re*(outxx[..., 0] + outyy[..., 0])
    E2 = outt[..., 1] + out[..., 0]*outx[..., 1] + out[..., 1]*outy[..., 1] + outy[..., 2] - 1/Re*(outxx[..., 1] + outyy[..., 1])
    E3 = outx[..., 0] + outy[..., 1]

    target = torch.zeros(E1.shape, device=E1.device)
    E1 = F.mse_loss(E1,target)
    E2 = F.mse_loss(E2,target)
    E3 = F.mse_loss(E3,target)

    return E1, E2, E3

def PINO_loss_FDM_f(out, Re=500):
    dx = 1 / (S+2)
    dy = 1 / (S+2)
    dt = 0.005*sub_t

    out = out[:,:S,:S,:T,:]
    out = F.pad(out, [0,0, 1,0, 1,1, 1,1])
    out[:, :, -1, :, 0] = 1

    outx = (out[:,2:,1:-1,1:-1] - out[:,:-2,1:-1,1:-1]) / (2*dx)
    outy = (out[:,1:-1,2:,1:-1] - out[:,1:-1,:-2,1:-1]) / (2*dy)
    outt = (out[:,1:-1,1:-1,2:] - out[:,1:-1,1:-1,:-2]) / (2*dt)
    outlap = (out[:,2:,1:-1,1:-1] + out[:,:-2,1:-1,1:-1] + out[:,1:-1,2:,1:-1] + out[:,1:-1,:-2,1:-1] - 4*out[:,1:-1,1:-1,1:-1]) / (dx*dy)

    out = out[:,1:-1,1:-1,1:-1]

    E1 = outt[..., 0] + out[..., 0]*outx[..., 0] + out[..., 1]*outy[..., 0] + outx[..., 2] - 1/Re*(outlap[..., 0])
    E2 = outt[..., 1] + out[..., 0]*outx[..., 1] + out[..., 1]*outy[..., 1] + outy[..., 2] - 1/Re*(outlap[..., 1])
    E3 = outx[..., 0] + outy[..., 1]

    target = torch.zeros(E1.shape, device=E1.device)
    E1 = F.mse_loss(E1,target)
    E2 = F.mse_loss(E2,target)
    E3 = F.mse_loss(E3,target)

    return E1, E2, E3



def PINO_loss_ic(out, y):
    myloss = LpLoss(size_average=True)
    # target = torch.zeros(out.shape, device=out.device)
    # target[:, :, -1, 0] = 1
    # IC = myloss(out, target)
    # return IC

    IC = F.mse_loss(out, y)
    return IC

def PINO_loss_bc(out, y):
    myloss = LpLoss(size_average=True)
    # target = torch.zeros((batch_size,S,T,2), device=out.device)
    # target3 = torch.zeros((batch_size,S,T,2), device=out.device)
    # target3[..., 0] = 1
    # out = torch.stack([out[:,0,:], out[:,-1,:], out[:,:,-1], out[:,:,0]], -1)
    # target = torch.stack([target, target, target3, target], -1)
    # BC = myloss(out, target)
    # return BC

    BC1 = F.mse_loss(out[:,0,:], y[:,0,:])
    BC2 = F.mse_loss(out[:,-1,:], y[:,-1,:])
    BC3 = F.mse_loss(out[:,:,-1], y[:,:,-1])
    BC4 = F.mse_loss(out[:,:,0], y[:,:,0])
    return (BC1+BC2+BC3+BC4)/4

################################################################
# training and evaluation
################################################################



model = model = FNO3d(modes, modes, modes, width, padding).cuda()
print(count_params(model))

optimizer = Adam(model.parameters(), lr=0.0025, weight_decay=0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

myloss = LpLoss(size_average=False)
model.train()
x = data_input.cuda().reshape(batch_size,S,S,T,2)
y = data_output.cuda().reshape(batch_size,S,S,T,2)

for ep in range(5000):
    t1 = default_timer()
    optimizer.zero_grad()

    out = model(x)

    loss_l2 = myloss(out[:,:S,:S,:T,:2], y)
    IC = PINO_loss_ic(out[:,:S,:S,0,:2], y[:,:,:,0])
    BC = PINO_loss_bc(out[:,:S,:S,:T,:2], y)
    E1, E2, E3 = PINO_loss_Fourier_f(out)
    # E1, E2, E3 = PINO_loss_FDM_f(out)
    loss_pino = IC*1 + BC*1 + E1*1 + E2*1 + E3*1

    loss_pino.backward()

    optimizer.step()
    scheduler.step()
    t2 = default_timer()
    print(ep, t2-t1, IC.item(), BC.item(), E1.item(),  E2.item(), E3.item(), loss_l2.item())

    if ep % 1000 == 500:
        y_plot = y[0,:,:,:].cpu().numpy()
        out_plot = out[0,:S,:S,:T].detach().cpu().numpy()

        fig, ax = plt.subplots(2, 2)
        ax[0,0].imshow(y_plot[..., -1, 0])
        ax[0,1].imshow(y_plot[..., -1, 1])
        ax[1,0].imshow(out_plot[..., -1, 0])
        ax[1,1].imshow(out_plot[..., -1, 1])
        plt.show()