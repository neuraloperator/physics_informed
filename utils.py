import numpy as np
import torch
import torch.autograd as autograd


def get_sample(N, T, s, p, q):
    # sample p nodes from Initial Condition, p nodes from Boundary Condition, q nodes from Interior

    # sample IC
    index_ic = torch.randint(s, size=(N, p))
    sample_ic_t = torch.zeros(N, p)
    sample_ic_x = index_ic/s

    # sample BC
    sample_bc = torch.rand(size=(N, p//2))
    sample_bc_t =  torch.cat([sample_bc, sample_bc],dim=1)
    sample_bc_x = torch.cat([torch.zeros(N, p//2), torch.ones(N, p//2)],dim=1)

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


def get_grid(N, T, s):
    gridt = torch.tensor(np.linspace(0, 1, T), dtype=torch.float).reshape(1, T, 1).repeat(N, 1, s).cuda()
    gridt.requires_grad = True
    gridx = torch.tensor(np.linspace(0, 1, s+1)[:-1], dtype=torch.float).reshape(1, 1, s).repeat(N, T, 1).cuda()
    gridx.requires_grad = True
    grid = torch.stack([gridt, gridx], dim=-1).reshape(N, T*s, 2)
    return grid, gridt, gridx


def PDELoss(model, x, t, nu):
    '''
    Compute the residual of PDE: 
        residual = u_t + u * u_x - nu * u_{xx} : (N,1)

    Params: 
        - model 
        - x, t: (x, t) pairs, (N, 2) tensor
        - nu: constant of PDE
    Return: 
        - mean of residual : scalar 
    '''
    u = model(torch.cat([x, t], dim=1))
    # First backward to compute u_x (shape: N x 1), u_t (shape: N x 1)
    grad_x, grad_t = autograd.grad(outputs=[u.sum()], inputs=[
                                   x, t], create_graph=True)
    # grad_x = grad_xt[:, 0]
    # grad_t = grad_xt[:, 1]

    # Second backward to compute u_{xx} (shape N x 1)

    gradgrad_x, = autograd.grad(
        outputs=[grad_x.sum()], inputs=[x], create_graph=True)
    # gradgrad_x = gradgrad[:, 0]

    residual = grad_t + u * grad_x - nu * gradgrad_x
    return residual


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def zero_grad(params):
    '''
    set grad field to 0
    '''
    if isinstance(params, torch.Tensor):
        if params.grad is not None:
            params.grad.detach()
            params.grad.zero_()
    else:
        for p in params:
            if p.grad is not None:
                p.grad.detach()
                p.grad.zero_()


def count_params(net):
    count = 0
    for p in net.parameters():
        count += p.numel()
    return count



# def compl_mul2d(a, b):
#     # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
#     return torch.einsum("bixy,ioxy->boxy", a, b)
#
#     # return torch.stack([
#     #     op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
#     #     op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
#     # ], dim=-1)
#
#
# class SpectralConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, modes1, modes2):
#         super(SpectralConv2d, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         # Number of Fourier modes to multiply, at most floor(N/2) + 1
#         self.modes1 = modes1
#         self.modes2 = modes2
#
#         self.scale = (1 / (in_channels * out_channels))
#         self.weights1 = nn.Parameter(
#             self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
#         self.weights2 = nn.Parameter(
#             self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
#
#     def forward(self, x):
#         batchsize = x.shape[0]
#         # Compute Fourier coeffcients up to factor of e^(- something constant)
#         x_ft = torch.fft.rfftn(x, dim=[2, 3])
#
#         # Multiply relevant Fourier modes
#         out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2),
#                              x.size(-1)//2 + 1, device=x.device, dtype=torch.cfloat)
#         out_ft[:, :, :self.modes1, :self.modes2] = \
#             compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
#         out_ft[:, :, -self.modes1:, :self.modes2] = \
#             compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
#
#         # Return to physical space
#         x = torch.fft.irfftn(out_ft, s=(x.size(-2), x.size(-1)), dim=[2, 3])
#         return x
#
#
# class SimpleBlock2d(nn.Module):
#     def __init__(self, modes1, modes2, width, in_dim=3, out_dim=1):
#         super(SimpleBlock2d, self).__init__()
#
#         self.modes1 = modes1
#         self.modes2 = modes2
#
#         self.width_list = [width*2//4, width*3 //
#                            4, width*4//4, width*4//4, width*5//4]
#
#         self.fc0 = nn.Linear(in_dim, self.width_list[0])
#
#         self.conv0 = SpectralConv2d(
#             self.width_list[0], self.width_list[1], self.modes1*4//4, self.modes2*4//4)
#         self.conv1 = SpectralConv2d(
#             self.width_list[1], self.width_list[2], self.modes1*3//4, self.modes2*3//4)
#         self.conv2 = SpectralConv2d(
#             self.width_list[2], self.width_list[3], self.modes1*2//4, self.modes2*2//4)
#         self.conv3 = SpectralConv2d(
#             self.width_list[3], self.width_list[4], self.modes1*1//4, self.modes2*1//4)
#         self.w0 = nn.Conv1d(self.width_list[0], self.width_list[1], 1)
#         self.w1 = nn.Conv1d(self.width_list[1], self.width_list[2], 1)
#         self.w2 = nn.Conv1d(self.width_list[2], self.width_list[3], 1)
#         self.w3 = nn.Conv1d(self.width_list[3], self.width_list[4], 1)
#
#         self.fc1 = nn.Linear(self.width_list[4], self.width_list[4]*2)
#         self.fc2 = nn.Linear(self.width_list[4]*2, self.width_list[4]*2)
#         self.fc3 = nn.Linear(self.width_list[4]*2, out_dim)
#
#     def forward(self, x):
#
#         batchsize = x.shape[0]
#         size_x, size_y = x.shape[1], x.shape[2]
#
#         x = self.fc0(x)
#         x = x.permute(0, 3, 1, 2)
#
#         x1 = self.conv0(x)
#         x2 = self.w0(x.view(batchsize, self.width_list[0], size_x*size_y)).view(
#             batchsize, self.width_list[1], size_x, size_y)
#         # x2 = F.interpolate(x2, size=size_list[1], mode='bilinear')
#         x = x1 + x2
#         x = F.selu(x)
#
#         x1 = self.conv1(x)
#         x2 = self.w1(x.view(batchsize, self.width_list[1], size_x*size_y)).view(
#             batchsize, self.width_list[2], size_x, size_y)
#         # x2 = F.interpolate(x2, size=size_list[2], mode='bilinear')
#         x = x1 + x2
#         x = F.selu(x)
#
#         x1 = self.conv2(x)
#         x2 = self.w2(x.view(batchsize, self.width_list[2], size_x*size_y)).view(
#             batchsize, self.width_list[3], size_x, size_y)
#         # x2 = F.interpolate(x2, size=size_list[3], mode='bilinear')
#         x = x1 + x2
#         x = F.selu(x)
#
#         x1 = self.conv3(x)
#         x2 = self.w3(x.view(batchsize, self.width_list[3], size_x*size_y)).view(
#             batchsize, self.width_list[4], size_x, size_y)
#         # x2 = F.interpolate(x2, size=size_list[4], mode='bilinear')
#         x = x1 + x2
#
#         x = x.permute(0, 2, 3, 1)
#         x = self.fc1(x)
#         x = F.selu(x)
#         x = self.fc2(x)
#         x = F.selu(x)
#         x = self.fc3(x)
#         return x
#
#     def get_grid(self, S, batchsize, device):
#         gridx = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
#         gridx = gridx.reshape(1, 1, S, 1).repeat([batchsize, 1, 1, S])
#         gridy = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
#         gridy = gridy.reshape(1, 1, 1, S).repeat([batchsize, 1, S, 1])
#
#         return torch.cat((gridx, gridy), dim=1).to(device)
#
#
# class Net2d(nn.Module):
#     def __init__(self, modes, width):
#         super(Net2d, self).__init__()
#
#         self.conv1 = SimpleBlock2d(modes, modes,  width)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         return x
#
#     def count_params(self):
#         c = 0
#         for p in self.parameters():
#             c += reduce(operator.mul, list(p.size()))
#
#         return c
