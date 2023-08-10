import numpy as np
import torch
import torch.nn.functional as F
import gc


def FDM_Darcy(u, a, D=1):
    batchsize = u.size(0)
    size = u.size(1)
    u = u.reshape(batchsize, size, size)
    a = a.reshape(batchsize, size, size)
    dx = D / (size - 1)
    dy = dx

    # ux: (batch, size-2, size-2)
    ux = (u[:, 2:, 1:-1] - u[:, :-2, 1:-1]) / (2 * dx)
    uy = (u[:, 1:-1, 2:] - u[:, 1:-1, :-2]) / (2 * dy)

    # ax = (a[:, 2:, 1:-1] - a[:, :-2, 1:-1]) / (2 * dx)
    # ay = (a[:, 1:-1, 2:] - a[:, 1:-1, :-2]) / (2 * dy)
    # uxx = (u[:, 2:, 1:-1] -2*u[:,1:-1,1:-1] +u[:, :-2, 1:-1]) / (dx**2)
    # uyy = (u[:, 1:-1, 2:] -2*u[:,1:-1,1:-1] +u[:, 1:-1, :-2]) / (dy**2)

    a = a[:, 1:-1, 1:-1]
    # u = u[:, 1:-1, 1:-1]
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


def darcy_loss(u, a):
    batchsize = u.size(0)
    size = u.size(1)
    u = u.reshape(batchsize, size, size)
    a = a.reshape(batchsize, size, size)
    lploss = LpLoss(size_average=True)

    # index_x = torch.cat([torch.tensor(range(0, size)), (size - 1) * torch.ones(size), torch.tensor(range(size-1, 1, -1)),
    #                      torch.zeros(size)], dim=0).long()
    # index_y = torch.cat([(size - 1) * torch.ones(size), torch.tensor(range(size-1, 1, -1)), torch.zeros(size),
    #                      torch.tensor(range(0, size))], dim=0).long()

    # boundary_u = u[:, index_x, index_y]
    # truth_u = torch.zeros(boundary_u.shape, device=u.device)
    # loss_u = lploss.abs(boundary_u, truth_u)

    Du = FDM_Darcy(u, a)
    f = torch.ones(Du.shape, device=u.device)
    loss_f = lploss.rel(Du, f)

    # im = (Du-f)[0].detach().cpu().numpy()
    # plt.imshow(im)
    # plt.show()

    # loss_f = FDM_Darcy(u, a)
    # loss_f = torch.mean(loss_f)
    return loss_f


def FDM_NS_vorticity(w, v=1/40, t_interval=1.0):
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

    dt = t_interval / (nt-1)
    wt = (w[:, :, :, 2:] - w[:, :, :, :-2]) / (2 * dt)

    Du1 = wt + (ux*wx + uy*wy - v*wlap)[...,1:-1] #- forcing
    return Du1


def Autograd_Burgers(u, grid, v=1/100):
    from torch.autograd import grad
    gridt, gridx = grid

    ut = grad(u.sum(), gridt, create_graph=True)[0]
    ux = grad(u.sum(), gridx, create_graph=True)[0]
    uxx = grad(ux.sum(), gridx, create_graph=True)[0]
    Du = ut + ux*u - v*uxx
    return Du, ux, uxx, ut


def AD_loss(u, u0, grid, index_ic=None, p=None, q=None):
    batchsize = u.size(0)
    # lploss = LpLoss(size_average=True)

    Du, ux, uxx, ut = Autograd_Burgers(u, grid)

    if index_ic is None:
        # u in on a uniform grid
        nt = u.size(1)
        nx = u.size(2)
        u = u.reshape(batchsize, nt, nx)

        index_t = torch.zeros(nx,).long()
        index_x = torch.tensor(range(nx)).long()
        boundary_u = u[:, index_t, index_x]

        # loss_bc0 = F.mse_loss(u[:, :, 0], u[:, :, -1])
        # loss_bc1 = F.mse_loss(ux[:, :, 0], ux[:, :, -1])
    else:
        # u is randomly sampled, 0:p are BC, p:2p are ic, 2p:2p+q are interior
        boundary_u = u[:, :p]
        batch_index = torch.tensor(range(batchsize)).reshape(batchsize, 1).repeat(1, p)
        u0 = u0[batch_index, index_ic]

        # loss_bc0 = F.mse_loss(u[:, p:p+p//2], u[:, p+p//2:2*p])
        # loss_bc1 = F.mse_loss(ux[:, p:p+p//2], ux[:, p+p//2:2*p])

    loss_ic = F.mse_loss(boundary_u, u0)
    f = torch.zeros(Du.shape, device=u.device)
    loss_f = F.mse_loss(Du, f)
    return loss_ic, loss_f


class LpLoss(object):
    '''
    loss function with rel/abs Lp loss
    '''
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


def FDM_Burgers(u, v, D=1):
    batchsize = u.size(0)
    nt = u.size(1)
    nx = u.size(2)

    u = u.reshape(batchsize, nt, nx)
    dt = D / (nt-1)
    dx = D / (nx)

    u_h = torch.fft.fft(u, dim=2)
    # Wavenumbers in y-direction
    k_max = nx//2
    k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=u.device),
                     torch.arange(start=-k_max, end=0, step=1, device=u.device)), 0).reshape(1,1,nx)
    ux_h = 2j *np.pi*k_x*u_h
    uxx_h = 2j *np.pi*k_x*ux_h
    ux = torch.fft.irfft(ux_h[:, :, :k_max+1], dim=2, n=nx)
    uxx = torch.fft.irfft(uxx_h[:, :, :k_max+1], dim=2, n=nx)
    ut = (u[:, 2:, :] - u[:, :-2, :]) / (2 * dt)
    Du = ut + (ux*u - v*uxx)[:,1:-1,:]
    return Du


def PINO_loss(u, u0, v):
    batchsize = u.size(0)
    nt = u.size(1)
    nx = u.size(2)

    u = u.reshape(batchsize, nt, nx)
    # lploss = LpLoss(size_average=True)

    index_t = torch.zeros(nx,).long()
    index_x = torch.tensor(range(nx)).long()
    boundary_u = u[:, index_t, index_x]
    loss_u = F.mse_loss(boundary_u, u0)

    Du = FDM_Burgers(u, v)[:, :, :]
    f = torch.zeros(Du.shape, device=u.device)
    loss_f = F.mse_loss(Du, f)

    # loss_bc0 = F.mse_loss(u[:, :, 0], u[:, :, -1])
    # loss_bc1 = F.mse_loss((u[:, :, 1] - u[:, :, -1]) /
    #                       (2/(nx)), (u[:, :, 0] - u[:, :, -2])/(2/(nx)))
    return loss_u, loss_f


def PINO_loss3d(u, u0, forcing, v=1/40, t_interval=1.0):
    batchsize = u.size(0)
    nx = u.size(1)
    ny = u.size(2)
    nt = u.size(3)

    u = u.reshape(batchsize, nx, ny, nt)
    lploss = LpLoss(size_average=True)

    u_in = u[:, :, :, 0]
    loss_ic = lploss(u_in, u0)

    Du = FDM_NS_vorticity(u, v, t_interval)
    f = forcing.repeat(batchsize, 1, 1, nt-2)
    loss_f = lploss(Du, f)

    return loss_ic, loss_f


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
    grad_x, grad_t = torch.autograd.grad(outputs=[u.sum()], inputs=[x, t], create_graph=True)
    # Second backward to compute u_{xx} (shape N x 1)

    gradgrad_x, = torch.autograd.grad(outputs=[grad_x.sum()], inputs=[x], create_graph=True)

    residual = grad_t + u * grad_x - nu * gradgrad_x
    return residual


def get_forcing(S):
    x1 = torch.tensor(np.linspace(0, 2*np.pi, S, endpoint=False), dtype=torch.float).reshape(S, 1).repeat(1, S)
    x2 = torch.tensor(np.linspace(0, 2*np.pi, S, endpoint=False), dtype=torch.float).reshape(1, S).repeat(S, 1)
    return -4 * (torch.cos(4*(x2))).reshape(1,S,S,1)


## ML4Physics additions

def transvese_laplacian(E,input):
    '''
    calculates the transverse laplacian given E and coordinates.
    Args:
    E: The tensor to be derived of shape (batchsize,X,Y,Z,2) the last index is the real and imag part.
    input: The input to the network. a tensor of size (batchsize,X,Y,Z,9) where input[...,:3] = grid_x,grid_y,grid_z

    return:
    Tuple (E_z,E_xx_yy) each of which is a complex tensor in shape (batchsize,X,Y,Z) of the derived field according to the z axis and the tranverse laplecian
    NOTE:
    need to check if needs to multiply by minus
    '''
    #real part
    E_real = E[...,0]
    grad =torch.autograd.grad(outputs=E_real.sum(),inputs=input,retain_graph=True, create_graph=True)[0]
    E_x_real = grad[...,0]
    E_y_real = grad[...,1]
    E_z_real = grad[...,2]
    E_xx_real =torch.autograd.grad(outputs=E_x_real.sum(),inputs=input, create_graph=True)[0][...,0]
    E_yy_real =torch.autograd.grad(outputs=E_y_real.sum(),inputs=input, create_graph=True)[0][...,1]
    E_xx_yy_real = E_xx_real + E_yy_real

    #imag part
    E_imag = E[...,1]
    grad =torch.autograd.grad(outputs=E_imag.sum(),inputs=input,retain_graph=True, create_graph=True)[0]
    E_x_imag = grad[...,0]
    E_y_imag = grad[...,1]
    E_z_imag = grad[...,2]
    E_xx_imag =torch.autograd.grad(outputs=E_x_imag.sum(),inputs=input, create_graph=True)[0][...,0]
    E_yy_imag =torch.autograd.grad(outputs=E_y_imag.sum(),inputs=input, create_graph=True)[0][...,1]
    E_xx_yy_imag = E_xx_imag + E_yy_imag

    E_z = E_z_real + 1j*E_z_imag
    E_xx_yy = E_xx_yy_real + 1j*E_xx_yy_imag
    return (E_z,E_xx_yy)

def coupled_wave_eq_PDE_Loss(u,y,input,equation_dict): 
    '''
    A NAIVE coupled wave equation pde loss calculation.
    Args:
    u: The out put of the network, a tensor of (batchsize,X,Y,Z,2,2)
        The last index is the index (signal out, idler out)
        The one before is the real/imag part (real, imag)
    y: ground truth,  a tensor of size (batchsize,X,Y,Z,2,5)
        The last index is the index (pump,signal vac, idler vac, signal out, idler out)
    input: The input to the network. a tensor of size (batchsize,X,Y,3+2*nin = 9)
        The last index is the index (grid_x ,grid_y, grid_z, pump,signal vac, idler vac), each of the last 3 appears twice, once  real and once imag part
    equation_dict: A dictionary containing
        "chi" -  np.ndarray of the shape (X,Y,Z) contain the chi2 
        "k_pump" -  scalar, the k pump coef
        "k_signal" -  scalar, the k signal coef
        "k_idler" -  scalar, the k idler coef
        "kappa_signal" -  scalar, the kappa signal coef
        "kappa_idler" -  scalar, the kappa idler coef


    return:
        The residule of the equations in tensor shape (batchsize,X,Y,Z,2)
    '''


    delta_k= equation_dict["k_pump"].item() - equation_dict["k_signal"].item() - equation_dict["k_idler"].item()
    kappa_s = equation_dict["kappa_signal"].item()
    kappa_i = equation_dict["kappa_idler"].item()
    chi= equation_dict["chi"].to(u.device)


    signal_out = u[...,0]
    idler_out = u[...,1]

    signal_out_z, signal_out_xx_yy=transvese_laplacian(E=signal_out, input=input)
    idler_out_z, idler_out_xx_yy=transvese_laplacian(E=idler_out, input=input)

    u_full = u[...,0,:] + 1j*u[...,1,:]
    y_full = y[...,0,:] + 1j*y[...,1,:]

    pump = y_full[...,0]
    signal_vac = y_full[...,1]
    idler_vac = y_full[...,2]
    signal_out = u_full[...,0]
    idler_out = u_full[...,1]
    grid_z = input[...,2]

    res = lambda E1_z,E1_xx_yy,k1,kapa1,E2: (1j*E1_z + E1_xx_yy/(2*k1) - kapa1*chi*pump*torch.exp(-1j*delta_k*grid_z)*E2.conj())

    # print("idler_out_z",idler_out_z.shape)
    # print("idler_out_xx_yy",idler_out_xx_yy.shape)
    # print("signal_out_z",signal_out_z.shape)
    # print("signal_out_xx_yy",signal_out_xx_yy.shape)
    # print("signal_vac",signal_vac.shape)
    # print("idler_vac",idler_vac.shape)
    # print("chi",chi.shape)
    # print("pump",pump.shape)
    # print("grid_z",grid_z.shape)
    res1 = res(idler_out_z,idler_out_xx_yy, equation_dict["k_idler"].item(),kappa_i,signal_vac)
    res2 = res(signal_out_z,signal_out_xx_yy, equation_dict["k_signal"].item(),kappa_s,idler_vac)

    residual = torch.cat((res1,res2),dim=-1) # may need to add differend weights
    return torch.abs(residual).type(torch.float32)


# ------ Need to be updated ---------
def coupled_wave_eq_PDE_Loss_numeric(u,equation_dict,grid_z,pump):
    '''
    A NAIVE coupled wave equation pde loss calculation numercial.
    Args:
    u: The out put of the network, a tensor of (batchsize,X,Y,Z,4)
    equation_dict: A dictionary containing
        "chi" -  np.ndarray of the shape (X,Y,Z) contain the chi2 
        "k_pump" -  scalar, the k pump coef
        "k_signal" -  scalar, the k signal coef
        "k_idler" -  scalar, the k idler coef
        "kappa_signal" -  scalar, the kappa signal coef
        "kappa_idler" -  scalar, the kappa idler coef


    return:
        The residule of the equations in tensor shape (batchsize,X,Y,Z,4)
    '''

    delta_k= equation_dict["k_pump"].item() - equation_dict["k_signal"].item() - equation_dict["k_idler"].item()
    kappa_s = equation_dict["kappa_signal"].item()
    kappa_i = equation_dict["kappa_idler"].item()
    chi= equation_dict["chi"][1:-1,1:-1,1:-1,None].to(u.device)
    pump = pump.permute(1,2,3,0)[1:-1,1:-1,1:-1]
    grid_z = grid_z.permute(1,2,3,0)[1:-1,1:-1,1:-1]
    dx = 2e-6 # SHOULD not be hardcoded
    dy = 2e-6
    dz = 10e-6

    u = u.permute(1,2,3,0,4)
    signal_vac = u[...,0]
    idler_vac = u[...,1]
    signal_out = u[...,2]
    idler_out = u[...,3]

    dd_dxx = lambda E: (E[2:,1:-1]+E[:-2,1:-1]-2*E[1:-1,1:-1])/dx**2
    dd_dyy = lambda E: (E[1:-1,2:]+E[1:-1,:-2]-2*E[1:-1,1:-1])/dy**2
    trans_laplasian=  lambda E: (dd_dxx(E)+dd_dyy(E))
    d_dz = lambda E: (E[:,:,2:] - E[:,:,:-2])/(2*dz)

    res = lambda E1,k1,kapa1,E2: (1j*d_dz(E1)[1:-1,1:-1] + trans_laplasian(E1)[:,:,1:-1]/(2*k1) - kapa1*chi*pump*torch.exp(-1j*delta_k*grid_z)*E2[1:-1,1:-1,1:-1].conj())

    res1 = res(idler_out, equation_dict["k_idler"].item(),kappa_i,signal_vac)
    res2 = res(idler_vac, equation_dict["k_idler"].item(),kappa_i,signal_out)
    res3 = res(signal_out, equation_dict["k_signal"].item(),kappa_s,idler_vac)
    res4 = res(signal_vac, equation_dict["k_signal"].item(),kappa_s,idler_out)

    residual = torch.cat((res1,res2,res3,res4),dim=-1) # may need to add differend weights
    return torch.abs(residual).type(torch.float32)

def fourier_diff_of_E(E,k):
    factor=2j*np.pi
    return factor*k*E

def fourier_diffs_of_E(E,kx,ky,kz):
    return fourier_diff_of_E(E,kx), fourier_diff_of_E(E,ky), fourier_diff_of_E(E,kz)

def coupled_wave_eq_PDE_Loss_fourier(u,input,k_arr, kappa_i, kappa_s):
    delta_k=k_arr[0]-(k_arr[1]+k_arr[2])
    x=input[1]
    y=input[2]
    z=input[3]
    batchsize = u.size(0)
    nx = u.size(1)
    ny = u.size(2)
    nz = u.size(3)
    nf = u.size(4)
    u = u.reshape(batchsize, nx, ny, nz,nf)

    signal_vac= u[...,0]
    idler_out=u[...,3]
    signal_out= u[...,2]
    idler_vac=u[...,1]
    grid_z = input[...,-1]

    E_out_i_h = torch.fft.fftn(idler_out, dim=[1, 2, 3])
    E_vac_i_h = torch.fft.fftn(signal_out, dim=[1, 2, 3])
    E_out_s_h = torch.fft.fftn(idler_vac, dim=[1, 2, 3])
    E_vac_s_h = torch.fft.fftn(signal_vac, dim=[1, 2, 3])

    k_x_max = nx//2
    k_y_max = ny//2
    k_z_max = nz//2

    k_x = torch.cat((torch.arange(start=0, end=k_x_max, step=1),
                        torch.arange(start=-k_x_max, end=0, step=1)), 0).reshape(nx, 1).repeat(1, nx).reshape(1,nx,nx,1)
    k_y = torch.cat((torch.arange(start=0, end=k_y_max, step=1),
                        torch.arange(start=-k_y_max, end=0, step=1)), 0).reshape(1, ny).repeat(ny, 1).reshape(1,ny,ny,1)
    k_z = torch.cat((torch.arange(start=0, end=k_z_max, step=1),
                        torch.arange(start=-k_z_max, end=0, step=1)), 0).reshape(1, nz).repeat(nz, 1).reshape(1,nz,nz,1)

    E_out_i_div_x_h, E_out_i_div_y_h, E_out_i_div_z_h= fourier_diffs_of_E(E_out_i_h,k_x,k_y,k_z)
    E_vac_i_div_x_h, E_vac_i_div_y_h, E_vac_i_div_z_h= fourier_diffs_of_E(E_vac_i_h,k_x,k_y,k_z)
    E_out_s_div_x_h, E_out_s_div_y_h, E_out_s_div_z_h= fourier_diffs_of_E(E_out_s_h,k_x,k_y,k_z)
    E_vac_s_div_x_h, E_vac_s_div_y_h, E_vac_s_div_z_h= fourier_diffs_of_E(E_vac_s_h,k_x,k_y,k_z)

    E_out_i_div_xx_h = fourier_diff_of_E(E_out_i_div_x_h,k_x)
    E_vac_i_div_xx_h = fourier_diff_of_E(E_vac_i_div_x_h,k_x)
    E_out_s_div_xx_h = fourier_diff_of_E(E_out_s_div_x_h,k_x)
    E_vac_s_div_xx_h = fourier_diff_of_E(E_vac_s_div_x_h,k_x)
    
    E_out_i_div_yy_h = fourier_diff_of_E(E_out_i_div_y_h,k_x)
    E_vac_i_div_yy_h = fourier_diff_of_E(E_vac_i_div_y_h,k_x)
    E_out_s_div_yy_h = fourier_diff_of_E(E_out_s_div_y_h,k_x)
    E_vac_s_div_yy_h = fourier_diff_of_E(E_vac_s_div_y_h,k_x)

    E_out_i_trans_laplace_h = E_out_i_div_xx_h+E_out_i_div_yy_h
    E_vac_i_trans_laplace_h = E_vac_i_div_xx_h+E_vac_i_div_yy_h
    E_out_s_trans_laplace_h = E_out_s_div_xx_h+E_out_s_div_yy_h
    E_vac_s_trans_laplace_h = E_vac_s_div_xx_h+E_vac_s_div_yy_h

    E_out_i_trans_laplace = torch.fft.ifft(E_out_i_trans_laplace_h, dim=[1,2,3])
    E_vac_i_trans_laplace = torch.fft.ifft(E_vac_i_trans_laplace_h, dim=[1,2,3])
    E_out_s_trans_laplace = torch.fft.ifft(E_out_s_trans_laplace_h, dim=[1,2,3])
    E_vac_s_trans_laplace = torch.fft.ifft(E_vac_s_trans_laplace_h, dim=[1,2,3])

    E_out_i_div_z = torch.fft.ifft(E_out_i_div_z_h, dim=[1,2,3])
    E_vac_i_div_z = torch.fft.ifft(E_vac_i_div_z_h, dim=[1,2,3])
    E_out_s_div_z = torch.fft.ifft(E_out_s_div_z_h, dim=[1,2,3])
    E_vac_s_div_z = torch.fft.ifft(E_vac_s_div_z_h, dim=[1,2,3])

    
    residual_1 = -E_out_i_trans_laplace/(2*k_arr[2]) + kappa_i*torch.exp(-1j*delta_k*z)*signal_vac.conj()-1j*E_out_i_div_z
    residual_2 = -E_out_s_trans_laplace/(2*k_arr[1]) + kappa_s*torch.exp(-1j*delta_k*z)*idler_vac.conj()-1j*E_vac_i_div_z
    residual_3 = -E_vac_s_trans_laplace/(2*k_arr[2]) + kappa_i*torch.exp(-1j*delta_k*z)*signal_out.conj()-1j*E_out_s_div_z
    residual_4 = -E_vac_i_trans_laplace/(2*k_arr[1]) + kappa_s*torch.exp(-1j*delta_k*z)*idler_out.conj()-1j*E_vac_s_div_z

    return torch.sum(abs(residual_1))+torch.sum(abs(residual_2))+torch.sum(abs(residual_3))+torch.sum(abs(residual_4))

def SPDC_loss(u,y,input,equation_dict, grad="autograd"):
    '''
    Calcultae and return the data loss, pde loss and ic (Initial condition) loss
    Args:
    u: The output of the network - tensor of shape (batch size, Nx, Ny, Nz, 2*nout) - where nout is the number of out fields (*2 because of both real and imag part). The fields order: (signal out, idler out)
    y: The entire ground truth solution - tensor of shape 
        (batch size, Nx, Ny, Nz, 2*(nin+nout)) - where nout is the number of out fields (*2 because of both real and imag part). The fields order:      (pump,signal vac, idler vac, signal out, idler out)
    input: The input to the network. a tensor of size (batchsize,X,Y,3+2*nin = 9)
        The last index is the index (grid_x ,grid_y, grid_z, pump,signal vac, idler vac), each of the last 3 appears twice, once  real and once imag part
    equation_dict: A dictionary containing
        "chi" -  np.ndarray of the shape (X,Y,Z) contain the chi2 
        "k_pump" -  scalar, the k pump coef
        "k_signal" -  scalar, the k signal coef
        "k_idler" -  scalar, the k idler coef
        "kappa_signal" -  scalar, the kappa signal coef
        "kappa_idler" -  scalar, the kappa idler coef
    grad: Method of derivation: 
        "autograd" - torch.guto gard
        "numeric" - numericaly
        "fourier_diff" - fourier diffrentiention
        "none" - does not calculate pde loss

    Return: (data_loss,ic_loss,pde_loss)
    '''

    mse_loss = lambda x: F.mse_loss(torch.abs(x),torch.zeros(x.shape,device=x.device,dtype=input.dtype))
    # mse_loss = lambda x: F.l1_loss(torch.abs(x),torch.zeros(x.shape,device=x.device,dtype=input.dtype)) # trying L1 loss
    batchsize = u.size(0)
    nx = u.size(1)
    ny = u.size(2)
    nz = u.size(3)
    u_nfields = u.size(4)//2 # should be 2
    y_nfields = y.size(4)//2 # should be 5

    u = u.reshape(batchsize,nx, ny, nz,2,u_nfields)
    y = y.reshape(batchsize,nx, ny, nz,2,y_nfields)
# calc pde losse
    if grad == "autograd":
        pde_res = coupled_wave_eq_PDE_Loss(u=u,y=y,input=input,equation_dict=equation_dict)
    elif grad == "numeric":
        pde_res = coupled_wave_eq_PDE_Loss_numeric(u=u,equation_dict=equation_dict,grid_z=input[...,-1])
    elif grad == "none":
        pde_res = torch.zeros(u.shape,dtype=input.dtype)
    

    u_full = u[...,0,:] + 1j*u[...,1,:] # real part + j * imag part
    y_full = y[...,0,:] + 1j*y[...,1,:] # real part + j * imag part
    

    u0 = u_full[..., 0,:]
    y0 = y_full[..., 0,:]
    ic_loss = mse_loss(u0-y0[...,-2:])
    data_loss = mse_loss(u_full-y_full[...,-2:])/mse_loss(y_full[...,-2:])
    pde_loss = mse_loss(pde_res)/1e5/0.7578
    gc.collect()
    torch.cuda.empty_cache()

    return data_loss,ic_loss,pde_loss
