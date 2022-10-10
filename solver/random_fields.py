import torch

import math

torch.manual_seed(0)


class GaussianRF(object):
    def __init__(self, dim, size, length=1.0, alpha=2.0, tau=3.0, sigma=None, boundary="periodic", constant_eig=False, device=None):

        self.dim = dim
        self.device = device

        if sigma is None:
            sigma = tau**(0.5*(2*alpha - self.dim))

        k_max = size//2

        const = (4*(math.pi**2))/(length**2)

        if dim == 1:
            k = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                           torch.arange(start=-k_max, end=0, step=1, device=device)), 0)

            self.sqrt_eig = size*math.sqrt(2.0)*sigma*((const*(k**2) + tau**2)**(-alpha/2.0))

            if constant_eig:
                self.sqrt_eig[0] = size*sigma*(tau**(-alpha))
            else:
                self.sqrt_eig[0] = 0.0

        elif dim == 2:
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size,1)

            k_x = wavenumers.transpose(0,1)
            k_y = wavenumers

            self.sqrt_eig = (size**2)*math.sqrt(2.0)*sigma*((const*(k_x**2 + k_y**2) + tau**2)**(-alpha/2.0))

            if constant_eig:
                self.sqrt_eig[0,0] = (size**2)*sigma*(tau**(-alpha))
            else:
                self.sqrt_eig[0,0] = 0.0

        elif dim == 3:
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size,size,1)

            k_x = wavenumers.transpose(1,2)
            k_y = wavenumers
            k_z = wavenumers.transpose(0,2)

            self.sqrt_eig = (size**3)*math.sqrt(2.0)*sigma*((const*(k_x**2 + k_y**2 + k_z**2) + tau**2)**(-alpha/2.0))

            if constant_eig:
                self.sqrt_eig[0,0,0] = (size**3)*sigma*(tau**(-alpha))
            else:
                self.sqrt_eig[0,0,0] = 0.0

        self.size = []
        for j in range(self.dim):
            self.size.append(size)

        self.size = tuple(self.size)

    def sample(self, N):

        coeff = torch.randn(N, *self.size, dtype=torch.cfloat, device=self.device)
        coeff = self.sqrt_eig*coeff

        u = torch.fft.irfftn(coeff, self.size, norm="backward")
        return u


class GaussianRF2d(object):

    def __init__(self, s1, s2, L1=2*math.pi, L2=2*math.pi, alpha=2.0, tau=3.0, sigma=None, mean=None, boundary="periodic", device=None, dtype=torch.float64):

        self.s1 = s1
        self.s2 = s2

        self.mean = mean

        self.device = device
        self.dtype = dtype

        if sigma is None:
            self.sigma = tau**(0.5*(2*alpha - 2.0))
        else:
            self.sigma = sigma

        const1 = (4*(math.pi**2))/(L1**2)
        const2 = (4*(math.pi**2))/(L2**2)

        freq_list1 = torch.cat((torch.arange(start=0, end=s1//2, step=1),\
                                torch.arange(start=-s1//2, end=0, step=1)), 0)
        k1 = freq_list1.view(-1,1).repeat(1, s2//2 + 1).type(dtype).to(device)

        freq_list2 = torch.arange(start=0, end=s2//2 + 1, step=1)

        k2 = freq_list2.view(1,-1).repeat(s1, 1).type(dtype).to(device)

        self.sqrt_eig = s1*s2*self.sigma*((const1*k1**2 + const2*k2**2 + tau**2)**(-alpha/2.0))
        self.sqrt_eig[0,0] = 0.0

    def sample(self, N, xi=None):
        if xi is None:
            xi  = torch.randn(N, self.s1, self.s2//2 + 1, 2, dtype=self.dtype, device=self.device)
        
        xi[...,0] = self.sqrt_eig*xi [...,0]
        xi[...,1] = self.sqrt_eig*xi [...,1]
        
        u = torch.fft.irfft2(torch.view_as_complex(xi), s=(self.s1, self.s2))

        if self.mean is not None:
            u += self.mean
        
        return u