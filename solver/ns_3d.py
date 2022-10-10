import torch
import torch.fft as fft

import math

#Setup for indexing in the 'ij' format

#Solve: -Lap(u) = f
class Poisson2d(object):

    def __init__(self, s1, s2, L1=2*math.pi, L2=2*math.pi, device=None, dtype=torch.float64):

        self.s1 = s1
        self.s2 = s2

        #Inverse negative Laplacian
        freq_list1 = torch.cat((torch.arange(start=0, end=s1//2, step=1),\
                                torch.arange(start=-s1//2, end=0, step=1)), 0)
        k1 = freq_list1.view(-1,1).repeat(1, s2//2 + 1).type(dtype).to(device)

        freq_list2 = torch.arange(start=0, end=s2//2 + 1, step=1)
        k2 = freq_list2.view(1,-1).repeat(s1, 1).type(dtype).to(device)

        self.inv_lap = ((4*math.pi**2)/(L1**2))*k1**2 + ((4*math.pi**2)/(L2**2))*k2**2
        self.inv_lap[0,0] = 1.0
        self.inv_lap = 1.0/self.inv_lap
    
    def solve(self, f):
        return fft.irfft2(fft.rfft2(f)*self.inv_lap, s=(self.s1, self.s2))
    
    def __call__(self, f):
        return self.solve(f)

#Solve: w_t = - u . grad(w) + (1/Re)*Lap(w) + f
#       u = (psi_y, -psi_x)
#       -Lap(psi) = w
#Note: Adaptive time-step takes smallest step across the batch
class NavierStokes2d(object):

    def __init__(self, s1, s2, L1=2*math.pi, L2=2*math.pi, device=None, dtype=torch.float64):

        self.s1 = s1
        self.s2 = s2

        self.L1 = L1
        self.L2 = L2

        self.h = 1.0/max(s1, s2)

        #Wavenumbers for first derivatives
        freq_list1 = torch.cat((torch.arange(start=0, end=s1//2, step=1),\
                                torch.zeros((1,)),\
                                torch.arange(start=-s1//2 + 1, end=0, step=1)), 0)
        self.k1 = freq_list1.view(-1,1).repeat(1, s2//2 + 1).type(dtype).to(device)


        freq_list2 = torch.cat((torch.arange(start=0, end=s2//2, step=1), torch.zeros((1,))), 0)
        self.k2 = freq_list2.view(1,-1).repeat(s1, 1).type(dtype).to(device)

        #Negative Laplacian
        freq_list1 = torch.cat((torch.arange(start=0, end=s1//2, step=1),\
                                torch.arange(start=-s1//2, end=0, step=1)), 0)
        k1 = freq_list1.view(-1,1).repeat(1, s2//2 + 1).type(dtype).to(device)

        freq_list2 = torch.arange(start=0, end=s2//2 + 1, step=1)
        k2 = freq_list2.view(1,-1).repeat(s1, 1).type(dtype).to(device)

        self.G = ((4*math.pi**2)/(L1**2))*k1**2 + ((4*math.pi**2)/(L2**2))*k2**2

        #Inverse of negative Laplacian
        self.inv_lap = self.G.clone()
        self.inv_lap[0,0] = 1.0
        self.inv_lap = 1.0/self.inv_lap

        #Dealiasing mask using 2/3 rule
        self.dealias = (self.k1**2 + self.k2**2 <= 0.6*(0.25*s1**2 + 0.25*s2**2)).type(dtype).to(device)
        #Ensure mean zero
        self.dealias[0,0] = 0.0

    #Compute stream function from vorticity (Fourier space)
    def stream_function(self, w_h, real_space=False):
        #-Lap(psi) = w
        psi_h = self.inv_lap*w_h

        if real_space:
            return fft.irfft2(psi_h, s=(self.s1, self.s2))
        else:
            return psi_h

    #Compute velocity field from stream function (Fourier space)
    def velocity_field(self, stream_f, real_space=True):
        #Velocity field in x-direction = psi_y
        q_h = (2*math.pi/self.L2)*1j*self.k2*stream_f

        #Velocity field in y-direction = -psi_x
        v_h = -(2*math.pi/self.L1)*1j*self.k1*stream_f

        if real_space:
            return fft.irfft2(q_h, s=(self.s1, self.s2)), fft.irfft2(v_h, s=(self.s1, self.s2))
        else:
            return q_h, v_h

    #Compute non-linear term + forcing from given vorticity (Fourier space)
    def nonlinear_term(self, w_h, f_h=None):
        #Physical space vorticity
        w = fft.irfft2(w_h, s=(self.s1, self.s2))

        #Physical space velocity
        q, v = self.velocity_field(self.stream_function(w_h, real_space=False), real_space=True)

        #Compute non-linear term in Fourier space
        nonlin = -1j*((2*math.pi/self.L1)*self.k1*fft.rfft2(q*w) + (2*math.pi/self.L1)*self.k2*fft.rfft2(v*w))

        #Add forcing function
        if f_h is not None:
            nonlin += f_h

        return nonlin
    
    def time_step(self, q, v, f, Re):
        #Maxixum speed
        max_speed = torch.max(torch.sqrt(q**2 + v**2)).item()

        #Maximum force amplitude
        if f is not None:
            xi = torch.sqrt(torch.max(torch.abs(f))).item()
        else:
            xi = 1.0
        
        #Viscosity
        mu = (1.0/Re)*xi*((self.L1/(2*math.pi))**(3.0/4.0))*(((self.L2/(2*math.pi))**(3.0/4.0)))

        if max_speed == 0:
            return 0.5*(self.h**2)/mu
        
        #Time step based on CFL condition
        return min(0.5*self.h/max_speed, 0.5*(self.h**2)/mu)

    def advance(self, w, f=None, T=1.0, Re=100, adaptive=True, delta_t=1e-3):

        #Rescale Laplacian by Reynolds number
        GG = (1.0/Re)*self.G

        #Move to Fourier space
        w_h = fft.rfft2(w)

        if f is not None:
            f_h = fft.rfft2(f)
        else:
            f_h = None
        
        if adaptive:
            q, v = self.velocity_field(self.stream_function(w_h, real_space=False), real_space=True)
            delta_t = self.time_step(q, v, f, Re)

        time  = 0.0
        #Advance solution in Fourier space
        while time < T:
            if time + delta_t > T:
                current_delta_t = T - time
            else:
                current_delta_t = delta_t

            #Inner-step of Heun's method
            nonlin1 = self.nonlinear_term(w_h, f_h)
            w_h_tilde = (w_h + current_delta_t*(nonlin1 - 0.5*GG*w_h))/(1.0 + 0.5*current_delta_t*GG)

            #Cranck-Nicholson + Heun update
            nonlin2 = self.nonlinear_term(w_h_tilde, f_h)
            w_h = (w_h + current_delta_t*(0.5*(nonlin1 + nonlin2) - 0.5*GG*w_h))/(1.0 + 0.5*current_delta_t*GG)

            #De-alias
            w_h *= self.dealias

            #Update time
            time += current_delta_t

            #New time step
            if adaptive:
                q, v = self.velocity_field(self.stream_function(w_h, real_space=False), real_space=True)
                delta_t = self.time_step(q, v, f, Re)
        
        return fft.irfft2(w_h, s=(self.s1, self.s2))
    
    def __call__(self, w, f=None, T=1.0, Re=100, adaptive=True, delta_t=1e-3):
        return self.advance(w, f, T, Re, adaptive, delta_t)