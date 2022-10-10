import torch
import math


class KolmogorovFlow2d(object):

    def __init__(self, w0, Re, n):

        # Grid size

        self.s = w0.size()[-1]

        assert self.s == w0.size()[-2], "Grid must be uniform in both directions."

        assert math.log2(self.s).is_integer(), "Grid size must be power of 2."

        assert n >= 0 and isinstance(n, int), "Forcing number must be non-negative integer."

        assert n < self.s // 2 - 1, "Forcing number too large for grid size."

        # Forcing number
        self.n = n

        assert Re > 0, "Reynolds number must be positive."

        # Reynolds number
        self.Re = Re

        # Device
        self.device = w0.device

        # Current time
        self.time = 0.0

        # Current vorticity in Fourier space
        self.w_h = torch.fft.fft2(w0, norm="backward")

        # Wavenumbers in y and x directions
        self.k_y = torch.cat((torch.arange(start=0, end=self.s // 2, step=1, dtype=torch.float32, device=self.device), \
                              torch.arange(start=-self.s // 2, end=0, step=1, dtype=torch.float32, device=self.device)),
                             0).repeat(self.s, 1)

        self.k_x = self.k_y.clone().transpose(0, 1)

        # Negative inverse Laplacian in Fourier space
        self.inv_lap = (self.k_x ** 2 + self.k_y ** 2)
        self.inv_lap[0, 0] = 1.0
        self.inv_lap = 1.0 / self.inv_lap

        # Negative scaled Laplacian
        self.G = (1.0 / self.Re) * (self.k_x ** 2 + self.k_y ** 2)

        # Dealiasing mask using 2/3 rule
        self.dealias = (self.k_x ** 2 + self.k_y ** 2 <= (self.s / 3.0) ** 2).float()
        # Ensure mean zero
        self.dealias[0, 0] = 0.0

    # Get current vorticity from stream function (Fourier space)
    def vorticity(self, stream_f=None, real_space=True):
        if stream_f is not None:
            w_h = self.Re * self.G * stream_f
        else:
            w_h = self.w_h

        if real_space:
            return torch.fft.irfft2(w_h, s=(self.s, self.s), norm="backward")
        else:
            return w_h

    # Compute stream function from vorticity (Fourier space)
    def stream_function(self, w_h=None, real_space=False):
        if w_h is None:
            psi_h = self.w_h.clone()
        else:
            psi_h = w_h.clone()

        # Stream function in Fourier space: solve Poisson equation
        psi_h = self.inv_lap * psi_h

        if real_space:
            return torch.fft.irfft2(psi_h, s=(self.s, self.s), norm="backward")
        else:
            return psi_h

    # Compute velocity field from stream function (Fourier space)
    def velocity_field(self, stream_f=None, real_space=True):
        if stream_f is None:
            stream_f = self.stream_function(real_space=False)

        # Velocity field in x-direction = psi_y
        q_h = stream_f * 1j * self.k_y

        # Velocity field in y-direction = -psi_x
        v_h = stream_f * -1j * self.k_x

        if real_space:
            q = torch.fft.irfft2(q_h, s=(self.s, self.s), norm="backward")
            v = torch.fft.irfft2(v_h, s=(self.s, self.s), norm="backward")
            return q, v
        else:
            return q_h, v_h

    # Compute non-linear term + forcing from given vorticity (Fourier space)
    def nonlinear_term(self, w_h):
        # Physical space vorticity
        w = torch.fft.ifft2(w_h, s=(self.s, self.s), norm="backward")

        # Velocity field in physical space
        q, v = self.velocity_field(self.stream_function(w_h, real_space=False), real_space=True)

        # Compute non-linear term
        t1 = torch.fft.fft2(q * w, s=(self.s, self.s), norm="backward")
        t1 = self.k_x * t1

        t2 = torch.fft.fft2(v * w, s=(self.s, self.s), norm="backward")
        t2 = self.k_y * t2

        nonlin = -1j * (t1 + t2)

        # Apply forcing: -ncos(ny)
        if self.n > 0:
            nonlin[..., 0, self.n] -= (float(self.n) / 2.0) * (self.s ** 2)
            nonlin[..., 0, -self.n] -= (float(self.n) / 2.0) * (self.s ** 2)

        return nonlin

    def advance(self, t, delta_t=1e-3):

        # Final time
        T = self.time + t

        # Advance solution in Fourier space
        while self.time < T:

            if self.time + delta_t > T:
                current_delta_t = T - self.time
            else:
                current_delta_t = delta_t

            # Inner-step of Heun's method
            nonlin1 = self.nonlinear_term(self.w_h)
            w_h_tilde = (self.w_h + current_delta_t * (nonlin1 - 0.5 * self.G * self.w_h)) / (
                        1.0 + 0.5 * current_delta_t * self.G)

            # Cranck-Nicholson + Heun update
            nonlin2 = self.nonlinear_term(w_h_tilde)
            self.w_h = (self.w_h + current_delta_t * (0.5 * (nonlin1 + nonlin2) - 0.5 * self.G * self.w_h)) / (
                        1.0 + 0.5 * current_delta_t * self.G)

            # De-alias
            self.w_h *= self.dealias
            self.time += current_delta_t



