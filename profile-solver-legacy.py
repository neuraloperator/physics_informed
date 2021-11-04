import math

import torch
from solver.legacy_solver import navier_stokes_2d, GaussianRF

import scipy.io
from timeit import default_timer


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Resolution
    s = 2048
    sub = 1

    # Number of solutions to generate
    N = 1

    # Set up 2d GRF with covariance parameters
    GRF = GaussianRF(2, s, alpha=2.5, tau=7, device=device)

    # Forcing function: 0.1*(sin(2pi(x+y)) + cos(2pi(x+y)))
    t = torch.linspace(0, 1, s + 1, device=device)
    t = t[0:-1]

    X, Y = torch.meshgrid(t, t)
    f = 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y)))

    # Number of snapshots from solution
    record_steps = 200

    # Inputs
    a = torch.zeros(N, s, s)
    # Solutions
    u = torch.zeros(N, s, s, record_steps)

    # Solve equations in batches (order of magnitude speed-up)

    # Batch size
    bsize = 1

    c = 0
    t0 = default_timer()
    for j in range(N // bsize):
        # Sample random feilds
        w0 = GRF.sample(bsize)

        # Solve NS
        sol, sol_t = navier_stokes_2d(w0, f, 1e-3, 50.0, 1e-4, record_steps)

        a[c:(c + bsize), ...] = w0
        u[c:(c + bsize), ...] = sol

        c += bsize
        t1 = default_timer()
        print(f'Time cost {t1 - t0} s')
    torch.save(
        {
            'a': a.cpu(),
            'u': u.cpu(),
            't': sol_t.cpu()
        },
        'data/ns_data.pt'
    )
    # scipy.io.savemat('data/ns_data.mat', mdict={'a': a.cpu().numpy(), 'u': u.cpu().numpy(), 't': sol_t.cpu().numpy()})