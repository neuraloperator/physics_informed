import math
import numpy as np
import os
from tqdm import tqdm

import torch
from solver.random_fields import GaussianRF, GaussianRF2d
from solver.kolmogorov_flow import KolmogorovFlow2d
from solver.periodic import NavierStokes2d
from timeit import default_timer
import argparse



def legacy_solver(args):
    save_dir = args.outdir
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda:0')
    s = 1024
    sub = s // args.res_x
    
    n = 4   # forcing
    Re = args.re

    T_in = 100.0
    T = args.T
    t = args.t_res
    dt = 1.0 / t

    GRF = GaussianRF(2, s, 2 * math.pi, alpha=2.5, tau=7, device=device)
    u0 = GRF.sample(1)

    NS = KolmogorovFlow2d(u0, Re, n)
    NS.advance(T_in, delta_t=1e-3)

    sol = np.zeros((T, t + 1, s // sub, s // sub))
    sol_ini = NS.vorticity().squeeze(0).cpu().numpy()[::sub, ::sub]
    pbar = tqdm(range(T))
    for i in pbar:
        sol[i, 0, :, :] = sol_ini
        for j in range(t):
            t1 = default_timer()
            NS.advance(dt, delta_t=1e-3)
            sol[i, j + 1, :, :] = NS.vorticity().squeeze(0).cpu().numpy()[::sub, ::sub]
            t2 = default_timer()
        pbar.set_description(
            (
                f'{i}, time cost: {t2-t1}'
            )
        )
        sol_ini = sol[i, -1, :, :]
    
    save_path = os.path.join(save_dir, f'NS-Re{int(Re)}_T{t}.npy')
    # np.save('NS_fine_Re500_S512_s64_T500_t128.npy', sol)
    np.save(save_path, sol)


def gen_data(args):
    dtype = torch.float64
    device = torch.device('cuda:0')
    save_dir = args.outdir
    os.makedirs(save_dir, exist_ok=True)
    
    T = args.T  # total time
    bsize = args.batchsize
    L = 2 * math.pi
    s =args.x_res
    x_sub = args.x_sub

    t_res = args.t_res
    dt = 1 / t_res
    re = args.re

    solver = NavierStokes2d(s,s,L,L,device=device,dtype=dtype)
    grf = GaussianRF2d(s,s,L,L,alpha=2.5,tau=3.0,sigma=None,device=device,dtype=dtype)

    t = torch.linspace(0, L, s+1, dtype=dtype, device=device)[0:-1]
    _, Y = torch.meshgrid(t, t, indexing='ij')
    f = -4*torch.cos(4.0*Y)
    vor = np.zeros((bsize, T, t_res + 1, s // x_sub, s // x_sub))

    pbar = tqdm(range(T))
    w = grf.sample(bsize)
    w = solver.advance(w, f, T=100, Re=re, adaptive=True)
    
    init_vor = w[:, ::x_sub, ::x_sub].cpu().type(torch.float32).numpy()
    for j in pbar:
        vor[:, j, 0, :, :] = init_vor

        for k in range(t_res):
            t1 = default_timer()

            w = solver.advance(w, f, T=dt, Re=re, adaptive=True)
            vor[:, j, k+1, :, :] = w[:,::x_sub,::x_sub].cpu().type(torch.float32).numpy()

            t2 = default_timer()

            pbar.set_description(
            (
                f'{j}, time cost: {t2-t1}'
            )
        )
        init_vor = vor[:, j, -1, :, :]

    for i in range(bsize):
        save_path = os.path.join(save_dir, f'NS-Re{int(re)}_T{T}_id{i}.npy')
        # np.save('NS_fine_Re500_S512_s64_T500_t128.npy', sol)
        np.save(save_path, vor[i])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--re', type=float, default=40.0)
    parser.add_argument('--x_res', type=int, default=512)
    parser.add_argument('--x_sub', type=int, default=2)
    parser.add_argument('--T', type=int, default=300)
    parser.add_argument('--outdir', type=str, default='../data')
    parser.add_argument('--t_res', type=int, default=512)
    parser.add_argument('--batchsize', type=int, default=1)
    parser.add_argument('--num_batchs', type=int, default=1)
    args = parser.parse_args()
    gen_data(args)