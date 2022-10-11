import math
import numpy as np
import os
from tqdm import tqdm

import torch
from solver.random_fields import GaussianRF
from solver.kolmogorov_flow import KolmogorovFlow2d
from timeit import default_timer
import argparse



def legacy_solver(args):
    save_dir = args.outdir
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda:0')
    s = 1024
    sub = args.sub_x
    
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
    
    save_path = os.path.join(save_dir, f'NS-Re{Re}_T{t}.npy')
    # np.save('NS_fine_Re500_S512_s64_T500_t128.npy', sol)
    np.save(save_path, sol)


def gen_data(args):
    save_dir = args.outdir
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda:0')
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--re', type=float, default=40.0)
    parser.add_argument('--sub_x', type=int, default=4)
    parser.add_argument('--T', type=int, default=2000)
    parser.add_argument('--outdir', type=str, default='data')
    parser.add_argument('--t_res', type=int, default=256)
    args = parser.parse_args()
    gen_data(args)