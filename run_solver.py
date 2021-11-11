import random
import math
import numpy as np
import yaml
from argparse import ArgumentParser
from timeit import default_timer

import torch

from train_utils.datasets import NSLoader
from train_utils.losses import LpLoss
from solver.kolmogorov_flow import KolmogorovFlow2d


def solve(a,
          res_x,
          res_t,
          end,
          Re,
          n=4,
          delta_t=1e-3):
    '''
    Given initial condition a, solve for u in time interval [0, end]
    Args:
        a: initial condition, res_x by res_x tensor
        res_x: resolution in space
        res_t: record step in time
        end: end of the time interval
        Re: Reynolds number
        n: forcing number
    Returns:
        tensor of shape (res_x, res_x, res_t)
    '''
    dt = end / res_t

    solver = KolmogorovFlow2d(a, Re, n)
    sol = torch.zeros((res_x, res_x, res_t + 1), device=a.device)
    sol[:, :, 0] = a
    for j in range(res_t):
        solver.advance(dt, delta_t=delta_t)
        sol[:, :, 1 + j] = solver.vorticity().squeeze(0)
    return sol


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    parser = ArgumentParser(description='Basic paser')
    parser.add_argument('--config_path', type=str, help='Path to the configuration file')
    parser.add_argument('--deltat', type=float, default=1e-3, help='delta T')
    args = parser.parse_args()
    config_file = args.config_path
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)

    data_config = config['data']
    loader = NSLoader(datapath1=data_config['datapath'],
                      nx=data_config['nx'], nt=data_config['nt'],
                      sub=data_config['sub'], sub_t=data_config['sub_t'],
                      N=data_config['total_num'],
                      t_interval=data_config['time_interval'])
    a_loader = loader.make_loader(data_config['n_sample'],
                                  batch_size=config['train']['batchsize'],
                                  start=data_config['offset'],
                                  train=data_config['shuffle'])

    print(f'Solver starts on device: {device}')
    myloss = LpLoss(size_average=True)
    test_err = []
    time_cost = []
    # run solver
    for _, u in a_loader:
        u = u[0].to(device)
        torch.cuda.synchronize()
        t1 = default_timer()
        pred = solve(u[:, :, 0],
                     res_x=loader.S,
                     res_t=loader.T - 1,
                     end=data_config['time_interval'],
                     Re=data_config['Re'],
                     n=4,
                     delta_t=args.deltat)
        torch.cuda.synchronize()
        t2 = default_timer()

        # report test error
        test_l2 = myloss(pred, u)

        test_err.append(test_l2.item())
        print(f'Test l2: {test_l2.item()}')
        time_cost.append(t2 - t1)

    test_err = np.array(test_err)
    time_cost = np.array(time_cost)

    idx = data_config['offset']
    n_sample = data_config['n_sample']
    print(f'Test instance: {idx} to {idx+n_sample}; \n'
          f'Time cost = mean: {time_cost.mean()}s; std_err: {time_cost.std(ddof=1) / math.sqrt(len(a_loader))}s; \n'
          f'Solver resolution: {loader.S} x {loader.S} x {loader.T}; \n'
          f'Test L2 error = mean: {test_err.mean()}; std_err: {test_err.std(ddof=1) / math.sqrt(len(a_loader))}')


