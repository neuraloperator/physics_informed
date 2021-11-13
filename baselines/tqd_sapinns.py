import random
import numpy as np


import tensordiffeq as tdq
from tensordiffeq.models import CollocationSolverND
from tensordiffeq.boundaries import DomainND

from baselines.data import NSdata


def train(offset, config, args):
    seed = random.randint(1, 10000)
    print(f'Random seed :{seed}')
    np.random.seed(seed)
    # construct dataloader
    data_config = config['data']
    if 'datapath2' in data_config:
        dataset = NSdata(datapath1=data_config['datapath'],
                         datapath2=data_config['datapath2'],
                         offset=offset, num=1,
                         nx=data_config['nx'], nt=data_config['nt'],
                         sub=data_config['sub'], sub_t=data_config['sub_t'],
                         vel=True,
                         t_interval=data_config['time_interval'])
    else:
        dataset = NSdata(datapath1=data_config['datapath'],
                         offset=offset, num=1,
                         nx=data_config['nx'], nt=data_config['nt'],
                         sub=data_config['sub'], sub_t=data_config['sub_t'],
                         vel=True,
                         t_interval=data_config['time_interval'])
    domain = DomainND(['x', 'y', 't'], time_var='t')
    domain.add('x', [0.0, 2 * np.pi], dataset.S)
    domain.add('y', [0.0, 2 * np.pi], dataset.S)
    domain.add('t', [0.0, data_config['time_interval']], dataset.T)
    domain.generate_collocation_points(config['train']['num_domain'])
