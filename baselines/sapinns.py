import csv
import random
from timeit import default_timer
from tqdm import tqdm
import deepxde as dde
import numpy as np
from baselines.data import NSdata
import torch
from torch.optim import Adam

from tensordiffeq.boundaries import DomainND, periodicBC
from .tqd_utils import PointsIC
from .model import SAWeight

from models.FCN import DenseNet
from train_utils.negadam import NAdam


Re = 500


def forcing(x):
    return - 4 * torch.cos(4 * x[:, 1:2])


def pde(x, u):
    '''
    Args:
        x: (x, y, t)
        u: (u, v, w), where (u,v) is the velocity, w is the vorticity
    Returns: list of pde loss

    '''
    u_vel, v_vel, w = u[:, 0:1], u[:, 1:2], u[:, 2:3]

    u_vel_x = dde.grad.jacobian(u, x, i=0, j=0)
    u_vel_xx = dde.grad.hessian(u, x, component=0, i=0, j=0)
    u_vel_yy = dde.grad.hessian(u, x, component=0, i=1, j=1)

    v_vel_y = dde.grad.jacobian(u, x, i=1, j=1)
    v_vel_xx = dde.grad.hessian(u, x, component=1, i=0, j=0)
    v_vel_yy = dde.grad.hessian(u, x, component=1, i=1, j=1)

    w_vor_x = dde.grad.jacobian(u, x, i=2, j=0)
    w_vor_y = dde.grad.jacobian(u, x, i=2, j=1)
    w_vor_t = dde.grad.jacobian(u, x, i=2, j=2)

    w_vor_xx = dde.grad.hessian(u, x, component=2, i=0, j=0)
    w_vor_yy = dde.grad.hessian(u, x, component=2, i=1, j=1)

    eqn1 = w_vor_t + u_vel * w_vor_x + v_vel * w_vor_y - \
           1 / Re * (w_vor_xx + w_vor_yy) - forcing(x)
    eqn2 = u_vel_x + v_vel_y
    eqn3 = u_vel_xx + u_vel_yy + w_vor_y
    eqn4 = v_vel_xx + v_vel_yy - w_vor_x
    return [eqn1, eqn2, eqn3, eqn4]


def eval(model, dataset,
         step, time_cost,
         offset, config):
    '''
    evaluate test error for the model over dataset
    '''
    test_points, test_vals = dataset.get_test_xyt()

    test_points = torch.tensor(test_points, dtype=torch.float32)
    with torch.no_grad():
        pred = model(test_points).cpu().numpy()
    vel_u_truth = test_vals[:, 0]
    vel_v_truth = test_vals[:, 1]
    vor_truth = test_vals[:, 2]

    vel_u_pred = pred[:, 0]
    vel_v_pred = pred[:, 1]
    vor_pred = pred[:, 2]

    u_err = dde.metrics.l2_relative_error(vel_u_truth, vel_u_pred)
    v_err = dde.metrics.l2_relative_error(vel_v_truth, vel_v_pred)
    vor_err = dde.metrics.l2_relative_error(vor_truth, vor_pred)
    print(f'Instance index : {offset}')
    print(f'L2 relative error in u: {u_err}')
    print(f'L2 relative error in v: {v_err}')
    print(f'L2 relative error in vorticity: {vor_err}')
    with open(config['log']['logfile'], 'a') as f:
        writer = csv.writer(f)
        writer.writerow([offset, u_err, v_err, vor_err, step, time_cost])


def train_sapinn(offset, config, args):
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
    num_collo = config['train']['num_domain']
    domain.generate_collocation_points(num_collo)
    init_vals = dataset.get_init_cond()
    num_inits = config['train']['num_init']
    if num_inits > dataset.S ** 2:
        num_inits = dataset.S ** 2
    init_cond = PointsIC(domain, init_vals, var=['x', 'y'], n_values=num_inits)
    bd_cond = periodicBC(domain, ['x', 'y'], n_values=config['train']['num_boundary'])

    # prepare initial condition inputs
    init_input = torch.tensor(init_cond.input, dtype=torch.float32)
    init_val = torch.tensor(init_cond.val, dtype=torch.float32)

    # prepare boundary condition inputs
    upper_input0 = torch.tensor(bd_cond.upper[0], dtype=torch.float32).squeeze().t()     # shape N x 3
    upper_input1 = torch.tensor(bd_cond.upper[1], dtype=torch.float32).squeeze().t()
    lower_input0 = torch.tensor(bd_cond.lower[0], dtype=torch.float32).squeeze().t()
    lower_input1 = torch.tensor(bd_cond.lower[1], dtype=torch.float32).squeeze().t()

    # prepare collocation points
    collo_input = torch.tensor(domain.X_f, dtype=torch.float32, requires_grad=True)

    weight_net = SAWeight(out_dim=3,
                          num_init=[num_inits],
                          num_bd=[upper_input0.shape[0]] * 2,
                          num_collo=[num_collo] * 4)
    net = DenseNet(config['model']['layers'], config['model']['activation'])
    weight_optim = NAdam(weight_net.parameters(), lr=config['train']['base_lr'])
    net_optim = Adam(net.parameters(), lr=config['train']['base_lr'])

    pbar = tqdm(range(config['train']['epochs']), dynamic_ncols=True)

    start_time = default_timer()
    for e in pbar:
        net.zero_grad()
        weight_net.zero_grad()
        if collo_input.grad is not None:
            collo_input.grad.zero_()

        init_pred = net(init_input) - init_val

        bd_0 = net(upper_input0) - net(lower_input0)
        bd_1 = net(upper_input1) - net(lower_input1)

        predu = net(collo_input)
        pde_residual = pde(collo_input, predu)

        loss = weight_net(init_cond=[init_pred], bd_cond=[bd_0, bd_1], residual=pde_residual)
        loss.backward()
        weight_optim.step()
        net_optim.step()
        dde.gradients.clear()
        pbar.set_description(
            (
                f'Epoch: {e}, loss: {loss.item()}'
            )
        )

        if e % config['train']['log_step'] == 0:
            end_time = default_timer()
            eval(net, dataset, e, time_cost=end_time - start_time, offset=offset, config=config)
            start_time = default_timer()
    print('Done!')





    
