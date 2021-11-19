'''
training for Navier Stokes with viscosity 0.001
spatial domain: (0, 1) ** 2
temporal domain: [0, 49]
'''
import csv
import random
from timeit import default_timer
import deepxde as dde
from deepxde.optimizers.config import set_LBFGS_options
import numpy as np
from baselines.data import NSLong

import tensorflow as tf


def forcing(x):
    theta = x[:, 0:1] + x[:, 1:2]
    return 0.1 * (tf.math.sin(2 * np.pi * theta) + tf.math.cos(2 * np.pi * theta))


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
           0.001 * (w_vor_xx + w_vor_yy) - forcing(x)
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

    pred = model.predict(test_points)
    vel_u_truth = test_vals[:, 0]
    vel_v_truth = test_vals[:, 1]
    vor_truth = test_vals[:, 2]

    vel_u_pred = pred[:, 0]
    vel_v_pred = pred[:, 1]
    vor_pred = pred[:, 2]

    u_err = dde.metrics.l2_relative_error(vel_u_truth, vel_u_pred)
    v_err = dde.metrics.l2_relative_error(vel_v_truth, vel_v_pred)
    vor_err = dde.metrics.l2_relative_error(vor_truth, vor_pred)

    total_num = test_vals.shape[0]
    u50 = test_vals[dataset.T - 1: total_num: dataset.T, 0]
    v50 = test_vals[dataset.T - 1: total_num: dataset.T, 1]
    vor50 = test_vals[dataset.T - 1: total_num: dataset.T, 2]

    u50_pred = pred[dataset.T - 1: total_num: dataset.T, 0]
    v50_pred = pred[dataset.T - 1: total_num: dataset.T, 1]
    vor50_pred = pred[dataset.T - 1: total_num: dataset.T, 2]

    u50_err = dde.metrics.l2_relative_error(u50, u50_pred)
    v50_err = dde.metrics.l2_relative_error(v50, v50_pred)
    vor50_err = dde.metrics.l2_relative_error(vor50, vor50_pred)

    print(f'Instance index : {offset}')
    print(f'L2 relative error in u: {u_err}')
    print(f'L2 relative error in v: {v_err}')
    print(f'L2 relative error in vorticity: {vor_err}')

    print(f'Time {dataset.T - 1} L2 relative error of u : {u50_err}')
    print(f'Time {dataset.T - 1} L2 relative error of v : {v50_err}')
    print(f'Time {dataset.T - 1} L2 relative error of vor : {vor50_err}')
    with open(config['log']['logfile'], 'a') as f:
        writer = csv.writer(f)
        writer.writerow([offset, u_err, v_err, vor_err, step, time_cost, u50_err, v50_err, vor50_err])


def train_longtime(offset, config, args):
    seed = random.randint(1, 10000)
    print(f'Random seed :{seed}')
    np.random.seed(seed)
    # construct dataloader
    data_config = config['data']
    spatial_domain = dde.geometry.Rectangle(xmin=[0, 0], xmax=[1, 1])
    temporal_domain = dde.geometry.TimeDomain(0, data_config['time_scale'])
    st_domain = dde.geometry.GeometryXTime(spatial_domain, temporal_domain)

    dataset = NSLong(datapath=data_config['datapath'],
                     nx=data_config['nx'], nt=data_config['nt'],
                     time_scale=data_config['time_scale'],
                     offset=offset, num=data_config['n_sample'],
                     vel=True)

    points = dataset.get_boundary_points(dataset.S, dataset.S, dataset.T)
    u_value = dataset.get_boundary_value(component=0)
    v_value = dataset.get_boundary_value(component=1)
    w_value = dataset.get_boundary_value(component=2)
    # u, v are velocity, w is vorticity
    boundary_u = dde.PointSetBC(points=points, values=u_value, component=0)
    boundary_v = dde.PointSetBC(points=points, values=v_value, component=1)
    boundary_w = dde.PointSetBC(points=points, values=w_value, component=2)

    data = dde.data.TimePDE(
        st_domain,
        pde,
        [
            boundary_u,
            boundary_v,
            boundary_w
        ],
        num_domain=config['train']['num_domain'],
        num_boundary=config['train']['num_boundary'],
        num_test=config['train']['num_test'],
    )

    net = dde.maps.FNN(config['model']['layers'],
                       config['model']['activation'],
                       'Glorot normal')
    # net = dde.maps.STMsFFN([3] + 4 * [50] + [3], 'tanh', 'Glorot normal', [50], [50])
    model = dde.Model(data, net)

    model.compile('adam', lr=config['train']['base_lr'], loss_weights=[1, 1, 1, 1, 100, 100, 100])
    if 'log_step' in config['train']:
        step_size = config['train']['log_step']
    else:
        step_size = 100
    epochs = config['train']['epochs'] // step_size
    for i in range(epochs):
        time_start = default_timer()
        model.train(epochs=step_size, display_every=step_size)
        time_end = default_timer()
        eval(model, dataset, i * step_size,
             time_cost=time_end - time_start,
             offset=offset,
             config=config)
    print('Done!')

