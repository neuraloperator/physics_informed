'''
training for Navier Stokes with Reynolds number 500, 0.5 second time period
'''
import csv
import random
from timeit import default_timer
import deepxde as dde
from deepxde.optimizers.config import set_LBFGS_options
import numpy as np
from baselines.data import NSdata

import tensorflow as tf

Re = 500


def forcing(x):
    return - 4 * tf.math.cos(4 * x[:, 1:2])


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
    print(f'Instance index : {offset}')
    print(f'L2 relative error in u: {u_err}')
    print(f'L2 relative error in v: {v_err}')
    print(f'L2 relative error in vorticity: {vor_err}')
    with open(config['log']['logfile'], 'a') as f:
        writer = csv.writer(f)
        writer.writerow([offset, u_err, v_err, vor_err, step, time_cost])


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
    spatial_domain = dde.geometry.Rectangle(xmin=[0, 0], xmax=[2 * np.pi, 2 * np.pi])
    temporal_domain = dde.geometry.TimeDomain(0, data_config['time_interval'])
    st_domain = dde.geometry.GeometryXTime(spatial_domain, temporal_domain)
    num_boundary_points = dataset.S
    points = dataset.get_boundary_points(num_x=num_boundary_points, num_y=num_boundary_points,
                                         num_t=dataset.T)
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
    # set_LBFGS_options(maxiter=10000)
    # model.compile('L-BFGS', loss_weights=[1, 1, 1, 1, 100, 100, 100])
    # model.train()

    # test_points, test_vals = dataset.get_test_xyt()
    #
    # pred = model.predict(test_points)
    # vel_u_truth = test_vals[:, 0]
    # vel_v_truth = test_vals[:, 1]
    # vor_truth = test_vals[:, 2]
    #
    # vel_u_pred = pred[:, 0]
    # vel_v_pred = pred[:, 1]
    # vor_pred = pred[:, 2]
    #
    # u_err = dde.metrics.l2_relative_error(vel_u_truth, vel_u_pred)
    # v_err = dde.metrics.l2_relative_error(vel_v_truth, vel_v_pred)
    # vor_err = dde.metrics.l2_relative_error(vor_truth, vor_pred)
    # print(f'Instance index : {offset}')
    # print(f'L2 relative error in u: {u_err}')
    # print(f'L2 relative error in v: {v_err}')
    # print(f'L2 relative error in vorticity: {vor_err}')
    # with open(args.logfile, 'a') as f:
    #     writer = csv.writer(f)
    #     writer.writerow([offset, u_err, v_err, vor_err])
