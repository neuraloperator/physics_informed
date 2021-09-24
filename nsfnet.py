import random
from argparse import ArgumentParser

import yaml
import tensorflow as tf
import deepxde as dde
import numpy as np
from baselines.data import NSdata

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


if __name__ == '__main__':
    spatial_domain = dde.geometry.Rectangle(xmin=[0, 0], xmax=[2 * np.pi, 2 * np.pi])
    temporal_domain = dde.geometry.TimeDomain(0, 0.5)
    st_domain = dde.geometry.GeometryXTime(spatial_domain, temporal_domain)

    seed = random.randint(1, 10000)
    print(f'Random seed :{seed}')

    parser = ArgumentParser(description='Basic paser')
    parser.add_argument('--config_path', type=str, help='Path to the configuration file')
    parser.add_argument('--log', action='store_true', help='Turn on the wandb')
    parser.add_argument('--num_gpus', type=int, help='Number of GPUs', default=1)
    parser.add_argument('--start', type=int, default=-1, help='start index')
    args = parser.parse_args()

    config_file = args.config_path
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)

    # construct dataloader
    data_config = config['data']
    if 'datapath2' in data_config:
        dataset = NSdata(datapath1=data_config['datapath'],
                         datapath2=data_config['datapath2'],
                         nx=data_config['nx'], nt=data_config['nt'],
                         sub=data_config['sub'], sub_t=data_config['sub_t'],
                         N=data_config['total_num'],
                         t_interval=data_config['time_interval'])
    else:
        dataset = NSdata(datapath1=data_config['datapath'],
                          nx=data_config['nx'], nt=data_config['nt'],
                          sub=data_config['sub'], sub_t=data_config['sub_t'],
                          N=data_config['total_num'],
                          t_interval=data_config['time_interval'])
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
        num_domain=10000,
        num_boundary=40000,
        num_test=1000,
    )

    net = dde.maps.FNN([3] + 6 * [50] + [3], 'tanh', 'Glorot normal')
    model = dde.Model(data, net)

    model.compile('adam', lr=1e-3, loss_weights=[1, 1, 1, 1, 100, 100, 100])
    model.train(epochs=15000)
    model.compile('L-BFGS', loss_weights=[1, 1, 1, 1, 100, 100, 100])
    model.train()

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

    print(f'L2 relative error in u: {u_err}')
    print(f'L2 relative error in v: {v_err}')
    print(f'L2 relative error in vorticity: {vor_err}')


