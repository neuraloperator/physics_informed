import random
import numpy as np
import csv
from timeit import default_timer

import tensorflow as tf
import deepxde as dde
import tensordiffeq as tdq
from tensordiffeq.models import CollocationSolverND
from tensordiffeq.boundaries import DomainND, periodicBC

from .tqd_utils import PointsIC
from baselines.data import NSdata


Re = 500


def forcing(x):
    return - 4 * tf.math.cos(4 * x)


def bd_model(u_model, x, y, t):
    u = u_model(tf.concat([x, y, t], 1))
    u_vel, v_vel, w = u[:, 0:1], u[:, 1:2], u[:, 2:3]
    return u_vel, v_vel, w


def f_model(u_model, x, y, t):
    inp = tf.concat([x, y, t], 1)
    u = u_model(inp)
    u_vel, v_vel, w = u[:, 0:1], u[:, 1:2], u[:, 2:3]

    u_vel_x = tf.gradients(u_vel, x)[0]
    u_vel_xx = tf.gradients(u_vel_x, x)[0]
    u_vel_y = tf.gradients(u_vel, y)[0]
    u_vel_yy = tf.gradients(u_vel_y, y)[0]

    v_vel_y = tf.gradients(v_vel, y)[0]
    v_vel_x = tf.gradients(v_vel, x)[0]
    v_vel_xx = tf.gradients(v_vel_x, x)[0]
    v_vel_yy = tf.gradients(v_vel_y, y)[0]

    w_vor_x = tf.gradients(w, x)[0]
    w_vor_y = tf.gradients(w, y)[0]
    w_vor_t = tf.gradients(w, t)[0]

    w_vor_xx = tf.gradients(w_vor_x, x)[0]
    w_vor_yy = tf.gradients(w_vor_y, y)[0]

    c1 = tdq.utils.constant(1 / Re)
    eqn1 = w_vor_t + u_vel * w_vor_x + v_vel * w_vor_y - c1 * (w_vor_xx + w_vor_yy) - forcing(x)
    eqn2 = u_vel_x + v_vel_y
    eqn3 = u_vel_xx + u_vel_yy + w_vor_y
    eqn4 = v_vel_xx + v_vel_yy - w_vor_x
    return eqn1, eqn2, eqn3, eqn4


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


def train_sa(offset, config, args):
    seed = random.randint(1, 10000)
    print(f'Random seed :{seed}')
    np.random.seed(seed)
    # tf.config.run_functions_eagerly(True)
    # print(tf.executing_eagerly())
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
    model = CollocationSolverND()
    init_vals = dataset.get_init_cond()
    num_inits = config['train']['num_init']
    if num_inits > dataset.S ** 2:
        num_inits = dataset.S ** 2
    init_cond = PointsIC(domain, init_vals, var=['x', 'y'], n_values=num_inits)
    bd_cond = periodicBC(domain, ['x', 'y'], [bd_model], n_values=config['train']['num_boundary'])
    BCs = [init_cond, bd_cond]

    dict_adaptive = {'residual': [True, True, True, True],
                     'BCs': [True, False]}
    init_weights = {
        'residual': [tf.random.uniform([config['train']['num_domain'], 1]),
                     tf.random.uniform([config['train']['num_domain'], 1]),
                     tf.random.uniform([config['train']['num_domain'], 1]),
                     tf.random.uniform([config['train']['num_domain'], 1])],
        'BCs': [100 * tf.random.uniform([num_inits, 1]),
                100 * tf.ones([config['train']['num_boundary'], 1])]
    }

    model.compile(config['model']['layers'], f_model, domain, BCs,
                  isAdaptive=True, dict_adaptive=dict_adaptive, init_weights=init_weights)

    if 'log_step' in config['train']:
        step_size = config['train']['log_step']
    else:
        step_size = 100
    epochs = config['train']['epochs'] // step_size

    for i in range(epochs):
        time_start = default_timer()
        model.fit(tf_iter=step_size)
        time_end = default_timer()
        eval(model, dataset, i * step_size,
             time_cost=time_end - time_start,
             offset=offset,
             config=config)
    print('Done!')
