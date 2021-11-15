import numpy as np

import torch
import torch.autograd as autograd


def weighted_mse(pred, target, weight=None):
    if weight is None:
        return torch.mean((pred - target) ** 2)
    else:
        return torch.mean(weight * (pred - target) ** 2)


def get_3dboundary_points(num_x,                # number of points on x axis
                          num_y,                # number of points on y axis
                          num_t,                # number of points on t axis
                          bot=(0, 0, 0),        # lower bound
                          top=(1.0, 1.0, 1.0)   # upper bound
                          ):
    x_top, y_top, t_top = top
    x_bot, y_bot, t_bot = bot

    x_arr = np.linspace(x_bot, x_top, num=num_x, endpoint=False)
    y_arr = np.linspace(y_bot, y_top, num=num_y, endpoint=False)
    xx, yy = np.meshgrid(x_arr, y_arr, indexing='ij')
    xarr = np.ravel(xx)
    yarr = np.ravel(yy)
    tarr = np.ones_like(xarr) * t_bot
    point0 = np.stack([xarr, yarr, tarr], axis=0).T  # (SxSx1, 3), boundary on t=0

    t_arr = np.linspace(t_bot, t_top, num=num_t)
    yy, tt = np.meshgrid(y_arr, t_arr, indexing='ij')
    yarr = np.ravel(yy)
    tarr = np.ravel(tt)
    xarr = np.ones_like(yarr) * x_bot
    point2 = np.stack([xarr, yarr, tarr], axis=0).T  # (1xSxT, 3), boundary on x=0

    xarr = np.ones_like(yarr) * x_top
    point3 = np.stack([xarr, yarr, tarr], axis=0).T  # (1xSxT, 3), boundary on x=2pi

    xx, tt = np.meshgrid(x_arr, t_arr, indexing='ij')
    xarr = np.ravel(xx)
    tarr = np.ravel(tt)
    yarr = np.ones_like(xarr) * y_bot
    point4 = np.stack([xarr, yarr, tarr], axis=0).T  # (128x1x65, 3), boundary on y=0

    yarr = np.ones_like(xarr) * y_top
    point5 = np.stack([xarr, yarr, tarr], axis=0).T  # (128x1x65, 3), boundary on y=2pi

    points = np.concatenate([point0,
                             point2, point3,
                             point4, point5],
                            axis=0)
    return points


def get_3dboundary(value):
    boundary0 = value[0, :, :, 0:1]  # 128x128x1, boundary on t=0
    # boundary1 = value[0, :, :, -1:]     # 128x128x1, boundary on t=0.5
    boundary2 = value[0, 0:1, :, :]  # 1x128x65, boundary on x=0
    boundary3 = value[0, -1:, :, :]  # 1x128x65, boundary on x=1
    boundary4 = value[0, :, 0:1, :]  # 128x1x65, boundary on y=0
    boundary5 = value[0, :, -1:, :]  # 128x1x65, boundary on y=1

    part0 = np.ravel(boundary0)
    # part1 = np.ravel(boundary1)
    part2 = np.ravel(boundary2)
    part3 = np.ravel(boundary3)
    part4 = np.ravel(boundary4)
    part5 = np.ravel(boundary5)
    boundary = np.concatenate([part0,
                               part2, part3,
                               part4, part5],
                              axis=0)[:, np.newaxis]
    return boundary


def get_xytgrid(S, T, bot=[0, 0, 0], top=[1, 1, 1]):
    '''
    Args:
        S: number of points on each spatial domain
        T: number of points on temporal domain including endpoint
        bot: list or tuple, lower bound on each dimension
        top: list or tuple, upper bound on each dimension

    Returns:
        (S * S * T, 3) array
    '''
    x_arr = np.linspace(bot[0], top[0], num=S, endpoint=False)
    y_arr = np.linspace(bot[1], top[1], num=S, endpoint=False)
    t_arr = np.linspace(bot[2], top[2], num=T)
    xgrid, ygrid, tgrid = np.meshgrid(x_arr, y_arr, t_arr, indexing='ij')
    xaxis = np.ravel(xgrid)
    yaxis = np.ravel(ygrid)
    taxis = np.ravel(tgrid)
    points = np.stack([xaxis, yaxis, taxis], axis=0).T
    return points


def get_2dgird(num=31):
    x = np.linspace(-1, 1, num)
    y = np.linspace(-1, 1, num)
    gridx, gridy = np.meshgrid(x, y)
    xs = gridx.reshape(-1, 1)
    ys = gridy.reshape(-1, 1)
    result = np.hstack((xs, ys))
    return result


def get_3dgrid(num=11):
    x = np.linspace(-1, 1, num)
    y = np.linspace(-1, 1, num)
    z = np.linspace(-1, 1, num)
    gridx, gridy, gridz = np.meshgrid(x, y, z)
    xs = gridx.reshape(-1, 1)
    ys = gridy.reshape(-1, 1)
    zs = gridz.reshape(-1, 1)
    return np.hstack((xs, ys, zs))


def get_4dgrid(num=11):
    '''
    4-D meshgrid
    Args:
        num: number of collocation points of each dimension

    Returns:
        (num**4, 4) tensor
    '''
    t = np.linspace(0, 1, num)
    x = np.linspace(-1, 1, num)
    y = np.linspace(-1, 1, num)
    z = np.linspace(-1, 1, num)
    gridx, gridy, gridz, gridt = np.meshgrid(x, y, z, t)
    xs = gridx.reshape(-1, 1)
    ys = gridy.reshape(-1, 1)
    zs = gridz.reshape(-1, 1)
    ts = gridt.reshape(-1, 1)
    result = np.hstack((xs, ys, zs, ts))
    return result


def vel2vor(u, v, x, y):
    u_y, = autograd.grad(outputs=[u.sum()], inputs=[y], create_graph=True)
    v_x, = autograd.grad(outputs=[v.sum()], inputs=[x], create_graph=True)
    vorticity = - u_y + v_x
    return vorticity


def sub_mse(vec):
    '''
    Compute mse of two parts of a vector
    Args:
        vec:

    Returns:

    '''
    length = vec.shape[0] // 2
    diff = (vec[:length] - vec[length: 2 * length]) ** 2
    return diff.mean()


def get_sample(npt=100):
    num = npt // 2
    bc1_y_sample = torch.rand(size=(num, 1)).repeat(2, 1)
    bc1_t_sample = torch.rand(size=(num, 1)).repeat(2, 1)

    bc1_x_sample = torch.cat([torch.zeros(num, 1), torch.ones(num, 1)], dim=0)

    bc2_x_sample = torch.rand(size=(num, 1)).repeat(2, 1)
    bc2_t_sample = torch.rand(size=(num, 1)).repeat(2, 1)

    bc2_y_sample = torch.cat([torch.zeros(num, 1), torch.ones(num, 1)], dim=0)
    return bc1_x_sample, bc1_y_sample, bc1_t_sample, \
           bc2_x_sample, bc2_y_sample, bc2_t_sample


def concat(xy, z, t=0.0, offset=0):
    '''
    Args:
        xy: (N, 2)
        z: (N, 1）
        t: (N, 1)
        offset: start index of xy
    Returns:
        (N, 4) array
    '''
    output = np.zeros((z.shape[0], 4)) * t
    if offset < 2:
        output[:, offset: offset+2] = xy
        output[:, (offset+2) % 3: (offset+2) % 3 + 1] = z
    else:
        output[:, 2:] = xy[:, 0:1]
        output[:, 0:1] = xy[:, 1:]
        output[:, 1:2] = z
    return output


def cal_mixgrad(outputs, inputs):
    out_grad, = autograd.grad(outputs=[outputs.sum()], inputs=[inputs], create_graph=True)
    out_x2, = autograd.grad(outputs=[out_grad[:, 0].sum()], inputs=[inputs], create_graph=True)
    out_xx = out_x2[:, 0]
    out_y2, = autograd.grad(outputs=[out_grad[:, 1].sum()], inputs=[inputs], create_graph=True)
    out_yy = out_y2[:, 1]
    out_z2, = autograd.grad(outputs=[out_grad[:, 2].sum()], inputs=[inputs], create_graph=True)
    out_zz = out_z2[:, 2]
    return out_grad, out_xx, out_yy, out_zz