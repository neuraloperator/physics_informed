import torch
import torch.autograd as autograd


def vel2vor(u, v, x, y):
    u_y, = autograd.grad(outputs=[u.sum()], inputs=[y], create_graph=True)
    v_x, = autograd.grad(outputs=[v.sum()], inputs=[x], create_graph=True)
    vorticity = - u_y + v_x
    return vorticity


def net_NS(x, y, t, model):
    out = model(torch.cat([x, y, t], dim=1))
    u = out[:, 0]
    v = out[:, 1]
    p = out[:, 2]
    return u, v, p


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