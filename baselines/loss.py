import torch
import torch.autograd as autograd
from train_utils.utils import set_grad
from .utils import get_sample, net_NS, sub_mse


def boundary_loss(model, npt=100):
    device = next(model.parameters()).device

    bc1_x_sample, bc1_y_sample, bc1_t_sample, bc2_x_sample, bc2_y_sample, bc2_t_sample \
        = get_sample(npt)

    bc1_x_sample, bc1_y_sample, bc1_t_sample, bc2_x_sample, bc2_y_sample, bc2_t_sample \
        = bc1_x_sample.to(device), bc1_y_sample.to(device), bc1_t_sample.to(device), \
          bc2_x_sample.to(device), bc2_y_sample.to(device), bc2_t_sample.to(device)
    set_grad([bc1_x_sample, bc1_y_sample, bc1_t_sample, bc2_x_sample, bc2_y_sample, bc2_t_sample])

    u1, v1, _ = net_NS(bc1_x_sample, bc1_y_sample, bc1_t_sample, model)
    u2, v2, _ = net_NS(bc2_x_sample, bc2_y_sample, bc2_t_sample, model)
    bc_loss = sub_mse(u1) + sub_mse(v1) + sub_mse(u2) + sub_mse(v2)
    return 0.5 * bc_loss  # 0.5 is the normalization factor


def resf_NS(u, v, p, x, y, t, re=40):
    '''
    Args:
        u: x-component, tensor
        v: y-component, tensor
        x: x-dimension, tensor
        y: y-dimension, tensor
        t: time dimension, tensor
    Returns:
        Residual f error
    '''
    u_x, u_y, u_t = autograd.grad(outputs=[u.sum()], inputs=[x, y, t], create_graph=True)
    v_x, v_y, v_t = autograd.grad(outputs=[v.sum()], inputs=[x, y, t], create_graph=True)
    u_xx, = autograd.grad(outputs=[u_x.sum()], inputs=[x], create_graph=True)
    u_yy, = autograd.grad(outputs=[u_y.sum()], inputs=[y], create_graph=True)
    v_xx, = autograd.grad(outputs=[v_x.sum()], inputs=[x], create_graph=True)
    v_yy, = autograd.grad(outputs=[v_y.sum()], inputs=[y], create_graph=True)
    p_x, = autograd.grad(outputs=[p.sum()], inputs=[x], create_graph=True)
    p_y, = autograd.grad(outputs=[p.sum()], inputs=[y], create_graph=True)
    res_x = u_t + u * u_x + v * u_y + p_x - 1 / re * (u_xx + u_yy) - torch.sin(4 * y)
    res_y = v_t + u * v_x + v * v_y + p_y - 1 / re * (v_xx + v_yy)
    evp3 = u_x + v_y
    return res_x, res_y, evp3

