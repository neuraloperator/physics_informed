import torch
import torch.autograd as autograd


def PDELoss(model, x, t, nu):
    '''
    Compute the residual of PDE: 
        residual = u_t + u * u_x - nu * u_{xx} : (N,1)

    Params: 
        - model 
        - x, t: (x, t) pairs, (N, 2) tensor
        - nu: constant of PDE
    Return: 
        - mean of residual : scalar 
    '''
    u = model(torch.cat([x, t], dim=1))
    # First backward to compute u_x (shape: N x 1), u_t (shape: N x 1)
    grad_x, grad_t = autograd.grad(outputs=[u.sum()], inputs=[
                                   x, t], create_graph=True)
    # grad_x = grad_xt[:, 0]
    # grad_t = grad_xt[:, 1]

    # Second backward to compute u_{xx} (shape N x 1)

    gradgrad_x, = autograd.grad(
        outputs=[grad_x.sum()], inputs=[x], create_graph=True)
    # gradgrad_x = gradgrad[:, 0]

    residual = grad_t + u * grad_x - nu * gradgrad_x
    return residual


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def zero_grad(params):
    '''
    set grad field to 0
    '''
    if isinstance(params, torch.Tensor):
        if params.grad is not None:
            params.grad.detach()
            params.grad.zero_()
    else:
        for p in params:
            if p.grad is not None:
                p.grad.detach()
                p.grad.zero_()
