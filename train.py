import numpy as np
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.autograd as autograd

from torch.optim import Adam, SGD, LBFGS
from torch.utils.data import DataLoader

from models import FNN1d, FNN2d
from utils import PDELoss, zero_grad


def predict(model, x, t, nu):
    '''
    Params:
        - model: model
        - xt: (N, 2) tensor
    Return: 
        - u: (N, 1) tensor
        - residual: (N, 1) tensor
    '''
    model.eval()

    x.requires_grad = True
    t.requires_grad = True

    u = model(torch.cat([x, t], dim=1))

    grad_x, grad_t = autograd.grad(outputs=u.sum(), inputs=[x, t],
                                   create_graph=True)
    gradgrad_x, = autograd.grad(outputs=grad_x.sum(), inputs=[x])

    residual = grad_t + u * grad_x - nu * gradgrad_x
    return u.detach(), residual.detach()


def train(model, X_u, u, X_f,
          nu=1.0, num_epoch=100,
          device=torch.device('cpu'), optim='LBFGS'):
    model.to(device)
    model.train()
    optimizer = LBFGS(model.parameters(),
                      lr=1.0,
                      max_iter=50000,
                      max_eval=50000,
                      history_size=50,
                      tolerance_grad=1e-5,
                      tolerance_change=1.0 * np.finfo(float).eps,
                      line_search_fn="strong_wolfe")
    mse = nn.MSELoss()
    # training stage
    xts = torch.from_numpy(X_u).float().to(device)
    us = torch.from_numpy(u).float().to(device)

    xs = torch.from_numpy(X_f[:, 0:1]).float().to(device)
    ts = torch.from_numpy(X_f[:, 1:2]).float().to(device)
    xs.requires_grad = True
    ts.requires_grad = True
    iter = 0

    def loss_closure():
        nonlocal iter
        iter = iter + 1

        optimizer.zero_grad()

        zero_grad(xs)
        zero_grad(ts)
        # print(xs.grad)
        # MSE loss of prediction error
        pred_u = model(xts)
        mse_u = mse(pred_u, us)

        # MSE loss of PDE constraint
        f = PDELoss(model, xs, ts, nu)

        mse_f = torch.mean(f ** 2)
        loss = mse_u + mse_f
        loss.backward()

        if iter % 200 == 0:
            print('Iter: {}, total loss: {}, mse_u: {}, mse_f: {}'.
                  format(iter, loss.item(), mse_u.item(), mse_f.item()))
        return loss

    optimizer.step(loss_closure)

    return model


if __name__ == '__main__':

    # # net = FNN1d(modes=16, width=64)
    net = FNN2d(modes1=16, modes2=16, width=64)
    # # net2 = FNNd(modes1=16, modes2=16, width=64)
    data = torch.randn((1, 100, 256, 2))
    pred = net(data)
    print(pred.shape)
