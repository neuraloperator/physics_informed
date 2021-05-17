import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.autograd as autograd

from torch.optim import Adam, SGD, LBFGS
from data_utils import NS40data
from utils import set_grad, save_checkpoint
from losses import LpLoss
from models import FCNet
from torch.utils.data import DataLoader

try:
    import wandb
except ImportError:
    wandb = None

torch.manual_seed(2022)


def net_NS(x, y, t, model):
    psi = model(torch.cat([x, y, t], dim=1))
    u, v = autograd.grad(outputs=[psi.sum()], inputs=[y, x], create_graph=True)
    v = - v
    u_y, = autograd.grad(outputs=[u.sum()], inputs=[y], create_graph=True)
    v_x, = autograd.grad(outputs=[v.sum()], inputs=[x], create_graph=True)
    vorticity = u_y - v_x
    return u, v, vorticity


def resf_NS(u, v, x, y, t, re=40):
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
    u_xx,  = autograd.grad(outputs=[u_x.sum()], inputs=[x], create_graph=True)
    u_yy, = autograd.grad(outputs=[u_y.sum()], inputs=[y], create_graph=True)
    v_xx, = autograd.grad(outputs=[v_x.sum()], inputs=[x], create_graph=True)
    v_yy, = autograd.grad(outputs=[v_y.sum()], inputs=[y], create_graph=True)
    res_x = u_t + u * u_x + v * u_y - 1 / re * (u_xx + u_yy) - torch.sin(y)
    res_y = v_t + u * v_x + v * v_y - 1 / re * (v_xx + v_yy)
    return res_x, res_y


def train(model, dataset, device):
    model.train()
    criterion = LpLoss(size_average=True)
    optimizer = LBFGS(model.parameters(),
                      lr=1.0,
                      max_iter=50000,
                      max_eval=50000,
                      history_size=50,
                      tolerance_change=1.0 * np.finfo(float).eps,
                      line_search_fn="strong_wolfe")
    bd_x, bd_y, bd_t, bd_vor = dataset.get_boundary()
    bd_x, bd_y, bd_t, bd_vor = bd_x.to(device), bd_y.to(device), bd_t.to(device), bd_vor.to(device)
    x, y, t, vor = dataset.sample_xyt(sub=2)
    x, y, t, vor = x.to(device), y.to(device), t.to(device), vor.to(device)
    set_grad([bd_x, bd_y, bd_t, x, y, t])
    iter = 0
    def loss_closure():
        nonlocal iter
        iter += 1
        optimizer.zero_grad()
        u, v, pred_vor = net_NS(bd_x, bd_y, bd_t, model)
        loss_bc = criterion(pred_vor, bd_vor)

        u, v, pred_vor = net_NS(x, y, t, model)
        res_x, res_y = resf_NS(u, v, x, y, t, re=40)
        loss_f = torch.mean(res_x ** 2) + torch.mean(res_y ** 2)
        test_error = criterion(pred_vor, vor)
        total_loss = loss_bc + loss_f
        total_loss.backward()
        if wandb and log:
            wandb.log(
                {
                    'Train f error': loss_f.item(),
                    'Train L2 error': loss_bc.item(),
                    'Test L2 error': test_error.item()
                }
            )
        return total_loss

    optimizer.step(loss_closure)
    return model


def train_adam(model, dataset, device):
    epoch_num = 8000
    dataloader = DataLoader(dataset, batch_size=5000, shuffle=True, drop_last=True)

    model.train()
    criterion = LpLoss(size_average=True)
    optimizer = Adam(model.parameters(), lr=0.001)
    milestones = [500, 1500, 2500, 3500, 4500, 5500]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)
    bd_x, bd_y, bd_t, bd_vor = dataset.get_boundary()
    bd_x, bd_y, bd_t, bd_vor = bd_x.to(device), bd_y.to(device), bd_t.to(device), bd_vor.to(device)

    set_grad([bd_x, bd_y, bd_t])
    for e in range(epoch_num):
        for x, y, t, vor in dataloader:
            optimizer.zero_grad()
            u, v, pred_vor = net_NS(bd_x, bd_y, bd_t, model)
            loss_bc = criterion(pred_vor, bd_vor)

            x, y, t, vor = x.to(device), y.to(device), t.to(device), vor.to(device)
            set_grad([x, y, t])

            u, v, pred_vor = net_NS(x, y, t, model)
            res_x, res_y = resf_NS(u, v, x, y, t, re=40)
            loss_f = torch.mean(res_x ** 2) + torch.mean(res_y ** 2)
            test_error = criterion(pred_vor, vor)
            total_loss = loss_bc
            total_loss.backward()
            optimizer.step()
            if wandb and log:
                wandb.log(
                    {
                        'Train f error': loss_f.item(),
                        'Train L2 error': loss_bc.item(),
                        'Test L2 error': test_error.item(),
                        'Total loss': total_loss.item()
                    }
                )
        scheduler.step()
    return model


if __name__ == '__main__':
    log = True
    if wandb and log:
        wandb.init(project='PINO-NS40-pinns',
                   entity='hzzheng-pino',
                   group='AD',
                   tags=['last instance'])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    datapath = 'data/NS_fine_Re40_s64_T1000.npy'
    dataset = NS40data(datapath, nx=64, nt=64, sub=1, sub_t=1, N=1000, index=1)
    layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 1]
    model = FCNet(layers).to(device)
    model = train_adam(model, dataset, device)
    save_checkpoint('checkpoints/pinns', name='NS40.pt', model=model)


