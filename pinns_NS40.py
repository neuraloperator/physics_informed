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
from tqdm import tqdm

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
    vorticity = - u_y + v_x
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
    res_x = u_t + u * u_x + v * u_y - 1 / re * (u_xx + u_yy) - torch.sin(4 * y)
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
    epoch_num = 500
    dataloader = DataLoader(dataset, batch_size=5000, shuffle=True, drop_last=True)

    model.train()
    criterion = LpLoss(size_average=True)
    mse = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    milestones = [60, 120, 180, 240, 300, 360]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.2)
    bd_x, bd_y, bd_t, bd_vor, u_gt, v_gt = dataset.get_boundary()
    bd_x, bd_y, bd_t, bd_vor, u_gt, v_gt = bd_x.to(device), bd_y.to(device), bd_t.to(device), \
                                           bd_vor.to(device), u_gt.to(device), v_gt.to(device)
    pbar = tqdm(range(epoch_num), dynamic_ncols=True, smoothing=0.01)

    set_grad([bd_x, bd_y, bd_t])
    for e in pbar:
        total_train_loss = 0.0
        train_error = 0.0
        f_error = 0.0
        model.train()
        for x, y, t, vor, true_u, true_v in dataloader:
            optimizer.zero_grad()
            u, v, pred_vor = net_NS(bd_x, bd_y, bd_t, model)
            loss_bc = mse(u, u_gt) + mse(v, v_gt)

            x, y, t, vor, true_u, true_v = x.to(device), y.to(device), t.to(device), \
                                           vor.to(device), true_u.to(device), true_v.to(device)
            set_grad([x, y, t])

            u, v, pred_vor = net_NS(x, y, t, model)
            velu_loss = mse(u, true_u)
            velv_loss = mse(v, true_v)
            res_x, res_y = resf_NS(u, v, x, y, t, re=40)
            loss_f = torch.mean(res_x ** 2) + torch.mean(res_y ** 2)
            total_loss = velu_loss + velv_loss + loss_f
            total_loss.backward()
            optimizer.step()

            total_train_loss += total_loss.item()
            train_error += loss_bc.item()
            f_error += loss_f.item()
        total_train_loss /= len(dataloader)

        train_error /= len(dataloader)
        f_error /= len(dataloader)

        u_error = 0.0
        v_error = 0.0
        test_error = 0.0
        model.eval()
        for x, y, t, vor, true_u, true_v in dataloader:
            x, y, t, vor, true_u, true_v = x.to(device), y.to(device), t.to(device), \
                                           vor.to(device), true_u.to(device), true_v.to(device)
            set_grad([x, y, t])
            u, v, pred_vor = net_NS(x, y, t, model)
            velu_loss = criterion(u, true_u)
            velv_loss = criterion(v, true_v)
            test_loss = criterion(pred_vor, vor)
            u_error += velu_loss.item()
            v_error += velv_loss.item()
            test_error += test_loss.item()

        u_error /= len(dataloader)
        v_error /= len(dataloader)
        test_error /= len(dataloader)
        pbar.set_description(
            (
                f'Train f error: {f_error:.5f}; Train l2 error: {train_error:.5f}. '
                f'Train loss: {total_train_loss:.5f}; Test l2 error: {test_error:.5f}'
            )
        )
        if wandb and log:
            wandb.log(
                {
                    'Train f error': f_error,
                    'Train L2 error': train_error,
                    'Test L2 error': test_error,
                    'Total loss': total_train_loss,
                    'u error': u_error,
                    'v error': v_error
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
                   tags=['batch sample'])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    datapath = 'data/NS_fine_Re40_s64_T1000.npy'
    dataset = NS40data(datapath, nx=64, nt=64, sub=1, sub_t=1, N=1000, index=1)
    layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 1]
    model = FCNet(layers).to(device)
    model = train_adam(model, dataset, device)
    save_checkpoint('checkpoints/pinns', name='NS40.pt', model=model)


