import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd

from torch.optim import Adam, LBFGS
from train_utils.data_utils import NS40data
from train_utils.utils import set_grad, save_checkpoint
from train_utils.losses import LpLoss
from models import FCNet
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None

torch.manual_seed(2022)


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


def boundary_loss(model, npt=100):
    device = next(model.parameters()).device

    bc1_x_sample, bc1_y_sample, bc1_t_sample, bc2_x_sample, bc2_y_sample, bc2_t_sample \
        = get_sample(npt)

    bc1_x_sample, bc1_y_sample, bc1_t_sample, bc2_x_sample, bc2_y_sample, bc2_t_sample \
        = bc1_x_sample.to(device), bc1_y_sample.to(device), bc1_t_sample.to(device), \
          bc2_x_sample.to(device), bc2_y_sample.to(device), bc2_t_sample.to(device)
    set_grad([bc1_x_sample, bc1_y_sample, bc1_t_sample, bc2_x_sample, bc2_y_sample, bc2_t_sample])

    u1, v1 = net_NS(bc1_x_sample, bc1_y_sample, bc1_t_sample, model)
    u2, v2 = net_NS(bc2_x_sample, bc2_y_sample, bc2_t_sample, model)
    bc_loss = sub_mse(u1) + sub_mse(v1) + sub_mse(u2) + sub_mse(v2)
    return 0.5 * bc_loss  # 0.5 is the normalization factor


def net_NS(x, y, t, model):
    out = model(torch.cat([x, y, t], dim=1))
    u = out[:, 0]
    v = out[:, 1]
    return u, v


def vel2vor(u, v, x, y):
    u_y, = autograd.grad(outputs=[u.sum()], inputs=[y], create_graph=True)
    v_x, = autograd.grad(outputs=[v.sum()], inputs=[x], create_graph=True)
    vorticity = - u_y + v_x
    return vorticity


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
    evp3 = u_x + v_y
    return res_x, res_y, evp3


def train(model, dataset, device):
    # TODO: update code for new forward function and network
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
        u, v = net_NS(bd_x, bd_y, bd_t, model)
        pred_vor = vel2vor(u, v, bd_x, bd_y)
        loss_bc = criterion(pred_vor, bd_vor)

        u, v = net_NS(x, y, t, model)
        res_x, res_y, evp3 = resf_NS(u, v, x, y, t, re=40)
        loss_f = torch.mean(res_x ** 2) + torch.mean(res_y ** 2)
        pred_vor = vel2vor(u, v, x, y)
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
    alpha = 1
    beta = 1
    epoch_num = 3000
    dataloader = DataLoader(dataset, batch_size=5000, shuffle=True, drop_last=True)

    model.train()
    criterion = LpLoss(size_average=True)
    mse = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=0.002)
    milestones = [100, 500, 1500, 2000]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)
    bd_x, bd_y, bd_t, bd_vor, u_gt, v_gt = dataset.get_boundary()
    bd_x, bd_y, bd_t, bd_vor, u_gt, v_gt = bd_x.to(device), bd_y.to(device), bd_t.to(device), \
                                           bd_vor.to(device), u_gt.to(device), v_gt.to(device)
    pbar = tqdm(range(epoch_num), dynamic_ncols=True, smoothing=0.01)

    set_grad([bd_x, bd_y, bd_t])
    for e in pbar:
        total_train_loss = 0.0
        bc_error = 0.0
        ic_error = 0.0
        f_error = 0.0
        model.train()
        for x, y, t, vor, true_u, true_v in dataloader:
            optimizer.zero_grad()
            # initial condition
            u, v = net_NS(bd_x, bd_y, bd_t, model)
            loss_ic = mse(u, u_gt.view(-1)) + mse(v, v_gt.view(-1))
            #  boundary condition
            loss_bc = boundary_loss(model, 100)

            # collocation points
            x, y, t, vor, true_u, true_v = x.to(device), y.to(device), t.to(device), \
                                           vor.to(device), true_u.to(device), true_v.to(device)
            set_grad([x, y, t])
            u, v = net_NS(x, y, t, model)
            # velu_loss = criterion(u, true_u)
            # velv_loss = criterion(v, true_v)
            res_x, res_y, evp3 = resf_NS(u, v, x, y, t, re=40)
            loss_f = mse(res_x, torch.zeros_like(res_x)) \
                     + mse(res_y, torch.zeros_like(res_y)) \
                     + mse(evp3, torch.zeros_like(evp3))

            total_loss = loss_f + loss_bc * alpha + loss_ic * beta
            total_loss.backward()
            optimizer.step()

            total_train_loss += total_loss.item()
            bc_error += loss_bc.item()
            ic_error += loss_ic.item()
            f_error += loss_f.item()
        total_train_loss /= len(dataloader)

        ic_error /= len(dataloader)
        f_error /= len(dataloader)

        u_error = 0.0
        v_error = 0.0
        test_error = 0.0
        model.eval()
        for x, y, t, vor, true_u, true_v in dataloader:
            x, y, t, vor, true_u, true_v = x.to(device), y.to(device), t.to(device), \
                                           vor.to(device), true_u.to(device), true_v.to(device)
            set_grad([x, y, t])
            u, v = net_NS(x, y, t, model)
            pred_vor = vel2vor(u, v, x, y)
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
                f'Train f error: {f_error:.5f}; Train IC error: {ic_error:.5f}. '
                f'Train loss: {total_train_loss:.5f}; Test l2 error: {test_error:.5f}'
            )
        )
        if wandb and log:
            wandb.log(
                {
                    'Train f error': f_error,
                    'Train IC error': ic_error,
                    'Train BC error': bc_error,
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
        wandb.init(project='PINO-NS40-NSFnet',
                   entity='hzzheng-pino',
                   group='1-1',
                   tags=['5x100'])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    datapath = 'data/NS_fine_Re40_s64_T1000.npy'
    dataset = NS40data(datapath, nx=64, nt=64, sub=1, sub_t=1, N=1000, index=1)
    layers = [3, 50, 50, 50, 50, 2]
    model = FCNet(layers).to(device)
    model = train_adam(model, dataset, device)
    save_checkpoint('checkpoints/pinns', name='NS40.pt', model=model)


