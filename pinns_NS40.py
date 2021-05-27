import torch
import torch.nn as nn

from torch.optim import Adam
from train_utils.data_utils import NS40data
from train_utils.utils import set_grad, save_checkpoint
from train_utils.losses import LpLoss
from train_utils.baseline_utils import net_NS, vel2vor
from train_utils.baseline_loss import resf_NS, boundary_loss

from models import FCNet
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None

torch.manual_seed(2022)


def train_adam(model, dataset, device):
    alpha = 100
    beta = 100
    epoch_num = 3000
    dataloader = DataLoader(dataset, batch_size=5000, shuffle=True, drop_last=True)

    model.train()
    criterion = LpLoss(size_average=True)
    mse = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=0.0005)
    milestones = [100, 500, 1500, 2000]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.9)
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
            u, v, _ = net_NS(bd_x, bd_y, bd_t, model)
            loss_ic = mse(u, u_gt.view(-1)) + mse(v, v_gt.view(-1))
            #  boundary condition
            loss_bc = boundary_loss(model, 100)

            # collocation points
            x, y, t, vor, true_u, true_v = x.to(device), y.to(device), t.to(device), \
                                           vor.to(device), true_u.to(device), true_v.to(device)
            set_grad([x, y, t])
            u, v, p = net_NS(x, y, t, model)
            # velu_loss = criterion(u, true_u)
            # velv_loss = criterion(v, true_v)
            res_x, res_y, evp3 = resf_NS(u, v, p, x, y, t, re=40)
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
            u, v, _ = net_NS(x, y, t, model)
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
                   group='with pressure',
                   tags=['4x50'])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    datapath = 'data/NS_fine_Re40_s64_T1000.npy'
    dataset = NS40data(datapath, nx=64, nt=64, sub=1, sub_t=1, N=1000, index=1)
    layers = [3, 50, 50, 50, 50, 3]
    model = FCNet(layers).to(device)
    model = train_adam(model, dataset, device)
    save_checkpoint('checkpoints/pinns', name='NS40.pt', model=model)


