import os
import numpy as np

import torch

from models import FNN2d

from tqdm import tqdm
from timeit import default_timer
from train_utils.utils import count_params, save_checkpoint
from train_utils.datasets import BurgersLoader, sample_data
from train_utils.losses import LpLoss, PINO_loss

try:
    import wandb
except ImportError:
    wandb = None

torch.manual_seed(0)
np.random.seed(0)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# datapath = '/mnt/md1/zongyi/burgers_pino.mat'
datapath = 'data/burgers_pino.mat'
sub = 1
sub_t = 1
constructor = BurgersLoader(datapath, nx=128, nt=100, sub=sub, sub_t=sub_t, new=True)


def train():
    batch_size = 20
    learning_rate = 0.001

    epochs = 2500
    step_size = 100
    gamma = 0.25

    modes = 12  # 20
    width = 32  # 64

    config_defaults = {
        'ntrain': 800,
        'nlabels': 10,
        'ntest': 200,
        'lr': learning_rate,
        'batch_size': batch_size,
        'modes': modes,
        'width': width
    }
    wandb.init(config=config_defaults, tags=['Epoch'])
    config = wandb.config
    print('config: ', config)

    ntrain = config.ntrain
    nlabels = config.nlabels
    ntest = config.ntest


    image_dir = 'figs/FDM-burgers'
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    ckpt_dir = 'Burgers-FDM'
    name = 'PINO_FDM_burgers_N' + \
           str(ntrain) + '_L' + str(nlabels) + '-' + str(ntest) + '.pt'

    train_loader = constructor.make_loader(n_sample=ntrain, batch_size=batch_size, train=True)
    test_loader = constructor.make_loader(n_sample=ntest, batch_size=batch_size, train=False)
    if config.nlabels > 0:
        supervised_loader = constructor.make_loader(n_sample=nlabels,
                                                    batch_size=batch_size,
                                                    start=ntrain,
                                                    train=True)
        supervised_loader = sample_data(loader=supervised_loader)
    else:
        supervised_loader = None
    layers = [width * 2 // 4, width * 3 // 4, width * 3 // 4, width * 4 // 4, width * 4 // 4]
    modes = [modes * (5 - i) // 4 for i in range(4)]

    model = FNN2d(modes1=modes, modes2=modes, width=width, layers=layers).to(device)
    num_param = count_params(model)
    print('Number of model parameters', num_param)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500, 1000, 2000], gamma=gamma)


    myloss = LpLoss(size_average=True)
    pbar = tqdm(range(epochs), dynamic_ncols=True, smoothing=0.01)

    for ep in pbar:
        model.train()
        t1 = default_timer()
        train_pino = 0.0
        train_l2 = 0.0
        train_loss = 0.0
        dp_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
            loss_u, loss_f = PINO_loss(out, x[:, 0, :, 0])
            total_loss = loss_u * 10 + loss_f

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            train_l2 += loss.item()
            train_pino += loss_f.item()
            train_loss += total_loss.item()

        for x, y in supervised_loader:
            x, y = x.to(device), y.to(device)

            out = model(x)
            datapoint_loss = myloss(out.view(batch_size, -1), y.view(batch_size, -1))

            optimizer.zero_grad()
            datapoint_loss.backward()
            optimizer.step()

            dp_loss += datapoint_loss.item()
        scheduler.step()

        model.eval()
        test_l2 = 0.0
        test_pino = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)

                out = model(x)

                test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()
                test_u, test_f = PINO_loss(out, x[:, 0, :, 0])
                test_pino = test_f.item()

        train_l2 /= len(train_loader)
        test_l2 /= len(test_loader)
        train_pino /= len(train_loader)
        test_pino /= len(test_loader)
        train_loss /= len(train_loader)
        dp_loss /= len(supervised_loader)

        t2 = default_timer()
        pbar.set_description(
            (
                f'Time cost: {t2 - t1:.2f}; Train f error: {train_pino:.5f}; Train l2 error: {train_l2:.5f}. '
                f'Test f error: {test_pino:.5f}; Test l2 error: {test_l2:.5f}'
            )
        )
        if wandb:
            wandb.log(
                {
                    'Train f error': train_pino,
                    'Train L2 error': train_l2,
                    'Train DP error': dp_loss,
                    'Train loss': train_loss,
                    'Test f error': test_pino,
                    'Test L2 error': test_l2,
                    'Time cost': t2 - t1
                }
            )

    save_checkpoint(ckpt_dir, name, model, optimizer)


if __name__ == '__main__':
    sweep_config = {
        'method': 'grid',  # grid, random
        'metric': {
            'name': 'Test L2 error',
            'goal': 'minimize'
        },
        'parameters': {
            'nlabels': {
                'values': [20, 40, 60, 80, 100, 150, 200]
            }
        }
    }
    sweep_id = wandb.sweep(sweep_config, entity="hzzheng-pino", project="PINO-burgers-sweep")
    wandb.agent(sweep_id, train)
