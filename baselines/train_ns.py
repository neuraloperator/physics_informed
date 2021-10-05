from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from baselines.model import DeepONet, DeepONetCP
from baselines.data import DeepOnetNS, DeepONetCPNS
from train_utils.losses import LpLoss
from train_utils.utils import save_checkpoint
from train_utils.data_utils import sample_data


def train_deeponet_cp(config):
    '''
    Train Cartesian product DeepONet
    Args:
        config:

    Returns:
    '''
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_config = config['data']
    batch_size = config['train']['batchsize']
    dataset = DeepONetCPNS(datapath=data_config['datapath'],
                           nx=data_config['nx'], nt=data_config['nt'],
                           sub=data_config['sub'], sub_t=data_config['sub_t'],
                           offset=data_config['offset'], num=data_config['n_sample'],
                           t_interval=data_config['time_interval'])
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    u0_dim = dataset.S ** 2
    model = DeepONetCP(branch_layer=[u0_dim] + config['model']['branch_layers'],
                       trunk_layer=[3] + config['model']['trunk_layers']).to(device)
    optimizer = Adam(model.parameters(), lr=config['train']['base_lr'])
    scheduler = MultiStepLR(optimizer, milestones=config['train']['milestones'],
                            gamma=config['train']['scheduler_gamma'])
    pbar = range(config['train']['epochs'])
    pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.1)
    myloss = LpLoss(size_average=True)
    model.train()

    for e in pbar:
        train_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)  # initial condition, (batchsize, u0_dim)
            grid = dataset.xyt
            grid = grid.to(device)  # grid value, (SxSxT, 3)
            y = y.to(device)  # ground truth, (batchsize, SxSxT)

            pred = model(x, grid)
            loss = myloss(pred, y)

            model.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * y.shape[0]
        train_loss /= len(dataset)
        scheduler.step()

        pbar.set_description(
            (
                f'Epoch: {e}; Averaged train loss: {train_loss:.5f}; '
            )
        )
        if e % 500 == 0:
            print(f'Epoch: {e}, averaged train loss: {train_loss:.5f}')
            save_checkpoint(config['train']['save_dir'],
                            config['train']['save_name'].replace('.pt', f'_{e}.pt'),
                            model, optimizer)


def train_deeponet(config):
    '''
    train plain DeepOnet
    Args:
        config:

    Returns:

    '''
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_config = config['data']
    dataset = DeepOnetNS(datapath=data_config['datapath'],
                         nx=data_config['nx'], nt=data_config['nt'],
                         sub=data_config['sub'], sub_t=data_config['sub_t'],
                         offset=data_config['offset'], num=data_config['n_sample'],
                         t_interval=data_config['time_interval'])
    train_loader = DataLoader(dataset, batch_size=config['train']['batchsize'], shuffle=False)

    u0_dim = dataset.S ** 2
    model = DeepONet(branch_layer=[u0_dim] + config['model']['branch_layers'],
                     trunk_layer=[3] + config['model']['trunk_layers']).to(device)
    optimizer = Adam(model.parameters(), lr=config['train']['base_lr'])
    scheduler = MultiStepLR(optimizer, milestones=config['train']['milestones'],
                            gamma=config['train']['scheduler_gamma'])

    pbar = range(config['train']['epochs'])
    pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.1)
    myloss = LpLoss(size_average=True)
    model.train()
    loader = sample_data(train_loader)
    for e in pbar:
        u0, x, y = next(loader)
        u0 = u0.to(device)
        x = x.to(device)
        y = y.to(device)
        pred = model(u0, x)
        loss = myloss(pred, y)
        model.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        pbar.set_description(
            (
                f'Epoch: {e}; Train loss: {loss.item():.5f}; '
            )
        )
    save_checkpoint(config['train']['save_dir'],
                    config['train']['save_name'],
                    model, optimizer)
    print('Done!')
