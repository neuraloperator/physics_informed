from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from baselines.model import DeepONetCP
from train_utils.losses import LpLoss
from train_utils.utils import save_checkpoint
from baselines.data import DarcyFlow


def train_deeponet_darcy(config):
    '''
    train deepONet for darcy flow
    '''
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_config = config['data']
    batch_size = config['train']['batchsize']
    dataset = DarcyFlow(data_config['datapath'],
                        nx=data_config['nx'], sub=data_config['sub'],
                        offset=data_config['offset'], num=data_config['n_sample'])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    u0_dim = dataset.S ** 2
    model = DeepONetCP(branch_layer=[u0_dim] + config['model']['branch_layers'],
                       trunk_layer=[2] + config['model']['trunk_layers']).to(device)
    optimizer = Adam(model.parameters(), lr=config['train']['base_lr'])
    scheduler = MultiStepLR(optimizer, milestones=config['train']['milestones'],
                            gamma=config['train']['scheduler_gamma'])
    pbar = range(config['train']['epochs'])
    pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.1)
    myloss = LpLoss(size_average=True)
    model.train()
    grid = dataset.mesh
    grid = grid.reshape(-1, 2).to(device)  # grid value, (SxS, 2)
    for e in pbar:
        train_loss = 0.0
        for x, y in dataloader:
            x = x.to(device)  # initial condition, (batchsize, u0_dim)

            y = y.to(device)  # ground truth, (batchsize, SxS)

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
        if e % 1000 == 0:
            print(f'Epoch: {e}, averaged train loss: {train_loss:.5f}')
            save_checkpoint(config['train']['save_dir'],
                            config['train']['save_name'].replace('.pt', f'_{e}.pt'),
                            model, optimizer)
    save_checkpoint(config['train']['save_dir'],
                    config['train']['save_name'],
                    model, optimizer)