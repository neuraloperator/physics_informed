import numpy as np
import torch
from tqdm import tqdm
from .utils import save_checkpoint
from .losses import LpLoss, darcy_loss

try:
    import wandb
except ImportError:
    wandb = None


def train_2d_operator(model,
                      train_loader,
                      optimizer, scheduler,
                      config,
                      rank=0, log=False,
                      project='PINO-2d-default',
                      group='default',
                      tags=['default'],
                      use_tqdm=True,
                      profile=False):
    if rank == 0 and wandb and log:
        run = wandb.init(project=project,
                         entity='hzzheng-pino',
                         group=group,
                         config=config,
                         tags=tags, reinit=True,
                         settings=wandb.Settings(start_method="fork"))

    data_weight = config['train']['xy_loss']
    f_weight = config['train']['f_loss']
    model.train()
    myloss = LpLoss(size_average=True)
    pbar = range(config['train']['epochs'])
    if use_tqdm:
        pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.1)
    mesh = train_loader.dataset.mesh
    mollifier = torch.sin(np.pi * mesh[..., 0]) * torch.sin(np.pi * mesh[..., 1]) * 0.001
    mollifier = mollifier.to(rank)
    for e in pbar:
        loss_dict = {'train_loss': 0.0,
                     'data_loss': 0.0,
                     'f_loss': 0.0,
                     'test_error': 0.0}
        for x, y in train_loader:
            x, y = x.to(rank), y.to(rank)

            optimizer.zero_grad()

            pred = model(x).reshape(y.shape)
            pred = pred * mollifier

            data_loss = myloss(pred, y)

            a = x[..., 0]
            f_loss = darcy_loss(pred, a)

            loss = data_weight * data_loss + f_weight * f_loss
            loss.backward()
            optimizer.step()

            loss_dict['train_loss'] += loss.item() * y.shape[0]
            loss_dict['f_loss'] += f_loss.item() * y.shape[0]
            loss_dict['data_loss'] += data_loss.item() * y.shape[0]

        scheduler.step()
        train_loss_val = loss_dict['train_loss'] / len(train_loader.dataset)
        f_loss_val = loss_dict['f_loss'] / len(train_loader.dataset)
        data_loss_val = loss_dict['data_loss'] / len(train_loader.dataset)

        if use_tqdm:
            pbar.set_description(
                (
                    f'Epoch: {e}, train loss: {train_loss_val:.5f}, '
                    f'f_loss: {f_loss_val:.5f}, '
                    f'data loss: {data_loss_val:.5f}'
                )
            )
        if wandb and log:
            wandb.log(
                {
                    'train loss': train_loss_val,
                    'f loss': f_loss_val,
                    'data loss': data_loss_val
                }
            )
    save_checkpoint(config['train']['save_dir'],
                    config['train']['save_name'],
                    model, optimizer)
    if wandb and log:
        run.finish()
    print('Done!')