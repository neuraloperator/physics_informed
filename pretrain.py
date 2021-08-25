import yaml
from argparse import ArgumentParser
import math
from tqdm import tqdm
from timeit import default_timer

import torch
import torch.nn.functional as F

from solver.random_fields import GaussianRF
from train_utils.utils import save_checkpoint, convert_ic
from train_utils.losses import LpLoss, PINO_loss3d, get_forcing
from train_utils import Adam
from train_utils.datasets import NSLoader
from models import FNN3d

try:
    import wandb
except ImportError:
    wandb = None


def mixed_train(model,              # model of neural operator
                train_loader,       # dataloader for training with data
                S1, T1,             # spacial and time dimension for training with data
                a_loader,           # dataloader for training with equation loss only
                S2, T2,             # spacial and time dimension for training with equation only
                optimizer,          # optimizer
                scheduler,          # learning rate scheduler
                config,             # configuration dict
                device=torch.device('cpu'),
                log=False,          # turn on the wandb
                project='PINO-default', # project name
                group='FDM',        # group name
                tags=['Nan'],       # tags
                use_tqdm=True):     # turn on tqdm
    if wandb and log:
        run = wandb.init(project=project,
                         entity='hzzheng-pino',
                         group=group,
                         config=config,
                         tags=tags, reinit=True,
                         settings=wandb.Settings(start_method="fork"))

    # data parameters
    v = 1 / config['data']['Re']
    t_interval = config['data']['time_interval']
    forcing_1 = get_forcing(S1).to(device)
    forcing_2 = get_forcing(S2).to(device)
    # training settings
    batch_size = config['train']['batchsize']
    ic_weight = config['train']['ic_loss']
    f_weight = config['train']['f_loss']
    xy_weight = config['train']['xy_loss']

    model.train()
    myloss = LpLoss(size_average=True)
    pbar = range(config['train']['epochs'])
    if use_tqdm:
        pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.05)
    zero = torch.zeros(1).to(device)
    for ep in pbar:
        model.train()
        t1 = default_timer()
        train_loss = 0.0
        train_ic = 0.0
        train_f = 0.0
        test_l2 = 0.0
        err_eqn = 0.0
        for (x, y), new_a in zip(train_loader, a_loader):
            # Stage 1: train with data
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            x_in = F.pad(x, (0, 0, 0, 5), "constant", 0)
            out = model(x_in).reshape(batch_size, S1, S1, T1 + 5)
            out = out[..., :-5]
            x = x[:, :, :, 0, -1]

            loss_l2 = myloss(out.view(batch_size, S1, S1, T1),
                             y.view(batch_size, S1, S1, T1))

            if ic_weight != 0 or f_weight != 0:
                loss_ic, loss_f = PINO_loss3d(out.view(batch_size, S1, S1, T1),
                                              x, forcing_1,
                                              v, t_interval)
            else:
                loss_ic, loss_f = zero, zero

            total_loss = loss_l2 * xy_weight + loss_f * f_weight + loss_ic * ic_weight

            total_loss.backward()
            optimizer.step()

            train_ic = loss_ic.item()
            test_l2 += loss_l2.item()
            train_loss += total_loss.item()
            train_f += loss_f.item()

            # Stage 2: train with equation loss only
            new_a = new_a[0].to(device)
            optimizer.zero_grad()
            x_in = F.pad(new_a, (0, 0, 0, 5), "constant", 0)
            out = model(x_in).reshape(batch_size, S2, S2, T2 + 5)
            out = out[..., :-5]
            new_a = new_a[:, :, :, 0, -1]
            loss_ic, loss_f = PINO_loss3d(out.view(batch_size, S2, S2, T2),
                                          new_a, forcing_2,
                                          v, t_interval)
            eqn_loss = loss_f * f_weight + loss_ic * ic_weight
            eqn_loss.backward()
            optimizer.step()

            err_eqn += eqn_loss.item()

        scheduler.step()

        train_ic /= len(train_loader)
        train_f /= len(train_loader)
        train_loss /= len(train_loader)
        test_l2 /= len(train_loader)
        err_eqn /= len(a_loader)
        t2 = default_timer()
        if use_tqdm:
            pbar.set_description(
                (
                    f'Train f error: {train_f:.5f}; Train ic l2 error: {train_ic:.5f}. '
                    f'Train loss: {train_loss:.5f}; Test l2 error: {test_l2:.5f}'
                    f'Eqn loss: {err_eqn:.5f}'
                )
            )
        if wandb and log:
            wandb.log(
                {
                    'Train f error': train_f,
                    'Train L2 error': train_ic,
                    'Train loss': train_loss,
                    'Test L2 error': test_l2,
                    'Train equation loss': err_eqn,
                    'Time cost': t2 - t1
                }
            )

    save_checkpoint(config['train']['save_dir'],
                    config['train']['save_name'],
                    model, optimizer)
    if wandb and log:
        run.finish()


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # parse options
    parser = ArgumentParser(description='Basic paser')
    parser.add_argument('--config_path', type=str, help='Path to the configuration file')
    parser.add_argument('--log', action='store_true', help='Turn on the wandb')
    parser.add_argument('--new', action='store_true', help='Use new data loader')
    options = parser.parse_args()

    config_file = options.config_path
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_config = config['data']

    # prepare dataloader for training with data
    if options.new:
        datapath2 = data_config['datapath'].replace('part0', 'part1')
        loader = NSLoader(datapath1=data_config['datapath'], datapath2=datapath2,
                          nx=data_config['nx'], nt=data_config['nt'],
                          sub=data_config['sub'], sub_t=data_config['sub_t'],
                          N=data_config['total_num'],
                          t_interval=data_config['time_interval'])
    else:
        loader = NSLoader(datapath1=data_config['datapath'],
                          nx=data_config['nx'], nt=data_config['nt'],
                          sub=data_config['sub'], sub_t=data_config['sub_t'],
                          N=data_config['total_num'],
                          t_interval=data_config['time_interval'])

    train_loader = loader.make_loader(data_config['n_sample'],
                                      batch_size=config['train']['batchsize'],
                                      start=data_config['offset'],
                                      train=data_config['shuffle'])
    # prepare dataloader for training with only equations
    gr_sampler = GaussianRF(2, data_config['S2'], 2 * math.pi, alpha=2.5, tau=7, device=device)
    u0 = gr_sampler.sample(data_config['n_sample']).to('cpu')
    a_data = convert_ic(u0,
                        data_config['n_sample'],
                        data_config['S2'],
                        data_config['T2'])
    a_dataset = torch.utils.data.TensorDataset(a_data)
    a_loader = torch.utils.data.DataLoader(a_dataset,
                                           batch_size=config['train']['batchsize'],
                                           shuffle=True)
    # create model
    model = FNN3d(modes1=config['model']['modes1'],
                  modes2=config['model']['modes2'],
                  modes3=config['model']['modes3'],
                  fc_dim=config['model']['fc_dim'],
                  layers=config['model']['layers']).to(device)
    # Load from checkpoint
    if 'ckpt' in config['train']:
        ckpt_path = config['train']['ckpt']
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        print('Weights loaded from %s' % ckpt_path)
    # create optimizer and learning rate scheduler
    optimizer = Adam(model.parameters(), betas=(0.9, 0.999),
                     lr=config['train']['base_lr'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=config['train']['milestones'],
                                                     gamma=config['train']['scheduler_gamma'])
    mixed_train(model,
                train_loader,
                loader.S, loader.T,
                a_loader,
                data_config['S2'], data_config['T2'],
                optimizer,
                scheduler,
                config,
                device,
                log=options.log,
                project=config['others']['project'],
                group=config['others']['group'])