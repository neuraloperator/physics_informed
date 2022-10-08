import os
from unittest import loader
import yaml
import random
from argparse import ArgumentParser
import math
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


from models import FNN3d
from train_utils.adam import Adam

from train_utils.losses import LpLoss, PINO_loss3d, get_forcing
from train_utils.datasets import NS3DDataset
from train_utils.utils import save_ckpt

try:
    import wandb
except ImportError:
    wandb = None


def pad_input(x, num_pad):
    if num_pad >0:
        res = F.pad(x, (0, 0, 0, num_pad), 'constant', 0)
    else:
        res = x
    return res


def train_ns(model, 
             train_loader, 
             val_loader,
             optimizer, 
             scheduler,
             device, config, args):
    # parse configuration
    v = 1/ config['data']['Re']
    t_duration = config['data']['time_duration']
    num_pad = config['model']['num_pad']
    save_step = config['log']['save_step']
    ic_weight = config['train']['ic_loss']
    f_weight = config['train']['f_loss']
    xy_weight = config['train']['xy_loss']
    # set up directory
    base_dir = os.path.join('exp', config['log']['logdir'])
    ckpt_dir = os.path.join(base_dir, 'ckpts')
    os.makedirs(ckpt_dir, exist_ok=True)

    # loss fn
    lploss = LpLoss(size_average=True)
    S = config['data']['pde_res'][0]
    data_s_step = train_loader.dataset.data_s_step
    data_t_step = train_loader.dataset.data_t_step
    forcing = get_forcing(S)
    # set up wandb
    if wandb and args.log:
        run = wandb.init(project=config['log']['project'], 
                         entity=config['log']['entity'], 
                         group=config['log']['group'], 
                         config=config, reinit=True, 
                         settings=wandb.Settings(start_method='fork'))
    pbar = range(config['train']['epochs'])
    pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.2)

    for e in pbar:
        loss_dict = {
            'train_loss': 0.0, 
            'ic_loss': 0.0, 
            'pde_loss': 0.0
        }

        # train 
        model.train()
        for u, a in train_loader:
            u, a = u.to(device), a.to(device)
            optimizer.zero_grad()

            if ic_weight == 0.0 and f_weight == 0.0:
                # FNO
                a = a[:, ::data_s_step, ::data_s_step, ::data_t_step]
                a_in = pad_input(a, num_pad=num_pad)
                out = model(a_in)[:, :, :, :-num_pad, 0]

                loss_ic, loss_f = 0, 0
                loss = lploss(out, u)
            else:
                # PINO
                a_in = pad_input(a, num_pad=num_pad)
                out = model(a_in)[:, :, :, :-num_pad, 0]
                # PDE loss
                loss_ic, loss_f = PINO_loss3d(out, a, forcing, v, t_duration)
                # data loss
                data_loss = lploss(out[:, ::data_s_step, ::data_s_step, ::data_t_step], u)
                loss = data_loss * xy_weight + loss_f * f_weight + loss_ic * ic_weight
            
            loss.backward()
            optimizer.step()

            loss_dict['train_loss'] += loss.item()
            loss_dict['ic_loss'] += loss_ic.item()
            loss_dict['pde_loss'] += loss_f.item()
        scheduler.step()
        
        loader_size = len(train_loader)
        train_loss = loss_dict['train_loss'] / loader_size
        ic_loss = loss_dict['ic_loss'] / loader_size
        pde_loss = loss_dict['pde_loss'] / loader_size

        # eval
        model.eval()
        with torch.no_grad():
            val_error = 0.0
            for u, a in val_loader:
                u, a = u.to(device), a.to(device)

                if ic_weight == 0.0 and f_weight == 0.0:
                    # FNO
                    a = a[:, ::data_s_step, ::data_s_step, ::data_t_step]
                    a_in = pad_input(a, num_pad=num_pad)
                    out = model(a_in)[:, :, :, :-num_pad, 0]
                    data_loss = lploss(out, u)
                else:
                    # PINO
                    a_in = pad_input(a, num_pad=num_pad)
                    out = model(a_in)[:, :, :, :-num_pad, 0]
                    # data loss
                    data_loss = lploss(out[:, ::data_s_step, ::data_s_step, ::data_t_step], u)
                val_error += data_loss.item()
            avg_val_error = val_error / len(val_loader)

        pbar.set_description(
            (
                f'Train loss: {train_loss}. IC loss: {ic_loss}, PDE loss: {pde_loss}, val error: {avg_val_error}'
            )
        )
        log_dict = {
            'Train loss': train_loss, 
            'IC loss': ic_loss, 
            'PDE loss': pde_loss, 
            'Val error': avg_val_error
        }

        if wandb and args.log:
            wandb.log(log_dict)
        if e % save_step:
            ckpt_path = os.path.join(ckpt_dir, f'model-{e}.pt')
            save_ckpt(ckpt_path, model, optimizer)
    # clean up wandb
    if wandb and args.log:
        run.finish()


def subprocess(args):
    with open(args.config, 'r') as f:
        config = yaml.load(f, yaml.FullLoader)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # set random seed
    config['seed'] = args.seed
    seed = args.seed
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # prepare dataset
    batchsize = config['train']['batchsize']

    trainset = NS3DDataset(paths=config['data']['paths'], 
                           data_res=config['data']['data_res'], 
                           pde_res=config['data']['pde_res'], 
                           n_samples=config['data']['n_samples'], 
                           offset=config['data']['offset'], 
                           t_duration=config['data']['t_duration'], 
                           train=True)
    train_loader = DataLoader(trainset, batch_size=batchsize, num_workers=4, shuffle=True)

    valset = NS3DDataset(paths=config['data']['paths'], 
                         data_res=config['data']['data_res'], 
                         pde_res=config['data']['pde_res'], 
                         n_samples=config['data']['n_samples'], 
                         offset=config['data']['offset'], 
                         t_duration=config['data']['t_duration'], 
                         train=False)
    val_loader = DataLoader(valset, batch_size=batchsize)
    # create model 
    model = FNN3d(modes1=config['model']['modes1'],
                  modes2=config['model']['modes2'],
                  modes3=config['model']['modes3'],
                  fc_dim=config['model']['fc_dim'],
                  layers=config['model']['layers'], 
                  act=config['model']['act']).to(device)
    # Load from checkpoint
    if 'ckpt' in config['train']:
        ckpt_path = config['train']['ckpt']
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        print('Weights loaded from %s' % ckpt_path)

    optimizer = Adam(model.parameters(), lr=config['train']['base_lr'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                     milestones=config['train']['milestones'], 
                                                     gamma=config['train']['scheduler_gamma'])


    train_ns(model, train_loader, val_loader, 
             optimizer, scheduler, device, config, args)
    print('Done!')


if __name__ == '__main__':

    torch.backends.cudnn.benchmark = True
    # parse options
    parser = ArgumentParser(description='Basic paser')
    parser.add_argument('--config', type=str, help='Path to the configuration file')
    parser.add_argument('--log', action='store_true', help='Turn on the wandb')
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()
    if args.seed is None:
        args.seed = random.randint(0, 100000)
    subprocess(args)