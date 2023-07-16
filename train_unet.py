from datetime import datetime
import os
import yaml
import random
from argparse import ArgumentParser
import math
from tqdm import tqdm

import numpy as np

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from baselines.unet3d import UNet3D

from train_utils.losses import LpLoss
from train_utils.datasets import KFDataset, KFaDataset, sample_data
from train_utils.utils import save_ckpt, count_params, dict2str

try:
    import wandb
except ImportError:
    wandb = None


@torch.no_grad()
def eval_ns(model, val_loader, criterion, device):
    model.eval()
    val_err = []
    for u, a in val_loader:
        u, a = u.to(device), a.to(device)
        a = a.permute(0, 4, 3, 1, 2)
        out = model(a)
        out = out.squeeze(1).permute(0, 2, 3, 1)
        val_loss = criterion(out, u)
        val_err.append(val_loss.item())

    N = len(val_loader)

    avg_err = np.mean(val_err)
    std_err = np.std(val_err, ddof=1) / np.sqrt(N)
    return avg_err, std_err


def train_ns(model, 
             train_u_loader,        # training data
             val_loader,            # validation data
             optimizer, 
             scheduler,
             device, config, args):
    start_iter = config['train']['start_iter']
    v = 1/ config['data']['Re']
    save_step = config['train']['save_step']
    eval_step = config['train']['eval_step']

    # set up directory
    base_dir = os.path.join('exp', config['log']['logdir'])
    ckpt_dir = os.path.join(base_dir, 'ckpts')
    os.makedirs(ckpt_dir, exist_ok=True)

    # loss fn
    lploss = LpLoss(size_average=True)
    
    S = config['data']['pde_res'][0]
    # set up wandb
    if wandb and args.log:
        run = wandb.init(project=config['log']['project'], 
                         entity=config['log']['entity'], 
                         group=config['log']['group'], 
                         config=config, reinit=True, 
                         settings=wandb.Settings(start_method='fork'))
    
    pbar = range(start_iter, config['train']['num_iter'])
    if args.tqdm:
        pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.2)

    u_loader = sample_data(train_u_loader)

    for e in pbar:
        log_dict = {}

        optimizer.zero_grad()
        # data loss
        u, a_in = next(u_loader)
        u = u.to(device)
        a_in = a_in.to(device).permute(0, 4, 3, 1, 2)   # B, C, T, X, Y
        out = model(a_in)
        out = out.squeeze(1).permute(0, 2, 3, 1)   # B, X, Y, T
        data_loss = lploss(out, u)

        loss = data_loss

        loss.backward()
        optimizer.step()
        scheduler.step()

        log_dict['train loss'] = loss.item()
        if e % eval_step == 0:
            eval_err, std_err = eval_ns(model, val_loader, lploss, device)
            log_dict['val error'] = eval_err
        
        if args.tqdm:
            logstr = dict2str(log_dict)
            pbar.set_description(
                (
                    logstr
                )
            )
        if wandb and args.log:
            wandb.log(log_dict)
        if e % save_step == 0 and e > 0:
            ckpt_path = os.path.join(ckpt_dir, f'model-{e}.pt')
            save_ckpt(ckpt_path, model, optimizer, scheduler)

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

    # create model 
    model = UNet3D(in_channels=4, out_channels=1, f_maps=64, final_sigmoid=False).to(device)
    num_params = count_params(model)
    config['num_params'] = num_params
    print(f'Number of parameters: {num_params}')
    # Load from checkpoint
    if args.ckpt:
        ckpt_path = args.ckpt
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        print('Weights loaded from %s' % ckpt_path)
    
    if args.test:
        batchsize = config['test']['batchsize']
        testset = KFDataset(paths=config['data']['paths'], 
                            raw_res=config['data']['raw_res'],
                            data_res=config['test']['data_res'], 
                            pde_res=config['test']['data_res'], 
                            n_samples=config['data']['n_test_samples'], 
                            offset=config['data']['testoffset'], 
                            t_duration=config['data']['t_duration'])
        testloader = DataLoader(testset, batch_size=batchsize, num_workers=4)
        criterion = LpLoss()
        test_err, std_err = eval_ns(model, testloader, criterion, device)
        print(f'Averaged test relative L2 error: {test_err}; Standard error: {std_err}')
    else:
        # training set
        batchsize = config['train']['batchsize']
        u_set = KFDataset(paths=config['data']['paths'], 
                          raw_res=config['data']['raw_res'],
                          data_res=config['data']['data_res'], 
                          pde_res=config['data']['data_res'], 
                          n_samples=config['data']['n_data_samples'], 
                          offset=config['data']['offset'], 
                          t_duration=config['data']['t_duration'])
        u_loader = DataLoader(u_set, batch_size=batchsize, num_workers=4, shuffle=True)

        # val set
        valset = KFDataset(paths=config['data']['paths'], 
                           raw_res=config['data']['raw_res'],
                           data_res=config['test']['data_res'], 
                           pde_res=config['test']['data_res'], 
                           n_samples=config['data']['n_test_samples'], 
                           offset=config['data']['testoffset'], 
                           t_duration=config['data']['t_duration'])
        val_loader = DataLoader(valset, batch_size=batchsize, num_workers=4)
        print(f'Train set: {len(u_set)}; Test set: {len(valset)}.')
        optimizer = Adam(model.parameters(), lr=config['train']['base_lr'])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                         milestones=config['train']['milestones'], 
                                                         gamma=config['train']['scheduler_gamma'])
        if args.ckpt:
            ckpt = torch.load(ckpt_path)
            optimizer.load_state_dict(ckpt['optim'])
            scheduler.load_state_dict(ckpt['scheduler'])
            config['train']['start_iter'] = scheduler.last_epoch
        train_ns(model, 
                 u_loader, 
                 val_loader, 
                 optimizer, scheduler, 
                 device, 
                 config, args)
    print('Done!')
        
        

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    # parse options
    parser = ArgumentParser(description='Basic paser')
    parser.add_argument('--config', type=str, help='Path to the configuration file')
    parser.add_argument('--log', action='store_true', help='Turn on the wandb')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--test', action='store_true', help='Test')
    parser.add_argument('--tqdm', action='store_true', help='Turn on the tqdm')
    args = parser.parse_args()
    if args.seed is None:
        args.seed = random.randint(0, 100000)
    subprocess(args)
