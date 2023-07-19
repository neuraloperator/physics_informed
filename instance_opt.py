import os
import yaml
import random
from argparse import ArgumentParser
import math
from tqdm import tqdm

import torch

from torch.optim import Adam
from torch.utils.data import DataLoader, Subset

from models import FNO3d

from train_utils.losses import LpLoss, PINO_loss3d, get_forcing
from train_utils.datasets import KFDataset, KFaDataset, sample_data
from train_utils.utils import save_ckpt, count_params, dict2str

try:
    import wandb
except ImportError:
    wandb = None


def train_ns(model, 
             u_loader,        # training data
             optimizer, 
             scheduler,
             device, config, args):

    v = 1/ config['data']['Re']
    t_duration = config['data']['t_duration']
    save_step = config['train']['save_step']

    ic_weight = config['train']['ic_loss']
    f_weight = config['train']['f_loss']
    # set up directory
    base_dir = os.path.join('exp', config['log']['logdir'])
    ckpt_dir = os.path.join(base_dir, 'ckpts')
    os.makedirs(ckpt_dir, exist_ok=True)

    # loss fn
    lploss = LpLoss(size_average=True)
    
    S = config['data']['pde_res'][0]
    forcing = get_forcing(S).to(device)
    # set up wandb
    if wandb and args.log:
        run = wandb.init(project=config['log']['project'], 
                         entity=config['log']['entity'], 
                         group=config['log']['group'], 
                         config=config, reinit=True, 
                         settings=wandb.Settings(start_method='fork'))
    
    pbar = range(config['train']['num_iter'])
    if args.tqdm:
        pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.2)

    u_loader = sample_data(u_loader)

    for e in pbar:
        log_dict = {}

        optimizer.zero_grad()
        # data loss
        u, a_in = next(u_loader)
        u = u.to(device)
        a_in = a_in.to(device)
        out = model(a_in)
        data_loss = lploss(out, u)
            
        u0  = a_in[:, :, :, 0, -1]
        loss_ic, loss_f = PINO_loss3d(out, u0, forcing, v, t_duration)
        log_dict['IC'] = loss_ic.item()
        log_dict['PDE'] = loss_f.item()
        loss = loss_f * f_weight + loss_ic * ic_weight

        loss.backward()
        optimizer.step()
        scheduler.step()

        log_dict['train loss'] = loss.item()
        log_dict['test error'] = data_loss.item()
        
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
            save_ckpt(ckpt_path, model, optimizer)

    # clean up wandb
    if wandb and args.log:
        run.finish()
        
    # save prediction and truth
    save_dir = os.path.join(base_dir, 'results')
    os.makedirs(save_dir, exist_ok=True)
    result_path = os.path.join(save_dir, f'results-{args.idx}.pt')

    criterion = LpLoss()

    model.eval()
    with torch.no_grad():
        u, a_in = next(u_loader)
        u = u.to(device)
        a_in = a_in.to(device)
        out = model(a_in)
        error = criterion(out, u)
        print(f'Test error: {error.item()}')
        torch.save({'truth': u.cpu(), 'pred': out.cpu()}, result_path)
    print(f'Results saved to {result_path}')



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
    model = FNO3d(modes1=config['model']['modes1'],
                  modes2=config['model']['modes2'],
                  modes3=config['model']['modes3'],
                  fc_dim=config['model']['fc_dim'],
                  layers=config['model']['layers'], 
                  act=config['model']['act'], 
                  pad_ratio=config['model']['pad_ratio']).to(device)
    num_params = count_params(model)
    config['num_params'] = num_params
    print(f'Number of parameters: {num_params}')
    # Load from checkpoint
    if args.ckpt:
        ckpt_path = args.ckpt
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        print('Weights loaded from %s' % ckpt_path)
    
    # training set
    batchsize = config['train']['batchsize']
    dataset = KFDataset(paths=config['data']['paths'], 
                        raw_res=config['data']['raw_res'],
                        data_res=config['data']['data_res'], 
                        pde_res=config['data']['data_res'], 
                        n_samples=config['data']['n_test_samples'], 
                        total_samples=1,
                        idx=args.idx,
                        offset=config['data']['testoffset'], 
                        t_duration=config['data']['t_duration'])
    u_loader = DataLoader(dataset, batch_size=1)

    optimizer = Adam(model.parameters(), lr=config['train']['base_lr'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                     milestones=config['train']['milestones'], 
                                                     gamma=config['train']['scheduler_gamma'])
    train_ns(model, 
             u_loader, 
             optimizer, 
             scheduler, 
             device, 
             config, 
             args)
    print('Done!')
        
        

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    # parse options
    parser = ArgumentParser(description='Basic paser')
    parser.add_argument('--config', type=str, help='Path to the configuration file')
    parser.add_argument('--idx', type=int, default=0, help='Index of the instance')
    parser.add_argument('--log', action='store_true', help='Turn on the wandb')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--tqdm', action='store_true', help='Turn on the tqdm')
    args = parser.parse_args()
    if args.seed is None:
        args.seed = random.randint(0, 100000)
    subprocess(args)