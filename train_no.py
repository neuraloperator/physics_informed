import os
import yaml
import random
from argparse import ArgumentParser
import math
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


from models import FNO3d
from train_utils.adam import Adam

from train_utils.losses import LpLoss, PINO_loss3d, get_forcing
from train_utils.datasets import NS3DDataset, KFDataset
from train_utils.utils import save_ckpt, count_params

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
    t_duration = config['data']['t_duration']
    num_pad = config['model']['num_pad']
    save_step = config['train']['save_step']
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
    data_s_step = train_loader.dataset.dataset.data_s_step
    data_t_step = train_loader.dataset.dataset.data_t_step
    forcing = get_forcing(S).to(device)
    # set up wandb
    if wandb and args.log:
        run = wandb.init(project=config['log']['project'], 
                         entity=config['log']['entity'], 
                         group=config['log']['group'], 
                         config=config, reinit=True, 
                         settings=wandb.Settings(start_method='fork'))
    pbar = range(config['train']['epochs'])
    pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.2)
    zero = torch.zeros(1).to(device)
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
                a_in = a[:, ::data_s_step, ::data_s_step, ::data_t_step]
                out = model(a_in)
                loss_ic, loss_f = zero, zero
                loss = lploss(out, u)
            else:
                # PINO
                a_in = a
                out = model(a_in)
                # PDE loss
                u0 = a[:, :, :, 0, -1]
                loss_ic, loss_f = PINO_loss3d(out, u0, forcing, v, t_duration)
                # data loss
                # print(out.shape)
                # print(u.shape)
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
                    a_in = a
                    out = model(a_in)
                    data_loss = lploss(out, u)
                else:
                    # PINO
                    a_in = a
                    out = model(a_in)
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
        if e % save_step == 0:
            ckpt_path = os.path.join(ckpt_dir, f'model-{e}.pt')
            save_ckpt(ckpt_path, model, optimizer)
    # clean up wandb
    if wandb and args.log:
        run.finish()


def eval_ns(model, val_loader, device, config, args):
    # parse configuration
    v = 1/ config['data']['Re']
    t_duration = config['data']['t_duration']
    num_pad = config['model']['num_pad']

    model.eval()
    # loss fn
    lploss = LpLoss(size_average=True)
    S = config['data']['pde_res'][0]
    data_s_step = val_loader.dataset.data_s_step
    data_t_step = val_loader.dataset.data_t_step

    with torch.no_grad():
        val_error = 0.0
        for u, a in tqdm(val_loader):
            u, a = u.to(device), a.to(device)
            # a = a[:, ::data_s_step, ::data_s_step, ::data_t_step]
            a_in = a
            out = model(a_in)
            out = out[:, ::data_s_step, ::data_s_step, ::data_t_step]
            data_loss = lploss(out, u)
            val_error += data_loss.item()
        avg_val_err = val_error / len(val_loader)

    print(f'Average relative L2 error {avg_val_err}')


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

    datasets = {
        'KF': KFDataset, 
        'NS': NS3DDataset
    }
    if 'name' in config['data']:
        dataname = config['data']['name']
    else:
        dataname = 'NS'

    if args.test:
        batchsize = config['test']['batchsize']
        testset = datasets[dataname](paths=config['data']['paths'], 
                                     raw_res=config['data']['raw_res'],
                                     data_res=config['test']['data_res'], 
                                     pde_res=config['data']['pde_res'], 
                                     n_samples=config['data']['n_test_samples'], 
                                     offset=config['data']['testoffset'], 
                                     t_duration=config['data']['t_duration'])
        test_loader = DataLoader(testset, batch_size=batchsize, num_workers=4, shuffle=True)
        eval_ns(model, test_loader, device, config, args)

    else:
        # prepare datast
        batchsize = config['train']['batchsize']

        dataset = datasets[dataname](paths=config['data']['paths'], 
                                     raw_res=config['data']['raw_res'],
                                     data_res=config['data']['data_res'], 
                                     pde_res=config['data']['pde_res'], 
                                     n_samples=config['data']['n_samples'], 
                                     offset=config['data']['offset'], 
                                     t_duration=config['data']['t_duration'])
        idxs = torch.randperm(len(dataset))
        # setup train and test
        num_test = config['data']['n_test_samples']
        num_train = len(idxs) - num_test
        print(f'Number of training samples: {num_train};\nNumber of test samples: {num_test}.')
        train_idx = idxs[:num_train]
        test_idx = idxs[num_train:]

        trainset = Subset(dataset, indices=train_idx)
        valset = Subset(dataset, indices=test_idx)

        train_loader = DataLoader(trainset, batch_size=batchsize, num_workers=4, shuffle=True)

        val_loader = DataLoader(valset, batch_size=batchsize, num_workers=4)
        optimizer = Adam(model.parameters(), lr=config['train']['base_lr'])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                         milestones=config['train']['milestones'], 
                                                         gamma=config['train']['scheduler_gamma'])
        print(dataset.data.shape)
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
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--test', action='store_true', help='Test')
    args = parser.parse_args()
    if args.seed is None:
        args.seed = random.randint(0, 100000)
    subprocess(args)