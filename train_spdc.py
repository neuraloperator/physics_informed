from argparse import ArgumentParser
import yaml

import torch
import numpy as np
from models import FNO3d
from train_utils import Adam
from train_utils.datasets import SPDCLoader
from train_utils.utils import save_checkpoint
from train_utils.losses import LpLoss, darcy_loss, PINO_loss, SPDC_loss
from tqdm import tqdm
import torch.nn.functional as F
import gc
import torch.nn as nn
import wandb

def train_SPDC(model,
                    train_loader, 
                    optimizer, scheduler,
                    config,
                    equation_dict,
                    rank=0, 
                    log=False,
                    validate=False,
                    val_dataloader = None,
                    padding = 0,
                    project='PINO-3d-default',
                    group='default',
                    tags=['default'],
                    use_tqdm=True):
    if wandb and log:
        run = wandb.init(project=project,
                         entity=config['log']['entity'],
                         group=group,
                         config=config,
                         tags=tags, reinit=True,
                         settings=wandb.Settings(start_method="fork"))
        print(f"wandb is activated")

    data_weight = config['train']['xy_loss']
    f_weight = config['train']['f_loss']
    ic_weight = config['train']['ic_loss']
    nout = config['data']['nout']
    grad = config['model']['grad']

    model.train()
    pbar = range(config['train']['epochs'])
    if use_tqdm:
        pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.1)

    min_train_loss=float('inf')
    for e in pbar:
        model.train()
        train_pino = 0.0
        data_l2 = 0.0
        train_loss = 0.0

        for x, y in train_loader:
            gc.collect()
            torch.cuda.empty_cache()
            x, y = x.to(rank), y.to(rank)
            # y = torch.ones_like(y).to(rank)
            x_in = F.pad(x,(0,0,0,padding),"constant",0).type(torch.float32)
            out = model(x_in).reshape(y.shape[0],y.shape[1],y.shape[2],y.shape[3] + padding, 2*nout)
            # out = out[...,:-padding,:, :] # if padding is not 0

            data_loss,ic_loss,f_loss = SPDC_loss(u=out,y=y,input=x,equation_dict=equation_dict, grad=grad)
            total_loss = ic_loss * ic_weight + f_loss * f_weight + data_loss * data_weight

            gc.collect()
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            data_l2 += data_loss.item()
            train_pino += f_loss.item() 
            train_loss += total_loss.item()
        scheduler.step()
        data_l2 /= len(train_loader)
        train_pino /= len(train_loader)
        train_loss /= len(train_loader)

        if validate:
            validation_loss = eval_SPDC(
                                        model=model,
                                        dataloader=val_dataloader,
                                        config=config,
                                        equation_dict=equation_dict,
                                        device=rank,
                                        use_tqdm=False,
                                        validation=True)
            if use_tqdm:
                pbar.set_description(
                    (
                        f'Epoch {e}, train loss: {train_loss:.5f}; '
                        f'equation f error: {train_pino:.5f}; '
                        f'data l2 error: {data_l2:.5f}; '
                        f'val l2 error: {validation_loss:.5f}; '
                    )
                )
            if wandb and log:
                wandb.log(
                    {
                        'Equation f error': train_pino,
                        'Data L2 error': data_l2,
                        'Train loss': train_loss,
                        'Validation (data) L2 loss': validation_loss
                    }
                )

        else:
            if use_tqdm:
                pbar.set_description(
                    (
                        f'Epoch {e}, train loss: {train_loss:.5f}; '
                        f'equation f error: {train_pino:.5f}; '
                        f'data l2 error: {data_l2:.5f}; '
                    )
                )
            if wandb and log:
                wandb.log(
                    {
                        'Equation f error': train_pino,
                        'Data L2 error': data_l2,
                        'Train loss': train_loss
                    }
                )



        if e % 100 == 0:
            save_checkpoint(config['train']['save_dir'],
                            config['train']['save_name'].replace('.pt', f'_{e}.pt'),
                            model, optimizer)
        if train_loss < min_train_loss:
            min_train_loss=train_loss
            save_checkpoint(config['train']['save_dir'],
                            config['train']['save_name'].replace('.pt', f'_best_yet.pt'),
                            model, optimizer)

    save_checkpoint(config['train']['save_dir'],
                    config['train']['save_name'],
                    model, optimizer)
    print('Done!')


def eval_SPDC(model,
                 dataloader,
                 config,
                 equation_dict,
                 device,
                 padding = 0,
                 use_tqdm=True,
                 validation=False):
    model.eval()
    nout = config['data']['nout']
    grad = config['model']['grad']

    if use_tqdm:
        pbar = tqdm(dataloader, dynamic_ncols=True, smoothing=0.05)
    else:
        pbar = dataloader

    test_err = []
    f_err = []
    ic_err = []

    for x, y in pbar:
        gc.collect()
        torch.cuda.empty_cache()
        x, y = x.to(device), y.to(device)
        x_in = F.pad(x,(0,0,0,padding),"constant",0)
        out = model(x_in).reshape(y.shape[0],y.shape[1],y.shape[2],y.shape[3] + padding, 2*nout)
            # out = out[...,:-padding,:, :] # if padding is not 0

        data_loss,ic_loss,f_loss = SPDC_loss(u=out,y=y,input=x,equation_dict=equation_dict, grad=grad)
        test_err.append(data_loss.item())
        f_err.append(f_loss.item())
        ic_err.append(ic_loss.item())

    if validation:
        return np.mean(test_err)

    mean_f_err = np.mean(f_err)
    std_f_err = np.std(f_err, ddof=1) / np.sqrt(len(f_err))

    mean_err = np.mean(test_err)
    std_err = np.std(test_err, ddof=1) / np.sqrt(len(test_err))

    mean_ic_err = np.mean(ic_err)
    std_ic_err = np.std(ic_err, ddof=1) / np.sqrt(len(ic_err))

    print(f'==Averaged relative L2 error mean: {mean_err}, std error: {std_err}==\n'
          f'==Averaged equation error mean: {mean_f_err}, std error: {std_f_err}==\n'
          f'==Averaged initial condition error mean: {mean_ic_err}, std error: {std_ic_err}==')
    
def eval_dummy_SPDC(dataloader,
                 config,
                 equation_dict,
                 device,
                 padding = 0,
                 use_tqdm=True):
    nout = config['data']['nout']
    grad = config['model']['grad']
    if use_tqdm:
        pbar = tqdm(dataloader, dynamic_ncols=True, smoothing=0.05)
    else:
        pbar = dataloader

    test_err = []
    f_err = []
    ic_err = []

    for x, y in pbar:
        gc.collect()
        torch.cuda.empty_cache()
        x, y = x.to(device), y.to(device)
        out = torch.zeros_like(y)[...,:2*nout]
        data_loss,ic_loss,f_loss = SPDC_loss(u=out,y=y,input=x,equation_dict=equation_dict,grad=grad)
        test_err.append(data_loss.item())
        f_err.append(f_loss.item())
        ic_err.append(ic_loss.item())

    mean_f_err = np.mean(f_err)
    std_f_err = np.std(f_err, ddof=1) / np.sqrt(len(f_err))

    mean_err = np.mean(test_err)
    std_err = np.std(test_err, ddof=1) / np.sqrt(len(test_err))

    mean_ic_err = np.mean(ic_err)
    std_ic_err = np.std(ic_err, ddof=1) / np.sqrt(len(ic_err))

    print(f'==Averaged relative L2 error mean: {mean_err}, std error: {std_err}==\n'
          f'==Averaged equation error mean: {mean_f_err}, std error: {std_f_err}==\n'
          f'==Averaged initial condition error mean: {mean_ic_err}, std error: {std_ic_err}==')


def run(args, config):

    model = FNO3d(modes1=config['model']['modes1'],
                  modes2=config['model']['modes2'],
                  modes3=config['model']['modes3'],
                  fc_dim=config['model']['fc_dim'],
                  layers=config['model']['layers'],
                  in_dim=config['model']['in_dim'],
                  out_dim=config['model']['out_dim'],
                  activation_func=config['model']['act'])
  #
  #if torch.cuda.device_count() > 1:
  #   print("Let's use", torch.cuda.device_count(), "GPUs!")
  #   model = nn.DataParallel(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    torch.cuda.empty_cache()

    data_config = config['data']
    dataset = SPDCLoader(   datapath = data_config['datapath'],
                            nx=data_config['nx'], 
                            ny=data_config['ny'],
                            nz=data_config['nz'],
                            nin = data_config['nin'],
                            nout = data_config['nout'],
                            sub_xy=data_config['sub_xy'],
                            sub_z=data_config['sub_z'],
                            N=data_config['total_num'],
                            device=device)
    
    equation_dict = dataset.data_dict
    train_loader = dataset.make_loader(n_sample=data_config['n_sample'],
                                       batch_size=config['train']['batchsize'],
                                       start=data_config['offset'],train=True)

    val_dataloader = None
    if args.validate:
        val_dataloader = dataset.make_loader(
                                     n_sample=data_config['total_num'] - data_config['n_sample'],
                                     batch_size=config['train']['batchsize'],
                                     start=data_config['n_sample'],train=False)
        del dataset
        gc.collect()
    torch.cuda.empty_cache()

    # Load from checkpoint
    if 'ckpt' in config['train']:
        ckpt_path = config['train']['ckpt']
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        print('Weights loaded from %s' % ckpt_path)
    optimizer = Adam(model.parameters(), betas=(0.9, 0.999),
                     lr=config['train']['base_lr'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=config['train']['milestones'],
                                                     gamma=config['train']['scheduler_gamma'])
    train_SPDC(model,
                    train_loader, 
                    optimizer, 
                    scheduler,
                    config,
                    equation_dict,
                    rank=device, 
                    log=args.log,
                    project=config['log']['project'],
                    group=config['log']['group'],
                    validate=args.validate,
                    val_dataloader=val_dataloader)

def test(config):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    data_config = config['data']
    dataset = SPDCLoader(   datapath = data_config['datapath'],
                            nx=data_config['nx'], 
                            ny=data_config['ny'],
                            nz=data_config['nz'],
                            nin = data_config['nin'],
                            nout = data_config['nout'],
                            sub_xy=data_config['sub_xy'],
                            sub_z=data_config['sub_z'],
                            N=data_config['total_num'],
                            device=device)
    
    equation_dict = dataset.data_dict
    dataloader = dataset.make_loader(n_sample=data_config['n_sample'],
                                     batch_size=config['test']['batchsize'],
                                     start=data_config['offset'])
    del dataset
    gc.collect()
    torch.cuda.empty_cache()

    model = FNO3d(modes1=config['model']['modes1'],
                  modes2=config['model']['modes2'],
                  modes3=config['model']['modes3'],
                  fc_dim=config['model']['fc_dim'],
                  layers=config['model']['layers'],
                  in_dim=config['model']['in_dim'],
                  out_dim=config['model']['out_dim'],
                  activation_func=config['model']['act']).to(device)
    # Load from checkpoint
    if 'ckpt' in config['test']:
        ckpt_path = config['test']['ckpt']
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        print('Weights loaded from %s' % ckpt_path)
    eval_SPDC(model=model,dataloader=dataloader, config=config, equation_dict=equation_dict, device=device)

def dummy(config):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    data_config = config['data']
    dataset = SPDCLoader(   datapath = data_config['datapath'],
                            nx=data_config['nx'], 
                            ny=data_config['ny'],
                            nz=data_config['nz'],
                            nin = data_config['nin'],
                            nout = data_config['nout'],
                            sub_xy=data_config['sub_xy'],
                            sub_z=data_config['sub_z'],
                            N=data_config['total_num'],
                            device=device)
    
    equation_dict = dataset.data_dict
    dataloader = dataset.make_loader(n_sample=data_config['n_sample'],
                                     batch_size=config['train']['batchsize'],
                                     start=data_config['offset'])
    del dataset
    gc.collect()
    torch.cuda.empty_cache()
    eval_dummy_SPDC(dataloader=dataloader, config=config, equation_dict=equation_dict, device=device)

if __name__ == '__main__':
    parser = ArgumentParser(description='Basic paser')
    parser.add_argument('--config_path', type=str, help='Path to the configuration file')
    parser.add_argument('--log', action='store_true', help='Turn on the wandb')
    parser.add_argument('--validate', action='store_true', help='Calculate validation error')
    parser.add_argument('--mode', type=str, help='train, test or dummy')
    args = parser.parse_args()

    config_file = args.config_path
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)
    if args.mode == 'train':
        run(args, config)
    elif args.mode == 'dummy':
        dummy(config)
    else:
        test(config)
