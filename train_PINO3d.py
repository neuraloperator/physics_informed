import yaml
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from train_utils import Adam
from train_utils.datasets import NSLoader
from train_utils.data_utils import data_sampler
from train_utils.losses import get_forcing
from train_utils.train_3d import train
from train_utils.distributed import setup, cleanup

from models import FNN3d


def subprocess_fn(rank, args):
    if args.distributed:
        setup(rank, args.num_gpus)
    print(f'Running on rank {rank}')

    config_file = args.config_path
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)

    assert (config['train']['batchsize'] > 0), 'batchsize must >= 1'
    assert (config['train']['batchsize'] % args.num_gpus == 0), \
        'batchsize must be a multiple of the number of gpus'

    # construct dataloader
    data_config = config['data']
    if args.new:
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

    trainset = loader.make_dataset(data_config['n_sample'],
                                   start=data_config['offset'],
                                   train=data_config['shuffle'])
    train_loader = DataLoader(trainset, batch_size=config['train']['batchsize'],
                              sampler=data_sampler(trainset,
                                                   shuffle=data_config['shuffle'],
                                                   distributed=args.distributed),
                              drop_last=True)

    # construct model
    model = FNN3d(modes1=config['model']['modes1'],
                  modes2=config['model']['modes2'],
                  modes3=config['model']['modes3'],
                  fc_dim=config['model']['fc_dim'],
                  layers=config['model']['layers']).to(rank)

    if 'ckpt' in config['train']:
        ckpt_path = config['train']['ckpt']
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        print('Weights loaded from %s' % ckpt_path)

    if args.distributed:
        model = DDP(model, device_ids=[rank], broadcast_buffers=False)

    optimizer = Adam(model.parameters(), betas=(0.9, 0.999),
                     lr=config['train']['base_lr'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=config['train']['milestones'],
                                                     gamma=config['train']['scheduler_gamma'])
    forcing = get_forcing(loader.S).to(rank)
    train(model,
          loader, train_loader,
          optimizer, scheduler,
          forcing, config,
          rank,
          log=args.log,
          project=config['others']['project'],
          group=config['others']['group'])

    if args.distributed:
        cleanup()
    print(f'Process {rank} done!...')


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    parser =ArgumentParser(description='Basic paser')
    parser.add_argument('--config_path', type=str, help='Path to the configuration file')
    parser.add_argument('--log', action='store_true', help='Turn on the wandb')
    parser.add_argument('--new', action='store_true', help='Use new data loader')
    parser.add_argument('--num_gpus', type=int, help='Number of GPUs', default=1)
    args = parser.parse_args()
    args.distributed = args.num_gpus > 1

    if args.distributed:
        mp.spawn(subprocess_fn, args=(args, ), nprocs=args.num_gpus)
    else:
        subprocess_fn(0, args)

