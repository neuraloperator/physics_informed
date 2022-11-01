import yaml
from argparse import ArgumentParser
import random

import torch

from models import FNO2d
from train_utils import Adam
from torch.utils.data import DataLoader
from train_utils.datasets import DarcyFlow
from train_utils.train_2d import train_2d_operator


def train(args, config):
    seed = random.randint(1, 10000)
    print(f'Random seed :{seed}')
    torch.manual_seed(seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_config = config['data']
    dataset = DarcyFlow(data_config['datapath'],
                        nx=data_config['nx'], sub=data_config['sub'],
                        offset=data_config['offset'], num=data_config['n_sample'])
    dataloader = DataLoader(dataset, batch_size=config['train']['batchsize'])
    model = FNO2d(modes1=config['model']['modes1'],
                  modes2=config['model']['modes2'],
                  fc_dim=config['model']['fc_dim'],
                  layers=config['model']['layers'],
                  act=config['model']['act']).to(device)
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
    train_2d_operator(model,
                      dataloader,
                      optimizer, scheduler,
                      config, rank=0, log=args.log,
                      project=config['log']['project'],
                      group=config['log']['group'])


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    parser = ArgumentParser(description='Basic paser')
    parser.add_argument('--config_path', type=str, help='Path to the configuration file')
    parser.add_argument('--start', type=int, help='Start index of test instance')
    parser.add_argument('--stop', type=int, help='Stop index of instances')
    parser.add_argument('--log', action='store_true', help='Turn on the wandb')
    args = parser.parse_args()

    config_file = args.config_path
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)

    for i in range(args.start, args.stop):
        print(f'Start solving instance {i}')
        config['data']['offset'] = i
        train(args, config)
    print(f'{args.stop - args.start} instances are solved')