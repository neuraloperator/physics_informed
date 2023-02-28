from argparse import ArgumentParser
import yaml

import torch
from models import FNO2d
from train_utils import Adam
from train_utils.datasets import BurgersLoader
from train_utils.train_2d import train_2d_burger
from train_utils.eval_2d import eval_burgers


def run(args, config):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_config = config['data']
    dataset = BurgersLoader(data_config['datapath'],
                            nx=data_config['nx'], nt=data_config['nt'],
                            sub=data_config['sub'], sub_t=data_config['sub_t'], new=True)
    train_loader = dataset.make_loader(n_sample=data_config['n_sample'],
                                       batch_size=config['train']['batchsize'],
                                       start=data_config['offset'])

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
    train_2d_burger(model,
                    train_loader,
                    dataset.v,
                    optimizer,
                    scheduler,
                    config,
                    rank=0,
                    log=args.log,
                    project=config['log']['project'],
                    group=config['log']['group'])


def test(config):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_config = config['data']
    dataset = BurgersLoader(data_config['datapath'],
                            nx=data_config['nx'], nt=data_config['nt'],
                            sub=data_config['sub'], sub_t=data_config['sub_t'], new=True)
    dataloader = dataset.make_loader(n_sample=data_config['n_sample'],
                                     batch_size=config['test']['batchsize'],
                                     start=data_config['offset'])

    model = FNO2d(modes1=config['model']['modes1'],
                  modes2=config['model']['modes2'],
                  fc_dim=config['model']['fc_dim'],
                  layers=config['model']['layers'],
                  act=config['model']['act']).to(device)
    # Load from checkpoint
    if 'ckpt' in config['test']:
        ckpt_path = config['test']['ckpt']
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        print('Weights loaded from %s' % ckpt_path)
    eval_burgers(model, dataloader, dataset.v, config, device)


if __name__ == '__main__':
    parser = ArgumentParser(description='Basic paser')
    parser.add_argument('--config_path', type=str, help='Path to the configuration file')
    parser.add_argument('--log', action='store_true', help='Turn on the wandb')
    parser.add_argument('--mode', type=str, help='train or test')
    args = parser.parse_args()

    config_file = args.config_path
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)
    if args.mode == 'train':
        run(args, config)
    else:
        test(config)
