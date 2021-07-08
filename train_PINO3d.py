import yaml

import torch

from train_utils import Adam
from train_utils.data_utils import NS500Loader, NSLoader
from train_utils.losses import get_forcing
from train_utils.train_3d import train
from models import FNN3d
from argparse import ArgumentParser


if __name__ == '__main__':
    parser =ArgumentParser(description='Basic paser')
    parser.add_argument('--config_path', type=str, help='Path to the configuration file')
    parser.add_argument('--log', action='store_true', help='Turn on the wandb')
    parser.add_argument('--new', action='store_true', help='Use new data loader')
    options = parser.parse_args()

    config_file = options.config_path
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_config = config['data']
    if options.new:
        datapath2 = data_config['datapath'].replace('part0', 'part1')
        loader = NSLoader(datapath1=data_config['datapath'], datapath2=datapath2,
                          nx=data_config['nx'], nt=data_config['nt'],
                          sub=data_config['sub'], sub_t=data_config['sub_t'],
                          N=data_config['total_num'],
                          t_interval=data_config['time_interval'])
    else:
        loader = NS500Loader(data_config['datapath'],
                             nx=data_config['nx'], nt=data_config['nt'],
                             sub=data_config['sub'], sub_t=data_config['sub_t'],
                             N=data_config['total_num'],
                             t_interval=data_config['time_interval'])

    train_loader = loader.make_loader(data_config['n_sample'],
                                      batch_size=config['train']['batchsize'],
                                      start=data_config['offset'],
                                      train=data_config['shuffle'])

    model = FNN3d(modes1=config['model']['modes1'],
                  modes2=config['model']['modes2'],
                  modes3=config['model']['modes3'],
                  fc_dim=config['model']['fc_dim'],
                  layers=config['model']['layers']).to(device)
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
    forcing = get_forcing(loader.S).to(device)
    train(model,
          loader, train_loader,
          optimizer, scheduler,
          forcing, config,
          device,
          log=options.log,
          project=config['others']['project'],
          group=config['others']['group'])
