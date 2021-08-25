import numpy as np
from tqdm import tqdm
import yaml
from argparse import ArgumentParser


import torch
import torch.nn.functional as F

from train_utils import Adam
from train_utils.data_utils import NSLoader
from train_utils.losses import get_forcing
from train_utils.train_3d import train
from models import FNN3d


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
    print(device)
    data_config = config['data']
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

    train_loader = loader.make_loader(data_config['n_sample'],
                                      batch_size=config['train']['batchsize'],
                                      start=data_config['offset'],
                                      train=data_config['shuffle'])
    # data parameters
    S, T = loader.S, loader.T
    batch_size = config['train']['batchsize']
    pred_sol = torch.zeros_like(loader.data)
    # training settings
    batch_size = config['train']['batchsize']
    model.eval()
    with torch.no_grad():
        for i, (x, y) in tqdm(enumerate(train_loader)):
            x = x.to(device)
            x_in = F.pad(x, (0, 0, 0, 5), 'constant', 0)
            pred = model(x_in).reshape(batch_size, S, S, T + 5)
            pred = pred[..., :-5]
            pred_sol[i * batch_size: (i + 1) * batch_size] = pred

    print(f'Result shape: {pred_sol.shape}')
    torch.save(
        {
            'pred': pred_sol
        },
        'data/Re500-pretrained-part2.pt'
    )

