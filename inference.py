import os
import yaml
from argparse import ArgumentParser
import random

import torch
from torch.utils.data import DataLoader

from models import FNO3d

from train_utils.datasets import KFDataset
from train_utils.losses import LpLoss
from train_utils.utils import count_params


@torch.no_grad()
def get_pred(args):
    with open(args.config, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # prepare data
    dataset = KFDataset(paths=config['data']['paths'], 
                        raw_res=config['data']['raw_res'],
                        data_res=config['data']['data_res'], 
                        pde_res=config['data']['data_res'], 
                        n_samples=config['data']['n_test_samples'], 
                        offset=config['data']['testoffset'], 
                        t_duration=config['data']['t_duration'])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)

    # create model
    model = FNO3d(modes1=config['model']['modes1'],
                modes2=config['model']['modes2'],
                modes3=config['model']['modes3'],
                fc_dim=config['model']['fc_dim'],
                layers=config['model']['layers'], 
                act=config['model']['act'], 
                pad_ratio=config['model']['pad_ratio']).to(device)
    num_params = count_params(model)
    print(f'Number of parameters: {num_params}')
    if args.ckpt_path:
        ckpt = torch.load(args.ckpt_path)
        model.load_state_dict(ckpt['model'])
        print('Weights loaded from %s' % args.ckpt_path)
    # metric
    lploss = LpLoss(size_average=True)
    model.eval()
    for u, a_in in dataloader:
        u, a_in = u.to(device), a_in.to(device)
        out = model(a_in)
        data_loss = lploss(out, u)
        print(data_loss.item())
        torch.save({
            'truth': u.cpu(),
            'pred': out.cpu(),
        }, 're500-1_8s-800-fno-50k-prediction.pt')
        break


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--ckpt_path', type=str, default=None)
    args = parser.parse_args()
    get_pred(args)