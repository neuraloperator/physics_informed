import os
from re import sub
from matplotlib import test
import yaml

import torch
import numpy as np

import matplotlib.pyplot as plt
from models import FNN3d


def test_config():
    config_path = 'configs/operator/Re500-PINO.yaml'

    with open(config_path, 'r') as f:
        config = yaml.load(f, yaml.FullLoader)
    print(config['data']['paths'])


def test_data():
    # data_path = '../data/NS-T4000.npy'
    # data_path = '../data/NS-Re500_T256.npy'
    data_path = '../data/NS-Re500_T10_ID3.npy'
    plot_dir = 'exp/debug/plots'
    os.makedirs(plot_dir, exist_ok=True)
    data = np.load(data_path)[0]
    T = data.shape[0]
    for i in range(0, T, 32):
        fig, ax = plt.subplots()
        im = ax.imshow(data[i])
        ax.set_title(f'T:{i}')
        fig.colorbar(im, ax=ax)
        plot_path = os.path.join(plot_dir, f'N10T{i}-id3.png')
        plt.savefig(plot_path)


def convert_data():
    sub_x = 4
    sub_t = 4
    offset = 0
    n_samples = 2400
    data_path = '../data/NS-Re500_T3000_id0.npy'
    raw_data = np.load(data_path, mmap_mode='r+')

    data = raw_data[offset: offset+n_samples, ::sub_t, ::sub_x, ::sub_x]
    data = torch.from_numpy(data).to(torch.float32)
    a_data = raw_data[offset: offset+n_samples, 0, :, :]
    a_data = torch.from_numpy(a_data).to(torch.float32)
    print(data.shape)
    print(a_data.shape)


def test_fno3d():
    config_path = 'configs/operator/Re500-3000-FNO.yaml'
    with open(config_path, 'r') as f:
        config = yaml.load(f, yaml.FullLoader)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = FNN3d(modes1=config['model']['modes1'],
                modes2=config['model']['modes2'],
                modes3=config['model']['modes3'],
                fc_dim=config['model']['fc_dim'],
                layers=config['model']['layers'], 
                act=config['model']['act']).to(device)
    B, S, T = 1, 256, 260
    a = torch.randn((B, S, S, T, 4), device=device)
    output = model(a)
    print(output.shape)


if __name__ == '__main__':
    # test_config()
    # test_data()
    # convert_data()
    test_fno3d()