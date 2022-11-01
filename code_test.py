import os
from re import sub
from matplotlib import test
import yaml

import torch
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image


from models import FNO3d


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


def plot_data(datapath, id=1):
    plot_dir = 'exp/debug/plots/Re500'
    os.makedirs(plot_dir, exist_ok=True)
    raw = np.load(datapath, mmap_mode='r')
    data = raw[id]
    t_duration = 1
    T = int(data.shape[0] * t_duration)
    for i in range(0, T):
        fig, ax = plt.subplots()
        im = ax.imshow(data[i])
        ax.set_title(f'T={i} / {T-1}s')
        fig.colorbar(im, ax=ax)
        plot_path = os.path.join(plot_dir, f'test_t{i}.jpeg')
        plt.savefig(plot_path)
        plt.close()


def make_gif(id=1):
    plot_dir = 'exp/debug/plots/Re500'
    os.makedirs(plot_dir, exist_ok=True)
    # Create new figure for GIF

    # Adjust figure so GIF does not have extra whitespace
    ims = []
    T = 513
    for i in range(T):
        img_path = os.path.join(plot_dir, f'test_t{i}.jpeg')
        im = Image.open(img_path)
        ims.append(im)
    
    gif = ims[0]
    gif_dir = 'exp/debug/plots/gifs'
    os.makedirs(gif_dir, exist_ok=True)
    gif_path = os.path.join(gif_dir, f'ns-Re500-1s_id{id}.gif')
    gif.save(gif_path, format='GIF', append_images=ims, save_all=True, duration=50, loop=0)


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

    model = FNO3d(modes1=config['model']['modes1'],
                modes2=config['model']['modes2'],
                modes3=config['model']['modes3'],
                fc_dim=config['model']['fc_dim'],
                layers=config['model']['layers'], 
                act=config['model']['act']).to(device)
    B, S, T = 1, 256, 260
    a = torch.randn((B, S, S, T, 4), device=device)
    output = model(a)
    print(output.shape)


def test_opt():
    ckpt_path = 'exp/Re500-1_8s-2200-PINO-s/ckpts/model-30000.pt'
    ckpt = torch.load(ckpt_path)
    adam = ckpt['optim']
    print('optim')


if __name__ == '__main__':
    # test_config()
    # test_data()
    # convert_data()
    # test_fno3d()
    datapath = '/raid/hongkai/NS-Re500_T300_id0-shuffle.npy'
    # id = 10
    ids = [0, 50, 100, 200, 250, 299]
    # for id in ids:
        # plot_data(datapath, id=id)
        # make_gif(id=id)
    
    test_opt()