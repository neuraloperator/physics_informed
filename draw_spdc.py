from argparse import ArgumentParser
import yaml

import torch
import numpy as np
from models import FNO3d
from train_utils.datasets import SPDCLoader
from train_utils.utils import save_checkpoint
from tqdm import tqdm
import torch.nn.functional as F
import gc
import matplotlib.pyplot as plt
from matplotlib import cm



def plot_av_sol(u,y):
    # y = torch.ones_like(y)
    N,nx,ny,nz,u_nfields = u.shape
    y_nfields = y.shape[4]
    u = u.reshape(N,nx, ny, nz,2,u_nfields//2)
    y = y.reshape(N,nx, ny, nz,2,y_nfields//2)[...,-2:]
    u = (u[...,0,:] + 1j*u[...,1,:]).detach().numpy()
    y = (y[...,0,:] + 1j*y[...,1,:]).detach().numpy()
    for sol,src in zip([u,y],["prediction", "grt"]):
        dict = {0:"signal out", 1:"idler out"}
        maxXY = 120e-6
        XY = u.shape[1]
        xy = np.linspace(-maxXY, maxXY, XY + 1)[:-1]
        X,Y = np.meshgrid(xy,xy)
        for i in range(2):
            fig, ax = plt.subplots(dpi=150,subplot_kw={"projection": "3d"})
            surf = ax.plot_surface(X, Y, (np.mean(np.abs(sol[...,-1,i])**2,axis=0)), cmap=cm.coolwarm,linewidth=0, antialiased=False)
            fig.colorbar(surf, shrink=0.5, aspect=5)
            plt.title(f"{dict[i]}-{src}")
            plt.savefig(f"tmp_fig/{dict[i]}-{src}.jpg")

def plot_singel_sol(u,y,j):

    N,nx,ny,nz,nfields = u.shape
    u = u.reshape(N,nx, ny, nz,2,nfields//2)
    y = y.reshape(N,nx, ny, nz,2,nfields//2)
    u = (u[...,0,:] + 1j*u[...,1,:]).detach().numpy()
    y = (y[...,0,:] + 1j*y[...,1,:]).detach().numpy()
    for sol,src in zip([u,y],["prediction", "grt"]):
        dict = {0:"signal vac", 1:"idler vac", 2:"single out", 3:"idler out"}
        maxXY = 120e-6
        XY = u.shape(1)
        xy = np.linspace(-maxXY, maxXY, XY + 1)[:-1]
        X,Y = np.meshgrid(xy,xy)
        for i in range(4):
            fig, ax = plt.subplots(dpi=150,subplot_kw={"projection": "3d"})
            surf = ax.plot_surface(X, Y, np.real(sol[j,...,-1,i]), cmap=cm.coolwarm,linewidth=0, antialiased=False)
            fig.colorbar(surf, shrink=0.5, aspect=5)
            plt.title(f"{dict[i]}-{src}")
            plt.savefig(f"tmp_fig/{dict[i]}-{src}-real.jpg")
            fig, ax = plt.subplots(dpi=150,subplot_kw={"projection": "3d"})
            surf = ax.plot_surface(X, Y, np.imag(sol[j,...,-1,i]), cmap=cm.coolwarm,linewidth=0, antialiased=False)
            fig.colorbar(surf, shrink=0.5, aspect=5)
            plt.title(f"{dict[i]}-{src}")
            plt.savefig(f"tmp_fig/{dict[i]}-{src}-imag.jpg")

def draw_SPDC(model,
                 dataloader,
                 config,
                 equation_dict,
                 device,
                 padding = 0,
                 use_tqdm=True):
    model.eval()
    nout = config['data']['nout']
    if use_tqdm:
        pbar = tqdm(dataloader, dynamic_ncols=True, smoothing=0.05)
    else:
        pbar = dataloader

    total_out = torch.tensor([])
    total_y = torch.tensor([])
    for x, y in pbar:
        gc.collect()
        torch.cuda.empty_cache()
        x, y = x.to(device), y.to(device)
        x_in = F.pad(x,(0,0,0,padding),"constant",0)
        out = model(x_in).reshape(dataloader.batch_size,y.size(1),y.size(2),y.size(3) + padding, 2*nout)
            # out = out[...,:-padding,:, :] # if padding is not 0
        total_out = torch.cat((total_out,out.to("cpu")),dim=0)
        total_y = torch.cat((total_y,y.to("cpu")),dim=0)
    plot_av_sol(total_out,total_y)
    # plot_singel_sol(total_out,total_y,1)



def run(args, config):
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
    draw_SPDC(model=model,dataloader=dataloader, config=config, equation_dict=equation_dict, device=device)


if __name__ == '__main__':
    parser = ArgumentParser(description='Basic paser')
    parser.add_argument('--config_path', type=str, help='Path to the configuration file')
    args = parser.parse_args()

    config_file = args.config_path
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)
        run(args, config)
