from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader
from baselines.model import DeepONetCP
from baselines.data import DeepONetCPNS, DarcyFlow
from train_utils.losses import LpLoss


def test(model,
         test_loader,
         grid,
         device):
    pbar = tqdm(test_loader, dynamic_ncols=True, smoothing=0.1)
    myloss = LpLoss(size_average=True)
    model.eval()

    test_error = []
    with torch.no_grad():
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)

            grid = grid.to(device)

            pred = model(x, grid)
            loss = myloss(pred, y)

            test_error.append(loss.item())
            pbar.set_description(
                (
                    f'test error: {loss.item():.5f}'
                )
            )

    mean = np.mean(test_error)
    std = np.std(test_error, ddof=1) / np.sqrt(len(test_error))
    print(f'Averaged test error :{mean}, standard error: {std}')


def test_deeponet_ns(config):
    '''
    Evaluate deeponet model on Navier Stokes equation
    Args:
        config: configurations

    Returns:

    '''
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_config = config['data']
    batch_size = config['test']['batchsize']
    dataset = DeepONetCPNS(datapath=data_config['datapath'],
                           nx=data_config['nx'], nt=data_config['nt'],
                           sub=data_config['sub'], sub_t=data_config['sub_t'],
                           offset=data_config['offset'], num=data_config['n_sample'],
                           t_interval=data_config['time_interval'])
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    u0_dim = dataset.S ** 2
    model = DeepONetCP(branch_layer=[u0_dim] + config['model']['branch_layers'],
                       trunk_layer=[3] + config['model']['trunk_layers']).to(device)
    if 'ckpt' in config['test']:
        ckpt = torch.load(config['test']['ckpt'])
        model.load_state_dict(ckpt['model'])
    grid = test_loader.dataset.xyt
    test(model, test_loader, grid, device=device)


def test_deeponet_darcy(config):
    '''
    Evaluate deeponet mode on Darcy Flow
    '''

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_config = config['data']
    batch_size = config['test']['batchsize']
    dataset = DarcyFlow(data_config['datapath'],
                        nx=data_config['nx'], sub=data_config['sub'],
                        offset=data_config['offset'], num=data_config['n_sample'])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    u0_dim = dataset.S ** 2
    model = DeepONetCP(branch_layer=[u0_dim] + config['model']['branch_layers'],
                       trunk_layer=[2] + config['model']['trunk_layers']).to(device)
    if 'ckpt' in config['test']:
        ckpt = torch.load(config['test']['ckpt'])
        model.load_state_dict(ckpt['model'])
        print('Load model weights from %s' % config['test']['ckpt'])

    grid = dataset.mesh.reshape(-1, 2)
    test(model, dataloader, grid, device)