from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from baselines.model import DeepONet, DeepONetCP
from baselines.data import DeepOnetNS, DeepONetCPNS
from train_utils.losses import LpLoss


def test(model,
         test_loader,
         device):
    pbar = tqdm(test_loader, dynamic_ncols=True, smoothing=0.1)
    myloss = LpLoss(size_average=True)
    model.eval()
    grid = test_loader.dataset.xyt
    test_error = 0.0
    with torch.no_grad():
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)

            grid = grid.to(device)

            pred = model(x, grid)
            loss = myloss(pred, y)

            test_error += loss.item() * y.shape[0]
            pbar.set_description(
                (
                    f'test error: {loss.item():.5f}'
                )
            )
        test_error /= len(test_loader.dataset)
    print(f'Averaged test error :{test_error}')


def test_deeponet_cp(config):
    '''
    Evaluate deeponet model
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
    test(model, test_loader, device=device)
