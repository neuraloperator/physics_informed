import yaml
import torch
import torch.nn.functional as F

from tqdm import tqdm
from timeit import default_timer

from train_utils import Adam
from train_utils.datasets import NSLoader
from train_utils.utils import save_checkpoint
from train_utils.losses import LpLoss, PINO_loss3d, get_forcing
from train_utils.train_3d import train
from models.trial import FNet
from argparse import ArgumentParser


def make_loader(path, n_sample=1, offset=0, batch_size=1, shuffle=False):
    data = torch.load(path)
    pred = data['pred'][offset: offset + n_sample]
    dataset = torch.utils.data.TensorDataset(pred)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


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
    print('Data loaded')
    train_loader = loader.make_loader(data_config['n_sample'],
                                      batch_size=config['train']['batchsize'],
                                      start=data_config['offset'],
                                      train=data_config['shuffle'])
    a_loader = make_loader('data/Re500-pretrained-part2.pt',
                           data_config['n_sample'],
                           offset=data_config['offset'])
    model = FNet(mode1=config['model']['modes1'][0],
                 mode2=config['model']['modes2'][0],
                 mode3=config['model']['modes3'][0]).to(device)
    if 'ckpt' in config['train']:
        ckpt_path = config['train']['ckpt']
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        print('Weights loaded from %s' % ckpt_path)
    # optimizer = SGD(model.parameters(),
    #                 lr=config['train']['base_lr'],
    #                 momentum=0.0)
    optimizer = Adam(model.parameters(), betas=(0.9, 0.999),
                     lr=config['train']['base_lr'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=config['train']['milestones'],
                                                     gamma=config['train']['scheduler_gamma'])
    forcing = get_forcing(loader.S).to(device)

    # train(model,
    #       loader, train_loader,
    #       optimizer, scheduler,
    #       forcing, config,
    #       device,
    #       log=options.log,
    #       project=config['others']['project'],
    #       group=config['others']['group'])
    # # data parameters
    v = 1 / config['data']['Re']
    S, T = loader.S, loader.T
    t_interval = config['data']['time_interval']

    # training settings
    batch_size = config['train']['batchsize']
    ic_weight = config['train']['ic_loss']
    f_weight = config['train']['f_loss']
    xy_weight = config['train']['xy_loss']

    model.train()
    myloss = LpLoss(size_average=True)
    pbar = range(config['train']['epochs'])
    pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.05)
    zero = torch.zeros(1).to(device)

    for ep in pbar:
        model.train()
        t1 = default_timer()
        train_loss = 0.0
        train_ic = 0.0
        train_f = 0.0
        test_l2 = 0.0
        for (_, y), pred in zip(train_loader, a_loader):
            y = y.to(device)
            x = torch.unsqueeze(pred[0].to(device), dim=0)

            optimizer.zero_grad()
            x_in = F.pad(x, (0, 0, 0, 5), "constant", 0)
            out = model(x_in).reshape(batch_size, S, S, T + 5)
            out = out[..., :-5]
            x = x[:, :, :, 0, -1]

            loss_l2 = myloss(out.view(batch_size, S, S, T), y.view(batch_size, S, S, T))

            if ic_weight != 0 or f_weight != 0:
                loss_ic, loss_f = PINO_loss3d(out.view(batch_size, S, S, T), x, forcing, v, t_interval)
            else:
                loss_ic, loss_f = zero, zero

            total_loss = loss_l2 * xy_weight + loss_f * f_weight + loss_ic * ic_weight

            total_loss.backward()

            optimizer.step()
            train_ic = loss_ic.item()
            test_l2 += loss_l2.item()
            train_loss += total_loss.item()
            train_f += loss_f.item()
        scheduler.step()

        train_ic /= len(train_loader)
        train_f /= len(train_loader)
        train_loss /= len(train_loader)
        test_l2 /= len(train_loader)
        t2 = default_timer()
        pbar.set_description(
            (
                f'Train f error: {train_f:.5f}; Train ic l2 error: {train_ic:.5f}. '
                f'Train loss: {train_loss:.5f}; Test l2 error: {test_l2:.5f}'
            )
        )
