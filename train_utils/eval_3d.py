import torch
import torch.nn.functional as F

from tqdm import tqdm
from timeit import default_timer

from .losses import LpLoss, PINO_loss3d

try:
    import wandb
except ImportError:
    wandb = None


def eval_ns(model,  # model
            loader,  # dataset instance
            dataloader,  # dataloader
            forcing,  # forcing
            config,  # configuration dict
            device,  # device id
            log=False,
            project='PINO-default',
            group='FDM',
            tags=['Nan'],
            use_tqdm=True):
    '''
    Evaluate the model for Navier Stokes equation
    '''
    if wandb and log:
        run = wandb.init(project=project,
                         entity='hzzheng-pino',
                         group=group,
                         config=config,
                         tags=tags, reinit=True,
                         settings=wandb.Settings(start_method="fork"))
    # data parameters
    v = 1 / config['data']['Re']
    S, T = loader.S, loader.T
    t_interval = config['data']['time_interval']
    # eval settings
    batch_size = config['test']['batchsize']

    model.eval()
    myloss = LpLoss(size_average=True)
    if use_tqdm:
        pbar = tqdm(dataloader, dynamic_ncols=True, smoothing=0.05)
    else:
        pbar = dataloader
    loss_dict = {'train_f': 0.0,
                 'test_l2': 0.0}
    start_time = default_timer()
    with torch.no_grad():
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            x_in = F.pad(x, (0, 0, 0, 5), "constant", 0)
            out = model(x_in).reshape(batch_size, S, S, T + 5)
            out = out[..., :-5]
            x = x[:, :, :, 0, -1]
            loss_l2 = myloss(out.view(batch_size, S, S, T), y.view(batch_size, S, S, T))
            loss_ic, loss_f = PINO_loss3d(out.view(batch_size, S, S, T), x, forcing, v, t_interval)

            loss_dict['train_f'] += loss_f
            loss_dict['test_l2'] += loss_l2
            if device == 0 and use_tqdm:
                pbar.set_description(
                    (
                        f'Train f error: {loss_f.item():.5f}; Test l2 error: {loss_l2.item():.5f}'
                    )
                )
    end_time = default_timer()
    test_l2 = loss_dict['test_l2'].item() / len(dataloader)
    loss_f = loss_dict['train_f'].item() / len(dataloader)
    print(f'==Averaged relative L2 error is: {test_l2}==\n'
          f'==Averaged equation error is: {loss_f}==')
    print(f'Time cost: {end_time - start_time} s')
    if device == 0:
        if wandb and log:
            wandb.log(
                {
                    'Train f error': loss_f,
                    'Test L2 error': test_l2,
                }
            )
            run.finish()
