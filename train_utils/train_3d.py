import torch
from tqdm import tqdm
from timeit import default_timer
import torch.nn.functional as F
from train_utils.utils import save_checkpoint
from train_utils.losses import LpLoss, PINO_loss3d

try:
    import wandb
except ImportError:
    wandb = None


def train(model,
          loader, train_loader,
          optimizer, scheduler,
          forcing, config,
          device=torch.device('cpu'),
          log=False,
          project='PINO-default',
          group='FDM',
          tags=['Nan'],
          use_tqdm=True):
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

    # training settings
    batch_size = config['train']['batchsize']
    ic_weight = config['train']['ic_loss']
    f_weight = config['train']['f_loss']
    xy_weight = config['train']['xy_loss']

    model.train()
    myloss = LpLoss(size_average=True)
    pbar = range(config['train']['epochs'])
    if use_tqdm:
        pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.05)
    zero = torch.zeros(1).to(device)
    for ep in pbar:
        model.train()
        t1 = default_timer()
        train_loss = 0.0
        train_ic = 0.0
        train_f = 0.0
        test_l2 = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

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
        if use_tqdm:
            pbar.set_description(
                (
                    f'Train f error: {train_f:.5f}; Train ic l2 error: {train_ic:.5f}. '
                    f'Train loss: {train_loss:.5f}; Test l2 error: {test_l2:.5f}'
                )
            )
        if wandb and log:
            wandb.log(
                {
                    'Train f error': train_f,
                    'Train L2 error': train_ic,
                    'Train loss': train_loss,
                    'Test L2 error': test_l2,
                    'Time cost': t2 - t1
                }
            )

    save_checkpoint(config['train']['save_dir'],
                    config['train']['save_name'],
                    model, optimizer)
    if wandb and log:
        run.finish()
