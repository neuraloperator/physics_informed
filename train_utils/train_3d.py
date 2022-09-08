import torch
from tqdm import tqdm
from timeit import default_timer
import torch.nn.functional as F
from .utils import save_checkpoint
from .losses import LpLoss, PINO_loss3d, get_forcing
from .distributed import reduce_loss_dict
from .data_utils import sample_data

try:
    import wandb
except ImportError:
    wandb = None
    

def train(model,
          loader, train_loader,
          optimizer, scheduler,
          forcing, config,
          rank=0,
          log=False,
          project='PINO-default',
          group='FDM',
          tags=['Nan'],
          use_tqdm=True,
          profile=False):
    if rank == 0 and wandb and log:
        run = wandb.init(project=project,
                         entity=config['log']['entity'],
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
    zero = torch.zeros(1).to(rank)

    for ep in pbar:
        loss_dict = {'train_loss': 0.0,
                     'train_ic': 0.0,
                     'train_f': 0.0,
                     'test_l2': 0.0}
        log_dict = {}
        if rank == 0 and profile:
                torch.cuda.synchronize()
                t1 = default_timer()
        # start solving
        for x, y in train_loader:
            x, y = x.to(rank), y.to(rank)

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
            loss_dict['train_ic'] += loss_ic
            loss_dict['test_l2'] += loss_l2
            loss_dict['train_loss'] += total_loss
            loss_dict['train_f'] += loss_f

        if rank == 0 and profile:
            torch.cuda.synchronize()
            t2 = default_timer()
            log_dict['Time cost'] = t2 - t1
        scheduler.step()
        loss_reduced = reduce_loss_dict(loss_dict)
        train_ic = loss_reduced['train_ic'].item() / len(train_loader)
        train_f = loss_reduced['train_f'].item() / len(train_loader)
        train_loss = loss_reduced['train_loss'].item() / len(train_loader)
        test_l2 = loss_reduced['test_l2'].item() / len(train_loader)
        log_dict = {
            'Train f error': train_f,
            'Train L2 error': train_ic,
            'Train loss': train_loss,
            'Test L2 error': test_l2
            }

        if rank == 0:
            if use_tqdm:
                pbar.set_description(
                    (
                        f'Train f error: {train_f:.5f}; Train ic l2 error: {train_ic:.5f}. '
                        f'Train loss: {train_loss:.5f}; Test l2 error: {test_l2:.5f}'
                    )
                )
            if wandb and log:
                wandb.log(log_dict)

    if rank == 0:
        save_checkpoint(config['train']['save_dir'],
                        config['train']['save_name'],
                        model, optimizer)
        if wandb and log:
            run.finish()


def mixed_train(model,              # model of neural operator
                train_loader,       # dataloader for training with data
                S1, T1,             # spacial and time dimension for training with data
                a_loader,           # generator for  ICs
                S2, T2,             # spacial and time dimension for training with equation only
                optimizer,          # optimizer
                scheduler,          # learning rate scheduler
                config,             # configuration dict
                device=torch.device('cpu'),
                log=False,          # turn on the wandb
                project='PINO-default', # project name
                group='FDM',        # group name
                tags=['Nan'],       # tags
                use_tqdm=True):     # turn on tqdm
    if wandb and log:
        run = wandb.init(project=project,
                         entity=config['log']['entity'],
                         group=group,
                         config=config,
                         tags=tags, reinit=True,
                         settings=wandb.Settings(start_method="fork"))

    # data parameters
    v = 1 / config['data']['Re']
    t_interval = config['data']['time_interval']
    forcing_1 = get_forcing(S1).to(device)
    forcing_2 = get_forcing(S2).to(device)
    # training settings
    batch_size = config['train']['batchsize']
    ic_weight = config['train']['ic_loss']
    f_weight = config['train']['f_loss']
    xy_weight = config['train']['xy_loss']
    num_data_iter = config['train']['data_iter']
    num_eqn_iter = config['train']['eqn_iter']

    model.train()
    myloss = LpLoss(size_average=True)
    pbar = range(config['train']['epochs'])
    if use_tqdm:
        pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.05)
    zero = torch.zeros(1).to(device)
    train_loader = sample_data(train_loader)
    for ep in pbar:
        model.train()
        t1 = default_timer()
        train_loss = 0.0
        train_ic = 0.0
        train_f = 0.0
        test_l2 = 0.0
        err_eqn = 0.0
        # train with data
        for _ in range(num_data_iter):
            x, y = next(train_loader)
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            x_in = F.pad(x, (0, 0, 0, 5), "constant", 0)
            out = model(x_in).reshape(batch_size, S1, S1, T1 + 5)
            out = out[..., :-5]
            x = x[:, :, :, 0, -1]

            loss_l2 = myloss(out.view(batch_size, S1, S1, T1),
                             y.view(batch_size, S1, S1, T1))

            if ic_weight != 0 or f_weight != 0:
                loss_ic, loss_f = PINO_loss3d(out.view(batch_size, S1, S1, T1),
                                              x, forcing_1,
                                              v, t_interval)
            else:
                loss_ic, loss_f = zero, zero

            total_loss = loss_l2 * xy_weight + loss_f * f_weight + loss_ic * ic_weight

            total_loss.backward()
            optimizer.step()

            train_ic = loss_ic.item()
            test_l2 += loss_l2.item()
            train_loss += total_loss.item()
            train_f += loss_f.item()
        if num_data_iter != 0:
            train_ic /= num_data_iter
            train_f /= num_data_iter
            train_loss /= num_data_iter
            test_l2 /= num_data_iter
        # train with random ICs
        for _ in range(num_eqn_iter):
            new_a = next(a_loader)
            new_a = new_a.to(device)
            optimizer.zero_grad()
            x_in = F.pad(new_a, (0, 0, 0, 5), "constant", 0)
            out = model(x_in).reshape(batch_size, S2, S2, T2 + 5)
            out = out[..., :-5]
            new_a = new_a[:, :, :, 0, -1]
            loss_ic, loss_f = PINO_loss3d(out.view(batch_size, S2, S2, T2),
                                          new_a, forcing_2,
                                          v, t_interval)
            eqn_loss = loss_f * f_weight + loss_ic * ic_weight
            eqn_loss.backward()
            optimizer.step()

            err_eqn += eqn_loss.item()

        scheduler.step()
        t2 = default_timer()
        if num_eqn_iter != 0:
            err_eqn /= num_eqn_iter
        if use_tqdm:
            pbar.set_description(
                (
                    f'Data f error: {train_f:.5f}; Data ic l2 error: {train_ic:.5f}. '
                    f'Data train loss: {train_loss:.5f}; Data l2 error: {test_l2:.5f}'
                    f'Eqn loss: {err_eqn:.5f}'
                )
            )
        if wandb and log:
            wandb.log(
                {
                    'Data f error': train_f,
                    'Data IC L2 error': train_ic,
                    'Data train loss': train_loss,
                    'Data L2 error': test_l2,
                    'Random IC Train equation loss': err_eqn,
                    'Time cost': t2 - t1
                }
            )

    save_checkpoint(config['train']['save_dir'],
                    config['train']['save_name'],
                    model, optimizer)
    if wandb and log:
        run.finish()


def progressive_train(model,
                      loader, train_loader,
                      optimizer, scheduler,
                      milestones, config,
                      device=torch.device('cpu'),
                      log=False,
                      project='PINO-default',
                      group='FDM',
                      tags=['Nan'],
                      use_tqdm=True):
    if wandb and log:
        run = wandb.init(project=project,
                         entity=config['log']['entity'],
                         group=group,
                         config=config,
                         tags=tags, reinit=True,
                         settings=wandb.Settings(start_method="fork"))

    # data parameters
    v = 1 / config['data']['Re']
    T = loader.T
    t_interval = config['data']['time_interval']

    # training settings
    batch_size = config['train']['batchsize']
    ic_weight = config['train']['ic_loss']
    f_weight = config['train']['f_loss']
    xy_weight = config['train']['xy_loss']

    model.train()
    myloss = LpLoss(size_average=True)
    zero = torch.zeros(1).to(device)
    for milestone, epochs in zip(milestones, config['train']['epochs']):
        pbar = range(epochs)
        if use_tqdm:
            pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.05)
        S = loader.S // milestone
        print(f'Resolution :{S}')
        forcing = get_forcing(S).to(device)
        for ep in pbar:
            model.train()
            t1 = default_timer()
            train_loss = 0.0
            train_ic = 0.0
            train_f = 0.0
            test_l2 = 0.0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                x = x[:, ::milestone, ::milestone, :, :]
                y = y[:, ::milestone, ::milestone, :]
                optimizer.zero_grad()
                x_in = F.pad(x, (0, 0, 0, 5), "constant", 0)
                out = model(x_in).reshape(batch_size, S, S, T + 5)
                out = out[..., :-5]
                x = x[:, :, :, 0, -1]

                loss_l2 = myloss(out.view(batch_size, S, S, T), y.view(batch_size, S, S, T))

                if ic_weight != 0 or f_weight != 0:
                    loss_ic, loss_f = PINO_loss3d(out.view(batch_size, S, S, T),
                                                  x, forcing, v, t_interval)
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


