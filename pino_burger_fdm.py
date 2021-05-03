import os
import numpy as np
import matplotlib.pyplot as plt

import torch

from models import FNN2d

from tqdm import tqdm
from timeit import default_timer
from utils import count_params
from data_utils import DataConstructor
from losses import LpLoss, PINO_loss

try:
    import wandb
except ImportError:
    wandb = None

torch.manual_seed(0)
np.random.seed(0)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

ntrain = 1000
ntest = 200

sub = 8  # subsampling rate
h = 2**10 // sub
s = h
sub_t = 1
T = 100 // sub_t

batch_size = 100
learning_rate = 0.001

epochs = 2500
step_size = 100
gamma = 0.5

modes = 20
width = 64

datapath = '/mnt/md1/zongyi/burgers_v100_t100_r1024_N2048.mat'
log = True

if wandb and log:
    wandb.init(project='PINO-burgers',
               group='FDM',
               config={'lr': learning_rate,
                       'schedule_step': step_size,
                       'batch_size': batch_size,
                       'modes': modes,
                       'width': width})

constructor = DataConstructor(datapath, sub=sub, sub_t=sub_t)
train_loader = constructor.make_loader(n_sample=ntrain, batch_size=batch_size, train=True)
test_loader = constructor.make_loader(n_sample=ntest, batch_size=batch_size, train=False)

if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')
if not os.path.exists('figs'):
    os.makedirs('figs')

path = 'PINO_FDM_burgers_N' + \
    str(ntrain)+'_ep' + str(epochs) + '_m' + str(modes) + '_w' + str(width)
path_model = 'checkpoints/' + path + '.pt'


layers = [width * (2+i) // 4 for i in range(5)]
modes = [modes * (4-i) // 4 for i in range(4)]

model = FNN2d(modes1=modes, modes2=modes, width=width, layers=layers).to(device)
num_param = count_params(model)
print('Number of model parameters', num_param)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs//5, gamma=gamma/2)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

myloss = LpLoss(size_average=True)
pbar = tqdm(range(epochs), dynamic_ncols=True, smoothing=0.01)

for ep in pbar:
    model.train()
    t1 = default_timer()
    train_pino = 0.0
    train_l2 = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        out = model(x)
        # out = y_normalizer.decode(out)
        # y = y_normalizer.decode(y)

        loss = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
        loss_u, loss_f = PINO_loss(out, x[:, 0, :, 0])
        total_loss = loss_u * 10 + loss_f
        total_loss.backward()

        optimizer.step()
        train_l2 += loss.item()
        train_pino += loss_f.item()

    scheduler.step()

    model.eval()
    test_l2 = 0.0
    test_pino = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            out = model(x)
            # out = y_normalizer.decode(out)

            test_l2 += myloss(out.view(batch_size, -1),
                              y.view(batch_size, -1)).item()
            test_u, test_f = PINO_loss(out, x[:, 0, :, 0])
            test_pino = test_f.item()

    if ep % step_size == 0:
        plt.imsave('figs/y_%d.png' % ep, y[0, :, :].cpu().numpy())
        plt.imsave('figs/out_%d.png' % ep, out[0, :, :, 0].cpu().numpy())

    train_l2 /= ntrain
    test_l2 /= ntest
    train_pino /= len(train_loader)
    test_pino /= len(test_loader)

    t2 = default_timer()
    pbar.set_description(
        (
            f'Time cost: {t2- t1:.2f}; Train f error: {train_pino:.5f}; Train l2 error: {train_l2:.5f}. '
            f'Test f error: {test_pino:.5f}; Test l2 error: {test_l2:.5f}'
        )
    )
    if wandb and log:
        wandb.log(
            {
                'Train f error': train_pino,
                'Train L2 error': train_l2,
                'Test f error': test_pino,
                'Test L2 error': test_l2,
                'Time cost': t2 - t1
            }
        )

torch.save(model, path_model)
