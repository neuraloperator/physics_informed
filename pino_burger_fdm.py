import os
import numpy as np
import matplotlib.pyplot as plt

import torch

from models import FNN2d

from tqdm import tqdm
from timeit import default_timer
from utils import count_params, save_checkpoint
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
ntest = 100  #200

sub = 1  #8  # subsampling rate
# h = 2**10 // sub
# s = h
sub_t = 1
# T = 100 // sub_t

batch_size = 20  # 100
learning_rate = 0.001

epochs = 2500
step_size = 100
gamma = 0.25

modes = 12  # 20
width = 32  # 64

# datapath = '/mnt/md1/zongyi/burgers_v100_t100_r1024_N2048.mat'

datapath = '/mnt/md1/zongyi/burgers_pino.mat'
log = True

if wandb and log:
    wandb.init(project='PINO-burgers',
               entity='hzzheng-pino',
               group='FDM',
               config={'lr': learning_rate,
                       'schedule_step': step_size,
                       'batch_size': batch_size,
                       'modes': modes,
                       'width': width})

constructor = DataConstructor(datapath, nx=128, nt=100, sub=sub, sub_t=sub_t, new=True)
train_loader = constructor.make_loader(n_sample=ntrain, batch_size=batch_size, train=True)
test_loader = constructor.make_loader(n_sample=ntest, batch_size=batch_size, train=False)

image_dir = 'figs/FDM-burgers'
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

ckpt_dir = 'Burgers-FDM'

name = 'PINO_FDM_burgers_N' + \
    str(ntrain)+'_ep' + str(epochs) + '_m' + str(modes) + '_w' + str(width) + '.pt'

layers = [width*2//4, width*3//4, width*3//4, width*4//4, width*4//4]
modes = [modes * (5-i) // 4 for i in range(4)]

model = FNN2d(modes1=modes, modes2=modes, width=width, layers=layers).to(device)
num_param = count_params(model)
print('Number of model parameters', num_param)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[800, 1000, 2000], gamma=gamma)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

myloss = LpLoss(size_average=True)
pbar = tqdm(range(epochs), dynamic_ncols=True, smoothing=0.01)

for ep in pbar:
    model.train()
    t1 = default_timer()
    train_pino = 0.0
    train_l2 = 0.0
    train_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        out = model(x)
        # out = y_normalizer.decode(out)
        # y = y_normalizer.decode(y)

        loss = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
        loss_u, loss_f = PINO_loss(out, x[:, 0, :, 0])
        total_loss = (loss_u * 10 + loss_f) * 100
        total_loss.backward()
        optimizer.step()

        train_l2 += loss.item()
        train_pino += loss_f.item()
        train_loss += total_loss.item()

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
        plt.imsave('%s/y_%d.png' % (image_dir, ep), y[0, :, :].cpu().numpy())
        plt.imsave('%s/out_%d.png' % (image_dir, ep), out[0, :, :, 0].cpu().numpy())

    train_l2 /= ntrain
    test_l2 /= ntest
    train_pino /= len(train_loader)
    test_pino /= len(test_loader)
    train_loss /= len(train_loader)

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
                'Train loss': train_loss,
                'Test f error': test_pino,
                'Test L2 error': test_l2,
                'Time cost': t2 - t1
            }
        )

save_checkpoint(ckpt_dir, name, model, optimizer)
