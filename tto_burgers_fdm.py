import os
import numpy as np

import torch

from models import FNN2d

from tqdm import tqdm
from timeit import default_timer
from train_utils.utils import count_params, save_checkpoint
from train_utils.datasets import BurgersLoader
from train_utils.losses import LpLoss, PINO_loss

try:
    import wandb
except ImportError:
    wandb = None

# train index: 1009
# test index: 1009
# 0.1%
torch.manual_seed(0)
np.random.seed(0)

sub = 1  #8  # subsampling rate
# h = 2**10 // sub
# s = h
sub_t = 1
# T = 100 // sub_t

batch_size = 1  # 100
learning_rate = 0.001

epochs = 5000
step_size = 100
gamma = 0.25

modes = 12  # 20
width = 16  # 64

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# datapath = 'data/burgers_pino.mat'
datapath = '/mnt/md1/zongyi/burgers_pino.mat'
log = True

if wandb and log:
    wandb.init(project='PINO-burgers-tto',
               entity='hzzheng-pino',
               group='FDM',
               config={'lr': learning_rate,
                       'schedule_step': step_size,
                       'batch_size': batch_size,
                       'modes': modes,
                       'width': width},
               tags=['Single instance'])

constructor = BurgersLoader(datapath, nx=128, nt=100, sub=sub, sub_t=sub_t, new=True)
dataloader = constructor.make_loader(n_sample=1, batch_size=1, start=1009, train=False)

image_dir = 'figs/FDM-burgers'
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

ckpt_dir = 'Burgers-FDM'

name = 'PINO_FDM_burgers_N' + '_ep' + str(epochs) + '_m' + str(modes) + '_w' + str(width) + '.pt'

layers = [width*2//4, width*3//4, width*3//4, width*4//4, width*4//4]
modes = [modes * (5-i) // 4 for i in range(4)]

model = FNN2d(modes1=modes, modes2=modes, widths=width, layers=layers).to(device)
num_param = count_params(model)
print('Number of model parameters', num_param)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
milestones = [i * 1000 for i in range(1, 5)]
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

myloss = LpLoss(size_average=True)
pbar = tqdm(range(epochs), dynamic_ncols=True, smoothing=0.01)

for ep in pbar:
    model.train()
    t1 = default_timer()
    train_pino = 0.0
    train_l2 = 0.0
    train_loss = 0.0
    test_l2 = 0.0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)

        loss = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
        loss_u, loss_f = PINO_loss(out, x[:, 0, :, 0])
        total_loss = (loss_u * 20 + loss_f) * 100
        total_loss.backward()
        optimizer.step()

        train_l2 += loss_u.item()
        test_l2 += loss.item()
        train_pino += loss_f.item()
        train_loss += total_loss.item()

    scheduler.step()

    # if ep % step_size == 0:
    #     plt.imsave('%s/y_%d.png' % (image_dir, ep), y[0, :, :].cpu().numpy())
    #     plt.imsave('%s/out_%d.png' % (image_dir, ep), out[0, :, :, 0].cpu().numpy())

    t2 = default_timer()
    pbar.set_description(
        (
            f'Time cost: {t2- t1:.2f}; Train f error: {train_pino:.5f}; Train l2 error: {train_l2:.5f}. '
            f'Test l2 error: {test_l2:.5f}'
        )
    )
    if wandb and log:
        wandb.log(
            {
                'Train f error': train_pino,
                'Train L2 error': train_l2,
                'Train loss': train_loss,
                'Test L2 error': test_l2,
                'Time cost': t2 - t1
            }
        )

save_checkpoint(ckpt_dir, name, model, optimizer)


# 80 pretrain, 100 epoch
# 100
# 6401 x 256 x 256 x 128