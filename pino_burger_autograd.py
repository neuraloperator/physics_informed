import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from models import PINO2d

from tqdm import tqdm
from timeit import default_timer
from losses import LpLoss, AD_loss
from data_utils import BurgersLoader
from utils import get_grid, count_params, save_checkpoint

try:
    import wandb
except ImportError:
    wandb = None

torch.manual_seed(0)
np.random.seed(0)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

ntrain = 1000
ntest = 100

sub = 2 #subsampling rate
h = 128 // sub
s = h
sub_t = 2
T = 100 // sub_t + 1

batch_size = 20
learning_rate = 0.001

epochs = 1500
step_size = 200
gamma = 0.5

modes = 12
width = 32

log = True
if wandb and log:
    wandb.init(project='PINO-Burgers',
               entity='hzzheng-pino',
               group='AD',
               config={'lr': learning_rate,
                       'schedule_step': step_size,
                       'batch_size': batch_size,
                       'modes': modes,
                       'width': width,
                       'sub_x': sub,
                       'sub_t': sub_t},
               tags=['full grid'])

datapath = '/mnt/md1/zongyi/burgers_pino.mat'
constructor = BurgersLoader(datapath, nx=128, nt=100, sub=sub, sub_t=sub_t, new=True)
train_loader = constructor.make_loader(n_sample=ntrain, batch_size=batch_size, train=True)
test_loader = constructor.make_loader(n_sample=ntest, batch_size=batch_size, train=False)

image_dir = 'figs/AD-burgers'
ckpt_dir = 'AD-burgers'
name = 'PINO_autograd_burgers_N'+str(ntrain)+'_ep' + str(epochs) + '_m' + str(modes) + '_w' + str(width) + '.pt'


if not os.path.exists(image_dir):
    os.makedirs(image_dir)
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

layers = [width*2//4, width*3//4, width*3//4, width*4//4, width*4//4]
modes = [modes * (5-i) // 4 for i in range(4)]

model = PINO2d(modes1=modes, modes2=modes, width=width, layers=layers).to(device)
num_param = count_params(model)
print('Number of model parameters', num_param)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 400, 600, 800, 1000], gamma=gamma)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

myloss = LpLoss(size_average=True)
# myloss = HpLoss(size_average=False, k=2, group=True)

pbar = tqdm(range(epochs), dynamic_ncols=True, smoothing=0.01)

for ep in pbar:
    model.train()
    t1 = default_timer()
    train_pino = 0.0
    train_l2 = 0.0
    train_loss = 0.0

    # train with ground truth
    # N = 10
    # ux, uy = x_train[:N].to(device), y_train[:N].to(device)
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        grid, gridt, gridx = get_grid(batch_size, T, s)
        # p = 50
        # q = 400
        # P = p+p+q
        # sample, sample_t, sample_x, index_ic = get_sample(batch_size, T, s, p=p, q=q)

        optimizer.zero_grad()
        out = model(x, grid)

        pred = model(x)
        loss_u = myloss(pred.view(batch_size, -1), y.view(batch_size, -1))
        # uout = model(ux)
        # loss_u = myloss(uout.view(N, T, s), uy.view(N, T, s))
        # loss_ic, loss_f = AD_loss(out.view(batch_size, P), x[:, 0, :, 0], (sample_t, sample_x), index_ic, p, q)
        loss_ic, loss_f = AD_loss(out.view(batch_size, T, s), x[:, 0, :, 0], (gridt, gridx))
        total_loss = (20*loss_ic + loss_f) * 100
        total_loss.backward()

        optimizer.step()
        # loss = myloss(out.view(batch_size,T,s), y.view(batch_size,T,s))
        train_l2 += loss_u.item()
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

            test_l2 += myloss(out.view(batch_size, T, s), y.view(batch_size, T, s)).item()
            # test_pino += PINO_loss(out.view(batch_size,T,s), x[:, 0, :, 0], (gridt, gridx)).item()

    if ep % step_size == 0:
        plt.imsave('%s/y_%d.png' % (image_dir, ep), y[0, :, :].cpu().numpy())
        plt.imsave('%s/out_%d.png' % (image_dir, ep), out[0, :, :].cpu().numpy())

    train_l2 /= len(train_loader)
    test_l2 /= len(test_loader)
    train_pino /= len(train_loader)
    test_pino /= len(test_loader)
    train_loss /= len(train_loader)


    t2 = default_timer()
    pbar.set_description(
        (
            f'Time cost: {t2 - t1:.2f}; Train f error: {train_pino:.5f}; Train l2 error: {train_l2:.5f}. '
            f'Train loss: {train_loss:.5f}; Test l2 error: {test_l2:.5f}'
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
