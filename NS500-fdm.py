import os
import numpy as np

import torch
import torch.nn.functional as F

from models import FNN3d

from tqdm import tqdm
from timeit import default_timer
from train_utils.utils import count_params, save_checkpoint
from train_utils.data_utils import NS500Loader
from train_utils.losses import LpLoss, PINO_loss3d
try:
    import wandb
except ImportError:
    wandb = None


torch.manual_seed(2022)
Ntrain = 500
Ntest = 1
ntrain = Ntrain
ntest = Ntest
v = 1/500

modes = 20
width = 32

batch_size = 2
epochs = 1000
learning_rate = 0.0025
scheduler_gamma = 0.5

image_dir = 'figs/Re500'
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

ckpt_dir = 'Re500-FDM'

name = 'PINO_FDM_Re500_T500.pt'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

log = True

if wandb and log:
    wandb.init(project='PINO-NS500-operator-wy',
               entity='hzzheng-pino',
               group='FDM',
               config={'lr': learning_rate,
                       'batch_size': batch_size,
                       'scheduler_gamma': scheduler_gamma,
                       'modes': modes,
                       'width': width},
               tags=['f loss * 2'])


sub = 1
nx = 64
nt = 128
S = nx // sub
# T_in = 1
sub_t = 1
T = nt // sub_t + 1
# datapath = '/mnt/md1/visiondatasets/PINO-data/NS_fine_Re40_s64_T1000.npy'
datapath = 'data/NS_fine_Re500_S2048_s64_T500_t128.npy'
data = np.load(datapath)
loader = NS500Loader(datapath, nx=nx, nt=nt, sub=sub, sub_t=sub_t, N=500)
train_loader = loader.make_loader(ntrain, batch_size=batch_size, train=True)
# test_loader = loader.make_loader(ntest, batch_size=batch_size, train=False)
# train_loader = test_loader

# model_ckpt = 'checkpoints/Re500-FDM/PINO_FDM_Re500_T500.pt'
model_ckpt = None
layers = [width*4//4, width * 4 // 4, width*4//4, width * 4 // 4, width*4//4]
modes = [modes, modes, modes, modes]

model = FNN3d(modes1=modes, modes2=modes, modes3=modes, fc_dim=256, layers=layers).to(device)
num_param = count_params(model)
print('Number of model parameters', num_param)

if model_ckpt is not None:
    ckpt_state = torch.load(model_ckpt)
    model.load_state_dict(ckpt_state['model'])

optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=learning_rate)
milestones = [100, 150, 200, 300, 400, 500]
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=scheduler_gamma)
# scheduler = OneCycleLR(optimizer, max_lr=0.005, steps_per_epoch=len(train_loader), epochs=epochs)

x1 = torch.tensor(np.linspace(0, 2*np.pi, S+1)[:-1], dtype=torch.float).reshape(S, 1).repeat(1, S)
x2 = torch.tensor(np.linspace(0, 2*np.pi, S+1)[:-1], dtype=torch.float).reshape(1, S).repeat(S, 1)

forcing = -4 * (torch.cos(4*(x2))).reshape(1,S,S,1).to(device)

myloss = LpLoss(size_average=True)
pbar = tqdm(range(epochs), dynamic_ncols=True, smoothing=0.01)

for ep in pbar:
    model.train()
    t1 = default_timer()
    train_loss = 0.0
    train_l2 = 0.0
    train_f = 0.0
    test_l2 = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        x_in = F.pad(x, (0, 0, 0, 5), "constant", 0)
        out = model(x_in).reshape(batch_size, S, S, T+5)
        out = out[..., :-5]
        x = x[:, :, :, 0, -1]

        loss_l2 = myloss(out.view(batch_size, S, S, T), y.view(batch_size, S, S, T))
        loss_ic, loss_f = PINO_loss3d(out.view(batch_size, S, S, T), x, forcing, v)
        total_loss = loss_l2 + loss_f

        total_loss.backward()

        optimizer.step()
        train_l2 = loss_ic.item()
        test_l2 += loss_l2.item()
        train_loss += total_loss.item()
        train_f += loss_f.item()
        scheduler.step()
    train_l2 /= len(train_loader)
    train_f /= len(train_loader)
    train_loss /= len(train_loader)
    test_l2 /= len(train_loader)
    t2 = default_timer()

    pbar.set_description(
        (
            f'Train f error: {train_f:.5f}; Train l2 error: {train_l2:.5f}. '
            f'Train loss: {train_loss:.5f}; Test l2 error: {test_l2:.5f}'
        )
    )
    if wandb and log:
        wandb.log(
            {
                'Train f error': train_f,
                'Train L2 error': train_l2,
                'Train loss': train_loss,
                'Test L2 error': test_l2,
                'Time cost': t2 - t1
            }
        )

save_checkpoint(ckpt_dir, name, model, optimizer)
