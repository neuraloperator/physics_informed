import os
import numpy as np
import matplotlib.pyplot as plt

import torch

from models import FNN2d

from tqdm import tqdm
from timeit import default_timer
from utils import count_params, save_checkpoint
from data_utils import DataConstructor, sample_data
from losses import LpLoss, PINO_loss

Ntrain = 999
Ntest = 1
ntrain = Ntrain
ntest = Ntest

modes = 12
width = 32

batch_size = 1
batch_size2 = batch_size


epochs = 5000
learning_rate = 0.002
scheduler_step = 500
scheduler_gamma = 0.5


sub = 1
S = 64 // sub
T_in = 1
sub_t = 1
T = 64 // sub_t + 1
datapath = 'data/NS_fine_Re40_s64_T1000.npy'

data = np.load(datapath)


