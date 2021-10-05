import os
import numpy as np

import torch

from models import FNN2d

from tqdm import tqdm
from timeit import default_timer
from train_utils.utils import count_params, save_checkpoint
from train_utils.datasets import BurgersLoader, sample_data
from train_utils.losses import LpLoss, PINO_loss
try:
    import wandb
except ImportError:
    wandb = None

