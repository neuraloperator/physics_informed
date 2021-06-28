from baselines.train import Baselinetrainer
from baselines.data import BelflowData
from models.FCN import FCNet

import torch
from torch.optim import Adam, LBFGS


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    layers = [4, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 4]
    model = FCNet(layers)
    trainer = Baselinetrainer(model=model, device=device)
    dataset = BelflowData(npt_col=11, npt_boundary=31, npt_init=11)

    alpha = 100.0
    beta = 100.0
    optimizer = Adam(model.parameters(), lr=1e-3)

    trainer.prepare_data(dataset)
    trainer.train_adam(optimizer, alpha, beta,
                       iter_num=10000)
