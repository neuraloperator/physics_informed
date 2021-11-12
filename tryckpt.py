import torch
from models import FNN3d
from models.trial import CIFAR10Model

from train_utils.losses import LpLoss
from train_utils import Adam

from argparse import ArgumentParser
'''
Profile the memory reduction by gradient checkpointing. 

'''


def try_pino(args, device):
    modes = [16] * 4
    layers = [64] * 5
    activations = ['tanh'] * 3 + ['none']
    batchsize = 2
    S = 64
    T = 33

    model = FNN3d(modes1=modes, modes2=modes, modes3=modes,
                  layers=layers,
                  fc_dim=128,
                  activation=activations,
                  use_checkpoint=args.use_ckpt).to(device)
    optimizer = Adam(model.parameters())
    criterion = LpLoss(size_average=True)
    print(f'Use gradient: {not args.test}; use checkpointing: {args.use_ckpt}')
    if args.test:
        model.eval()
        for i in range(50):
            with torch.no_grad():
                random_data = torch.randn((batchsize, S, S, T, 4)).to(device)
                ground_truth = torch.randn((batchsize, S, S, T)).to(device)

                pred = model(random_data).reshape(ground_truth.shape)
                loss = criterion(pred, ground_truth)
                if i % 10 == 0:
                    memory_used = torch.cuda.max_memory_allocated(device) / 1024.0 ** 2
                    print(f'Iter {i}, GPU memory used: {memory_used} MB')

    else:
        model.train()
        for i in range(50):
            random_data = torch.randn((batchsize, S, S, T, 4)).to(device)
            ground_truth = torch.randn((batchsize, S, S, T)).to(device)

            pred = model(random_data).reshape(ground_truth.shape)
            loss = criterion(pred, ground_truth)
            model.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                memory_used = torch.cuda.max_memory_allocated(device) / 1024.0 ** 2
                print(f'Iter {i}, GPU memory used: {memory_used} MB')
    print('Done!')


def try_cifar(args, device):
    model = CIFAR10Model().to(device)



if __name__ == '__main__':
    parser = ArgumentParser(description='Basic paser')
    parser.add_argument('--test', action='store_true', help='test with no grad')
    parser.add_argument('--use_ckpt', action='store_true', help='use gradient checkpointing')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    try_pino(args, device)

