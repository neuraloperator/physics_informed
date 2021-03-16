import torch.nn as nn


def linear_block(in_channel, out_channel):
    block = nn.Sequential(
        nn.Linear(in_channel, out_channel),
        nn.Tanh()
    )
    return block


class FCNet(nn.Module):
    '''
    Fully connected layers with Tanh as nonlinearity
    Reproduced from PINNs Burger equation
    '''

    def __init__(self, layers=[2, 10, 1]):
        super(FCNet, self).__init__()

        fc_list = [linear_block(in_size, out_size)
                   for in_size, out_size in zip(layers, layers[1:-1])]
        fc_list.append(nn.Linear(layers[-2], layers[-1]))
        self.fc = nn.Sequential(*fc_list)

    def forward(self, x):
        return self.fc(x)
