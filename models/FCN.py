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


class DenseNet(nn.Module):
    def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False):
        super(DenseNet, self).__init__()

        self.n_layers = len(layers) - 1
        assert self.n_layers >= 1
        if isinstance(nonlinearity, str):
            if nonlinearity == 'tanh':
                nonlinearity = nn.Tanh
            elif nonlinearity == 'relu':
                nonlinearity == nn.ReLU
            else:
                raise ValueError(f'{nonlinearity} is not supported')
        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j+1]))

            if j != self.n_layers - 1:
                if normalize:
                    self.layers.append(nn.BatchNorm1d(layers[j+1]))

                self.layers.append(nonlinearity())

        if out_nonlinearity is not None:
            self.layers.append(out_nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)

        return x


