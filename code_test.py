import os
from matplotlib import test
import yaml
import numpy as np

import matplotlib.pyplot as plt



def test_config():
    config_path = 'configs/operator/Re500-PINO.yaml'

    with open(config_path, 'r') as f:
        config = yaml.load(f, yaml.FullLoader)
    print(config['data']['paths'])


def test_data():
    # data_path = '../data/NS-T4000.npy'
    # data_path = '../data/NS-Re500_T256.npy'
    data_path = '../data/NS-'
    plot_dir = 'exp/debug/plots'
    os.makedirs(plot_dir, exist_ok=True)
    data = np.load(data_path)[0]
    T = data.shape[0]
    for i in range(T):
        fig, ax = plt.subplots()
        im = ax.imshow(data[i])
        ax.set_title(f'T:{i}')
        fig.colorbar(im, ax=ax)
        plot_path = os.path.join(plot_dir, f'N2000T{i}.png')
        plt.savefig(plot_path)


if __name__ == '__main__':
    # test_config()
    test_data()
