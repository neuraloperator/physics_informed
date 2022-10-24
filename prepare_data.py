from curses import raw
import numpy as np


def shuffle_data(datapath):
    data = np.load(datapath)
    rng = np.random.default_rng(123)
    rng.shuffle(data, axis=0)
    savepath = datapath.replace('.npy', '-shuffle.npy')
    np.save(savepath, data)


def test_data(datapath):
    raw = np.load(datapath, mmap_mode='r')
    print(raw[0, 0, 0, 0:10])
    newpath = datapath.replace('.npy', '-shuffle.npy')
    new = np.load(newpath, mmap_mode='r')
    print(new[0, 0, 0, 0:10])


if __name__ == '__main__':
    datapath = '../data/NS-Re500_T300_id0.npy'
    # shuffle_data(datapath)
    test_data(datapath)