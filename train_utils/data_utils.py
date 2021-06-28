import scipy.io
import numpy as np

try:
    from pyDOE import lhs
    # Only needed for PINN's dataset
except ImportError:
    lhs = None

import torch
from torch.utils.data import Dataset
from train_utils.utils import get_grid3d


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        self.data = scipy.io.loadmat(self.file_path)
        self.old_mat = True

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float


class BurgersLoader(object):
    def __init__(self, datapath, nx=2 ** 10, nt=100, sub=8, sub_t=1, new=False):
        dataloader = MatReader(datapath)
        self.sub = sub
        self.sub_t = sub_t
        self.s = nx // sub
        self.T = nt // sub_t
        self.new = new
        if new:
            self.T += 1
        self.x_data = dataloader.read_field('input')[:, ::sub]
        self.y_data = dataloader.read_field('output')[:, ::sub_t, ::sub]

    def make_loader(self, n_sample, batch_size, start=0, train=True):
        if train:
            Xs = self.x_data[start:start + n_sample]
            ys = self.y_data[start:start + n_sample]
        else:
            Xs = self.x_data[-n_sample:]
            ys = self.y_data[-n_sample:]

        if self.new:
            gridx = torch.tensor(np.linspace(0, 1, self.s + 1)[:-1], dtype=torch.float)
            gridt = torch.tensor(np.linspace(0, 1, self.T), dtype=torch.float)
        else:
            gridx = torch.tensor(np.linspace(0, 1, self.s), dtype=torch.float)
            gridt = torch.tensor(np.linspace(0, 1, self.T + 1)[1:], dtype=torch.float)
        gridx = gridx.reshape(1, 1, self.s)
        gridt = gridt.reshape(1, self.T, 1)

        Xs = Xs.reshape(n_sample, 1, self.s).repeat([1, self.T, 1])
        Xs = torch.stack([Xs, gridx.repeat([n_sample, self.T, 1]), gridt.repeat([n_sample, 1, self.s])], dim=3)
        dataset = torch.utils.data.TensorDataset(Xs, ys)
        if train:
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        else:
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return loader


class NS40Loader(object):
    def __init__(self, datapath, nx, nt, sub=1, sub_t=1, N=1000):
        self.N = N
        self.S = nx // sub
        self.T = nt // sub_t + 1
        data = np.load(datapath)
        data = torch.tensor(data, dtype=torch.float)[..., ::sub, ::sub]
        self.data = self.rearrange(data, sub_t)

    def rearrange(self, data, sub_t):
        new_data = torch.zeros(self.N, self.S, self.S, self.T)
        for i in range(self.N):
            new_data[i] = data[i * 64: (i + 1) * 64 + 1: sub_t].permute(1, 2, 0)
        return new_data

    def make_loader(self, n_sample, batch_size, start=0, train=True):
        if train:
            a_data = self.data[:n_sample, :, :, 0].reshape(n_sample, self.S, self.S)
            u_data = self.data[:n_sample].reshape(n_sample, self.S, self.S, self.T)
        else:
            a_data = self.data[-n_sample:, :, :, 0].reshape(n_sample, self.S, self.S)
            u_data = self.data[-n_sample:].reshape(n_sample, self.S, self.S, self.T)
        a_data = a_data.reshape(n_sample, self.S, self.S, 1, 1).repeat([1, 1, 1, self.T, 1])
        gridx, gridy, gridt = get_grid3d(self.S, self.T)
        a_data = torch.cat((gridx.repeat([n_sample, 1, 1, 1, 1]), gridy.repeat([n_sample, 1, 1, 1, 1]),
                            gridt.repeat([n_sample, 1, 1, 1, 1]), a_data), dim=-1)
        dataset = torch.utils.data.TensorDataset(a_data, u_data)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train)
        return loader


class NS500Loader(object):
    def __init__(self, datapath, nx, nt, sub=1, sub_t=1, N=1000, t_interval=1.0, rearrange=True):
        self.N = int(N / t_interval)
        self.S = nx // sub
        self.T = int(nt * t_interval) // sub_t + 1
        data = np.load(datapath)
        data = torch.tensor(data, dtype=torch.float)[..., ::sub_t, ::sub, ::sub]
        self.data = data.permute(0, 2, 3, 1)
        if t_interval == 0.5 and rearrange:
            self.data = self.rearrange(self.data)

    def make_loader(self, n_sample, batch_size, start=0, train=True):
        if train:
            a_data = self.data[start:start + n_sample, :, :, 0].reshape(n_sample, self.S, self.S)
            u_data = self.data[start:start + n_sample].reshape(n_sample, self.S, self.S, self.T)
        else:
            a_data = self.data[-n_sample:, :, :, 0].reshape(n_sample, self.S, self.S)
            u_data = self.data[-n_sample:].reshape(n_sample, self.S, self.S, self.T)
        a_data = a_data.reshape(n_sample, self.S, self.S, 1, 1).repeat([1, 1, 1, self.T, 1])
        gridx, gridy, gridt = get_grid3d(self.S, self.T)
        a_data = torch.cat((gridx.repeat([n_sample, 1, 1, 1, 1]), gridy.repeat([n_sample, 1, 1, 1, 1]),
                            gridt.repeat([n_sample, 1, 1, 1, 1]), a_data), dim=-1)
        dataset = torch.utils.data.TensorDataset(a_data, u_data)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train)
        return loader

    def rearrange(self, data):
        new_data = torch.zeros(self.N, self.S, self.S, self.T)
        for i in range(self.N // 2):
            new_data[2 * i] = data[i, :, :, :65]
            new_data[2 * i + 1] = data[i, :, :, 64:]
        return new_data


class NSLoader(object):
    def __init__(self, datapath1, datapath2,
                 nx, nt, sub=1, sub_t=1,
                 N=100, t_interval=1.0):
        self.N = N
        self.S = nx // sub
        self.T = int(nt * t_interval) // sub_t + 1
        data1 = np.load(datapath1)
        data2 = np.load(datapath2)
        data1 = torch.tensor(data1, dtype=torch.float)[..., ::sub_t, ::sub, ::sub]
        data2 = torch.tensor(data2, dtype=torch.float)[..., ::sub_t, ::sub, ::sub]
        if t_interval == 0.5:
            data1 = self.extract(data1)
            data2 = self.extract(data2)
        part1 = data1.permute(0, 2, 3, 1)
        part2 = data2.permute(0, 2, 3, 1)
        self.data = torch.cat((part1, part2), dim=0)

    def make_loader(self, n_sample, batch_size, start=0, train=True):
        if train:
            a_data = self.data[start:start + n_sample, :, :, 0].reshape(n_sample, self.S, self.S)
            u_data = self.data[start:start + n_sample].reshape(n_sample, self.S, self.S, self.T)
        else:
            a_data = self.data[-n_sample:, :, :, 0].reshape(n_sample, self.S, self.S)
            u_data = self.data[-n_sample:].reshape(n_sample, self.S, self.S, self.T)
        a_data = a_data.reshape(n_sample, self.S, self.S, 1, 1).repeat([1, 1, 1, self.T, 1])
        gridx, gridy, gridt = get_grid3d(self.S, self.T)
        a_data = torch.cat((gridx.repeat([n_sample, 1, 1, 1, 1]), gridy.repeat([n_sample, 1, 1, 1, 1]),
                            gridt.repeat([n_sample, 1, 1, 1, 1]), a_data), dim=-1)
        dataset = torch.utils.data.TensorDataset(a_data, u_data)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train)
        return loader

    @staticmethod
    def extract(data):
        '''
        Extract data with time range 0-0.5, 0.25-0.75, 0.5-1.0, 0.75-1.25,...
        Args:
            data: tensor with size N x 129 x 128 x 128

        Returns:
            output: (4*N-1) x 64 x 128 x 128
        '''
        T = data.shape[1] // 2
        interval = data.shape[1] // 4
        N = data.shape[0]
        new_data = torch.zeros(4 * N - 1, T + 1, data.shape[2], data.shape[3])
        for i in range(N):
            for j in range(4):
                if i == N - 1 and j == 3:
                    # reach boundary
                    break
                if j != 3:
                    new_data[i * 4 + j] = data[i, interval * j:interval * j + T + 1]
                else:
                    new_data[i * 4 + j, 0: interval] = data[i, interval * j:interval * j + interval]
                    new_data[i * 4 + j, interval: T + 1] = data[i + 1, 0:interval + 1]
        return new_data


class BurgerData(Dataset):
    '''
    members: 
        - t, x, Exact: raw data
        - X, T: meshgrid 
        - X_star, u_star: flattened (x, t), u array
        - lb, ub: lower bound and upper bound vector
        - X_u, u: boundary condition data (x, t), u
    '''

    def __init__(self, datapath):
        data = scipy.io.loadmat(datapath)

        # raw 2D data
        self.t = data['t'].flatten()[:, None]  # (100,1)
        self.x = data['x'].flatten()[:, None]  # (256, 1)
        self.Exact = np.real(data['usol']).T  # (100, 256)

        # Flattened sequence
        self.get_flatten_data()
        self.get_boundary_data()

    def __len__(self):
        return self.Exact.shape[0]

    def __getitem__(self, idx):
        return self.X_star[idx], self.u_star[idx]

    def get_flatten_data(self):
        X, T = np.meshgrid(self.x, self.t)
        self.X, self.T = X, T
        self.X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
        self.u_star = self.Exact.flatten()[:, None]

        # lower bound of (x, t): 2-dimensional vector
        self.lb = self.X_star.min(0)
        # upper bound of (x, t): 2-dimensional vector
        self.ub = self.X_star.max(0)

    def get_boundary_data(self):
        xx1 = np.hstack((self.X[0:1, :].T, self.T[0:1, :].T))
        uu1 = self.Exact[0:1, :].T
        xx2 = np.hstack((self.X[:, 0:1], self.T[:, 0:1]))
        uu2 = self.Exact[:, 0:1]
        xx3 = np.hstack((self.X[:, -1:], self.T[:, -1:]))
        uu3 = self.Exact[:, -1:]
        self.X_u = np.vstack([xx1, xx2, xx3])
        self.u = np.vstack([uu1, uu2, uu3])

    def sample_xt(self, N=10000):
        '''
        Sample (x, t) pairs within the boundary
        Return:
            - X_f: (N, 2) array
        '''
        X_f = self.lb + (self.ub - self.lb) * lhs(2, N)
        X_f = np.vstack((X_f, self.X_u))
        return X_f

    def sample_xu(self, N=100):
        '''
        Sample N points from boundary data
        Return: 
            - X_u: (N, 2) array 
            - u: (N, 1) array
        '''
        idx = np.random.choice(self.X_u.shape[0], N, replace=False)
        X_u = self.X_u[idx, :]
        u = self.u[idx, :]
        return X_u, u


class NS40data(Dataset):
    def __init__(self, datapath, nx, nt, sub=1, sub_t=1, N=1000, index=1):
        self.N = N
        self.S = nx // sub
        self.T = nt // sub_t + 1
        data = np.load(datapath)
        data = torch.tensor(data, dtype=torch.float)[..., ::sub, ::sub]
        self.data = self.rearrange(data, sub_t)[-index]
        self.x, self.y, self.t, self.vor = self.sample_xyt()
        self.ux, self.uy = self.convert2vel()
        self.u_gt = self.ux.reshape(-1).unsqueeze(0).permute(1, 0)
        self.v_gt = self.ux.reshape(-1).unsqueeze(0).permute(1, 0)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.t[idx], self.vor[idx], self.u_gt[idx], self.v_gt[idx]

    def convert2vel(self):
        ux, uy = vor2vel(self.data.unsqueeze(0))
        return ux, uy

    def rearrange(self, data, sub_t):
        new_data = torch.zeros(self.N, self.S, self.S, self.T)
        for i in range(self.N):
            new_data[i] = data[i * 64: (i + 1) * 64 + 1: sub_t].permute(1, 2, 0)
        return new_data

    def get_boundary(self):
        bd_vor = self.data[:, :, 0].reshape(-1).unsqueeze(0).permute(1, 0)

        xs = torch.tensor(np.linspace(0, 1, self.S + 1)[:-1], dtype=torch.float)
        ys = torch.tensor(np.linspace(0, 1, self.S + 1)[:-1], dtype=torch.float)
        gridx, gridy = torch.meshgrid(xs, ys)
        bd_x = gridx.reshape(-1).unsqueeze(0).permute(1, 0)
        bd_y = gridy.reshape(-1).unsqueeze(0).permute(1, 0)
        bd_t = torch.zeros_like(bd_x)
        ux = self.ux[:, :, :, 0].reshape(-1).unsqueeze(0).permute(1, 0)
        uy = self.uy[:, :, :, 0].reshape(-1).unsqueeze(0).permute(1, 0)
        return bd_x, bd_y, bd_t, bd_vor, ux, uy

    def sample_xyt(self, sub=1):
        xs = torch.tensor(np.linspace(0, 1, self.S + 1)[:-1], dtype=torch.float)[::sub]
        ys = torch.tensor(np.linspace(0, 1, self.S + 1)[:-1], dtype=torch.float)[::sub]
        ts = torch.tensor(np.linspace(0, 1, self.T), dtype=torch.float)[::sub]
        gridx, gridy, gridt = torch.meshgrid(xs, ys, ts)
        x = gridx.reshape(-1).unsqueeze(0).permute(1, 0)
        y = gridy.reshape(-1).unsqueeze(0).permute(1, 0)
        t = torch.zeros_like(x)
        vor = self.data[::sub, ::sub, ::sub].reshape(-1).unsqueeze(0).permute(1, 0)
        return x, y, t, vor


def vor2vel(w):
    batchsize = w.size(0)
    nx = w.size(1)
    ny = w.size(2)
    nt = w.size(3)
    device = w.device
    w = w.reshape(batchsize, nx, ny, nt)

    w_h = torch.fft.fft2(w, dim=[1, 2])
    # Wavenumbers in y-direction
    k_max = nx // 2
    N = nx
    k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0) \
        .reshape(N, 1).repeat(1, N).reshape(1, N, N, 1)
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0) \
        .reshape(1, N).repeat(N, 1).reshape(1, N, N, 1)
    # Negative Laplacian in Fourier space
    lap = (k_x ** 2 + k_y ** 2)
    lap[0, 0, 0, 0] = 1.0
    f_h = w_h / lap

    ux_h = 1j * k_y * f_h
    uy_h = -1j * k_x * f_h

    ux = torch.fft.irfft2(ux_h[:, :, :k_max + 1], dim=[1, 2])
    uy = torch.fft.irfft2(uy_h[:, :, :k_max + 1], dim=[1, 2])
    return ux, uy
