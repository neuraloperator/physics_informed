import numpy as np
import torch
from torch.utils.data import Dataset
from .utils import get_xytgrid, get_3dboundary, get_3dboundary_points
from train_utils.utils import vor2vel, torch2dgrid
import scipy.io
import h5py


class DarcyFlow(Dataset):
    def __init__(self,
                 datapath,
                 nx, sub,
                 offset=0,
                 num=1):
        self.S = int(nx // sub) + 1
        data = scipy.io.loadmat(datapath)
        a = data['coeff']
        u = data['sol']
        self.a = torch.tensor(a[offset: offset + num, ::sub, ::sub], dtype=torch.float)
        self.u = torch.tensor(u[offset: offset + num, ::sub, ::sub], dtype=torch.float)
        self.mesh = torch2dgrid(self.S, self.S)

    def __len__(self):
        return self.a.shape[0]

    def __getitem__(self, item):
        fa = self.a[item]
        return fa.reshape(-1), self.u[item].reshape(-1)


class NSLong(object):
    def __init__(self,
                 datapath,
                 nx, nt,
                 time_scale,
                 offset=0,
                 num=1, vel=False):
        '''
        Load data from mat
        Args:
            datapath: path to data file
            nx: number of points in each spatial domain
            nt: number of points in temporal domain
            offset: index of the instance
            num: number of instances
            vel: compute velocity from vorticity if True
        '''

        self.time_scale = time_scale
        self.S = nx
        self.T = nt

        with h5py.File(datapath, mode='r') as file:
            raw = file['u']
            data = np.array(raw)
        vor = torch.tensor(data, dtype=torch.float).permute(3, 1, 2, 0)
        self.vor = vor[offset: offset + num, :, :, :]     # num x 64 x 64 x 50
        if vel:
            self.vel_u, self.vel_v = vor2vel(self.vor, L=1.0)

    def get_boundary_value(self, component=0):
        '''
            Get the boundary value for component-th output
            Args:
                component: int, 0: velocity_u; 1: velocity_v; 2: vorticity;
            Returns:
                value: N by 1 array, boundary value of the component
        '''
        if component == 0:
            value = self.vel_u
        elif component == 1:
            value = self.vel_v
        elif component == 2:
            value = self.vor
        else:
            raise ValueError(f'No component {component} ')

        boundary = get_3dboundary(value)
        return boundary

    def get_boundary_points(self, num_x, num_y, num_t):
        points = get_3dboundary_points(num_x, num_y, num_t,
                                       bot=(0,0,0),
                                       top=(1, 1, self.time_scale))
        return points

    def get_test_xyt(self):
        '''

        Returns:
            points: (x, y, t) array with shape (S * S * T, 3)
            values: (u, v, w) array with shape (S * S * T, 3)

        '''
        points = get_xytgrid(S=self.S, T=self.T,
                             bot=[0, 0, 0],
                             top=[1, 1, self.time_scale])
        u_val = np.ravel(self.vel_u)
        v_val = np.ravel(self.vel_v)
        w_val = np.ravel(self.vor)
        values = np.stack([u_val, v_val, w_val], axis=0).T
        return points, values


class NSdata(object):
    def __init__(self, datapath1,
                 nx, nt,
                 offset=0, num=1,
                 datapath2=None,
                 sub=1, sub_t=1,
                 vel=False, t_interval=1.0):
        '''
        Load data from npy and reshape to (N, X, Y, T)
        Args:
            datapath1: path to data
            nx: number of points in each spatial domain
            nt: number of points in temporal domain
            offset: index of the instance
            num: number of instances
            datapath2: path to second part of data, default None
            sub: downsample interval of spatial domain
            sub_t: downsample interval of temporal domain
            N:
            t_interval:
        '''
        self.S = nx // sub
        self.T = int(nt * t_interval) // sub_t + 1
        self.time_scale = t_interval
        data1 = np.load(datapath1)
        data1 = torch.tensor(data1, dtype=torch.float)[..., ::sub_t, ::sub, ::sub]

        if datapath2 is not None:
            data2 = np.load(datapath2)
            data2 = torch.tensor(data2, dtype=torch.float)[..., ::sub_t, ::sub, ::sub]
        if t_interval == 0.5:
            data1 = self.extract(data1)
            if datapath2 is not None:
                data2 = self.extract(data2)
        # transpose data into (N, S, S, T)
        part1 = data1.permute(0, 2, 3, 1)
        if datapath2 is not None:
            part2 = data2.permute(0, 2, 3, 1)
            self.data = torch.cat((part1, part2), dim=0)
        else:
            self.data = part1
        self.vor = self.data[offset: offset + num, :, :, :].cpu()
        if vel:
            self.vel_u, self.vel_v = vor2vel(self.vor)  # Compute velocity from vorticity

    def get_init_cond(self):
        values = np.stack([self.vel_u[0, :, :, 0],
                           self.vel_v[0, :, :, 0],
                           self.vor[0, :, :, 0]], axis=2)
        return values

    def get_boundary_value(self, component=0):
        '''
        Get the boundary value for component-th output
        Args:
            component: int, 0: velocity_u; 1: velocity_v; 2: vorticity;
        Returns:
            value: N by 1 array, boundary value of the component
        '''
        if component == 0:
            value = self.vel_u
        elif component == 1:
            value = self.vel_v
        elif component == 2:
            value = self.vor
        else:
            raise ValueError(f'No component {component} ')

        boundary = get_3dboundary(value)
        return boundary

    def get_boundary_points(self, num_x, num_y, num_t):
        '''
        Args:
            num_x:
            num_y:

        Returns:
            points: N by 3 array
        '''
        points = get_3dboundary_points(num_x, num_y, num_t,
                                       bot=(0, 0, 0),
                                       top=(2 * np.pi, 2 * np.pi, self.time_scale))
        # x_arr = np.linspace(0, 2 * np.pi, num=num_x, endpoint=False)
        # y_arr = np.linspace(0, 2 * np.pi, num=num_y, endpoint=False)
        # xx, yy = np.meshgrid(x_arr, y_arr, indexing='ij')
        # xarr = np.ravel(xx)
        # yarr = np.ravel(yy)
        # tarr = np.zeros_like(xarr)
        # point0 = np.stack([xarr, yarr, tarr], axis=0).T     # (128x128x1, 3), boundary on t=0
        #
        # # tarr = np.ones_like(xarr) * self.time_scale
        # # point1 = np.stack([xarr, yarr, tarr], axis=0).T     # (128x128x1, 3), boundary on t=0.5
        #
        # t_arr = np.linspace(0, self.time_scale, num=num_t)
        # yy, tt = np.meshgrid(y_arr, t_arr, indexing='ij')
        # yarr = np.ravel(yy)
        # tarr = np.ravel(tt)
        # xarr = np.zeros_like(yarr)
        # point2 = np.stack([xarr, yarr, tarr], axis=0).T     # (1x128x65, 3), boundary on x=0
        #
        # xarr = np.ones_like(yarr) * 2 * np.pi
        # point3 = np.stack([xarr, yarr, tarr], axis=0).T     # (1x128x65, 3), boundary on x=2pi
        #
        # xx, tt = np.meshgrid(x_arr, t_arr, indexing='ij')
        # xarr = np.ravel(xx)
        # tarr = np.ravel(tt)
        # yarr = np.zeros_like(xarr)
        # point4 = np.stack([xarr, yarr, tarr], axis=0).T     # (128x1x65, 3), boundary on y=0
        #
        # yarr = np.ones_like(xarr) * 2 * np.pi
        # point5 = np.stack([xarr, yarr, tarr], axis=0).T     # (128x1x65, 3), boundary on y=2pi
        #
        # points = np.concatenate([point0,
        #                          point2, point3,
        #                          point4, point5],
        #                         axis=0)
        return points

    def get_test_xyt(self):
        '''

        Returns:
            points: (x, y, t) array with shape (S * S * T, 3)
            values: (u, v, w) array with shape (S * S * T, 3)

        '''
        points = get_xytgrid(S=self.S, T=self.T,
                             bot=[0, 0, 0],
                             top=[2 * np.pi, 2 * np.pi, self.time_scale])
        u_val = np.ravel(self.vel_u)
        v_val = np.ravel(self.vel_v)
        w_val = np.ravel(self.vor)
        values = np.stack([u_val, v_val, w_val], axis=0).T
        return points, values


    @staticmethod
    def extract(data):
        '''
        Extract data with time range 0-0.5, 0.25-0.75, 0.5-1.0, 0.75-1.25,...
        Args:
            data: tensor with size N x 129 x 128 x 128

        Returns:
            output: (4*N-1) x 65 x 128 x 128
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


class DeepOnetNS(Dataset):
    '''
    Dataset class customized for DeepONet's input format
    '''
    def __init__(self, datapath,
                 nx, nt,
                 offset=0, num=1,
                 sub=1, sub_t=1,
                 t_interval=1.0):
        self.S = nx // sub
        self.T = int(nt * t_interval) // sub_t + 1
        self.time_scale = t_interval
        self.N = num
        data = np.load(datapath)
        data = torch.tensor(data, dtype=torch.float)[..., ::sub_t, ::sub, ::sub]
        if t_interval == 0.5:
                    data = NSdata.extract(data)
        # transpose data into (N, S, S, T)
        data = data.permute(0, 2, 3, 1)
        self.vor = data[offset: offset + num, :, :, :]
        points = get_xytgrid(S=self.S, T=self.T,
                             bot=[0, 0, 0],
                             top=[2 * np.pi, 2 * np.pi, self.time_scale])
        self.xyt = torch.tensor(points, dtype=torch.float)
        # (SxSxT, 3)

    def __len__(self):
        return self.N * self.S * self.S * self.T

    def __getitem__(self, idx):
        num_per_instance = self.S ** 2 * self.T
        instance_id = idx // num_per_instance
        pos_id = idx % num_per_instance
        point = self.xyt[pos_id]
        u0 = self.vor[instance_id, :, :, 0].reshape(-1)
        y = self.vor[instance_id].reshape(-1)[pos_id]
        return u0, point, y


class DeepONetCPNS(Dataset):
    '''
        Dataset class customized for DeepONet cartesian product's input format
        '''

    def __init__(self, datapath,
                 nx, nt,
                 offset=0, num=1,
                 sub=1, sub_t=1,
                 t_interval=1.0):
        self.S = nx // sub
        self.T = int(nt * t_interval) // sub_t + 1
        self.time_scale = t_interval
        self.N = num
        data = np.load(datapath)
        data = torch.tensor(data, dtype=torch.float)[..., ::sub_t, ::sub, ::sub]
        if t_interval == 0.5:
            data = NSdata.extract(data)
        # transpose data into (N, S, S, T)
        data = data.permute(0, 2, 3, 1)
        self.vor = data[offset: offset + num, :, :, :]
        points = get_xytgrid(S=self.S, T=self.T,
                             bot=[0, 0, 0],
                             top=[2 * np.pi, 2 * np.pi, self.time_scale])
        self.xyt = torch.tensor(points, dtype=torch.float)
        # (SxSxT, 3)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        '''

        Args:
            idx:

        Returns:
            u0: (batchsize, u0_dim)
            y: (batchsize, SxSxT)
        '''
        u0 = self.vor[idx, :, :, 0].reshape(-1)
        y = self.vor[idx, :, :, :].reshape(-1)
        return u0, y
