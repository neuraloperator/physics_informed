import numpy as np
import torch
from torch.utils.data import Dataset
from .utils import get_4dgrid, get_2dgird, concat, get_3dgrid
from train_utils.utils import vor2vel


class BelflowData(object):
    def __init__(self, npt_col=11, npt_boundary=31, npt_init=11):
        self.Ne = npt_col ** 4
        self.Nb = npt_boundary ** 2 * 6
        self.col_xyzt, self.col_uvwp = self.get_collocation(npt_col)
        self.bd_xyzt, self.bd_uvwp = self.sample_boundary(npt_boundary)
        self.ini_xyzt, self.ini_uvwp = self.get_init(npt_init)

    @staticmethod
    def get_collocation(num=11):
        xyzt = get_4dgrid(num)
        uvwp = BelflowData.cal_uvwp(xyzt)
        return xyzt, uvwp

    @staticmethod
    def get_init(num=11):
        xyz = get_3dgrid(num)
        ts = np.zeros((xyz.shape[0], 1))
        coord = np.hstack((xyz, ts))
        uvwp = BelflowData.cal_uvwp(coord)
        return coord, uvwp

    @staticmethod
    def sample_boundary(num=31):
        '''
        Sample boundary data on each face
        Args:
            num:

        Returns:

        '''
        samples = get_2dgird(num)
        dataList = []
        offset = range(3)
        z = np.ones((samples.shape[0], 1))
        signs = [-1, 1]
        for i in offset:
            for sign in signs:
                dataList.append(concat(samples, z*sign, offset=i))
        bd_xyzt = np.vstack(dataList)
        bd_uvwp = BelflowData.cal_uvwp(bd_xyzt)
        return bd_xyzt, bd_uvwp

    @staticmethod
    def cal_uvwp(xyzt, a=1, d=1):
        x, y, z = xyzt[:, 0:1], xyzt[:, 1:2], xyzt[:, 2:3]
        t = xyzt[:, -1:]
        comp_x = a * x + d * y
        comp_y = a * y + d * z
        comp_z = a * z + d * x
        u = -a * np.exp(- d ** 2 * t) * (np.exp(a * x) * np.sin(comp_y)
                                         + np.exp(a * z) * np.cos(comp_x))
        v = -a * np.exp(- d ** 2 * t) * (np.exp(a * y) * np.sin(comp_z)
                                         + np.exp(a * x) * np.cos(comp_y))
        w = -a * np.exp(- d ** 2 * t) * (np.exp(a * z) * np.sin(comp_x)
                                         + np.exp(a * y) * np.cos(comp_z))
        p = - 0.5 * a ** 2 * np.exp(-2 * d ** 2 * t) \
            * (np.exp(2 * a * x) + np.exp(2 * a * y) + np.exp(2 * a * z) +
               2 * np.sin(comp_x) * np.cos(comp_z) * np.exp(a * (y + z)) +
               2 * np.sin(comp_y) * np.cos(comp_x) * np.exp(a * (z + x)) +
               2 * np.sin(comp_z) * np.cos(comp_y) * np.exp(a * (x + y)))
        return np.hstack((u, v, w, p))


class NSdata(object):
    def __init__(self, datapath1,
                 nx, nt,
                 offset=0,
                 datapath2=None,
                 sub=1, sub_t=1,
                 N=100, t_interval=1.0):
        '''
        Load data from npy and reshape to (N, X, Y, T)
        Args:
            datapath1: path to data
            nx: number of points on each spatial domain
            nt: number of points on temporal domain
            offset: index of the instance
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
        part1 = data1.permute(0, 2, 3, 1)
        if datapath2 is not None:
            part2 = data2.permute(0, 2, 3, 1)
            self.data = torch.cat((part1, part2), dim=0)
        else:
            self.data = part1
        self.vor = self.data[offset: offset + 1, :, :, :]
        self.vel_u, self.vel_v = vor2vel(self.vor)  # Compute velocity from vorticity

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

        boundary0 = value[0, ::2, ::2, 0:1]     # 32x32x1

        boundary1 = value[0, ::2, ::2, -1:]     # 32x32x1

        part0 = np.ravel(boundary0)
        part1 = np.ravel(boundary1)
        boundary = np.concatenate([part0, part1], axis=0)[:, np.newaxis]
        return boundary
        # bd_solnx0 = self.data[0, 0:1, ::2, ::2]     # num_y//2 x (num_t //2 +1)
        # bd_solny0 = self.data[0, ::2, 0:1, ::2]     # num_x //2 x (num_t //2 + 1)
        # bd_solnx1 = self.data[0, -1:, ]

    def get_boundary_points(self, num_x, num_y):
        '''
        Args:
            num_x:
            num_y:

        Returns:
            points: N by 3 array
        '''
        x_arr = np.linspace(0, 1, num=num_x, endpoint=False)
        y_arr = np.linspace(0, 1, num=num_y, endpoint=False)
        xx, yy = np.meshgrid(x_arr, y_arr, indexing='ij')
        xarr = np.ravel(xx)
        yarr = np.ravel(yy)
        tarr = np.zeros_like(xarr)
        point0 = np.stack([xarr, yarr, tarr], axis=0).T

        x_arr = np.linspace(0, 1, num=num_x, endpoint=False)
        y_arr = np.linspace(0, 1, num=num_y, endpoint=False)
        xx, yy = np.meshgrid(x_arr, y_arr, indejing='ij')
        xarr = np.ravel(xx)
        yarr = np.ravel(yy)
        tarr = np.zeros_like(xarr)
        point1 = np.stack([xarr, yarr, tarr], axis=0).T
        points = np.concatenate([point0, point1], axis=0)

        return points






