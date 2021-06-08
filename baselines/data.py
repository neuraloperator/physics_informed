import numpy as np
import torch
from torch.utils.data import Dataset
from .utils import get_4dgrid, get_2dgird, concat, get_3dgrid


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





