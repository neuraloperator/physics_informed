import numpy as np

from tensordiffeq.boundaries import BC
from tensordiffeq.utils import flatten_and_stack, multimesh, MSE, convertTensor


class PointsIC(BC):
    '''
    Create Initial condition class from array on domain
    '''
    def __init__(self, domain, values, var, n_values=None):
        '''
        args:
            - domain:
            - values:
        '''
        super(PointsIC, self).__init__()
        self.isInit = True
        self.n_values = n_values
        self.domain = domain
        self.values = values
        self.vars = var
        self.isInit = True
        self.dicts_ = [item for item in self.domain.domaindict if item['identifier'] != self.domain.time_var]
        self.dict_ = next(item for item in self.domain.domaindict if item["identifier"] == self.domain.time_var)
        self.compile()
        self.create_target(self.values)

    def create_input(self):
        dims = self.get_not_dims(self.domain.time_var)
        mesh = flatten_and_stack(multimesh(dims))
        t_repeat = np.repeat(0.0, len(mesh))

        mesh = np.concatenate((mesh, np.reshape(t_repeat, (-1, 1))), axis=1)
        if self.n_values is not None:
            self.nums = np.random.randint(0, high=len(mesh), size=self.n_values)
            mesh = mesh[self.nums]
        return mesh

    def create_target(self, values):
        # for i, var_ in enumerate(self.vars):
        #     arg_list = []
        #     for j, var in enumerate(var_):
        #         var_dict = self.get_dict(var)
        #         arg_list.append(get_linspace(var_dict))
        #     inp = flatten_and_stack(multimesh(arg_list))
        #     fun_vals.append(self.fun[i](*inp.T))
        if self.n_values is not None:
            self.val = np.reshape(values, (-1, 3))[self.nums]
        else:
            self.val = np.reshape(values, (-1, 3))

    def loss(self):
        return MSE(self.preds, self.val)