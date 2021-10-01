import random
import deepxde as dde
from baselines.data import NSdata

'''
Training deepONet using deepxde implementation. 
Note that deepxde requires passing the whole dataset to Triple, which is very memory consuming. 
'''


def train(config):
    seed = random.randint(1, 10000)
    print(f'Random seed :{seed}')
    # construct dataloader
    data_config = config['data']
    train_set = NSdata(datapath1=data_config['datapath'],
                       offset=0, num=10,
                       nx=data_config['nx'], nt=data_config['nt'],
                       sub=data_config['sub'], sub_t=data_config['sub_t'],
                       vel=False,
                       t_interval=data_config['time_interval'])
    val_set = NSdata(datapath1=data_config['data_val'],
                     offset=310, num=10,
                     nx=data_config['val_nx'], nt=data_config['val_nt'],
                     sub=data_config['val_sub'], sub_t=data_config['val_subt'],
                     vel=False,
                     t_interval=data_config['time_interval'])
    # assert train_set.S == val_set.S
    dim_a = train_set.S ** 2
    dim_x = 3
    X_train, y_train = train_set.get_operator_data()
    X_val, y_val = val_set.get_operator_data()
    data = dde.data.Triple(X_train=X_train, y_train=y_train, X_test=X_val, y_test=y_val)

    activation = config['model']['activation']
    initializer = 'Glorot normal'   # He normal or Glorot normal

    net = dde.maps.DeepONet([dim_a] + config['model']['layers'],
                            [dim_x] + config['model']['layers'],
                            activation,
                            initializer,
                            use_bias=True,
                            stacked=False)
    model = dde.Model(data, net)
    model.compile('adam', lr=config['train']['base_lr'])
    checker = dde.callbacks.ModelCheckpoint(
        'checkpoints/deeponet.ckpt', save_better_only=True, period=10,
    )
    model.train(epochs=config['train']['epochs'], callbacks=[checker])
