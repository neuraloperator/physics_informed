import torch
import numpy as np

import torch.nn.functional as F
import torch.nn as nn

from torch_geometric.data import Data, DataLoader
import matplotlib.pyplot as plt
from utilities import *
from torch_geometric.nn import NNConv

from timeit import default_timer
import scipy.io

torch.manual_seed(0)
np.random.seed(0)

activation = F.relu


class KernelNN(torch.nn.Module):
    def __init__(self, width_node, width_kernel, depth, ker_in, in_width=1, out_width=1):
        super(KernelNN, self).__init__()
        self.depth = depth

        self.fc1 = torch.nn.Linear(in_width, width_node)

        kernel = DenseNet([ker_in, width_kernel // 2, width_kernel, width_node**2], torch.nn.ReLU)
        self.conv1 = NNConv(width_node, width_node, kernel, aggr='mean')

        self.fc2 = torch.nn.Linear(width_node, 128)
        self.fc3 = torch.nn.Linear(128, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.fc1(x)

        for k in range(self.depth):
            x = self.conv1(x, edge_index, edge_attr)
            if k != self.depth - 1:
                x = activation(x)

        x = self.fc2(x)
        x = activation(x)
        x = self.fc3(x)
        return x

# fourier features
class KernelNN_FF(torch.nn.Module):
    def __init__(self, width_node, width_kernel, depth, ker_in, in_width=1, out_width=1):
        super(KernelNN_FF, self).__init__()
        self.depth = depth

        self.fc1 = torch.nn.Linear(in_width, width_node)
        self.fc_ff1 = torch.nn.Linear(in_width, width_node)
        self.fc_ff2 = torch.nn.Linear(width_node, width_node)

        kernel = DenseNet([ker_in, width_kernel // 2, width_kernel, width_node**2], torch.nn.ReLU)
        self.conv1 = NNConv(width_node, width_node, kernel, aggr='mean')

        self.fc2 = torch.nn.Linear(width_node, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.fc_ff1(x)
        x = torch.sin(x)
        # x = F.relu(x)
        x = self.fc_ff2(x) + self.fc1(data.x)

        for k in range(self.depth):
            x = self.conv1(x, edge_index, edge_attr)
            if k != self.depth - 1:
                x = activation(x)

        x = self.fc2(x)
        return x

TRAIN_PATH = 'data/piececonst_r421_N1024_smooth1.mat'
TEST_PATH = 'data/piececonst_r421_N1024_smooth2.mat'


r = 5
s = int(((421 - 1)/r) + 1)
n = s**2
m = 200
k = 1

radius_train = 0.25
radius_test = 0.25
print('resolution', s)


ntrain = 1000
ntest = 100


batch_size = 20
batch_size2 = 20
width = 32
ker_width = 256
depth = 4
edge_features = 6
node_features = 6

epochs = 500
learning_rate = 0.001
scheduler_step = 50
scheduler_gamma = 0.5


path = 'GKN_activation_s'+str(s)+'_ntrain'+str(ntrain)+'_kerwidth'+str(ker_width) + '_m0' + str(m)
path_model = 'model/' + path
path_train_err = 'results/' + path + 'train.txt'
path_test_err = 'results/' + path + 'test.txt'
path_runtime = 'results/' + path + 'time.txt'
path_image = 'results/' + path

runtime = np.zeros(2, )
t1 = default_timer()


reader = MatReader(TRAIN_PATH)
train_a = reader.read_field('coeff')[:ntrain,::r,::r].reshape(ntrain,-1)
train_a_smooth = reader.read_field('Kcoeff')[:ntrain,::r,::r].reshape(ntrain,-1)
train_a_gradx = reader.read_field('Kcoeff_x')[:ntrain,::r,::r].reshape(ntrain,-1)
train_a_grady = reader.read_field('Kcoeff_y')[:ntrain,::r,::r].reshape(ntrain,-1)
train_u = reader.read_field('sol')[:ntrain,::r,::r].reshape(ntrain,-1)

reader.load_file(TEST_PATH)
test_a = reader.read_field('coeff')[:ntest,::r,::r].reshape(ntest,-1)
test_a_smooth = reader.read_field('Kcoeff')[:ntest,::r,::r].reshape(ntest,-1)
test_a_gradx = reader.read_field('Kcoeff_x')[:ntest,::r,::r].reshape(ntest,-1)
test_a_grady = reader.read_field('Kcoeff_y')[:ntest,::r,::r].reshape(ntest,-1)
test_u = reader.read_field('sol')[:ntest,::r,::r].reshape(ntest,-1)


a_normalizer = GaussianNormalizer(train_a)
train_a = a_normalizer.encode(train_a)
test_a = a_normalizer.encode(test_a)
as_normalizer = GaussianNormalizer(train_a_smooth)
train_a_smooth = as_normalizer.encode(train_a_smooth)
test_a_smooth = as_normalizer.encode(test_a_smooth)
agx_normalizer = GaussianNormalizer(train_a_gradx)
train_a_gradx = agx_normalizer.encode(train_a_gradx)
test_a_gradx = agx_normalizer.encode(test_a_gradx)
agy_normalizer = GaussianNormalizer(train_a_grady)
train_a_grady = agy_normalizer.encode(train_a_grady)
test_a_grady = agy_normalizer.encode(test_a_grady)

u_normalizer = UnitGaussianNormalizer(train_u)
train_u = u_normalizer.encode(train_u)
# test_u = y_normalizer.encode(test_u)



meshgenerator = RandomMeshGenerator([[0,1],[0,1]],[s,s], sample_size=m)
data_train = []
for j in range(ntrain):
    for i in range(k):
        idx = meshgenerator.sample()
        grid = meshgenerator.get_grid()
        edge_index = meshgenerator.ball_connectivity(radius_train)
        edge_attr = meshgenerator.attributes(theta=train_a[j,:])
        #data_train.append(Data(x=init_point.clone().view(-1,1), y=train_y[j,:], edge_index=edge_index, edge_attr=edge_attr))
        data_train.append(Data(x=torch.cat([grid, train_a[j, idx].reshape(-1, 1),
                                            train_a_smooth[j, idx].reshape(-1, 1), train_a_gradx[j, idx].reshape(-1, 1),
                                            train_a_grady[j, idx].reshape(-1, 1)
                                            ], dim=1),
                               y=train_u[j, idx], edge_index=edge_index, edge_attr=edge_attr, sample_idx=idx
                               ))


meshgenerator = RandomMeshGenerator([[0,1],[0,1]],[s,s], sample_size=m)
data_test = []
for j in range(ntest):
    idx = meshgenerator.sample()
    grid = meshgenerator.get_grid()
    edge_index = meshgenerator.ball_connectivity(radius_test)
    edge_attr = meshgenerator.attributes(theta=test_a[j,:])
    data_test.append(Data(x=torch.cat([grid, test_a[j, idx].reshape(-1, 1),
                                       test_a_smooth[j, idx].reshape(-1, 1), test_a_gradx[j, idx].reshape(-1, 1),
                                       test_a_grady[j, idx].reshape(-1, 1)
                                       ], dim=1),
                          y=test_u[j, idx], edge_index=edge_index, edge_attr=edge_attr, sample_idx=idx
                          ))
#
train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(data_test, batch_size=batch_size2, shuffle=False)

t2 = default_timer()

print('preprocessing finished, time used:', t2-t1)
device = torch.device('cuda')

model = KernelNN(width, ker_width,depth,edge_features,in_width=node_features).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

myloss = LpLoss(size_average=False)
u_normalizer.cuda()
ttrain = np.zeros((epochs, ))
ttest = np.zeros((epochs,))
model.train()
for ep in range(epochs):
    t1 = default_timer()
    train_mse = 0.0
    train_l2 = 0.0
    for batch in train_loader:
        batch = batch.to(device)

        optimizer.zero_grad()
        out = model(batch)
        mse = F.mse_loss(out.view(-1, 1), batch.y.view(-1,1))
        # mse.backward()

        l2 = myloss(
            u_normalizer.decode(out.view(batch_size, -1), sample_idx=batch.sample_idx.view(batch_size, -1)),
            u_normalizer.decode(batch.y.view(batch_size, -1), sample_idx=batch.sample_idx.view(batch_size, -1)))
        l2.backward()

        optimizer.step()
        train_mse += mse.item()
        train_l2 += l2.item()

    scheduler.step()
    t2 = default_timer()

    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)
            out = u_normalizer.decode(out.view(batch_size2,-1), sample_idx=batch.sample_idx.view(batch_size2,-1))
            test_l2 += myloss(out, batch.y.view(batch_size2, -1)).item()
            # test_l2 += myloss(out.view(batch_size2,-1), y_normalizer.encode(batch.y.view(batch_size2, -1))).item()

    t3 = default_timer()
    ttrain[ep] = train_l2/(ntrain * k)
    ttest[ep] = test_l2/ntest

    print(ep, t2-t1, t3-t2, train_mse/len(train_loader), train_l2/(ntrain * k), test_l2/ntest)

runtime[0] = t2-t1
runtime[1] = t3-t2
np.savetxt(path_train_err, ttrain)
np.savetxt(path_test_err, ttest)
np.savetxt(path_runtime, runtime)
torch.save(model, path_model)


