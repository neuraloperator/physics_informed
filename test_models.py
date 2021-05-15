import torch
from models import FNN3d

modes = 8
width = 12

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
layers = [width*2//4, width*3//4, width*3//4, width*4//4, width*4//4]
modes = [modes * (5-i) // 4 for i in range(4)]


data = torch.randn((1, 12, 32, 32, 4)).to(device)
model = FNN3d(modes1=modes, modes2=modes, modes3=modes, layers=layers).to(device)
x = model(data)
print(x.shape)
