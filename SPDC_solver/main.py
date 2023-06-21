from SPDC_solver import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pickle

with open(file='/home/dor-hay.sha/project/data/spdc/fixed_pump_10.bin',mode="rb") as file:
    dict = pickle.load(file)
fields = dict["fields"]

dict = {0:"pump", 1:"signal vac", 2:"idler vac", 3:"signal out", 4:"idler out"}
maxX = 120e-6
dx = 2e-6
x = np.arange(-maxX, maxX, dx)
X,Y = np.meshgrid(x,x)
for i in range(5):
    fig, ax = plt.subplots(dpi=150,subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, np.mean(np.abs(fields[:,i,:,:,-1])**2,axis=0), cmap=cm.coolwarm,linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title(f"{dict[i]}")
plt.show()
