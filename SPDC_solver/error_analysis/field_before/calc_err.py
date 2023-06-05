import sys
import pandas as pd
sys.path.insert(1, '/mnt/c/Users/dorsh/Documents/technion/semester8/project/physics_informed/SPDC_solver/')
import json
import numpy as np
from SPDC_solver import *


def stat(err):
    for mse in err:
        (np.max(mse),np.min(mse),np.mean(mse),np.median(mse),np.std(mse),mse[-1])

'''
Calculate the error on different sizes and save them into a file
'''
dxdy = [1e-6,2e-6,3e-6,4e-6,5e-6,6e-6]
dz = [2e-6,5e-6,10e-6,20e-6]
maxXY = [80e-6,120e-6,160e-6,200e-6,240e-6]
maxZ = [1e-4,2e-4,3e-4,4e-4,5e-4]


errors = {}



print("dxdy")
err = []
for d in  dxdy:
    shape = Shape(dx=d,dy=d)
    A = SPDC_solver(return_err=True,shape=shape)
    err.append(A.solve())
errors["dxdy"] = [err]


print("dz")
err = []
for d in  dz:
    shape = Shape(dz=d)
    A = SPDC_solver(return_err=True,shape=shape)
    err.append(A.solve())
errors["dz"] = [err]



print("maxXY")
err = []
for mxy in  maxXY:
    shape = Shape(maxX=mxy,maxY=mxy)
    A = SPDC_solver(return_err=True,shape=shape)
    err.append(A.solve())
errors["maxXY"] = [err]

print("maxZ")
err = []
for mz in  maxZ:
    shape = Shape(maxZ=mz)
    A = SPDC_solver(return_err=True,shape=shape)
    err.append(A.solve())
errors["maxZ"] = [err]


df = pd.DataFrame(errors)
df.to_pickle("./errors.txt")
print("Done!")