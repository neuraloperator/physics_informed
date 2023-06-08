from SPDC_solver import *
import numpy as np

A = SPDC_solver(N=100,config=Config(pump_waist=150e-6),data_creation=True)
A.solve()
print(A.data)