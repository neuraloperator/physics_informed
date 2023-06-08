from SPDC_solver import *
import numpy as np

# creating data with fixed pump

N_samples = 1000
file_name = "/home/dor-hay.sha/project/data/spdc/fixed_pump.npy"

print("creating data")
A = SPDC_solver(N=N_samples,config=Config(pump_waist=150e-6),data_creation=True)
A.solve()
print("saveing data")
np.save(file=file_name,arr=A.data)
print("Done!")