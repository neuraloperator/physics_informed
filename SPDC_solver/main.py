from SPDC_solver import *
from utils import *

A = SPDC_solver(draw_sol=True,N=100,print_err=True,config=Config(pump_waist=150e-6))
A.solve()
print("Done")