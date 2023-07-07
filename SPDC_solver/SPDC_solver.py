from abc import ABC
from jax import jit
import math
import jax.numpy as np
import jax.random as random
from utils import *
from solver import *
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import numpy

# second harmonic generation coupled wave equations
class SPDC_solver(object):
        """
    Class that holds all the problem's information and solve it numericly with split step

    Parameters
    ----------
    shape: A class that holds everything to do with the dimensions of the problem
    config: A class that holds everything to do with the constant in the problem
    N: Number of vacum field
    seed: the seed for the randomization
    pump_coef: Holds the coef for building the pump profile from LG basis
    is_crystal: True/False if there is a structrue to the crystal
    crystal_coef: Holds the coef for building the crystal profile from LG basis
    print_err: If True prints the log of MSE on each of the equations at each step
    return_err: If True return the MSE on each of the equations at each step
    draw_sol: draw a 3D graph of the intensety of each field at the end of the propogation
    data_creation: If true creates a dictonary conataina:
                   fields:
                    np ndarray at the shape (N,F = 5,X,Y,Z) where:
                    N - number of samples
                    F=5 - The 5 fields: pump, signal vac, idler vac, signal out, idler out
                    X - Nx i.e number of elements in the X array
                    Y - Ny i.e number of elements in the Y array
                    Z - Nz i.e number of elements in the Z array
                   chi:
                    np ndarray at shape (X,Y,Z) containg chi2 at each point of the grid
                  k_pump: k_pump
                  k_signal: k_signal
                  k_idler: k_idler
                  kappa_signal: kappa_signal
                  kappa_idler: kappa_idler
                    

    """

        def __init__(self,
                     shape = Shape(),
                     config = Config(),
                     N = 1,
                     seed = 1701,
                     pump_coef = {"max_mode1": 1, "max_mode2":0, "real_coef":np.array([1]),"img_coef":np.array([0])},
                     is_crystal = False,
                     crystal_coef = {"max_mode1": 1, "max_mode2":0, "real_coef":np.array([1]),"img_coef":np.array([0])},
                     print_err = False,
                     return_err = False,
                     draw_sol = False,
                     data_creation = False
                     ):
                         

                  self.shape = shape
                  self.config = config
                  self.N = N
                  self.pump_coef = pump_coef
                  self.crystal_coef = crystal_coef
                  self.is_crystal = is_crystal
                  self.check_sol = print_err or return_err
                  self.print_err = print_err
                  self.return_err = return_err
                  self.draw_sol = draw_sol
                  self.data_creation = data_creation
                  self.data = None
                  self.fields = None


                  if self.check_sol:
                        N = 1
                        is_crystal = False

                  self.pump = pump = Beam(lam=config.pump_lam, polarization="y", T=config.T, power=config.pump_power) 
                  self.signal = signal = Beam(lam=2*pump.lam, polarization="y", T=config.T, power=config.signal_power)
                  self.idler = idler = Beam(lam=SFG_idler_wavelength(pump.lam,signal.lam), polarization="z", T=config.T, power=config.idler_power)
                  self.signal_field = Field(beam = signal,dx=shape.dx,dy=shape.dy,maxZ=shape.maxZ)
                  self.idler_field = Field(beam = idler,dx=shape.dx,dy=shape.dy,maxZ=shape.maxZ)

                # change to gauusian, and 
                #pump
                  pump_max_mode1 = pump_coef["max_mode1"]
                  pump_max_mode2 = pump_coef["max_mode2"]
                  pump_real_coef = pump_coef["real_coef"]
                  pump_img_coef = pump_coef["img_coef"]
                  self.pump_profile = profile_laguerre_gauss(pump_real_coef,pump_img_coef,config.pump_waist,shape,pump_max_mode1,pump_max_mode2,pump,mode="pump")
                # crystal
                  crystal_profile = None
                  if is_crystal:
                    crystal_max_mode1 = crystal_coef["max_mode1"]
                    crystal_max_mode2 = crystal_coef["max_mode2"]
                    crystal_real_coef = crystal_coef["real_coef"]
                    crystal_img_coef = crystal_coef["img_coef"]
                    crystal_profile = profile_laguerre_gauss(crystal_real_coef,crystal_img_coef,config.r_scale0,shape,crystal_max_mode1,crystal_max_mode2,signal,mode="crystal")

                  delta_k = pump.k - signal.k - idler.k  
                  poling_period = config.dk_offset * delta_k

                  PP = PP_crystal_slab(delta_k=delta_k, shape=shape, crystal_profile=crystal_profile)
                  self.chi2 = PP*config.d33 

                  if data_creation:
                        self.data = {}
                        self.fields =  numpy.zeros(shape=(N,5,shape.Nx,shape.Ny,shape.Nz),dtype=complex)
                        self.data["fields"] = self.fields
                        self.data["chi"] = self.chi2
                        self.data["k_pump"] = pump.k
                        self.data["k_signal"] = signal.k
                        self.data["k_idler"] = idler.k
                        self.data["kappa_signal"] = self.signal_field.kappa
                        self.data["kappa_idler"] = self.idler_field.kappa


                  
                # Random N vacum state
                  key = random.PRNGKey(seed)
                  rand_key, subkey = random.split(key)
                  self.vacuum_states = random.normal(subkey,shape=(N,2,2,shape.Nx,shape.Ny))

        def solve(self):
                sol = crystal_prop(
                pump_profile = self.pump_profile, 
                pump = self.pump,
                signal_field = self.signal_field, 
                idler_field = self.idler_field,
                vacuum_states = self.vacuum_states,
                chi2 = self.chi2, 
                N = self.N,
                shape = self.shape,
                print_err = self.print_err,
                return_err = self.return_err,
                data = self.fields
                ) 

                if self.draw_sol:
                        dict = {0:"signal out", 1:"signal vac", 2:"idler out", 3:"idler vac"}
                        n = {0:self.signal.n, 1:self.signal.n, 2:self.idler.n, 3:self.idler.n}
                        X,Y = np.meshgrid(self.shape.x,self.shape.y, indexing='ij')
                        for i in range(4):
                                if (i%1==0):
                                         fig, ax = plt.subplots(dpi=150,subplot_kw={"projection": "3d"})
                                         surf = ax.plot_surface(X, Y, np.mean(I(sol[i],n[i]),axis=0), cmap=cm.coolwarm,linewidth=0, antialiased=False)
                                         fig.colorbar(surf, shrink=0.5, aspect=5)
                                         plt.title(f"{dict[i]}")
                        plt.show()

                if self.return_err:
                      err =  np.array(sol[-1]).T
                      return err
                
                return sol