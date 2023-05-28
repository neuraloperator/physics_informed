from abc import ABC
from jax import jit
import math
import jax.numpy as np
import jax.random as random
import os
from jax.ops import index_update
from typing import Dict
from utils import *
from solver import *
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import pandas as pd

# second harmonic generation coupled wave equations
class SHGCWE(object):
        """
    Class that holds all the problem's information and solve it numericly with split step

    Parameters
    ----------
    shape: A class that holds everything to do with the dimensions of the problem
    config: A class that holds everything to do with the constant in the problem
    N: Number of vacum field
    seed: the seed for the randomization
    pump_coef: Holds the coef for building the pump profile from LG basis
    crystal_coef: Holds the coef for building the crystal profile from LG basis
    check_sol: If True prints the log of MSE on each of the equations at each step
    interaction: A class that represents the SPDC interaction process, on all of its physical parameters.
    poling_period: Poling period (dk_offset * delta_k) :=
      # = interaction.dk_offset * self.delta_k, 
      delta_k= pump.k - signal.k - idler.k  
      # phase  mismatch
    N: number of vacuum_state elements
    crystal_hologram: 3D crystal hologram
    infer: (True/False) if in inference mode, we include more coefficients in the poling
                description for better validation
    signal_init: initial signal profile. If None, initiate to zero
    idler_init: initial idler profile. If None, initiate to zero
    check_sol: calculte the MSE of the solution on the equation at each dz


    """

        def __init__(self,
                     shape = Shape(),
                     config = Config(),
                     N = 1,
                     seed = 1701,
                     pump_coef = {"max_mode1": 1, "max_mode2":0, "real_coef":np.array([1]),"img_coef":np.array([0])},
                     crystal_coef = {"max_mode1": 1, "max_mode2":0, "real_coef":np.array([1]),"img_coef":np.array([0])},
                     check_sol = False,
                     draw_sol = False
                     ):
                         

                  self.shape = shape
                  self.config = config
                  self.N = N
                  self.pump_coef = pump_coef
                  self.crystal_coef = crystal_coef
                  self.check_sol = check_sol
                  self.draw_sol = draw_sol

                  self.pump = pump = Beam(lam=config.pump_lam, polarization="y", T=config.T, power=config.pump_power) 
                  self.signal = signal = Beam(lam=2*pump.lam, polarization="y", T=config.T, power=config.signal_power)
                  self.idler = idler = Beam(lam=SFG_idler_wavelength(pump.lam,signal.lam), polarization="z", T=config.T, power=config.idler_power)
                  self.signal_field = Field(beam = signal,dx=shape.dx,dy=shape.dy,maxZ=shape.maxZ)
                  self.idler_field = Field(beam = idler,dx=shape.dx,dy=shape.dy,maxZ=shape.maxZ)

                # change to gauusian, and 
                  X,Y = np.meshgrid(shape.x,shape.y)
                #pump
                  pump_max_mode1 = pump_coef["max_mode1"]
                  pump_max_mode2 = pump_coef["max_mode2"]
                  pump_real_coef = pump_coef["real_coef"]
                  pump_img_coef = pump_coef["img_coef"]
                  self.pump_profile = profile_laguerre_gauss(pump_real_coef,pump_img_coef,config.pump_waist,shape,pump_max_mode1,pump_max_mode2,pump,mode="pump")
                # crystal
                  crystal_max_mode1 = crystal_coef["max_mode1"]
                  crystal_max_mode2 = crystal_coef["max_mode2"]
                  crystal_real_coef = crystal_coef["real_coef"]
                  crystal_img_coef = crystal_coef["img_coef"]
                  crystal_profile = profile_laguerre_gauss(crystal_real_coef,crystal_img_coef,config.r_scale0,shape,crystal_max_mode1,crystal_max_mode2,signal,mode="crystal")

                  delta_k = pump.k - signal.k - idler.k  
                  poling_period = config.dk_offset * delta_k
                  if check_sol:
                        crystal_profile = None
      
                  PP = PP_crystal_slab(delta_k=delta_k, shape=shape, crystal_profile=crystal_profile)
                  self.chi2 = PP*config.d33 
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
                check_sol = self.check_sol) 

                if self.draw_sol:
                        dict = {0:"signal out", 1:"signal vac", 2:"idler out", 3:"idler vac"}
                        n = {0:self.signal.n, 1:self.signal.n, 2:self.idler.n, 3:self.idler.n}
                        X,Y = np.meshgrid(self.shape.x,self.shape.y)
                        for i in range(4):
                                if (i%1==0):
                                         fig, ax = plt.subplots(dpi=150,subplot_kw={"projection": "3d"})
                                         surf = ax.plot_surface(X, Y, np.mean(I(sol[i],n[i]),axis=0), cmap=cm.coolwarm,linewidth=0, antialiased=False)
                                         fig.colorbar(surf, shrink=0.5, aspect=5)
                                         plt.title(f"{dict[i]}")
                        plt.show()

                return sol