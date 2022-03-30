import imp
from IMLCV.base.MdEngine import MDEngine, Bias

import os
from yaff.log import log
import yaff.analysis.biased_sampling
import numpy as np
import matplotlib.pyplot as plt
from molmod import units

from functools import partial
from jax import vmap
import jax.numpy as jnp

from thermolib.thermodynamics.fep import SimpleFreeEnergyProfile, FreeEnergySurface2D, plot_feps
from thermolib.thermodynamics.histogram import Histogram2D
import numpy as np, os
from molmod.units import *


class Observable:
    """class to convert data and CVs to different thermodynamic/ kinetic observables."""

    def __init__(self, bias: Bias, traj) -> None:
        self.mdb = bias
        self.traj = traj

    def fes(self):
        # fes = FreeEnergySurface2D.from_txt
        Histogram2D.from_single_trajectory()

    def plot_bias(self):
        n = 51
        if np.isnan(self.mdb.cvs.periodicity).any():
            raise NotImplementedError("add argument for range")

        x = [np.linspace(p[0][0], p[0][1], n, endpoint=False) for p in np.split(self.mdb.cvs.periodicity, 2, axis=0)]
        mg = np.array(np.meshgrid(*x))  #(ncv,n,n) matrix
        biases, _ = jnp.apply_along_axis(self.mdb.compute, axis=0, arr=mg, diff=False)
        biases = -np.array(biases)

        if mg.shape[0] != 2:
            raise NotImplementedError("todo sum over extra dims to visualise")

        fes = biases - np.amin(biases)

        plt.clf()
        plt.contourf(mg[0, :, :], mg[1, :, :], fes / units.kjmol)
        plt.xlabel("CV1")
        plt.ylabel("CV2")
        plt.title("$F\,[\mathrm{kJ}\,\mathrm{mol}^{-1}]$")
        plt.colorbar()
        plt.savefig('output/ala_dipep.png')