import imp
from IMLCV.base.MdEngine import MDEngine
from IMLCV.base.bias import Bias, GridBias

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
from thermolib.thermodynamics.histogram import Histogram2D, plot_histograms
from thermolib.thermodynamics.bias import BiasPotential2D
import numpy as np, os
from molmod.units import *


class Observable:
    """class to convert data and CVs to different thermodynamic/ kinetic observables."""

    def __init__(self, bias: Bias, traj, temp) -> None:
        self.mdb = bias
        self.traj = traj
        self.temp = temp

        self.fes = None

    def fes_2D(self, round, plot=True):
        # fes = FreeEnergySurface2D.from_txt
        if self.fes is not None:
            return self.fes

        cv = []
        bss = []

        for (traj, bias) in zip(self.traj, self.mdb):
            arr = np.array([bias.cvs.compute(t.positions, cell=t.cell.array)[0] for t in traj])
            cv.append(arr)
            bss.append(Observable._thermo_bias2D(bias))

        grid = self._grid(n=50)

        histo = Histogram2D.from_wham(bins=grid,
                                      trajectories=cv,
                                      biasses=bss,
                                      temp=self.temp,
                                      error_estimate='mle_f',
                                      plot_biases=True)

        fes = FreeEnergySurface2D.from_histogram(histo, self.temp)
        fes.set_ref()

        if plot:
            fes.plot('output/ala_fes_thermolib_{}.png'.format(round))

        self.fes = fes

        return fes

    class _thermo_bias2D(BiasPotential2D):

        def __init__(self, bias: Bias) -> None:
            self.bias = bias

            super().__init__("IMLCV_bias")

        def __call__(self, cv1, cv2):
            cvs = jnp.array([cv1, cv2])
            b, _ = jnp.apply_along_axis(self.bias.compute, axis=0, arr=cvs, diff=False)
            return b

        def print_pars(self, *pars_units):
            pass

    def fes_Bias(self):
        fes = self.fes_2D(plot=False)

        #adapt bias, apparently tranpose is needed here
        bias = -np.transpose(fes.fs)
        bias -= bias[~np.isnan(bias)].min()
        bias[np.isnan(bias)] = 0.0

        return GridBias(cvs=self.mdb[-1].cvs, vals=bias)

    def _plot_bias(self):
        """manual bias plot, convert to _thermo_bias2D and plot instead."""

        b = self.mdb[-1]

        x = self._grid(n=51)
        mg = np.array(np.meshgrid(*x))  #(ncv,n,n) matrix
        biases, _ = jnp.apply_along_axis(b.compute, axis=0, arr=mg, diff=False)
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

    def _grid(self, n=51):
        b = self.mdb[-1]
        if np.isnan(b.cvs.periodicity).any():
            raise NotImplementedError("add argument for range")

        x = [np.linspace(p[0][0], p[0][1], n, endpoint=False) for p in np.split(b.cvs.periodicity, 2, axis=0)]

        return x