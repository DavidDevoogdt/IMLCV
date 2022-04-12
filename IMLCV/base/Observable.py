from __future__ import annotations
from pickle import BINSTRING

from IMLCV.base.bias import Bias, GridBias

from thermolib.thermodynamics.fep import SimpleFreeEnergyProfile, FreeEnergySurface2D, plot_feps
from thermolib.thermodynamics.histogram import Histogram2D, plot_histograms
from thermolib.thermodynamics.bias import BiasPotential2D

from molmod.units import kjmol, femtosecond
from molmod.constants import boltzmann

import numpy as np
import jax.numpy as jnp
import scipy
from scipy.interpolate import griddata

import matplotlib.pyplot as plt

# from IMLCV.scheme import Rounds


class Observable:
    """class to convert data and CVs to different thermodynamic/ kinetic observables."""

    def __init__(self, rounds) -> None:
        self.rounds = rounds

        self.fes = None

    def fes_2D(self, plot=True):
        # fes = FreeEnergySurface2D.from_txt
        if self.fes is not None:
            return self.fes

        temp = self.rounds.T

        trajs, bss, bins = self._get_biasses(plot=plot)

        histo = Histogram2D.from_wham_c(
            bins=bins,
            traj_input=trajs,
            error_estimate='mle_f',
            biasses=bss,
            temp=temp,
        )

        fes = FreeEnergySurface2D.from_histogram(histo, temp)
        fes.set_ref()

        if plot:
            fes.plot('output/ala_fes_thermolib_{}.png'.format(self.rounds.round))

        self.fes = fes

        return fes

    def _get_biasses(self, plot=False):
        trajs = []
        tbss = []
        bss = []

        for (traj, bias) in self.rounds.get_trajectories_and_biases():
            arr = np.array([bias.cvs.compute(t.positions, cell=t.cell.array)[0] for t in traj], dtype=np.double)
            trajs.append(arr)
            bss.append(bias)
            tbss.append(Observable._thermo_bias2D(bias))

        temp = self.rounds.T

        bins = self._grid(n=51, endpoint=True)
        bin_centers = [0.5 * (row[:-1] + row[1:]) for row in bins]
        beta = 1 / (boltzmann * temp)
        cb = self.rounds.commom_bias()
        mg = np.meshgrid(*bin_centers)

        biases, _ = jnp.apply_along_axis(cb.compute, axis=0, arr=np.array(mg), diff=False)

        self.mg = mg

        # pinit = np.exp(-biases * beta)

        def pl(b, name, trajs):
            plt.clf()
            contourf = plt.contourf(mg[0], mg[1], b, cmap=plt.get_cmap('rainbow'))
            contour = plt.contour(mg[0], mg[1], b)

            plt.xlabel('cv1', fontsize=16)
            plt.ylabel('cv2', fontsize=16)
            cbar = plt.colorbar(contourf, extend='both')
            cbar.set_label('Bias boltzmann factor [-]', fontsize=16)
            plt.clabel(contour, inline=1, fontsize=10)

            for traj in trajs:
                plt.scatter(traj[:, 0], traj[:, 1])

            plt.title(name)

            fig = plt.gcf()
            fig.set_size_inches([12, 8])
            plt.savefig(name)

        if plot:
            pl(biases, f'output/mtd_{self.rounds.round}', trajs)

            for i, b in enumerate(bss):
                bias, _ = jnp.apply_along_axis(b.compute, axis=0, arr=np.array(mg), diff=False)
                pl(bias, f'output/umbrella {i}', [trajs[i]])

        return trajs, tbss, bins

    class _thermo_bias2D(BiasPotential2D):

        def __init__(self, bias: Bias) -> None:
            self.bias = bias

            super().__init__("IMLCV_bias")

        def __call__(self, cv1, cv2):
            cvs = jnp.array([cv1, cv2])
            b, _ = jnp.apply_along_axis(self.bias.compute, axis=0, arr=cvs, diff=False)
            return np.array(b, dtype=np.double)

        def print_pars(self, *pars_units):
            pass

    def fes_Bias(self):
        fes = self.fes_2D(plot=False)

        #adapt bias, apparently tranpose is needed here
        bias = -np.transpose(fes.fs)

        # bias2 = griddata(   bias.shape  , bias, self.mg, method='linear')

        # Observable._thermo_bias2D(fesBias).plot('file', *Observable._grid(51, cvs= self.md.bias.cvs )  )

        bias -= bias[~np.isnan(bias)].min()
        bias[np.isnan(bias)] = 0.0

        return GridBias(cvs=self.rounds.commom_bias().cvs, vals=bias)

    def _plot_bias(self):
        """manual bias plot, convert to _thermo_bias2D and plot instead."""

        b = self.mdb[-1]

        x = self._grid(n=51, endpoint=False)
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

    def _grid(self, n=51, cvs=None, endpoint=True):
        if cvs is None:
            cvs = self.rounds._get_prop(0, 0, 'cv')
        if np.isnan(cvs.periodicity).any():
            raise NotImplementedError("add argument for range")

        return [np.linspace(p[0][0], p[0][1], n, endpoint=True) for p in np.split(cvs.periodicity, 2, axis=0)]
