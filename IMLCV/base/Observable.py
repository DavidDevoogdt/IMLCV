from __future__ import annotations
from ast import arg
from pickle import BINSTRING

import os

from IMLCV.base.bias import Bias, GridBias
from IMLCV.base.rounds import Rounds

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

import pathos

# from IMLCV.scheme import Rounds


class Observable:
    """class to convert data and CVs to different thermodynamic/ kinetic observables."""

    def __init__(self, rounds: Rounds) -> None:
        self.rounds = rounds

        self.folder = rounds.folder
        self.fes = None

    def fes_2D(self, plot=True):
        # fes = FreeEnergySurface2D.from_txt
        if self.fes is not None:
            return self.fes

        temp = self.rounds.T

        trajs, bss, bins, pinit = self._get_biasses(plot=plot)

        histo = Histogram2D.from_wham_c(
            bins=bins,
            pinit=pinit,
            traj_input=trajs,
            error_estimate='mle_f',
            biasses=bss,
            temp=temp,
        )

        # histo = Histogram2D.from_wham(
        #     bins=bins,
        #     pinit=pinit,
        #     trajectories=trajs,
        #     error_estimate='mle_f',
        #     biasses=bss,
        #     temp=temp,
        # )

        fes = FreeEnergySurface2D.from_histogram(histo, temp)
        fes.set_ref()

        if plot:
            Observable._plot({
                'bias': fes.fs,
                'name': f'{self.folder}/FES_thermolib_{self.rounds.round}',
            })

            #interp:
            fs2 = Observable._interp(fes.fs)

            Observable._plot({
                'bias': fs2,
                'name': f'{self.folder}/FES_thermolib_interp_{self.rounds.round}',
            })

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

        pinit = np.exp(-biases * beta)
        pinit = np.array(pinit, dtype=np.double)

        if plot:
            dir = f'{self.folder}/round_{self.rounds.round}'

            n = self.rounds.data[-1]['num']

            xlim = [mg[0].min(), mg[0].max()]
            ylim = [mg[1].min(), mg[1].max()]

            args = [{
                'bias': biases,
                'name': f'{dir}/combined',
                'trajs': trajs,
                'xlim': xlim,
                'ylim': ylim,
            }]

            for i, b in enumerate(bss[-n:]):
                bias, _ = jnp.apply_along_axis(b.compute, axis=0, arr=np.array(mg), diff=False)
                args.append({
                    'bias': bias,
                    'name': f'{dir}/umbrella_{i}',
                    'trajs': [trajs[i]],
                    'xlim': xlim,
                    'ylim': ylim,
                })

            #async plot and continue
            with pathos.pools.ProcessPool() as pool:
                pool.amap(Observable._plot, args)

        return trajs, tbss, bins, pinit

    @staticmethod
    def _plot(args):
        b = args.get('bias')
        trajs = args.get('trajs')
        name = args.get('name')
        xlim = args.get('xlim')
        ylim = args.get('ylim')

        if xlim is None or ylim is None:
            extent = None
        else:
            extent = [xlim[0], xlim[1], ylim[0], ylim[1]]

        plt.clf()
        p = plt.imshow(b / (kjmol), cmap=plt.get_cmap('rainbow'), origin='lower', extent=extent, vmin=0.0, vmax=100.0)

        plt.xlabel('cv1', fontsize=16)
        plt.ylabel('cv2', fontsize=16)
        cbar = plt.colorbar(p)
        cbar.set_label('Bias [kJ/mol]', fontsize=16)

        if trajs is not None:
            for traj in trajs:
                plt.scatter(traj[:, 0], traj[:, 1], s=3)

        plt.title(name)

        fig = plt.gcf()
        fig.set_size_inches([12, 8])
        plt.savefig(name)

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
        fes_interp = Observable._interp(fes.fs)
        bias = -np.transpose(fes_interp)
        return GridBias(cvs=self.rounds.commom_bias().cvs, vals=bias)

    def _grid(self, n=51, cvs=None, endpoint=True):
        if cvs is None:
            cvs = self.rounds._get_prop(0, 0, 'cv')
        if np.isnan(cvs.periodicity).any():
            raise NotImplementedError("add argument for range")

        return [np.linspace(p[0][0], p[0][1], n, endpoint=True) for p in np.split(cvs.periodicity, 2, axis=0)]

    @staticmethod
    def _interp(bias):
        #extend periodically
        dims = bias.shape

        interp = np.tile(bias, (3, 3))

        x, y = np.indices(interp.shape)
        interp[np.isnan(interp)] = griddata(
            (x[~np.isnan(interp)], y[~np.isnan(interp)]),
            interp[~np.isnan(interp)],
            (x[np.isnan(interp)], y[np.isnan(interp)]),
            method='cubic',
        )
        return interp[dims[0]:2 * dims[0], dims[1]:2 * dims[1]]
