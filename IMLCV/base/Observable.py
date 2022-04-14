from __future__ import annotations
from ast import arg
from pickle import BINSTRING

import os

from IMLCV.base.bias import Bias, GridBias
from IMLCV.base.rounds import RoundsMd

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

    samples_per_bin = 10

    def __init__(self, rounds: RoundsMd) -> None:
        self.rounds = rounds

        self.folder = rounds.folder
        self.fes = None

    def fes_2D(self, plot=True):
        # fes = FreeEnergySurface2D.from_txt
        if self.fes is not None:
            return self.fes

        temp = self.rounds.T

        if plot == True:
            self.plot()

        trajs, bss, bins, pinit = self._get_biasses(plot=plot)

        histo = Histogram2D.from_wham_c(
            bins=bins,
            pinit=pinit,
            traj_input=trajs,
            error_estimate='mle_f',
            biasses=bss,
            temp=temp,
        )

        fes = FreeEnergySurface2D.from_histogram(histo, temp)
        fes.set_ref()

        self.fes = fes

        if plot:
            bias = self.fes_Bias(internal=True)

            fesbias = Observable._sample_bias(bias, self.plot_mg)

            if plot:
                Observable._plot({
                    'bias': fesbias,
                    'name': f'{self.folder}/FES_thermolib_{self.rounds.round}',
                    'mg': self.plot_mg,
                })

        return fes

    def _get_biasses(self, plot=False):
        trajs = []
        tbss = []

        for dict in self.rounds.iter_md_runs():
            pos = dict["positions"][:]
            bias = Bias.load(dict.attrs["name_bias"])
            if 'cell' in dict:
                cell = dict["cell"][:]
                arr = np.array([bias.cvs.compute(coordinates=x, cell=y)[0] for (x, y) in zip(pos, cell)],
                               dtype=np.double)
            else:
                arr = np.array([bias.cvs.compute(coordinates=p, cell=None)[0] for p in pos], dtype=np.double)

            trajs.append(arr)
            tbss.append(Observable._thermo_bias2D(bias))

        temp = self.rounds.T

        n = 0
        for t in trajs:
            n += t.size

        #20 points per bin on average
        n = int(n**(1 / trajs[0].ndim) / self.samples_per_bin)

        assert n > 4, "sample more points"

        bins = self._grid(n=n, endpoint=True)
        bin_centers = [0.5 * (row[:-1] + row[1:]) for row in bins]
        beta = 1 / (boltzmann * temp)
        cb = self.rounds.get_bias()
        mg = np.meshgrid(*bin_centers)

        biases = Observable._sample_bias(cb, mg)

        self.mg = mg

        pinit = np.exp(-biases * beta)
        pinit = np.array(pinit, dtype=np.double)

        return trajs, tbss, bins, pinit

    def plot(self):
        trajs = []
        bss = []

        for dict in self.rounds.iter_md_runs():
            pos = dict["positions"][:]
            bias = Bias.load(dict.attrs["name_bias"])
            if 'cell' in dict:
                cell = dict["cell"][:]
                arr = np.array([bias.cvs.compute(coordinates=x, cell=y)[0] for (x, y) in zip(pos, cell)],
                               dtype=np.double)
            else:
                arr = np.array([bias.cvs.compute(coordinates=p, cell=None)[0] for p in pos], dtype=np.double)

            trajs.append(arr)
            bss.append(bias)

        bins = self._grid(n=50, endpoint=True)
        # bin_centers = [0.5 * (row[:-1] + row[1:]) for row in bins]
        # beta = 1 / (boltzmann * temp)
        cb = self.rounds.get_bias()
        plot_mg = np.meshgrid(*bins)
        self.plot_mg = plot_mg

        biases = Observable._sample_bias(cb, plot_mg)

        dir = f'{self.folder}/round_{self.rounds.round}'

        n = self.rounds.n()

        args = [{
            'bias': biases,
            'name': f'{dir}/combined',
            'trajs': trajs,
            'mg': plot_mg,
        }]

        for i, b in enumerate(bss[-n:]):
            bias = Observable._sample_bias(b, plot_mg)

            args.append({
                'bias': bias,
                'name': f'{dir}/umbrella_{i}',
                'trajs': [trajs[i]],
                'mg': plot_mg,
            })

        #async plot and continue
        with pathos.pools.ProcessPool() as pool:
            pool.amap(Observable._plot, args)

    @staticmethod
    def _sample_bias(b, mg):

        bias, _ = jnp.apply_along_axis(b.compute, axis=0, arr=np.array(mg), diff=False)
        return bias

    @staticmethod
    def _plot(args):
        b = args.get('bias')
        trajs = args.get('trajs')
        name = args.get('name')
        mg = args.get('mg')

        xlim = [mg[0].min(), mg[0].max()]
        ylim = [mg[1].min(), mg[1].max()]

        extent = [xlim[0], xlim[1], ylim[0], ylim[1]]

        plt.clf()
        p = plt.imshow(b / (kjmol), cmap=plt.get_cmap('rainbow'), origin='lower', extent=extent, vmin=0.0, vmax=100.0)
        # p = plt.contourf(mg[0], mg[1], b / kjmol, cmap=plt.get_cmap('rainbow'))

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

    def fes_Bias(self, internal=False):
        fes = self.fes_2D(plot=False)
        fes_interp = Observable._interp(fes.fs)
        bias = np.transpose(fes_interp)

        if internal == False:
            bias = -bias
            bias[:] -= bias.min()

        #append row cyclcally
        bias2 = np.zeros(np.array(bias.shape) + 1)
        bias2[:-1, :-1] = bias[:, :]
        bias2[-1, :] = bias2[0, :]
        bias2[:, -1] = bias2[:, 0]

        return GridBias(cvs=self.rounds.get_bias().cvs, vals=bias2)

    def _grid(self, n=51, cvs=None, endpoint=True):
        if cvs is None:
            cvs = self.rounds.get_bias().cvs
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
