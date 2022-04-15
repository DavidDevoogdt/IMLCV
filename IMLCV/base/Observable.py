from __future__ import annotations
from ast import arg
from pickle import BINSTRING

import os
from IMLCV.base.CV import CV

from IMLCV.base.bias import Bias, GridBias
from IMLCV.base.rounds import RoundsMd, Rounds, RoundsCV

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


class Observable:
    """class to convert data and CVs to different thermodynamic/ kinetic observables."""

    samples_per_bin = 10

    def __init__(self, rounds: Rounds, cvs: CV = None) -> None:
        self.rounds = rounds

        if isinstance(rounds, RoundsMd):
            assert cvs is None
            self.cvs = self.rounds.get_bias().cvs
        elif isinstance(rounds, RoundsCV):
            assert cvs is not None
            self.cvs = cvs
        else:
            raise NotImplementedError

        self.folder = rounds.folder
        self.fes = None

        #plot meshgrid
        bins = self._grid(n=50, endpoint=True)
        self.plot_mg = np.meshgrid(*bins)

    def _fes_2D(self, plot=True):
        # fes = FreeEnergySurface2D.from_txt
        if self.fes is not None:
            return self.fes

        temp = self.rounds.T
        beta = 1 / (boltzmann * temp)

        if isinstance(self.rounds, RoundsMd):
            if plot == True:
                self._plot_biases()

            trajs = []
            biases = []

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
                biases.append(Observable._thermo_bias2D(bias))

            mg, bins = self._FES_mg(trajs=trajs)

            pinit = Observable._sample_bias(self.rounds.get_bias(), mg)
            pinit = np.exp(-pinit * beta)
            pinit = np.array(pinit, dtype=np.double)

            histo = Histogram2D.from_wham_c(
                bins=bins,
                pinit=pinit,
                traj_input=trajs,
                error_estimate='mle_f',
                biasses=biases,
                temp=temp,
            )
        elif isinstance(self.rounds, RoundsCV):

            trajs = []
            for dict in self.rounds.iter_md_runs():
                pos = dict["positions"][:]

                if 'cell' in dict:
                    cell = dict["cell"][:]
                    arr = np.array([self.cvs.compute(coordinates=x, cell=y)[0] for (x, y) in zip(pos, cell)],
                                   dtype=np.double)
                else:
                    arr = np.array([self.cvs.compute(coordinates=p, cell=None)[0] for p in pos], dtype=np.double)

                trajs.append(arr)

            if plot == True:
                dir = f'{self.folder}/round_{self.rounds.round}'
                self._plot({
                    'bias': self.plot_mg[0] * 0,
                    'name': f'{dir}/combined',
                    'trajs': trajs,
                    'mg': self.plot_mg,
                })

            mg, bins = self._FES_mg(trajs=trajs)
            data = np.vstack(trajs)
            histo = Histogram2D.from_single_trajectory(
                data,
                bins,
                error_estimate='mle_f',
            )
        else:
            raise NotImplementedError

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

    def _FES_mg(self, trajs):
        n = 0
        for t in trajs:
            n += t.size

        #20 points per bin on average
        n = int(n**(1 / trajs[0].ndim) / self.samples_per_bin)

        assert n > 4, "sample more points"

        bins = self._grid(n=n, endpoint=True)
        bin_centers = [0.5 * (row[:-1] + row[1:]) for row in bins]

        mg = np.meshgrid(*bin_centers)

        return mg, bins

    def _plot_biases(self):
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

        # bin_centers = [0.5 * (row[:-1] + row[1:]) for row in bins]
        # beta = 1 / (boltzmann * temp)
        cb = self.rounds.get_bias()

        biases = Observable._sample_bias(cb, self.plot_mg)

        dir = f'{self.folder}/round_{self.rounds.round}'

        n = self.rounds.n()

        args = [{
            'bias': biases,
            'name': f'{dir}/combined',
            'trajs': trajs,
            'mg': self.plot_mg,
        }]

        for i, b in enumerate(bss[-n:]):
            bias = Observable._sample_bias(b, self.plot_mg)

            args.append({
                'bias': bias,
                'name': f'{dir}/umbrella_{i}',
                'trajs': [trajs[i]],
                'mg': self.plot_mg,
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

    def fes_Bias(self, kind='normal', plot=False, internal=False):
        fes = self._fes_2D(plot=plot)

        if kind == 'normal':
            fs = fes.fs
        elif kind == 'fupper':
            fs = fes.fupper

        fes_interp = Observable._interp(fs)
        bias = np.transpose(fes_interp)

        if internal == False:
            bias = -bias
            bias[:] -= bias.min()

        return GridBias(cvs=self.cvs, vals=bias)

    def _grid(self, n=51, cvs=None, endpoint=True):
        if cvs is None:
            cvs = self.cvs
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
