from __future__ import annotations
from ast import arg
from functools import partial
from pickle import BINSTRING

import os
from textwrap import wrap
from IMLCV.base.CV import CV

from IMLCV.base.bias import Bias, CompositeBias, CvMonitor, GridBias, MinBias
from IMLCV.base.rounds import RoundsMd, Rounds, RoundsCV

from thermolib.thermodynamics.fep import SimpleFreeEnergyProfile, FreeEnergySurface2D, plot_feps
from thermolib.thermodynamics.histogram import Histogram2D, plot_histograms
from thermolib.thermodynamics.bias import BiasPotential2D

from molmod.units import kjmol, femtosecond
from molmod.constants import boltzmann

import numpy as np
import jax.numpy as jnp
import scipy

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

    def _fes_2D(self, plot=True):
        # fes = FreeEnergySurface2D.from_txt
        if self.fes is not None:
            return self.fes, self.bounds

        temp = self.rounds.T
        beta = 1 / (boltzmann * temp)

        common_bias = self.rounds.get_bias()
        dir = f'{self.folder}/round_{self.rounds.round}'

        if isinstance(self.rounds, RoundsMd):

            trajs = []
            trajs_wrapped = []
            biases = []
            plot_args = []

        for dict in self.rounds.iter(num=1):
            pos = dict["positions"][:]
            bias = Bias.load(dict['attr']["name_bias"])

            if 'cell' in dict:
                cell = dict["cell"][:]
                arr = np.array([bias.cvs.compute(coordinates=x, cell=y)[0] for (x, y) in zip(pos, cell)],
                               dtype=np.double)
            else:
                arr = np.array([bias.cvs.compute(coordinates=p, cell=None)[0] for p in pos], dtype=np.double)

            arr_wrap = np.array(np.apply_along_axis(bias.cvs.metric.wrap, arr=arr, axis=1), dtype=np.double)

            trajs_wrapped.append(arr_wrap)
            trajs.append(arr)

            if plot == True:
                if dict['round']['round'] == self.rounds.round:
                    i = dict['i']

                    plot_args.append({
                        'self': bias,
                        'name': f'{dir}/umbrella_{i}',
                        'traj': [arr],
                    })

            biases.append(Observable._thermo_bias2D(bias))

        if plot == True:

            plot_args.append({
                'self': common_bias,
                'name': f'{dir}/combined',
                'traj': trajs,
            })

            def pl(args):
                Bias.plot(**args)

            #async plot and continue
            with pathos.pools.SerialPool() as pool:
                list(pool.map(pl, plot_args))

            mg, bins = self._FES_mg(trajs=trajs_wrapped)

            histo = Histogram2D.from_wham_c(
                bins=bins,
                # pinit=pinit,
                traj_input=trajs_wrapped,
                error_estimate='mle_f',
                biasses=biases,
                temp=temp,
            )
        elif isinstance(self.rounds, RoundsCV):

            trajs = []
            for dict in self.rounds.iter(num=np.Inf):
                pos = dict["positions"][:]

                if 'cell' in dict:
                    cell = dict["cell"][:]
                    arr = np.array([self.cvs.compute(coordinates=x, cell=y)[0] for (x, y) in zip(pos, cell)],
                                   dtype=np.double)
                else:
                    arr = np.array([self.cvs.compute(coordinates=p, cell=None)[0] for p in pos], dtype=np.double)

                trajs.append(arr)

            mg, bins = self._FES_mg(trajs=trajs, n=10)
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
        self.bounds = bins

        if plot:
            bias = self.fes_Bias(internal=True)
            bias.plot(name=f'{self.folder}/FES_thermolib_{self.rounds.round}')

        return fes, bins

    def new_metric(self, plot=False):
        assert isinstance(self.rounds, RoundsMd)

        trans = []
        cvs = None

        def find_monitor(bias):

            if isinstance(bias, CvMonitor):
                return bias

            if isinstance(bias, CompositeBias):
                for b in bias.biases:
                    ret = find_monitor(b)
                    if ret is not None:
                        return ret

            return None

        for dict in self.rounds.iter(num=1):
            bias = Bias.load(dict['attr']["name_bias"])
            if cvs is None:
                cvs = bias.cvs

            monitor = find_monitor(bias)
            assert monitor is not None

            trans.append(monitor.transitions)

        transitions = jnp.vstack(trans)
        return cvs.metric.update_metric(transitions, plot=plot)

    def _FES_mg(self, trajs, n=None):
        if n is None:
            n = 0
            for t in trajs:
                n += t.size

            #20 points per bin on average
            n = int(n**(1 / trajs[0].ndim) / self.samples_per_bin)

            assert n >= 4, "sample more points"

        trajs = np.vstack(trajs)

        bounds = [[trajs[:, i].min(), trajs[:, i].max()] for i in range(trajs.shape[1])]
        bins = [np.linspace(a, b, n, endpoint=True, dtype=np.double) for a, b in bounds]
        bin_centers = [0.5 * (row[:-1] + row[1:]) for row in bins]

        mg = np.meshgrid(*bin_centers)

        return mg, bins

    class _thermo_bias2D(BiasPotential2D):

        def __init__(self, bias: Bias) -> None:
            self.bias = bias

            super().__init__("IMLCV_bias")

        def __call__(self, cv1, cv2):
            cvs = jnp.array([cv1, cv2])
            #values are already wrapped
            b, _ = jnp.apply_along_axis(partial(self.bias.compute, wrap=False), axis=0, arr=cvs, diff=False)

            b = np.array(b, dtype=np.double)
            b[np.isnan(b)] = 0

            return b

        def print_pars(self, *pars_units):
            pass

    def fes_Bias(self, kind='normal', plot=False, fs=None, internal=False):
        if fs is None:
            fes, bounds = self._fes_2D(plot=plot)

            if kind == 'normal':
                fs = fes.fs
            elif kind == 'fupper':
                fs = fes.fupper

        # fes_interp = Observable._interp(fs)
        bias = np.transpose(fs)

        if internal == False:
            bias = -bias
            bias[:] -= bias[~np.isnan(bias)].min()
            fill = 'min'
        else:
            fill = 'max'

        return GridBias(cvs=self.cvs, fill=fill, vals=bias, wrap=True)
