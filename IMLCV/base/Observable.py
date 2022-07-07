from __future__ import annotations

from dataclasses import dataclass
from sys import stdout
from typing import List, Optional

import jax
import jax.numpy as jnp
import numpy as np
from IMLCV.base.bias import (Bias, BiasF, CompositeBias, CvMonitor, FesBias,
                             GridBias, plot_app)
from IMLCV.base.CV import CV
from IMLCV.base.rounds import Rounds, RoundsCV, RoundsMd
from molmod.units import picosecond
from parsl import File, python_app
from thermolib.thermodynamics.bias import BiasPotential2D
from thermolib.thermodynamics.fep import FreeEnergySurface2D
from thermolib.thermodynamics.histogram import Histogram2D

# @dataclass
# class plotArgs:
#     bias: Bias
#     name: File
#     n: int = 50
#     vmin: float = 0
#     vmax: float = 100
#     map: bool = True
#     traj: Optional[List[np.ndarray]] = None


class Observable:
    """class to convert data and CVs to different thermodynamic/ kinetic
    observables."""

    samples_per_bin = 20
    time_per_bin = 2*picosecond

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

    def _fes_2d(self, plot=True, throw_away=2*picosecond):
        # fes = FreeEnergySurface2D.from_txt

        temp = self.rounds.T

        common_bias = self.rounds.get_bias()
        directory = f'{self.folder}/round_{self.rounds.round}'

        if isinstance(self.rounds, RoundsMd):

            trajs = []
            trajs_mapped = []
            biases = []

            time = 0

            for dictionary in self.rounds.iter(num=2):

                bias = Bias.load(dictionary['attr']["name_bias"])

                index = np.argmax(dictionary['t'] > throw_away)
                time += dictionary['t'][-1]-dictionary['t'][index]

                pos = dictionary["positions"][index:]

                if 'cell' in dictionary:
                    cell = dictionary["cell"][index:]
                    arr = np.array(
                        [
                            bias.cvs.compute(coordinates=x, cell=y)[0]
                            for (x, y) in zip(pos, cell)
                        ],
                        dtype=np.double,
                    )
                else:
                    arr = np.array(
                        [
                            bias.cvs.compute(coordinates=p, cell=None)[0]
                            for p in pos
                        ],
                        dtype=np.double,
                    )

                arr_mapped = np.array(np.apply_along_axis(bias.cvs.metric.map,
                                                          arr=arr,
                                                          axis=1),
                                      dtype=np.double)

                trajs_mapped.append(arr_mapped)
                trajs.append(arr)

                if plot:
                    if dictionary['round']['round'] == self.rounds.round:
                        i = dictionary['i']

                        plot_app(bias=bias, outputs=[File(
                            f'{directory}/umbrella_{i}.pdf')], traj=[arr_mapped])

                biases.append(Observable._ThermoBias2D(bias))

            if plot:

                plot_app(bias=common_bias, outputs=[File(
                    f'{directory}/combined_unmapped.pdf')], map=False, traj=trajs)

                plot_app(bias=common_bias, outputs=[File(
                    f'{directory}/combined.pdf')], traj=trajs_mapped)

            bounds, bins = self._FES_mg(trajs=trajs_mapped, time=time)

            histo = Histogram2D.from_wham_c(
                bins=bins,
                # pinit=pinit,
                traj_input=trajs_mapped,
                error_estimate='mle_f',
                biasses=biases,
                temp=temp,
            )

        elif isinstance(self.rounds, RoundsCV):
            raise NotImplementedError("check this")
            trajs = []
            for dictionary in self.rounds.iter(num=np.Inf):
                pos = dictionary["positions"][:]

                if 'cell' in dictionary:
                    cell = dictionary["cell"][:]
                    arr = np.array(
                        [
                            self.cvs.compute(coordinates=x, cell=y)[0]
                            for (x, y) in zip(pos, cell)
                        ],
                        dtype=np.double,
                    )
                else:
                    arr = np.array(
                        [
                            self.cvs.compute(coordinates=p, cell=None)[0]
                            for p in pos
                        ],
                        dtype=np.double,
                    )

                trajs.append(arr)

            bounds, bins = self._FES_mg(trajs=trajs, n=10)
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

        return fes, bounds

    def new_metric(self, plot=False, r=None):
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

        for r in self.rounds.iter(num=1, r=r):
            bias = Bias.load(r['attr']["name_bias"])
            if cvs is None:
                cvs = bias.cvs

            monitor = find_monitor(bias)
            assert monitor is not None

            trans.append(monitor.transitions)

        transitions = jnp.vstack(trans)
        if plot:
            fn = f'{self.folder}/round_{self.rounds.round}/'
        else:
            fn = None

        return cvs.metric.update_metric(transitions, fn=fn)

    def _FES_mg(self, trajs, time=None, n=None):

        if n is None:
            n = 0
            for t in trajs:
                n += t.size

            # 20 points per bin on average
            n = int(n**(1 / trajs[0].ndim) / self.samples_per_bin)

        if time is not None:
            bins_max = int((time/self.time_per_bin)**(1 / trajs[0].ndim))
            if bins_max > n:
                n = bins_max

        assert n >= 4, "sample more points"

        trajs = np.vstack(trajs)

        bounds = [[trajs[:, i].min(), trajs[:, i].max()]
                  for i in range(trajs.shape[1])]
        bins = [
            np.linspace(a, b, n, endpoint=True, dtype=np.double)
            for a, b in bounds
        ]

        return bounds, bins

    class _ThermoBias2D(BiasPotential2D):

        def __init__(self, bias: Bias) -> None:
            self.bias = bias

            super().__init__("IMLCV_bias")

        def __call__(self, cv1, cv2):
            # CVs are already in mapped space
            cvs = jnp.array([cv1, cv2])
            b, _ = jnp.apply_along_axis(self.bias.compute,
                                        axis=0,
                                        arr=cvs,
                                        diff=False,
                                        map=False, #already mapped
                                        )

            b = np.array(b, dtype=np.double)
            b[np.isnan(b)] = 0

            return b

        def print_pars(self, *pars_units):
            pass

    def fes_bias(self, kind='normal', plot=False, fs=None, max_bias=np.inf):
        if fs is None:
            fes, bounds = self._fes_2d(plot=plot)

            if kind == 'normal':
                fs = fes.fs
            elif kind == 'fupper':
                fs = fes.fupper
            elif kind == 'flower':
                fs = fes.flower
            else:
                raise ValueError

        # fes is in 'xy'- indexing convention, convert to ij
        fs = np.transpose(fs)

        fs[:] = -fs[:] + np.min([max_bias, fs[~np.isnan(fs)].max()])

        fesBias = FesBias(GridBias(cvs=self.cvs,  vals=fs,
                                   bounds=bounds), T=self.rounds.T)

        fesBias = CompositeBias(biases=[fesBias,  BiasF(
            cvs=fesBias.cvs)], fun=lambda e: jax.numpy.where(e[0] > e[1], e[0], e[1]))

        if plot:
            plot_app(bias=fesBias, outputs=[File(
                f'{self.folder}/FES_thermolib_unmapped_{self.rounds.round}.pdf')], map=False)

            plot_app(bias=fesBias, outputs=[
                File(f'{self.folder}/FES_thermolib_{self.rounds.round}.pdf')])

        return fesBias
