from __future__ import annotations

import itertools
import sys
from typing import Type

import jax
import numpy as np
from molmod.constants import boltzmann
from molmod.units import kjmol

from IMLCV.base.bias import (BiasF, BiasMTD, CompositeBias, CvMonitor, FesBias,
                             HarmonicBias, NoneBias)
from IMLCV.base.CV import CV
from IMLCV.base.CVDiscovery import CVDiscovery
from IMLCV.base.MdEngine import MDEngine
from IMLCV.base.Observable import Observable
from IMLCV.base.rounds import RoundsMd


class Scheme:
    """base class that implements iterative scheme.

    args:
        format (String): intermediate file type between rounds
        CVs: list of CV instances.
    """

    def __init__(self,
                 cvd: CVDiscovery,
                 cvs: CV,
                 Engine: Type[MDEngine],
                 ener,
                 T,
                 P=None,
                 timestep=None,
                 timecon_thermo=None,
                 timecon_baro=None,
                 extension="extxyz",
                 folder='output',
                 write_step=100,
                 screenlog=1000,
                 max_energy=None) -> None:

        self.md = Engine(bias=NoneBias(cvs),
                         ener=ener,
                         T=T,
                         P=P,
                         timestep=timestep,
                         timecon_thermo=timecon_thermo,
                         timecon_baro=timecon_baro,
                         filename=None,
                         write_step=write_step,
                         screenlog=screenlog)

        self.cvd = cvd

        self.rounds = RoundsMd(
            extension=extension,
            folder=folder,
        )

        self.rounds.new_round(self.md)
        self.max_energy = max_energy

    @staticmethod
    def from_rounds(
        cvd: CVDiscovery,
        folder,
        max_energy=None,
    ) -> Scheme:

        self = Scheme.__new__(Scheme)

        rounds = RoundsMd.load(folder)
        self.md = rounds.get_engine()

        self.max_energy = max_energy
        self.rounds = rounds

        self.cvd = cvd

        return self

    def _MTDBias(self, steps, K=None, sigmas=None, start=500, step=250):
        """generate a metadynamics bias."""

        if sigmas is None:
            sigmas = (self.md.bias.cvs.metric[:, 1] -
                      self.md.bias.cvs.metric[:, 0]) / 20

        if K is None:
            K = 1.0 * self.md.T * boltzmann

        biasmtd = BiasMTD(self.md.bias.cvs, K, sigmas, start=start, step=step)
        bias = CompositeBias([self.md.bias, biasmtd])

        self.md = self.md.new_bias(bias, filename=None)
        self.md.run(steps)
        self.md.bias.finalize()

    def _FESBias(self, plot=True, max_bias=np.inf, kind='normal'):
        """replace the current md bias with the computed FES from current
        round."""
        obs = Observable(self.rounds)

        if max_bias is None:
            max_bias = self.max_energy
        fesBias = obs.fes_bias(kind=kind, plot=plot, max_bias=max_bias)
        self.md = self.md.new_bias(fesBias, filename=None)

    def _grid_umbrella(self, steps=1e4,  K=None, n=8):

        grid = self.md.bias.cvs.metric.grid(n)

        if K is None:
            K = 1.0 * self.md.T * boltzmann
        K /= (np.array([a[1]-a[0] for a in grid])/2) ** 2

        self.rounds.run_par([CompositeBias([
            HarmonicBias(self.md.bias.cvs, np.array(x), np.array(K)),
            CvMonitor(self.md.bias.cvs),
        ]) for x in itertools.product(*grid)], steps=steps)

    def _new_metric(self, plot=False, r=None):
        o = Observable(self.rounds)
        self.md.bias.cvs.metric = o.new_metric(plot=plot, r=r)

    def round(self, rnds=10, steps=5e4, K=None, update_metric=False, n=4):
        # startround = 0

        # update biases untill there are no discontinues jumps left
        for _ in range(rnds):
            self._grid_umbrella(steps=steps, n=n, K=K)
            if update_metric:
                self._new_metric(plot=True)
                update_metric = False
            else:
                self._FESBias(plot=True)

            self.rounds.new_round(self.md)
            self.rounds.save()

    def update_CV(self, plot=True):
        new_cv = self.cvd.compute(self.rounds, plot=plot)
        self.md.bias = NoneBias(new_cv)

        self.rounds.new_round(self.md)

    def save(self, filename):
        raise NotImplementedError

    @ classmethod
    def load(cls, filename):
        raise NotImplementedError
