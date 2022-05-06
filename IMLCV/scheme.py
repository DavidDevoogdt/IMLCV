from __future__ import annotations

import itertools
from typing import Type

import numpy as np
from molmod.constants import boltzmann
from molmod.units import kjmol

from IMLCV.base.bias import (BiasMTD, CompositeBias, CvMonitor, HarmonicBias,
                             NoneBias)
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
                 max_energy=80 * kjmol) -> None:

        # filename = f"{folder}/init.h5"

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
            max_energy=max_energy,
        )

        self.rounds.new_round(self.md)

        self.steps = 0
        self.cont_biases = None

    @staticmethod
    def from_rounds(
        cvd: CVDiscovery,
        folder,
    ) -> Scheme:

        self = Scheme.__new__(Scheme)

        rounds = RoundsMd.load(folder)
        self.md = rounds.get_engine()

        self.rounds = rounds

        self.cvd = cvd
        self.steps = 0

        self.cont_biases = None

        return self

    def _MTDBias(self, steps, K=None, sigmas=None, start=500, step=250):
        """generate a metadynamics bias."""

        if sigmas is None:
            sigmas = (self.md.bias.cvs.metric[:, 1] -
                      self.md.bias.cvs.metric[:, 0]) / 20

        if K is None:
            K = 0.1 * self.md.T * boltzmann

        biasmtd = BiasMTD(self.md.bias.cvs, K, sigmas, start=start, step=step)
        bias = CompositeBias([self.md.bias, biasmtd])

        self.md = self.md.new_bias(bias, filename=None)
        self.md.run(steps)
        self.md.bias.finalize()

    def _FESBias(self, plot=True, kind='flower'):
        """replace the current md bias with the computed FES from current
        round."""
        obs = Observable(self.rounds)
        fesBias = obs.fes_bias(kind=kind, plot=plot)

        self.md = self.md.new_bias(fesBias, filename=None)

    def _grid_umbrella(self, steps=1e4, US_grid=None, K=None, n=4):

        cvs = self.md.bias.cvs
        if ((cvs.metric.wrap_boundaries[:, 1] -
             cvs.metric.wrap_boundaries[:, 0]) <= 1e-6).any():
            raise NotImplementedError(
                "Metric provide boundaries or force constant K")

        if K is None:
            K = 1.0 * self.md.T * boltzmann * (
                n * 2 / (cvs.metric.wrap_boundaries[:, 1] -
                         cvs.metric.wrap_boundaries[:, 0]))**2

        grid = self.md.bias.cvs.metric.grid(n, endpoints=False, wrap=True)

        self.rounds.run_par([CompositeBias([
            HarmonicBias(self.md.bias.cvs, np.array(x), np.array(K)),
            CvMonitor(self.md.bias.cvs),
        ]) for x in itertools.product(*grid)], steps=steps)

    def round(self, rnds=10, steps=5e4, update_metric=False):
        # startround = 0

        # update biases untill there are no discontinues jumps left
        for i in range(rnds):

            if update_metric:
                self._grid_umbrella(steps=5e3)
                o = Observable(self.rounds)
                self.md.bias.cvs.metric = o.new_metric(plot=False)
                update_metric = False
            else:
                self._grid_umbrella(steps=steps)
                self._FESBias(plot=True)

            self.rounds.new_round(self.md)
            self.rounds.save()

    def update_CV(self):
        pass

    def save(self, filename):
        raise NotImplementedError

    @classmethod
    def load(cls, filename):
        raise NotImplementedError
