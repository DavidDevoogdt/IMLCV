from __future__ import annotations

from IMLCV.base.MdEngine import MDEngine
from IMLCV.base.bias import BiasMTD, Bias, CompositeBias, HarmonicBias, NoneBias
from IMLCV.base.CVDiscovery import CVDiscovery
from IMLCV.base.CV import CV
from IMLCV.base.Observable import Observable
from IMLCV.base.rounds import RoundsMd

from molmod.constants import boltzmann

import numpy as np

import itertools

from typing import Type


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
                 folder='output') -> None:

        # filename = f"{folder}/init.h5"
        write_step = 100
        screenlog = 1000

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

        self.rounds = RoundsMd(extension=extension, folder=folder)
        self.steps = 0

    def from_rounds(
        cvd: CVDiscovery,
        folder,
    ) -> Scheme:

        self = Scheme.__new__(Scheme)

        rounds = RoundsMd.load(folder)
        self.folder = folder
        self.md = rounds.engine

        self.rounds = rounds
        self.cvd = cvd
        self.steps = 0

        return self

    def _MTDBias(self, steps, K=None, sigmas=None, start=500, step=250) -> Bias:
        """generate a metadynamics bias"""

        if sigmas is None:
            sigmas = (self.md.bias.cvs.periodicity[:, 1] - self.md.bias.cvs.periodicity[:, 0]) / 20

        if K is None:
            K = 0.1 * self.md.T * boltzmann

        biasmtd = BiasMTD(self.md.bias.cvs, K, sigmas, start=start, step=step)
        bias = CompositeBias([self.md.bias, biasmtd])
        self.md = self.md.new_bias(bias, filename=None)
        self.md.run(steps)
        self.md.bias.finalize()

    def _FESBias(self, plot=True):
        """replace the current md bias with the computed FES from current round"""
        obs = Observable(self.rounds)
        fes = obs.fes_2D(plot=plot)
        fesBias = obs.fes_Bias()

        self.md = self.md.new_bias(fesBias, filename=None)

    def _grid_umbrella(self, steps=1e4, US_grid=None, K=None, n=4):

        cvs = self.md.bias.cvs
        if np.isnan(cvs.periodicity).any():
            raise NotImplementedError("impl non periodic")

        if K == None:
            K = 1.0 * self.md.T * boltzmann * (n * 2 / (cvs.periodicity[:, 1] - cvs.periodicity[:, 0]))**2

        if US_grid is None:
            grid = [np.linspace(row[0], row[1], n, endpoint=False) for row in self.md.bias.cvs.periodicity]

        self.rounds.run_par(
            [HarmonicBias(self.md.bias.cvs, np.array(x), np.array(K)) for x in itertools.product(*grid)], steps=steps)

    def calc_fes(self, rnds=10, steps=5e4):
        startround = 0

        for i in range(rnds):
            #create common bias
            if i != startround:
                self._FESBias()
            self.rounds.new_round(self.md)

            # self._MTDBias(steps=5e4)

            self._grid_umbrella(steps=steps)
            self.rounds.save()

        self._FESBias()
        self.rounds.save()

    def update_CV(self):
        pass

    def save(self, filename):
        raise NotImplementedError

    @classmethod
    def load(cls, filename):
        raise NotImplementedError
