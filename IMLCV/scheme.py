from __future__ import annotations

import itertools
from importlib import import_module
from pathlib import Path

import jax.numpy as jnp
from keras.api._v2 import keras as KerasAPI
from molmod.constants import boltzmann

from IMLCV.base.bias import BiasMTD, CompositeBias, HarmonicBias, NoneBias
from IMLCV.base.CV import CV
from IMLCV.base.CVDiscovery import CVDiscovery
from IMLCV.base.MdEngine import MDEngine
from IMLCV.base.Observable import ThermoLIB
from IMLCV.base.rounds import Rounds

keras: KerasAPI = import_module("tensorflow.keras")


class Scheme:
    """base class that implements iterative scheme.

    args:
        format (String): intermediate file type between rounds
        CVs: list of CV instances.
    """

    def __init__(
        self,
        Engine: MDEngine,
        folder="output",
    ) -> None:

        self.md = Engine
        self.rounds = Rounds(
            folder=folder,
        )
        self.rounds.add_round_from_md(self.md)

    @staticmethod
    def from_rounds(
        folder: str | Path,
        copy=False,
    ) -> Scheme:

        self = Scheme.__new__(Scheme)

        rounds = Rounds(folder, copy=copy)
        self.md = rounds.get_engine()

        self.rounds = rounds

        return self

    def MTDBias(self, steps, K=None, sigmas=None, start=500, step=250):
        """generate a metadynamics bias."""

        raise NotImplementedError("validate this")

        if sigmas is None:
            sigmas = (
                self.md.bias.collective_variable.metric[:, 1]
                - self.md.bias.collective_variable.metric[:, 0]
            ) / 20

        if K is None:
            K = 1.0 * self.md.T * boltzmann

        biasmtd = BiasMTD(
            self.md.bias.collective_variable, K, sigmas, start=start, step=step
        )
        bias = CompositeBias([self.md.bias, biasmtd])

        self.md = self.md.new_bias(bias, filename=None)
        self.md.run(steps)
        self.md.bias.finalize()

    def FESBias(self, **kwargs):
        """replace the current md bias with the computed FES from current
        round."""
        obs = ThermoLIB(self.rounds)
        fesBias = obs.fes_bias(**kwargs)
        self.md = self.md.new_bias(fesBias)

    def grid_umbrella(self, steps=1e4, k=None, n=8):

        grid = self.md.bias.collective_variable.metric.grid(n)

        if k is None:
            k = 2 * self.md.static_trajectory_info.T * boltzmann
        k *= n**2

        self.rounds.run_par(
            [
                HarmonicBias(
                    self.md.bias.collective_variable,
                    CV(cv=jnp.array(cv)),
                    k,
                )
                for cv in itertools.product(*grid)
            ],
            steps=steps,
        )

    def new_metric(self, plot=False, r=None):
        o = ThermoLIB(self.rounds)

        self.md.bias.collective_variable.metric = o.new_metric(plot=plot, r=r)

    def inner_loop(self, 
            rnds=10, 
            init=0, 
            steps=5e4, 
            K=None, 
            update_metric=False, 
            n=4,
            ):

        if init != 0:
            self.grid_umbrella(steps=init, n=n, k=K)
            self.rounds.add_round(self.md)
            self.rounds.save()

        for r in range(rnds):
            self.grid_umbrella(steps=steps, n=n, k=K)

            if update_metric:
                self.new_metric(plot=True)
                update_metric = False
            else:
                self.FESBias(plot=True, n=n)

            self.rounds.add_round_from_md(self.md)

    def update_CV(self, cvd: CVDiscovery, samples=2e3, plot=True, **kwargs):

        new_cv = cvd.compute(self.rounds, samples=samples, plot=plot, **kwargs)
        self.md.bias = NoneBias(new_cv)

        self.rounds.add_round(self.md.static_trajectory_info)

    def save(self, filename):
        raise NotImplementedError

    @classmethod
    def load(cls, filename):
        raise NotImplementedError


######################################
#           Test                     #
######################################
