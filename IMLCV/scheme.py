from __future__ import annotations

import itertools
from importlib import import_module

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
    def from_rounds(rounds: Rounds) -> Scheme:

        self = Scheme.__new__(Scheme)
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

    def grid_umbrella(self, steps=1e4, k=None, n=8, max_grad=None):

        m = self.md.bias.collective_variable.metric

        grid = m.grid(n)

        if k is None:
            k = 2 * self.md.static_trajectory_info.T * boltzmann
        k /= ((m.bounding_box[:, 1] - m.bounding_box[:, 0]) / (2 * n)) ** 2

        self.rounds.run_par(
            [
                HarmonicBias(
                    self.md.bias.collective_variable,
                    CV(cv=jnp.array(cv)),
                    k,
                    k_max=max_grad,
                )
                for cv in itertools.product(*grid)
            ],
            steps=steps,
        )

    def new_metric(self, plot=False, r=None):
        o = ThermoLIB(self.rounds)

        self.md.bias.collective_variable.metric = o.new_metric(plot=plot, r=r)

    def inner_loop(
        self,
        rnds=10,
        init=500,
        steps=5e4,
        K=None,
        update_metric=False,
        n=4,
        samples_per_bin=500,
        init_max_grad=None,
        max_grad=None,
    ):

        if init != 0:
            print(f"running init round with {init} steps")

            self.grid_umbrella(steps=init, n=n, k=K, max_grad=init_max_grad)
            self.rounds.invalidate_data()
            self.rounds.add_round_from_md(self.md)
        else:
            self.md.static_trajectory_info.max_grad = max_grad

        for _ in range(rnds):
            print(f"running round with {steps} steps")
            self.grid_umbrella(steps=steps, n=n, k=K, max_grad=max_grad)

            if update_metric:
                self.new_metric(plot=True)
                update_metric = False
            else:
                self.FESBias(plot=True, samples_per_bin=samples_per_bin)

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
