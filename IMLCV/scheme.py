from __future__ import annotations

import itertools
from pathlib import Path

import jax.numpy as jnp
from molmod.constants import boltzmann

from IMLCV.base.bias import BiasMTD, CompositeBias, HarmonicBias, NoneBias
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

    def __init__(
        self,
        Engine: MDEngine,
        cvd: CVDiscovery | None = None,
        folder="output",
        max_energy=None,
    ) -> None:

        self.md = Engine
        self.cvd = cvd
        self.rounds = RoundsMd(
            folder=folder,
        )
        self.rounds.new_round(self.md)
        self.max_energy = max_energy

    @staticmethod
    def from_rounds(
        folder: str | Path,
        cvd: CVDiscovery | None = None,
        max_energy=None,
    ) -> Scheme:

        self = Scheme.__new__(Scheme)

        rounds = RoundsMd.load(folder)
        self.md = rounds.get_engine()

        self.max_energy = max_energy
        self.rounds = rounds

        self.cvd = cvd

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

    def FESBias(self, plot=True, max_bias=None, kind="normal", n=None):
        """replace the current md bias with the computed FES from current
        round."""
        obs = Observable(self.rounds)

        if max_bias is None:
            if self.max_energy is not None:
                max_bias = self.max_energy
        fesBias = obs.fes_bias(
            kind=kind, plot=plot, max_bias=max_bias, update_bounds=True, n=n
        )
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
        o = Observable(self.rounds)
        self.md.bias.collective_variable.metric = o.new_metric(plot=plot, r=r)

    def round(self, rnds=10, init=None, steps=5e4, K=None, update_metric=False, n=4):

        if init is not None:
            self.grid_umbrella(steps=init, n=n, k=K)
        self.rounds.new_round(self.md)
        self.rounds.save()

        for r in range(rnds):
            self.grid_umbrella(steps=steps, n=n, k=K)

            if update_metric:
                self.new_metric(plot=True)
                update_metric = False
            else:
                self.FESBias(plot=True)

            self.rounds.new_round(self.md)
            self.rounds.save()

    def update_CV(self, samples=2e3, plot=True, **kwargs):
        assert self.cvd is not None, "Give cv deiscovery instance to scheme"

        new_cv = self.cvd.compute(self.rounds, samples=samples, plot=plot, **kwargs)
        self.md.bias = NoneBias(new_cv)

        self.rounds.new_round(self.md)
        self.rounds.save()

    def save(self, filename):
        raise NotImplementedError

    @classmethod
    def load(cls, filename):
        raise NotImplementedError
