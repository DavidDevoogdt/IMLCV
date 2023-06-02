from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
from IMLCV.base.bias import Bias
from IMLCV.base.bias import CompositeBias
from IMLCV.base.bias import NoneBias
from IMLCV.base.CV import CollectiveVariable
from IMLCV.base.CV import CV
from IMLCV.base.CV import CvMetric
from IMLCV.base.CV import CvTrans
from IMLCV.base.CV import SystemParams
from IMLCV.base.CVDiscovery import CVDiscovery
from IMLCV.base.MdEngine import EnergyResult
from IMLCV.base.MdEngine import MDEngine
from IMLCV.base.MdEngine import TrajectoryInfo
from IMLCV.base.Observable import ThermoLIB
from IMLCV.base.rounds import Rounds
from IMLCV.implementations.bias import BiasMTD
from IMLCV.implementations.bias import HarmonicBias
from IMLCV.implementations.bias import RbfBias
from molmod.constants import boltzmann


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
        self.rounds.add_cv_from_cv(self.md.bias.collective_variable)
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
                self.md.bias.collective_variable.metric[:, 1] - self.md.bias.collective_variable.metric[:, 0]
            ) / 20

        if K is None:
            K = 1.0 * self.md.T * boltzmann

        biasmtd = BiasMTD(
            self.md.bias.collective_variable,
            K,
            sigmas,
            start=start,
            step=step,
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

    def grid_umbrella(
        self,
        steps=1e4,
        k=None,
        n=8,
        max_grad=None,
        plot=True,
        scale_n=True,
    ):
        m = self.md.bias.collective_variable.metric

        grid = m.grid(n)

        if k is None:
            k = 2 * self.md.static_trajectory_info.T * boltzmann
        k /= ((m.bounding_box[:, 1] - m.bounding_box[:, 0]) / 2) ** 2
        if scale_n:
            k /= n**2

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
            plot=plot,
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
        plot=True,
        choice="grid_bias",
        fes_bias_rnds=4,
    ):
        if init != 0:
            print(f"running init round with {init} steps")

            self.grid_umbrella(steps=init, n=n, k=K, max_grad=init_max_grad, plot=plot)
            self.rounds.invalidate_data()
            self.rounds.add_round_from_md(self.md)
        else:
            self.md.static_trajectory_info.max_grad = max_grad

        for _ in range(rnds):
            print(f"running round with {steps} steps")
            self.grid_umbrella(steps=steps, n=n, k=K, max_grad=max_grad, plot=plot)

            if update_metric:
                self.new_metric(plot=plot)
                update_metric = False
            else:
                self.FESBias(plot=plot, samples_per_bin=samples_per_bin, choice=choice, num_rnds=fes_bias_rnds)

            self.rounds.add_round_from_md(self.md)

    def update_CV(
        self,
        cvd: CVDiscovery,
        chunk_size=None,
        samples=2e3,
        plot=True,
        new_r_cut=None,
        **kwargs,
    ):
        new_cv = cvd.compute(
            self.rounds,
            samples=samples,
            plot=plot,
            chunk_size=chunk_size,
            new_r_cut=new_r_cut,
            **kwargs,
        )

        # update state

        self.md.bias = NoneBias(new_cv)
        self.rounds.add_cv_from_cv(new_cv)
        self.md.static_trajectory_info.r_cut = new_r_cut
        self.rounds.add_round_from_md(self.md)

    def transform_CV(self, cv_trans: CvTrans, copy_samples=True, plot=True, num_copy=2, chunk_size=None):
        original_collective_variable = self.md.bias.collective_variable

        grid = original_collective_variable.metric.grid(n=50, endpoints=True, margin=0.1)
        grid = jnp.reshape(jnp.array(jnp.meshgrid(*grid)), (len(grid), -1)).T
        cv = CV(cv=grid)

        @jax.vmap
        def f(cv):
            bias_inter, _ = self.md.bias.compute_from_cv(cv)
            v, log_jac = cv_trans.compute_cv_trans(cv, log_Jf=True)

            return bias_inter, cv, v, log_jac

        bias_inter, cv, v, log_jac = f(cv)

        FES_offset = -boltzmann * self.md.static_trajectory_info.T * log_jac

        new_collective_variable = CollectiveVariable(
            f=original_collective_variable.f * cv_trans,
            metric=CvMetric(
                periodicities=[False] * v.shape[1],
                bounding_box=jnp.array([jnp.min(cv.cv, axis=0), jnp.max(cv.cv, axis=0)]).T,
            ),
        )

        self.md.bias = RbfBias(cvs=new_collective_variable, cv=v, vals=FES_offset + bias_inter, kernel="linear")

        self.rounds.add_cv_from_cv(new_collective_variable)
        self.rounds.add_round_from_md(self.md)

        if plot:
            self.md.bias.plot(name=self.rounds.path(self.rounds.cv) / "transformed_bias.pdf")
            self.md.bias.plot(name=self.rounds.path(self.rounds.cv) / "transformed_bias_inverted.pdf", inverted=True)

        if copy_samples:
            fes_offset_bias = RbfBias(cvs=new_collective_variable, cv=v, vals=FES_offset, kernel="linear")

            for ri, ti in self.rounds.iter(c=self.rounds.cv - 1, num=num_copy):
                i = ti.num
                round_path = self.rounds.path(c=self.rounds.cv, r=0, i=i)
                round_path.mkdir(parents=True, exist_ok=True)

                bias = CompositeBias(biases=[ti.get_bias(), fes_offset_bias])
                bias.save(round_path / "bias")

                traj_info = ti.ti

                sys_params: SystemParams = traj_info.sp
                nl = sys_params.get_neighbour_list(r_cut=ri.tic.r_cut, z_array=ri.tic.atomic_numbers)
                new_cv, energy_result = fes_offset_bias.compute_from_system_params(
                    sp=sys_params,
                    nl=nl,
                    gpos=True,
                    vir=True,
                    chunk_size=chunk_size,
                )
                energy_result: EnergyResult
                new_cv: CV

                new_traj_info = TrajectoryInfo(
                    _positions=traj_info.positions,
                    _cell=traj_info.cell,
                    _charges=traj_info.charges,
                    _e_pot=traj_info.e_pot,
                    _e_pot_gpos=traj_info.e_pot_gpos,
                    _e_pot_vtens=traj_info.e_pot_vtens,
                    _e_bias=traj_info.e_bias + energy_result.energy,
                    _e_bias_gpos=traj_info.e_bias_gpos + energy_result.gpos if energy_result.gpos is not None else None,
                    _e_bias_vtens=traj_info.e_bias_vtens + energy_result.vtens
                    if energy_result.vtens is not None
                    else None,
                    _cv=new_cv.cv,
                    _T=traj_info._T,
                    _P=traj_info._P,
                    _err=traj_info._err,
                    _t=traj_info._t,
                    _capacity=traj_info._capacity,
                    _size=traj_info._size,
                )

                self.rounds.add_md(
                    i=i,
                    d=new_traj_info,
                    attrs=None,
                    bias=self.rounds.rel_path(round_path / "bias"),
                    r=0,
                )
            self.rounds.add_round_from_md(self.md)

    def save(self, filename):
        raise NotImplementedError

    @classmethod
    def load(cls, filename):
        raise NotImplementedError


######################################
#           Test                     #
######################################
