from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
from IMLCV.base.bias import NoneBias
from IMLCV.base.CV import CollectiveVariable
from IMLCV.base.CV import CV
from IMLCV.base.CV import CvMetric
from IMLCV.base.CV import CvTrans
from IMLCV.base.CV import SystemParams
from IMLCV.base.CVDiscovery import Transformer
from IMLCV.base.MdEngine import MDEngine
from IMLCV.base.Observable import ThermoLIB
from IMLCV.base.rounds import Rounds
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
    def from_rounds(rounds: Rounds, md=None) -> Scheme:
        self = Scheme.__new__(Scheme)
        if md is None:
            self.md = rounds.get_engine()
        else:
            self.md = md

        self.rounds = rounds

        return self

    def FESBias(self, cv_round: int | None = None, chunk_size=None, **plotkwargs):
        """replace the current md bias with the computed FES from current
        round."""
        obs = ThermoLIB(self.rounds, cv_round=cv_round)
        fesBias = obs.fes_bias(chunk_size=chunk_size, **plotkwargs)
        self.md = self.md.new_bias(fesBias)

    def grid_umbrella(
        self,
        steps=1e4,
        k=None,
        n=8,
        max_grad=None,
        plot=True,
        scale_n: int | None = None,
        cv_round: int | None = None,
        ignore_invalid=False,
        eps=0.1,
        min_traj_length=None,
        recalc_cv=False,
        only_finished=True,
    ):
        m = self.md.bias.collective_variable.metric
        grid = m.grid(n)

        if k is None:
            # 0.1*N *Kb*T
            k = (
                eps
                * self.md.static_trajectory_info.T
                * boltzmann
                * self.md.static_trajectory_info.atomic_numbers.shape[0]
            )

        k /= ((m.bounding_box[:, 1] - m.bounding_box[:, 0]) / 2) ** 2
        if scale_n is None:
            scale_n = n
        k *= scale_n**2

        biases = [
            HarmonicBias.create(
                self.md.bias.collective_variable,
                CV(cv=jnp.array(cv)),
                k,
                k_max=max_grad,
            )
            for cv in itertools.product(*grid)
        ]

        if self.rounds.cv == 0 and self.rounds.round == 0:
            sp0 = SystemParams.stack(*[self.md.sp] * len(biases))
        else:
            sp0 = None

        self.rounds.run_par(
            biases=biases,
            steps=steps,
            plot=plot,
            cv_round=cv_round,
            ignore_invalid=ignore_invalid,
            min_traj_length=min_traj_length,
            recalc_cv=recalc_cv,
            only_finished=only_finished,
            sp0=sp0,
        )

    def new_metric(self, plot=False, r=None, cv_round: int | None = None):
        o = ThermoLIB(self.rounds, cv_round=cv_round)

        self.md.bias.collective_variable.metric = o.new_metric(plot=plot, r=r)

    def inner_loop(
        self,
        rnds=10,
        init=500,
        steps=5e4,
        K=None,
        update_metric=False,
        n=4,
        samples_per_bin=200,
        init_max_grad=None,
        max_grad=None,
        plot=True,
        plot_kwargs={},
        choice="rbf",
        fes_bias_rnds=4,
        scale_n: int | None = None,
        cv_round: int | None = None,
        chunk_size=None,
        eps_umbrella=0.1,
        plot_margin=0.1,
        enforce_min_traj_length=False,
        recalc_cv=False,
        only_finished=True,
        plot_umbrella=None,
        max_bias=None,
    ):
        if plot_umbrella is None:
            plot_umbrella = plot

        if cv_round is None:
            cv_round = self.rounds.cv

        print(f"{cv_round=}")

        if init != 0:
            print(f"running init round with {init} steps")

            self.grid_umbrella(
                steps=init,
                n=n,
                k=K,
                max_grad=init_max_grad,
                plot=plot_umbrella,
                cv_round=cv_round,
                eps=eps_umbrella,
            )
            self.rounds.invalidate_data(c=cv_round)
            self.rounds.add_round_from_md(self.md, cv=cv_round)
        else:
            self.md.static_trajectory_info.max_grad = max_grad

        i_0 = self.rounds.get_round(c=cv_round)

        print(f"{i_0=}")

        for i in range(i_0, rnds):
            print(f"running round {i=} with {steps} steps")
            self.grid_umbrella(
                steps=steps,
                n=n,
                k=K,
                max_grad=max_grad,
                plot=plot_umbrella,
                scale_n=scale_n,
                cv_round=cv_round,
                ignore_invalid=i <= 1,
                eps=eps_umbrella,
                min_traj_length=steps if (i > 1 and enforce_min_traj_length) else None,
                recalc_cv=recalc_cv,
                only_finished=i > 1 and only_finished,
            )

            if update_metric:
                self.new_metric(plot=plot)
                update_metric = False
            else:
                self.FESBias(
                    plot=plot,
                    samples_per_bin=samples_per_bin,
                    choice=choice,
                    num_rnds=fes_bias_rnds,
                    cv_round=cv_round,
                    chunk_size=chunk_size,
                    min_traj_length=steps if enforce_min_traj_length else None,
                    margin=plot_margin,
                    only_finished=only_finished,
                    max_bias=max_bias,
                    use_prev_fs=i > 1,
                    **plot_kwargs,
                )

            self.rounds.add_round_from_md(self.md, cv=cv_round)

    def update_CV(
        self,
        transformer: Transformer,
        dlo_kwargs={},
        dlo: Rounds.data_loader_output | None = None,
        chunk_size=None,
        plot=True,
        new_r_cut=None,
        save_samples=True,
        save_multiple_cvs=False,
        jac=jax.jacrev,
        cv_round_from=None,
        test=False,
        max_bias=None,
        transform_bias=False,
        samples_per_bin=100,
        min_samples_per_bin=20,
    ):
        if cv_round_from is None:
            cv_round_from = self.rounds.cv

        if "chunk_size" in dlo_kwargs:
            chunk_size = dlo_kwargs["chunk_size"]
            dlo_kwargs.pop("chunk_size")

        if dlo is None:
            dlo = self.rounds.data_loader(
                cv_round=cv_round_from,
                chunk_size=chunk_size,
                **dlo_kwargs,
            )

        cvs_new, new_collective_variable, new_bias = transformer.fit(
            dlo=dlo,
            chunk_size=chunk_size,
            plot=plot,
            plot_folder=self.rounds.path(c=self.rounds.cv + 1),
            jac=jac,
            test=test,
            max_fes_bias=max_bias,
            transform_FES=transform_bias,
            samples_per_bin=samples_per_bin,
            min_samples_per_bin=min_samples_per_bin,
        )

        # update state

        self.rounds.add_cv_from_cv(new_collective_variable)

        self.md.bias = new_bias
        self.md.static_trajectory_info.r_cut = new_r_cut
        self.rounds.add_round_from_md(self.md)

        if save_samples:
            first = True

            if save_multiple_cvs:
                for dlo_i, cv_new_i in zip(iter(dlo), CV.unstack(cvs_new)):
                    if not first:
                        self.md.bias = NoneBias.create(new_collective_variable)
                        self.rounds.add_cv_from_cv(new_collective_variable)
                        self.md.static_trajectory_info.r_cut = new_r_cut
                        self.rounds.add_round_from_md(self.md)

                    self.rounds.copy_from_previous_round(dlo=dlo_i, new_cvs=[cv_new_i], cv_round=cv_round_from)
                    self.rounds.add_round_from_md(self.md)

                    first = False

            else:
                print(f"{ len(dlo.cv)}, { len( CV.unstack(cvs_new))=  } ")

                self.rounds.copy_from_previous_round(dlo=dlo, new_cvs=CV.unstack(cvs_new), cv_round=cv_round_from)
                self.rounds.add_round_from_md(self.md)

    def transform_CV(
        self,
        cv_trans: CvTrans,
        copy_samples=True,
        plot=True,
        num_copy=2,
        chunk_size=None,
        kernel="thin_plate_spline",
    ):
        original_collective_variable = self.md.bias.collective_variable

        def cv_grid(margin):
            # take reasonable margin
            grid = original_collective_variable.metric.grid(n=50, endpoints=True, margin=margin)
            grid = jnp.reshape(jnp.array(jnp.meshgrid(*grid, indexing="ij")), (len(grid), -1)).T
            cv = CV(cv=grid)
            return cv

        @jax.vmap
        def f(cv):
            bias_inter, _ = self.md.bias.compute_from_cv(cv, chunk_size=chunk_size)
            v, _, log_jac = cv_trans.compute_cv_trans(cv, log_Jf=True)

            return bias_inter, v, log_jac

        cv_orig = cv_grid(margin=0.4)
        bias_inter, cv_new, log_jac = f(cv_orig)

        FES_offset = -boltzmann * self.md.static_trajectory_info.T * log_jac

        # determine metrix based on no margin extension
        cv_grid_strict = cv_grid(margin=0)

        new_collective_variable = CollectiveVariable(
            f=original_collective_variable.f * cv_trans,
            metric=CvMetric.create(
                periodicities=[False] * cv_new.shape[1],
                bounding_box=jnp.array(
                    [
                        jnp.min(cv_grid_strict.cv, axis=0),
                        jnp.max(cv_grid_strict.cv, axis=0),
                    ]
                ).T,
            ),
        )

        fes_offset_bias = RbfBias.create(cvs=new_collective_variable, cv=cv_new, vals=FES_offset, kernel=kernel)
        self.md.bias = RbfBias.create(
            cvs=new_collective_variable,
            cv=cv_new,
            vals=FES_offset + bias_inter,
            kernel="thin_plate_spline",
        )

        self.rounds.add_cv_from_cv(new_collective_variable)
        self.rounds.add_round_from_md(self.md)

        if plot:
            self.md.bias.plot(name=self.rounds.path(self.rounds.cv) / "transformed_bias.pdf")
            self.md.bias.plot(
                name=self.rounds.path(self.rounds.cv) / "transformed_bias_inverted.pdf",
                inverted=True,
            )

            fes_offset_bias.plot(
                name=self.rounds.path(self.rounds.cv) / "fes_offset_bias.pdf",
                inverted=True,
            )

        if copy_samples:
            self.rounds.copy_from_previous_round(cv_trans=cv_trans, chunk_size=chunk_size, num_copy=num_copy)
            self.rounds.add_round_from_md(self.md)

    def save(self, filename):
        raise NotImplementedError

    @classmethod
    def load(cls, filename):
        raise NotImplementedError
