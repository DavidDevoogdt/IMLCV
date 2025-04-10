from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
from IMLCV.base.UnitsConstants import boltzmann
from IMLCV.base.UnitsConstants import kjmol

from IMLCV.base.bias import Bias, NoneBias
from IMLCV.base.CV import CvTrans, SystemParams
from IMLCV.base.CVDiscovery import Transformer
from IMLCV.base.MdEngine import MDEngine
from IMLCV.base.Observable import Observable
from IMLCV.base.rounds import DataLoaderOutput, Rounds
from IMLCV.implementations.bias import HarmonicBias


@dataclass
class Scheme:
    """base class that implements iterative scheme.

    args:
        format (String): intermediate file type between rounds
        CVs: list of CV instances.
    """

    # md: MDEngine
    rounds: Rounds

    @staticmethod
    def from_refs(
        mde: MDEngine,
        folder: Path,
        refs: list[SystemParams] | None = None,
        steps=2e3,
    ) -> Scheme:
        bias = mde.bias

        cv = bias.collective_variable

        rnds = Rounds.create(folder=folder)
        rnds.add_cv(collective_variable=cv)

        rnds.add_round(bias=bias, stic=mde.static_trajectory_info, mde=mde, r=0)

        biases = []
        for _ in refs:
            biases.append(NoneBias.create(collective_variable=cv))

        rnds.run_par(biases=biases, steps=steps, sp0=refs, plot=False)

        rnds.add_round(bias=bias)

        return Scheme(rounds=rnds)

    def FESBias(
        self,
        rnd: int | None = None,
        cv_round: int | None = None,
        chunk_size=None,
        **plotkwargs,
    ) -> Bias:
        """replace the current md bias with the computed FES from current
        round."""

        return Observable.create(
            self.rounds,
            rnd=rnd,
            cv_round=cv_round,
        ).fes_bias(
            chunk_size=chunk_size,
            **plotkwargs,
        )

    @property
    def md(self):
        return self.rounds.get_engine()

    @property
    def bias(self):
        return self.rounds.get_bias()

    @property
    def sti(self):
        return self.rounds.static_trajectory_information()

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
        eps=0.1,  # overlap between wave functions
        min_traj_length=None,
        recalc_cv=False,
        only_finished=True,
        chunk_size=None,
        T_scale=10,
    ):
        m = self.bias.collective_variable.metric
        grid, _, cv_grid, _ = m.grid(
            n + 1,
            margin=0.0,
        )

        print(f"{grid=}")

        # sigma =

        if k is None:
            mu = jnp.array([a[1] - a[0] for a in grid])
            k = (2 / mu * jax.scipy.special.erfinv(1 - eps)) ** 2 * self.sti.T * boltzmann

            print(f"{k*kjmol=}")

        biases = [
            HarmonicBias.create(
                self.rounds.get_collective_variable(),
                cv,
                k,
                k_max=max_grad,
            )
            for cv in cv_grid
        ]

        print(f"{len(biases)=}")

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
            chunk_size=chunk_size,
            T_scale=T_scale,
            # use_fes_bias=True,
        )

    def inner_loop(
        self,
        rnds=10,
        convergence_kl=0.1,
        # init=0,
        steps=5e4,
        K=None,
        update_metric=False,
        n=4,
        samples_per_bin=5,
        min_samples_per_bin=1,
        init_max_grad=None,
        plot=True,
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
        plot_umbrella=False,
        max_bias=None,
        n_max_fes=1e5,
        thermolib=False,
        macro_chunk=10000,
        T_scale=10,
        koopman=True,
        lag_n=30,
        koopman_wham=None,
        out=-1,
        direct_bias=True,
        init=False,
        # use_fes_bias=True,
    ):
        if plot_umbrella is None:
            plot_umbrella = plot

        if cv_round is None:
            cv_round = self.rounds.cv

        print(f"{cv_round=}")

        # if init != 0:
        #     print(f"running init round with {init} steps")

        #     self.grid_umbrella(
        #         steps=init,
        #         n=n,
        #         k=K,
        #         max_grad=init_max_grad,
        #         plot=plot_umbrella,
        #         cv_round=cv_round,
        #         eps=eps_umbrella,
        #     )
        #     # self.rounds.invalidate_data(c=cv_round)

        i_0 = self.rounds.get_round(c=cv_round)

        if i_0 >= 2:
            prev_bias = self.rounds.get_bias(c=cv_round, r=i_0 - 1)
            bias = self.rounds.get_bias(c=cv_round, r=i_0)

            kl_div = bias.kl_divergence(prev_bias, T=self.rounds.T, symmetric=True)

            if kl_div < convergence_kl:
                print(f"already converged {kl_div=}")
                return
            else:
                print(f"not converged {kl_div=}")

        print(f"{i_0=}")

        for i in range(i_0, rnds):
            print(f"running round {i=} with {steps} steps")

            if i == 1:
                print(f"running first round wihtout biases")

            self.grid_umbrella(
                steps=steps,
                n=n,
                k=K,
                plot=plot_umbrella,
                scale_n=scale_n,
                cv_round=cv_round,
                ignore_invalid=False,
                eps=eps_umbrella if i > 1 else 1,
                min_traj_length=steps if (i > 1 and enforce_min_traj_length) else None,
                recalc_cv=recalc_cv,
                only_finished=i > 1 and only_finished,
                chunk_size=chunk_size,
                T_scale=T_scale,
                # use_fes_bias=True,
            )

            prev_bias = self.rounds.get_bias(c=cv_round, r=i)

            new_bias = self.FESBias(
                plot=plot,
                samples_per_bin=samples_per_bin,
                min_samples_per_bin=min_samples_per_bin,
                choice=choice,
                num_rnds=fes_bias_rnds,
                cv_round=cv_round,
                chunk_size=chunk_size,
                min_traj_length=steps if enforce_min_traj_length else None,
                margin=plot_margin,
                only_finished=only_finished,
                max_bias=max_bias,
                n_max=n_max_fes,
                thermolib=thermolib,
                macro_chunk=macro_chunk,
                vmax=max_bias,
                T_scale=T_scale,
                koopman=koopman,
                lag_n=lag_n,
                koopman_wham=koopman_wham,
                out=out,
                direct_bias=direct_bias,
            )

            self.rounds.add_round(bias=new_bias, c=cv_round)

            kl_div = self.md.bias.kl_divergence(
                prev_bias,
                T=self.rounds.T,
                symmetric=True,
            )

            print(f"{kl_div=}")

            if kl_div < convergence_kl:
                break
        print("done")

    def update_CV(
        self,
        transformer: Transformer,
        dlo_kwargs={},
        dlo: DataLoaderOutput | None = None,
        chunk_size=None,
        plot=True,
        new_r_cut=None,
        save_samples=True,
        save_multiple_cvs=False,
        jac=jax.jacrev,
        cv_round_from=None,
        test=False,
        max_bias=None,
        transform_bias=True,
        samples_per_bin=5,
        min_samples_per_bin=1,
        percentile=1e-1,
        use_executor=True,
        n_max=1e5,
        vmax=100 * kjmol,
        macro_chunk=2000,
        macro_chunk_nl=5000,
        verbose=False,
    ):
        self.rounds.update_CV(
            transformer=transformer,
            dlo_kwargs=dlo_kwargs,
            dlo=dlo,
            chunk_size=chunk_size,
            plot=plot,
            new_r_cut=new_r_cut,
            save_samples=save_samples,
            save_multiple_cvs=save_multiple_cvs,
            jac=jac,
            cv_round_from=cv_round_from,
            test=test,
            max_bias=max_bias,
            transform_bias=transform_bias,
            samples_per_bin=samples_per_bin,
            min_samples_per_bin=min_samples_per_bin,
            percentile=percentile,
            use_executor=use_executor,
            n_max=n_max,
            vmax=vmax,
            macro_chunk=macro_chunk,
            macro_chunk_nl=macro_chunk_nl,
            verbose=verbose,
        )

    def transform_CV(
        self,
        cv_trans: CvTrans,
        dlo_kwargs=None,
        dlo: DataLoaderOutput | None = None,
        chunk_size=None,
        new_r_cut=None,
        plot=True,
    ):
        self.rounds.transform_CV(
            cv_trans=cv_trans,
            chunk_size=chunk_size,
            dlo=dlo,
            dlo_kwargs=dlo_kwargs,
            cv_round_from=None,
            new_r_cut=new_r_cut,
            plot=plot,
        )
