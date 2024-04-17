from __future__ import annotations


import jax
from IMLCV.base.CV import SystemParams
from IMLCV.base.CVDiscovery import Transformer
from IMLCV.base.MdEngine import MDEngine
from IMLCV.base.Observable import ThermoLIB
from IMLCV.base.rounds import Rounds
from IMLCV.implementations.bias import HarmonicBias
from molmod.constants import boltzmann
from IMLCV.base.rounds import data_loader_output
from IMLCV.base.bias import NoneBias
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Scheme:
    """base class that implements iterative scheme.

    args:
        format (String): intermediate file type between rounds
        CVs: list of CV instances.
    """

    md: MDEngine
    rounds: Rounds

    @staticmethod
    def from_rounds(rounds: Rounds, md=None) -> Scheme:
        if md is None:
            md = rounds.get_engine()

        return Scheme(md=md, rounds=rounds)

    @staticmethod
    def from_refs(
        mde: MDEngine,
        folder: Path,
        refs: list[SystemParams] | None = None,
        steps=2e3,
    ) -> Scheme:
        cv = mde.bias.collective_variable

        rnds = Rounds.create(folder=folder)
        rnds.add_cv_from_cv(cv=cv)

        rnds.add_round_from_md(mde, r=0)

        biases = []
        for _ in refs:
            biases.append(NoneBias.create(collective_variable=cv))

        rnds.run_par(biases=biases, steps=steps, sp0=refs, plot=False)

        rnds.add_round_from_md(mde)

        return Scheme(rounds=rnds, md=mde)

    def FESBias(self, cv_round: int | None = None, chunk_size=None, **plotkwargs):
        """replace the current md bias with the computed FES from current
        round."""
        obs = ThermoLIB.create(self.rounds, cv_round=cv_round)
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
        _, cv_grid, _ = m.grid(n)

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
                cv,
                k,
                k_max=max_grad,
            )
            for cv in cv_grid
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
        o = ThermoLIB.create(rounds=self.rounds, cv_round=cv_round)

        self.md.bias.collective_variable.metric = o.new_metric(plot=plot, r=r)

    def inner_loop(
        self,
        rnds=10,
        convergence_kl=0.1,
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
        n_max_fes=30,
        # resample_num=20,
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

        if i_0 >= 2:
            prev_bias = self.rounds.get_bias(c=cv_round, r=i_0 - 1)
            bias = self.rounds.get_bias(c=cv_round, r=i_0)

            kl_div = bias.kl_divergence(prev_bias, T=self.rounds.T, symmetric=True)

            if kl_div < convergence_kl:
                print("already converged")
                return

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
                prev_bias = self.md.bias

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
                    n_max=n_max_fes,
                    **plot_kwargs,
                )

            self.rounds.add_round_from_md(self.md, cv=cv_round)

            kl_div = self.md.bias.kl_divergence(prev_bias, T=self.rounds.T, symmetric=True)

            print(f"{kl_div=}")

            if kl_div < convergence_kl:
                break
        print("done")

    def update_CV(
        self,
        transformer: Transformer,
        dlo_kwargs={},
        dlo: data_loader_output | None = None,
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
        samples_per_bin=100,
        min_samples_per_bin=20,
        percentile=1e-1,
        use_executor=True,
        n_max=30,
    ):
        md = self.rounds.update_CV(
            md=self.md,
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
        )

        self.md = md
