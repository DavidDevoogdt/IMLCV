from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from parsl import File

from IMLCV.base.bias import Bias, BiasModify, GridBias, StdBias
from IMLCV.base.CV import CV, CollectiveVariable, CvMetric
from IMLCV.base.datastructures import Partial_decorator
from IMLCV.base.rounds import DataLoaderOutput, Rounds
from IMLCV.base.UnitsConstants import kelvin, kjmol, picosecond, boltzmann
from IMLCV.configs.bash_app_python import bash_app_python
from IMLCV.configs.config_general import Executors
from IMLCV.implementations.bias import RbfBias, _clip


@dataclass(kw_only=True)
class Observable:
    """class to convert data and CVs to different thermodynamic/ kinetic
    observables."""

    cv_round: int
    rounds: Rounds
    rnd: int
    common_bias: Bias
    collective_variable: CollectiveVariable | None

    time_per_bin: float = 2 * picosecond

    @staticmethod
    def create(
        rounds: Rounds,
        rnd=None,
        cv_round: int | None = None,
        collective_variable: CollectiveVariable | None = None,
    ) -> Observable:
        if cv_round is None:
            cv_round = rounds.cv

        if rnd is None:
            rnd = rounds.get_round(c=cv_round)

        common_bias = rounds.get_bias(c=cv_round, r=rnd)

        if collective_variable is None:
            b = rounds.get_bias(c=cv_round, r=rnd)
            collective_variable = b.collective_variable

        return Observable(
            cv_round=cv_round,
            rnd=rnd,
            rounds=rounds,
            common_bias=common_bias,
            collective_variable=collective_variable,
        )

    @staticmethod
    def _fes_nd_thermolib(
        dlo_kwargs: dict,
        dlo: DataLoaderOutput | None = None,
        update_bounding_box=True,
        bounds_percentile=1,
        samples_per_bin=5,
        min_samples_per_bin=1,
        n_hist=None,
        n_max=1e5,
        temp=None,
        chunk_size=None,
        shmap=False,
        rounds: Rounds | None = None,
        time_correlation_method=None,
    ):
        print(f"{dlo_kwargs["cv"]=}")

        if dlo is None:
            assert rounds is not None
            dlo = rounds.data_loader(**dlo_kwargs)  # type: ignore

        assert dlo is not None

        if temp is None:
            temp = dlo.sti.T

        trajs = dlo.cv
        biases = dlo.bias

        assert biases is not None

        nd = dlo.collective_variable.metric.ndim

        trajectories = [
            np.array(
                traj.cv[:, 0] if nd == 1 else traj.cv,
                dtype=np.double,
            )
            for traj in trajs
        ]

        print(f"{trajectories[0].shape=}")

        if time_correlation_method is not None:
            from thermolib.tools import decorrelate

            corrtimes = decorrelate(trajectories, method=time_correlation_method, plot=False)

            print(f"correlation times: {corrtimes} ps")

        else:
            corrtimes = jnp.ones(len(trajectories))

        bd = jnp.sum(jnp.array([trajs[i].shape[0] / corrtimes[i] for i in jnp.arange(len(trajs))]))

        if n_hist is None:
            n_hist = CvMetric.get_n(
                samples_per_bin,
                bd,
                trajs[0].shape[1],
                max_bins=n_max,
            )

            print(f"n: {n_hist}")

        if update_bounding_box:
            bounds, _, _ = CvMetric.bounds_from_cv(
                trajs,
                0.0,
            )

        else:
            bounds = dlo.collective_variable.metric.bounding_box

        bins, mid, cv, cv_mid, b = dlo.collective_variable.metric.grid(n_hist, bounds=bounds)
        bins_hist = [np.asarray(bins_i, dtype=np.double) for bins_i in bins]
        if nd == 1:
            bins_hist = bins_hist[0]

        print(f"{bins_hist=} {bins=}")
        # from thermolib.thermodynamics.bias import BiasPotential2D

        # from IMLCV.base.CV import padded_shard_map

        # if nd == 1:
        #     cvs = trajectories
        # elif nd == 2:
        #     cvs = [[a[:, i] for a in trajectories] for i in range(nd)]

        if nd == 1:
            from thermolib.thermodynamics.histogram import Histogram1D
            from thermolib.thermodynamics.bias import BiasPotential2D

            BiasND = BiasPotential2D
            HistND = Histogram1D
        elif nd == 2:
            from thermolib.thermodynamics.histogram import Histogram2D
            from thermolib.thermodynamics.bias import BiasPotential2D

            BiasND = BiasPotential2D
            HistND = Histogram2D
        else:
            raise NotImplementedError("only 1D and 2D FES implemented")

        class _ThermoBiasND(BiasND):
            def __init__(self, bias: Bias, chunk_size=None, num=None) -> None:
                self.bias = bias
                self.chunk_size = chunk_size
                self.num = num
                self.already_printed = False

                super().__init__(f"IMLCV_bias_{num}")

            @staticmethod
            @jax.jit
            def _calc(bias, cvs: CV) -> jax.Array:
                print(".", end="")
                f = Partial_decorator(
                    bias.compute_from_cv,
                    diff=False,
                    chunk_size=chunk_size,
                    shmap=False,
                )

                out, _ = f(cvs)

                return out

            def __call__(self, *cv):
                # print(".", end="")
                # if self.num is not None:
                #     if self.num % 100 == 0:
                #         print(f"\n bias {self.num} ", end="")
                #         self.already_printed = True

                out = _ThermoBiasND._calc(
                    self.bias, CV.combine(*[CV(cv=jnp.asarray(cvi).reshape((-1, 1))) for cvi in cv])
                )

                return np.asarray(jnp.reshape(out, cv[0].shape), dtype=np.double)

            def print_pars(self, *pars_units):
                pass

        bias_wrapped = [_ThermoBiasND(bias=b, chunk_size=chunk_size, num=i) for i, b in enumerate(biases)]

        histo = HistND.from_wham(
            bins=bins_hist,
            traj_input=trajectories,
            biasses=bias_wrapped,
            temp=temp,
            verbosity="high",
            error_estimate="mle_f",
            corrtimes=np.asarray(corrtimes, dtype=np.double),
            Nscf=10000,
        )

        # print(f"{histo.ps=}, {histo.error.lstds=}, cv_mid")

        return histo.ps, histo.error.lstds, cv_mid, bounds

    def fes_nd_thermolib(
        self,
        num_rnds=4,
        start_r=1,
        update_bounding_box=True,
        samples_per_bin=5,
        min_samples_per_bin=1,
        chunk_size=None,
        n_max=1e5,
        min_traj_length=None,
        dlo=None,
        directory=None,
        temp=None,
        shmap=False,
        only_finished=True,
        bounds_percentile=1,
        max_bias=None,
        rbf_kernel="thin_plate_spline",
        rbf_degree=None,
        executors=Executors.training,
        time_correlation_method=None,
        return_std_bias=False,
        n_hist=None,
    ):
        if temp is None:
            temp = self.rounds.T

        if directory is None:
            directory = self.rounds.path(c=self.cv_round, r=self.rnd)

        dlo_kwargs = {
            "num": num_rnds,
            "cv": self.cv_round,
            "start": start_r,
            # "split_data": False,
            "new_r_cut": None,
            "min_traj_length": min_traj_length,
            "get_bias_list": True,
            "only_finished": only_finished,
            "weight": False,
            "out": -1,
        }

        kwargs = dict(
            dlo_kwargs=dlo_kwargs,
            dlo=dlo,
            update_bounding_box=update_bounding_box,
            bounds_percentile=bounds_percentile,
            samples_per_bin=samples_per_bin,
            min_samples_per_bin=min_samples_per_bin,
            n_max=n_max,
            n_hist=n_hist,
            temp=temp,
            chunk_size=chunk_size,
            shmap=shmap,
            rounds=self.rounds,
            time_correlation_method=time_correlation_method,
        )

        print(f"executors: {executors}")

        if executors is None:
            ps, sigma, cv, bounds = Observable._fes_nd_thermolib(**kwargs)  # type:ignore

        else:
            ps, sigma, cv, bounds = bash_app_python(
                Observable._fes_nd_thermolib,
                executors=executors,
                execution_folder=directory,
            )(
                **kwargs,  # type:ignore
            ).result()

        shape_pre = ps.shape

        # # invert to use as bias, center zero
        ps = ps.reshape(-1)
        sigma = sigma.reshape(-1)

        mask = jnp.logical_and(jnp.isfinite(ps), ps > 0)
        ps_mask = ps[mask]
        sigma_mask = sigma[mask]

        cv_mask = cv[mask]

        fs_mask = -(kelvin * temp * boltzmann) * jnp.log(ps_mask)
        sigma_mask = sigma_mask

        fs_mask -= jnp.min(fs_mask)

        print(f"FES thermolib : {fs_mask/kjmol=} {sigma_mask=}")

        eps = ps.shape[0] / (
            (bounds[:, 1] - bounds[:, 0])
            / (self.collective_variable.metric.bounding_box[:, 1] - self.collective_variable.metric.bounding_box[:, 0])
        )

        fes_bias_tot = RbfBias.create(
            cvs=self.collective_variable,
            vals=-fs_mask,
            cv=cv_mask,
            kernel=rbf_kernel,
            epsilon=eps,
            degree=rbf_degree,
            smoothing=None,
        )

        if not return_std_bias:
            return fes_bias_tot

        fs = -(kelvin * temp * boltzmann) * jnp.log(ps)
        fs -= jnp.nanmin(fs)

        ss = (kelvin * temp * boltzmann) * sigma

        bounds_adjusted = GridBias.adjust_bounds(bounds=jnp.array(bounds), n=shape_pre[0])

        fes_bias = GridBias(
            collective_variable=self.collective_variable,
            n=shape_pre[0],
            bounds=bounds_adjusted,
            vals=-fs.reshape(shape_pre),
            order=0,
        )

        fes_std_bias = GridBias(
            collective_variable=self.collective_variable,
            n=shape_pre[0],
            bounds=bounds_adjusted,
            vals=(jnp.log(ss) - fs / (kelvin * temp * boltzmann)).reshape(shape_pre),
            order=0,
        )

        std_bias = StdBias.create(
            bias=fes_bias,
            log_exp_sigma=fes_std_bias,
        )

        return fes_bias_tot, std_bias

    @staticmethod
    def _fes_nd_weights(
        rounds: Rounds,
        num_rnds=4,
        out=int(3e4),
        lag_n=10,
        start_r=1,
        min_traj_length=None,
        only_finished=True,
        chunk_size=None,
        macro_chunk=1000,
        # T_scale=10,
        n_max=1e5,
        n_max_lin: int = 50,
        n_hist=None,
        cv=None,
        koopman=True,
        plot_selected_points=True,
        # divide_by_histogram=True,
        verbose=True,
        max_bias: float = 100 * kjmol,
        vmax: float = 100 * kjmol,
        vmax_std: float = 5 * kjmol,
        smoothing=None,
        kooopman_wham=None,
        samples_per_bin=10,
        min_samples_per_bin=5,
        resample=False,
        direct_bias=False,
        time_correlation_method=None,
        return_std_bias=False,
    ):
        if cv is None:
            cv = rounds.cv

        # if kooopman_wham is None:
        #     kooopman_wham = cv_round == 1

        dlo = rounds.data_loader(
            num=num_rnds,
            out=out,
            lag_n=lag_n,
            cv=cv,
            start=start_r,
            new_r_cut=None,
            min_traj_length=min_traj_length,
            only_finished=only_finished,
            time_series=koopman,
            chunk_size=chunk_size,
            macro_chunk=macro_chunk,
            verbose=verbose,
            n_max=n_max,
            n_max_lin=n_max_lin,
            samples_per_bin=samples_per_bin,
            min_samples_per_bin=min_samples_per_bin,
            time_correlation_method=time_correlation_method,
            n_hist=n_hist,
        )

        fes_bias_wham, std_bias_wham, _, _ = dlo.get_fes_bias_from_weights(
            n_max=n_max,
            chunk_size=chunk_size,
            macro_chunk=macro_chunk,
            max_bias=max_bias,
            samples_per_bin=samples_per_bin,
            min_samples_per_bin=min_samples_per_bin,
            smoothing=smoothing,
            n_max_lin=n_max_lin,
        )

        if plot_selected_points:
            print("plotting wham")
            fes_bias_wham.plot(
                name="FES_bias_wham_data.png",
                # traj=dlo.cv,
                margin=0.1,
                vmax=vmax,
                inverted=False,
            )

            std_bias_wham.plot(
                name="FES_bias_wham_std.png",
                # traj=dlo.cv,
                margin=0.1,
                vmax=vmax_std,
                inverted=False,
                offset=False,
            )

            std_bias_wham._bias.plot(
                name="FES_bias_wham_grid.png",
                # traj=dlo.cv,
                margin=0.1,
                vmax=vmax,
                inverted=False,
            )

        if koopman:
            weights, weights_t, w_corr, _ = dlo.koopman_weight(
                max_bins=n_max,
                samples_per_bin=samples_per_bin,
                # min_samples_per_bin=min_samples_per_bin,
                chunk_size=chunk_size,
                macro_chunk=macro_chunk,
                verbose=True,
                output_w_corr=True,
                # koopman_eps=1e-6,
                # koopman_eps_pre=1e,
                correlation=True,
                # add_1=True,
            )

            assert weights is not None

            fes_bias_tot, _, _, _ = dlo.get_fes_bias_from_weights(
                weights=weights,
                n_max=n_max,
                max_bias=max_bias,
                chunk_size=chunk_size,
                macro_chunk=macro_chunk,
                samples_per_bin=samples_per_bin,
                min_samples_per_bin=min_samples_per_bin,
                n_max_lin=n_max_lin,
            )

            if plot_selected_points:
                fes_bias_tot.plot(
                    name="FES_bias_koopman.png",
                    # traj=dlo.cv,
                    margin=0.1,
                    vmax=vmax,
                    inverted=False,
                )

            assert w_corr is not None

            if plot_selected_points:
                fes_bias_tot_corr, _, _, _ = dlo.get_fes_bias_from_weights(
                    weights=w_corr,
                    rho=[jnp.ones_like(x) for x in w_corr],
                    n_max=n_max,
                    max_bias=max_bias,
                    chunk_size=chunk_size,
                    macro_chunk=macro_chunk,
                    samples_per_bin=samples_per_bin,
                    min_samples_per_bin=min_samples_per_bin,
                    n_max_lin=n_max_lin,
                )

                fes_bias_tot_corr.plot(
                    name="FES_bias_corr.png",
                    # traj=dlo.cv,
                    margin=0.1,
                    vmax=vmax / 4,
                    inverted=False,
                )

            # if resample:
            #     fes_bias_tot = fes_bias_tot.resample(n=n_max)

        else:
            fes_bias_tot = fes_bias_wham

        if plot_selected_points:
            print("plotting")
            fes_bias_tot.plot(
                name="FES_bias_points.png",
                traj=dlo.cv,
                margin=0.1,
                vmax=vmax,
                inverted=False,
            )

        if return_std_bias:
            return fes_bias_tot, std_bias_wham

        return fes_bias_tot

    def fes_nd_weights(
        self,
        num_rnds: int = 4,
        out: int = int(1e5),
        lag_n: int = 10,
        start_r: int = 1,
        min_traj_length: int | None = None,
        only_finished: bool = True,
        chunk_size: int | None = None,
        macro_chunk: int = 1000,
        # T_scale=10,
        n_max: int | float = 1e5,
        n_max_lin: int = 50,
        cv_round: int | None = None,
        directory: str | Path | None = None,
        koopman: bool = True,
        # divide_by_histogram=True,
        verbose=True,
        max_bias: float | None = 100 * kjmol,
        vmax: float = 100 * kjmol,
        kooopman_wham=None,
        samples_per_bin=5,
        min_samples_per_bin=1,
        executors=Executors.training,
        direct_bias=False,
        time_correlation_method=None,
        return_std_bias=False,
        n_hist=None,
    ):
        if cv_round is None:
            cv_round = self.cv_round

        if directory is None:
            directory = self.rounds.path(c=self.cv_round, r=self.rnd)

        directory = Path(directory)

        kwargs = dict(
            rounds=self.rounds,
            num_rnds=num_rnds,
            out=out,
            lag_n=lag_n,
            start_r=start_r,
            min_traj_length=min_traj_length,
            only_finished=only_finished,
            chunk_size=chunk_size,
            macro_chunk=macro_chunk,
            # T_scale=T_scale,
            n_max=n_max,
            n_max_lin=n_max_lin,
            cv=cv_round,
            koopman=koopman,
            # divide_by_histogram=divide_by_histogram,
            verbose=verbose,
            max_bias=max_bias,
            kooopman_wham=kooopman_wham,
            samples_per_bin=samples_per_bin,
            min_samples_per_bin=min_samples_per_bin,
            direct_bias=direct_bias,
            vmax=vmax,
            time_correlation_method=time_correlation_method,
            return_std_bias=return_std_bias,
            n_hist=n_hist,
        )

        if executors is None:
            return Observable._fes_nd_weights(**kwargs)  # type:ignore

        return bash_app_python(
            Observable._fes_nd_weights,
            executors=Executors.training,
            remove_stdout=False,
            execution_folder=directory,
        )(
            **kwargs,  # type:ignore
        ).result()

    def fes_bias(
        self,
        plot=True,
        fes=None,
        max_bias=None,
        choice="rbf",
        num_rnds=8,
        start_r=1,
        rbf_kernel="thin_plate_spline",
        rbf_degree=None,
        samples_per_bin=5,
        min_samples_per_bin=1,
        chunk_size=None,
        macro_chunk=10000,
        update_bounding_box=True,  # make boudning box bigger for FES calculation
        n_max=1e5,
        min_traj_length=None,
        margin=0.1,
        only_finished=True,
        shmap=False,
        thermolib=False,
        lag_n=30,
        out=int(1e5),
        # T_scale=10,
        vmax=100 * kjmol,
        koopman=True,
        # divide_by_histogram=True,
        verbose=True,
        koopman_wham=None,
        executors=Executors.training,
        direct_bias=False,
        n_max_lin: int = 50,
        time_correlation_method="blav",
        return_std_bias=False,
        n_hist=None,
    ):
        print(f"{n_max_lin=}")

        if plot:
            directory = self.rounds.path(c=self.cv_round, r=self.rnd)

            bash_app_python(
                function=Bias.plot,
                execution_folder=directory,
                stdout="combined.stdout",
                stderr="combined.stderr",
                outputs=[directory / "combined.png"],
            )(
                self=self.common_bias,
                name="combined.png",
                map=False,
                dlo=self.rounds,
                dlo_kwargs=dict(
                    out=-1,  # max number of points to plot
                    num=1,
                    ignore_invalid=False,
                    cv=self.cv_round,
                    split_data=True,
                    new_r_cut=None,
                    min_traj_length=min_traj_length,
                    only_finished=only_finished,
                    chunk_size=chunk_size,
                    macro_chunk=macro_chunk,
                    weight=False,
                ),
                plot_FES=False,
                margin=margin,
                vmax=vmax,
            )

        if thermolib:
            print("estimating bias from thermolib WHAM!")
            fes_bias_tot = self.fes_nd_thermolib(
                start_r=start_r,
                samples_per_bin=samples_per_bin,
                min_samples_per_bin=min_samples_per_bin,
                num_rnds=num_rnds,
                chunk_size=chunk_size,
                update_bounding_box=update_bounding_box,
                n_max=n_max,
                min_traj_length=min_traj_length,
                # margin=margin,
                only_finished=only_finished,
                shmap=shmap,
                max_bias=max_bias,
                rbf_kernel=rbf_kernel,
                rbf_degree=rbf_degree,
                executors=executors,
                time_correlation_method=time_correlation_method,
                return_std_bias=return_std_bias,
                n_hist=n_hist,
            )

        else:
            print("estimating bias from koopman Theory!")

            fes_bias_tot = self.fes_nd_weights(
                num_rnds=num_rnds,
                out=out,
                lag_n=lag_n,
                start_r=start_r,
                min_traj_length=min_traj_length,
                only_finished=only_finished,
                chunk_size=chunk_size,
                macro_chunk=macro_chunk,
                # T_scale=T_scale,
                n_max=n_max,
                koopman=koopman,
                # divide_by_histogram=divide_by_histogram,
                verbose=verbose,
                max_bias=max_bias,
                vmax=vmax,
                kooopman_wham=koopman_wham,
                samples_per_bin=samples_per_bin,
                min_samples_per_bin=min_samples_per_bin,
                executors=executors,
                direct_bias=direct_bias,
                n_max_lin=n_max_lin,
                time_correlation_method=time_correlation_method,
                return_std_bias=return_std_bias,
                n_hist=n_hist,
            )

        return fes_bias_tot
