from __future__ import annotations

import itertools
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax.tree_util import Partial
from parsl import File

from IMLCV.base.bias import Bias, BiasModify
from IMLCV.base.CV import CV, CollectiveVariable, CvMetric
from IMLCV.base.rounds import DataLoaderOutput, Rounds
from IMLCV.base.UnitsConstants import kelvin, kjmol, picosecond
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
    collective_variable: CollectiveVariable

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
        dlo_kwargs,
        dlo: DataLoaderOutput | None = None,
        update_bounding_box=True,
        bounds_percentile=1,
        samples_per_bin=5,
        min_samples_per_bin=1,
        n=None,
        n_max=1e5,
        temp=None,
        chunk_size=None,
        shmap=False,
        rounds: Rounds = None,
    ):
        if dlo is None:
            dlo = rounds.data_loader(**dlo_kwargs)

        if temp is None:
            temp = dlo.sti.T

        trajs = dlo.cv
        biases = dlo.bias

        # c = CV.stack(*trajs)
        # c = dlo.collective_variable.metric.periodic_wrap(c)

        if update_bounding_box:
            bounds, _, _ = CvMetric.bounds_from_cv(
                trajs,
                bounds_percentile,
            )

            # # do not update periodic bounds
            bounding_box = jnp.where(
                dlo.collective_variable.metric.periodicities,
                dlo.collective_variable.metric.bounding_box,
                bounds,
            )

            print(f"updated bounding box: {bounding_box}")

            # bounding_box = bounds

        else:
            bounding_box = dlo.collective_variable.metric.bounding_box

        bd = [i.shape[0] for i in trajs]

        if n is None:
            n = CvMetric.get_n(
                samples_per_bin,
                bd,
                trajs[0].shape[1],
                max_bins=n_max,
            )

            print(f"n: {n}")

        assert n >= 4, "sample more points"

        # TODO: use metric grid to generate bins and center
        # bins, cv_grid, mid_cv_grid = dlo.collective_variable.metric.grid(n, endpoints=True)

        bins = [np.linspace(mini, maxi, n, endpoint=True, dtype=np.double) for mini, maxi in bounding_box]

        from thermolib.thermodynamics.bias import BiasPotential2D
        from thermolib.thermodynamics.fep import FreeEnergyHypersurfaceND
        from thermolib.thermodynamics.histogram import HistogramND

        from IMLCV.base.CV import padded_shard_map

        class _ThermoBiasND(BiasPotential2D):
            def __init__(self, bias: Bias, chunk_size=None, num=None) -> None:
                self.bias = bias
                self.chunk_size = chunk_size
                self.num = num

                super().__init__(f"IMLCV_bias_{num}")

            def __call__(self, *cv):
                print(".", end="")

                cvs = CV.combine(*[CV(cv=jnp.asarray(cvi).reshape((-1, 1))) for cvi in cv])
                f = Partial(
                    self.bias.compute_from_cv,
                    diff=False,
                    chunk_size=chunk_size,
                    shmap=False,
                )

                if shmap:
                    f = padded_shard_map(f)

                out, _ = f(cvs)

                return np.asarray(jnp.reshape(out, cv[0].shape), dtype=np.double)

            def print_pars(self, *pars_units):
                pass

        bias_wrapped = [_ThermoBiasND(bias=b, chunk_size=chunk_size, num=i) for i, b in enumerate(biases)]

        histo = HistogramND.from_wham(
            # bins=[np.array(b, dtype=np.double) for b in bins],
            bins=bins,
            trajectories=[
                np.array(
                    traj.cv,
                    dtype=np.double,
                )
                for traj in trajs
            ],
            error_estimate=None,
            biasses=bias_wrapped,
            temp=temp,
            verbosity="high",
        )

        fes = FreeEnergyHypersurfaceND.from_histogram(histo, temp)
        fes.set_ref()

        # xy indexing
        # construc list with centers CVs
        bin_centers = [0.5 * (x[:-1] + x[1:]) for x in bins]
        Ngrid = np.array([len(bi) for bi in bin_centers])
        grid = []
        for idx in itertools.product(*(range(x) for x in Ngrid)):
            center = [bin_centers[j][k] for j, k in enumerate(idx)]
            grid.append((idx, CV(cv=jnp.array(center))))

        return fes, grid, bounding_box

    def fes_nd_thermolib(
        self,
        num_rnds=4,
        start_r=1,
        update_bounding_box=True,
        samples_per_bin=5,
        min_samples_per_bin=1,
        chunk_size=None,
        n_max=1e5,
        n=None,
        min_traj_length=None,
        dlo=None,
        directory=None,
        temp=None,
        shmap=False,
        only_finished=True,
        bounds_percentile=1,
        max_bias=None,
        rbf_kernel="multiquadric",
        rbf_degree=None,
    ):
        if temp is None:
            temp = self.rounds.T

        if directory is None:
            directory = self.rounds.path(c=self.cv_round, r=self.rnd)

        dlo_kwargs = {
            "num": num_rnds,
            "cv_round": self.cv_round,
            "start": start_r,
            # "split_data": False,
            "new_r_cut": None,
            "min_traj_length": min_traj_length,
            "get_bias_list": True,
            "only_finished": only_finished,
            "weight": True,
        }

        fes, grid, bounds = bash_app_python(
            Observable._fes_nd_thermolib,
            executors=Executors.training,
        )(
            dlo_kwargs=dlo_kwargs,
            dlo=dlo,
            update_bounding_box=update_bounding_box,
            bounds_percentile=bounds_percentile,
            samples_per_bin=samples_per_bin,
            min_samples_per_bin=min_samples_per_bin,
            n=n,
            n_max=n_max,
            temp=temp,
            execution_folder=directory,
            chunk_size=chunk_size,
            shmap=shmap,
            rounds=self.rounds,
        ).result()

        # fes is in 'xy'- indexing convention, convert to ij
        fs = np.transpose(fes.fs)

        # # invert to use as bias, center zero
        mask = ~np.isnan(fs)
        fs[:] = -(fs[:] - fs[mask].min())

        fs_max = fs[mask].max()  # = 0 kjmol
        fs_min = fs[mask].min()  # = -max_bias kjmol

        if max_bias is not None:
            fs_min = -max_bias

        print(f"min fs: {-fs_min / kjmol} kjmol")

        fslist = []
        # smoothing_list = []
        cv: list[CV] = []

        for idx, cvi in grid:
            if not np.isnan(fs[idx]):
                fslist.append(fs[idx])

                cv += [cvi]

                # smoothing_list.append(sigma[idx])
        cv = CV.stack(*cv)

        fslist = jnp.array(fslist)
        bounds = jnp.array(bounds)

        eps = fs.shape[0] / (
            (bounds[:, 1] - bounds[:, 0])
            / (self.collective_variable.metric.bounds[:, 1] - self.collective_variable.metric.bounds[:, 0])
        )

        fes_bias_tot = RbfBias.create(
            cvs=self.collective_variable,
            vals=fslist,
            cv=cv,
            kernel=rbf_kernel,
            epsilon=eps,
            degree=rbf_degree,
        )

        # clip value of bias to min and max of computed FES
        fes_bias_tot = BiasModify.create(
            bias=fes_bias_tot,
            fun=_clip,
            kwargs={"a_min": fs_min, "a_max": fs_max},
        )

        return fes_bias_tot

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
        T_scale=10,
        n_max=1e5,
        cv_round=None,
        koopman=True,
        plot_selected_points=True,
        # divide_by_histogram=True,
        verbose=True,
        max_bias=None,
        kooopman_wham=None,
        samples_per_bin=10,
        min_samples_per_bin=5,
        resample=False,
        direct_bias=False,
    ):
        if cv_round is None:
            cv_round = rounds.cv

        if kooopman_wham is None:
            kooopman_wham = cv_round == 1

        dlo, fb = rounds.data_loader(
            num=num_rnds,
            out=out,
            lag_n=lag_n,
            cv_round=cv_round,
            start=start_r,
            new_r_cut=None,
            min_traj_length=min_traj_length,
            only_finished=only_finished,
            time_series=koopman,
            chunk_size=chunk_size,
            macro_chunk=macro_chunk,
            T_scale=T_scale,
            verbose=verbose,
            # divide_by_histogram=divide_by_histogram,
            n_max=n_max,
            wham=kooopman_wham,
            # uniform=True,
            output_FES_bias=True,
            reweight_inverse_bincount=True,
            samples_per_bin=samples_per_bin,
            min_samples_per_bin=min_samples_per_bin,
        )

        fes_bias_wham_p = fb[0]

        # get weights based on koopman theory. the CVs are binned with indicators

        if plot_selected_points:
            print("plotting wham")
            fes_bias_wham_p.plot(
                name="FES_bias_wham.png",
                # traj=dlo.cv,
                margin=0.1,
                vmax=max_bias,
                inverted=False,
            )

            from IMLCV.base.CVDiscovery import Transformer

            Transformer.plot_app(
                collective_variables=[dlo.collective_variable],
                cv_data=[[dlo.cv]],
                duplicate_cv_data=False,
                plot_FES=True,
                T=300 * kelvin,
                margin=0.1,
                name="points.png",
            )

        if direct_bias:
            fes_bias_wham = fes_bias_wham_p
        else:
            fes_bias_wham = dlo.get_fes_bias_from_weights(
                n_max=n_max,
                chunk_size=chunk_size,
                macro_chunk=macro_chunk,
                max_bias=max_bias,
                samples_per_bin=samples_per_bin,
                min_samples_per_bin=min_samples_per_bin,
            )

            if plot_selected_points:
                print("plotting wham")
                fes_bias_wham.plot(
                    name="FES_bias_wham_data.png",
                    # traj=dlo.cv,
                    margin=0.1,
                    vmax=max_bias,
                    inverted=False,
                )

        if koopman:
            weights, w_corr = dlo.koopman_weight(
                max_bins=n_max,
                samples_per_bin=samples_per_bin,
                # min_samples_per_bin=min_samples_per_bin,
                chunk_size=chunk_size,
                macro_chunk=macro_chunk,
                verbose=verbose,
                output_w_corr=True,
                koopman_eps=1e-10,
                koopman_eps_pre=0,
                correlation=True,
                # add_1=True,
            )

            fes_bias_tot = dlo.get_fes_bias_from_weights(
                weights=weights,
                n_max=n_max,
                chunk_size=chunk_size,
                macro_chunk=macro_chunk,
                samples_per_bin=samples_per_bin,
                min_samples_per_bin=min_samples_per_bin,
            )

            if plot_selected_points:
                fes_bias_tot.plot(
                    name="FES_bias_koopman.png",
                    # traj=dlo.cv,
                    margin=0.1,
                    vmax=max_bias,
                    inverted=False,
                )

            if plot_selected_points:
                fes_bias_tot_corr = dlo.get_fes_bias_from_weights(
                    weights=w_corr,
                    rho=[jnp.ones_like(x) for x in w_corr],
                    n_max=n_max,
                    chunk_size=chunk_size,
                    macro_chunk=macro_chunk,
                    samples_per_bin=samples_per_bin,
                    min_samples_per_bin=min_samples_per_bin,
                )

                fes_bias_tot_corr.plot(
                    name="FES_bias_corr.png",
                    # traj=dlo.cv,
                    margin=0.1,
                    vmax=max_bias / 4,
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
                vmax=max_bias,
                inverted=False,
            )

        return fes_bias_tot

    def fes_nd_weights(
        self,
        num_rnds=4,
        out=int(1e5),
        lag_n=10,
        start_r=1,
        min_traj_length=None,
        only_finished=True,
        chunk_size=None,
        macro_chunk=1000,
        T_scale=10,
        n_max=1e5,
        cv_round=None,
        directory=None,
        koopman=True,
        # divide_by_histogram=True,
        verbose=True,
        max_bias=None,
        kooopman_wham=None,
        samples_per_bin=5,
        min_samples_per_bin=1,
        executors=Executors.training,
        direct_bias=False,
    ):
        if cv_round is None:
            cv_round = self.cv_round

        if directory is None:
            directory = self.rounds.path(c=self.cv_round, r=self.rnd)

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
            T_scale=T_scale,
            n_max=n_max,
            cv_round=cv_round,
            koopman=koopman,
            # divide_by_histogram=divide_by_histogram,
            verbose=verbose,
            max_bias=max_bias,
            kooopman_wham=kooopman_wham,
            samples_per_bin=samples_per_bin,
            min_samples_per_bin=min_samples_per_bin,
            direct_bias=direct_bias,
        )

        if executors is None:
            return Observable._fes_nd_weights(**kwargs)

        return bash_app_python(
            Observable._fes_nd_weights,
            executors=Executors.training,
            remove_stdout=False,
        )(
            **kwargs,
            execution_folder=directory,
        ).result()

    def fes_bias(
        self,
        plot=True,
        fes=None,
        max_bias=None,
        choice="rbf",
        num_rnds=8,
        start_r=1,
        rbf_kernel="multiquadric",
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
        T_scale=10,
        vmax=100 * kjmol,
        koopman=True,
        # divide_by_histogram=True,
        verbose=True,
        koopman_wham=None,
        executors=Executors.training,
        direct_bias=False,
    ):
        if plot:
            directory = self.rounds.path(c=self.cv_round, r=self.rnd)

            bash_app_python(function=Bias.static_plot)(
                bias=self.common_bias,
                outputs=[File(f"{directory}/combined.png")],  # png because heavy file
                name="combined.png",
                execution_folder=directory,
                stdout="combined.stdout",
                stderr="combined.stderr",
                map=False,
                dlo=self.rounds,
                dlo_kwargs=dict(
                    out=-1,  # max number of points to plot
                    num=1,
                    ignore_invalid=False,
                    cv_round=self.cv_round,
                    split_data=True,
                    new_r_cut=None,
                    min_traj_length=min_traj_length,
                    only_finished=only_finished,
                    chunk_size=chunk_size,
                    macro_chunk=macro_chunk,
                    weight=False,
                ),
                margin=margin,
                vmax=vmax,
            )

        if thermolib:
            fes_bias_tot = self.fes_nd_thermolib(
                start_r=start_r,
                samples_per_bin=samples_per_bin,
                min_samples_per_bin=min_samples_per_bin,
                num_rnds=num_rnds,
                chunk_size=chunk_size,
                update_bounding_box=update_bounding_box,
                n_max=n_max,
                min_traj_length=min_traj_length,
                margin=margin,
                only_finished=only_finished,
                shmap=shmap,
                max_bias=max_bias,
                rbf_kernel=rbf_kernel,
                rbf_degree=rbf_degree,
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
                T_scale=T_scale,
                n_max=n_max,
                koopman=koopman,
                # divide_by_histogram=divide_by_histogram,
                verbose=verbose,
                max_bias=max_bias,
                kooopman_wham=koopman_wham,
                samples_per_bin=samples_per_bin,
                min_samples_per_bin=min_samples_per_bin,
                executors=executors,
                direct_bias=direct_bias,
            )

        return fes_bias_tot
