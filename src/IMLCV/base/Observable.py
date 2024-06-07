from __future__ import annotations

import itertools

import jax.numpy as jnp
import numpy as np
from IMLCV.base.bias import Bias
from IMLCV.base.bias import BiasModify
from IMLCV.base.CV import CollectiveVariable
from IMLCV.base.CV import CV
from IMLCV.base.CV import CvMetric
from IMLCV.base.rounds import Rounds
from IMLCV.configs.bash_app_python import bash_app_python
from IMLCV.implementations.bias import _clip
from IMLCV.implementations.bias import RbfBias
from molmod.units import kjmol
from molmod.units import picosecond
from parsl import File
from thermolib.thermodynamics.bias import BiasPotential2D
from thermolib.thermodynamics.fep import FreeEnergyHypersurfaceND
from thermolib.thermodynamics.histogram import HistogramND
from jax.tree_util import Partial
from dataclasses import dataclass
from IMLCV.base.rounds import data_loader_output
from IMLCV.configs.config_general import Executors


@dataclass(kw_only=True)
class ThermoLIB:
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
    ) -> ThermoLIB:
        if cv_round is None:
            cv_round = rounds.cv

        if rnd is None:
            rnd = rounds.get_round(c=cv_round)

        common_bias = rounds.get_bias(c=cv_round, r=rnd)

        if collective_variable is None:
            b = rounds.get_bias(c=cv_round, r=rnd)
            collective_variable = b.collective_variable

        return ThermoLIB(
            cv_round=cv_round,
            rnd=rnd,
            rounds=rounds,
            common_bias=common_bias,
            collective_variable=collective_variable,
        )

    @staticmethod
    def _fes_nd_thermolib(
        dlo_kwargs,
        dlo: data_loader_output | None = None,
        update_bounding_box=True,
        bounds_percentile=1,
        samples_per_bin=20,
        min_samples_per_bin=2,
        n=None,
        n_max=30,
        temp=None,
        chunk_size=None,
        pmap=True,
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
            bounds, _, _ = CvMetric.bounds_from_cv(trajs, bounds_percentile)

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
            n = CvMetric.get_n(samples_per_bin, bd, trajs[0].shape[1])

            print(f"n: {n}")

        assert n >= 4, "sample more points"

        if n > n_max:
            print(f"truncating number of bins {n=} to {n_max=}")
            n = n_max

        # TODO: use metric grid to generate bins and center
        # bins, cv_grid, mid_cv_grid = dlo.collective_variable.metric.grid(n, endpoints=True)

        bins = [np.linspace(mini, maxi, n, endpoint=True, dtype=np.double) for mini, maxi in bounding_box]

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
                f = Partial(self.bias.compute_from_cv, diff=False, chunk_size=chunk_size)

                if pmap:
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
        samples_per_bin=20,
        min_samples_per_bin=2,
        chunk_size=None,
        n_max=60,
        n=None,
        min_traj_length=None,
        dlo=None,
        directory=None,
        temp=None,
        pmap=True,
        only_finished=True,
        bounds_percentile=1,
        max_bias=None,
        rbf_kernel="linear",
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
            "split_data": True,
            "new_r_cut": None,
            "min_traj_length": min_traj_length,
            "get_bias_list": True,
            "only_finished": only_finished,
        }

        fes, grid, bounds = bash_app_python(
            ThermoLIB._fes_nd_thermolib,
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
            pmap=pmap,
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

        print(f"min fs: {-fs_min/kjmol} kjmol")

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

        eps = fs.shape[0] / (bounds[:, 1] - bounds[:, 0])

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
        macro_chunk=10000,
        T_scale=10,
        n_max=30,
        cv_round=None,
    ):
        dlo = rounds.data_loader(
            num=num_rnds,
            out=out,
            lag_n=lag_n,
            cv_round=cv_round,
            start=start_r,
            new_r_cut=None,
            min_traj_length=min_traj_length,
            only_finished=only_finished,
            time_series=True,
            chunk_size=chunk_size,
            macro_chunk=macro_chunk,
            T_scale=T_scale,
        )

        # get weights based on koopman theory. the CVs are binned with indicators
        weights = dlo.weights(
            koopman=True,
            indicator_CV=True,
            n_max=n_max,
            chunk_size=chunk_size,
            macro_chunk=macro_chunk,
        )

        print("gettingg FES Bias")

        fes_bias_tot = dlo._get_fes_bias_from_weights(
            weights=weights,
            cv=dlo.cv,
            n_grid=n_max,
            T=dlo.sti.T,
            collective_variable=dlo.collective_variable,
            chunk_size=chunk_size,
            macro_chunk=macro_chunk,
        )

        return fes_bias_tot

    def fes_nd_weights(
        self,
        num_rnds=4,
        out=int(3e4),
        lag_n=10,
        start_r=1,
        min_traj_length=None,
        only_finished=True,
        chunk_size=None,
        macro_chunk=10000,
        T_scale=10,
        n_max=30,
        cv_round=None,
        directory=None,
    ):
        if cv_round is None:
            cv_round = self.cv_round

        if directory is None:
            directory = self.rounds.path(c=self.cv_round, r=self.rnd)

        return bash_app_python(
            ThermoLIB._fes_nd_weights,
            executors=Executors.training,
        )(
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
            execution_folder=directory,
        ).result()

    def fes_bias(
        self,
        plot=True,
        fes=None,
        max_bias=None,
        choice="rbf",
        num_rnds=4,
        start_r=1,
        rbf_kernel="linear",
        rbf_degree=None,
        samples_per_bin=200,
        min_samples_per_bin=2,
        chunk_size=None,
        macro_chunk=10000,
        update_bounding_box=True,  # make boudning box bigger for FES calculation
        n_max=60,
        min_traj_length=None,
        margin=0.1,
        only_finished=True,
        pmap=True,
        thermolib=True,
        lag_n=10,
        out=int(3e4),
        T_scale=10,
        vmax=None,
    ):
        if plot:
            directory = self.rounds.path(c=self.cv_round, r=self.rnd)

            trajs_plot = self.rounds.data_loader(
                num=1,
                ignore_invalid=False,
                cv_round=self.cv_round,
                split_data=True,
                new_r_cut=None,
                min_traj_length=min_traj_length,
                only_finished=only_finished,
                chunk_size=chunk_size,
                macro_chunk=macro_chunk,
            ).cv

            bash_app_python(function=Bias.static_plot)(
                bias=self.common_bias,
                outputs=[File(f"{directory}/combined.png")],  # png because heavy file
                name="combined.png",
                execution_folder=directory,
                stdout="combined.stdout",
                stderr="combined.stderr",
                map=False,
                traj=trajs_plot,
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
                pmap=pmap,
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
            )

        if plot:
            fold = str(self.rounds.path(c=self.cv_round))

            pf = []

            pf.append(
                bash_app_python(function=Bias.static_plot)(
                    bias=fes_bias_tot,
                    outputs=[File(f"{fold}/FES_bias_{self.rnd}_inverted_{choice}.png")],
                    execution_folder=fold,
                    name=f"FES_bias_{self.rnd}_inverted_{choice}.png",
                    inverted=True,
                    label="Free Energy [kJ/mol]",
                    stdout=f"FES_bias_{self.rnd}_inverted_{choice}.stdout",
                    stderr=f"FES_bias_{self.rnd}_inverted_{choice}.stderr",
                    margin=margin,
                    vmax=vmax,
                ),
            )

            pf.append(
                bash_app_python(function=Bias.static_plot)(
                    bias=fes_bias_tot,
                    outputs=[File(f"{fold}/FES_bias_{self.rnd}_{choice}.png")],
                    execution_folder=fold,
                    name=f"FES_bias_{self.rnd}_{choice}.png",
                    stdout=f"FES_bias_{self.rnd}_{choice}.stdout",
                    stderr=f"FES_bias_{self.rnd}_{choice}.stderr",
                    margin=margin,
                    vmax=vmax,
                ),
            )

            for f in pf:
                f.result()

        return fes_bias_tot
