from __future__ import annotations

import itertools

import jax.numpy as jnp
import numpy as np
from IMLCV.base.bias import Bias
from IMLCV.base.bias import BiasModify
from IMLCV.base.bias import CompositeBias
from IMLCV.base.CV import CollectiveVariable
from IMLCV.base.CV import CV
from IMLCV.base.CV import CvMetric
from IMLCV.base.rounds import Rounds
from IMLCV.configs.bash_app_python import bash_app_python
from IMLCV.implementations.bias import _clip
from IMLCV.implementations.bias import GridBias
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
        rounds=None,
    ):
        if dlo is None:
            dlo = rounds.data_loader(**dlo_kwargs)

        if temp is None:
            temp = dlo.sti.T

        trajs = dlo.cv
        biases = dlo.bias

        c = CV.stack(*trajs)
        # c = dlo.collective_variable.metric.periodic_wrap(c)

        if update_bounding_box:
            bounds, _ = CvMetric.bounds_from_cv(c, bounds_percentile)

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

        if n is None:
            n = CvMetric.get_n(samples_per_bin, c.batch_dim, c.dim)

            print(f"n: {n}")

        assert n >= 4, "sample more points"

        if n > n_max:
            print(f"truncating number of bins {n=} to {n_max=}")
            n = n_max

        # TODO: use metric grid to generate bins and center
        # bins, cv_grid, mid_cv_grid = dlo.collective_variable.metric.grid(n, endpoints=True)

        bins = [np.linspace(mini, maxi, n, endpoint=True, dtype=np.double) for mini, maxi in bounding_box]

        from IMLCV.base.CV import padded_pmap

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
                    f = padded_pmap(f)

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
        plot=True,
        num_rnds=4,
        start_r=1,
        update_bounding_box=True,
        samples_per_bin=20,
        min_samples_per_bin=2,
        chunk_size=None,
        n_max=60,
        n=None,
        min_traj_length=None,
        margin=None,
        dlo=None,
        directory=None,
        temp=None,
        pmap=True,
        only_finished=True,
        bounds_percentile=1,
        vmax=100 * kjmol,
        thermolib=False,
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

        # return ThermoLIB._fes_nd_thermolib(
        #     dlo_kwargs=dlo_kwargs,
        #     dlo=dlo,
        #     update_bounding_box=update_bounding_box,
        #     bounds_percentile=bounds_percentile,
        #     samples_per_bin=samples_per_bin,
        #     n=n,
        #     n_max=n_max,
        #     temp=temp,
        #     chunk_size=chunk_size,
        #     pmap=pmap,
        #     rounds=self.rounds,
        # )

        return bash_app_python(
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

    def new_metric(self, plot=False, r=None):
        assert isinstance(self.rounds, Rounds)

        raise NotImplementedError

    def fes_bias(
        self,
        plot=True,
        # max_bias: float | None = None,
        fes=None,
        max_bias=None,
        choice="rbf",
        num_rnds=4,
        start_r=1,
        rbf_kernel="linear",
        rbf_degree=None,
        smoothing_threshold=5 * kjmol,
        samples_per_bin=200,
        min_samples_per_bin=2,
        chunk_size=None,
        resample_bias=True,
        update_bounding_box=True,  # make boudning box bigger for FES calculation
        n_max=60,
        min_traj_length=None,
        margin=0.1,
        grid=None,
        bounds=None,
        use_prev_fs=False,
        collective_variable=None,
        only_finished=True,
        vmax=100 * kjmol,
        pmap=True,
        resample_num=30,
        thermolib=True,
        lag_n=10,
        **plot_kwargs,
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
            if fes is None:
                fes, grid, bounds = self.fes_nd_thermolib(
                    plot=plot,
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
                    vmax=vmax,
                    pmap=pmap,
                )

            # fes is in 'xy'- indexing convention, convert to ij
            fs = np.transpose(fes.fs)

            # remove previous fs
            cv_grid = CV.stack(*list(zip(*grid))[1])

            if collective_variable is None:
                collective_variable = self.collective_variable

            # # invert to use as bias, center zero
            mask = ~np.isnan(fs)
            fs[:] = -(fs[:] - fs[mask].min())

            fs_max = fs[mask].max()  # = 0 kjmol
            fs_min = fs[mask].min()  # = -max_bias kjmol

            if max_bias is not None:
                fs_min = -max_bias

            print(f"min fs: {-fs_min/kjmol} kjmol")

            # if max_bias is None:
            #     max_bias = fs_max - fs_min

            if use_prev_fs:
                prev_fs = jnp.reshape(self.common_bias.compute_from_cv(cv_grid)[0], fs.shape)
                fs -= np.array(prev_fs)

            if choice == "rbf":
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

                def get_b(fact):
                    eps = fs.shape[0] / (bounds[:, 1] - bounds[:, 0]) * fact

                    # 'cubic', 'thin_plate_spline', 'multiquadric', 'quintic', 'inverse_multiquadric', 'gaussian', 'inverse_quadratic', 'linear'

                    fesBias = RbfBias.create(
                        cvs=collective_variable,
                        vals=fslist,
                        cv=cv,
                        # kernel="linear",
                        kernel=rbf_kernel,
                        epsilon=eps,
                        # smoothing=sigmalist,
                        degree=rbf_degree,
                    )
                    return fesBias

                fesBias = get_b(1.0)

            elif choice == "gridbias":
                raise ValueError("choose choice='rbf' for the moment")

                fs[~mask] = 0.0
                fesBias = GridBias(cvs=collective_variable, vals=fs, bounds=bounds)
            else:
                raise ValueError

            if use_prev_fs:
                fes_bias_tot = CompositeBias.create(biases=[self.common_bias, fesBias])
            else:
                fes_bias_tot = fesBias

            if resample_bias:
                fes_bias_tot = fes_bias_tot.resample(
                    cv_grid=cv_grid,
                    n=n_max,
                )

            # clip value of bias to min and max of computed FES
            fes_bias_tot = BiasModify.create(
                bias=fes_bias_tot,
                fun=_clip,
                kwargs={"a_min": fs_min, "a_max": fs_max},
            )
        else:
            use_prev_fs = False

            print("estimating bias from koopman Theory!")

            dlo = self.rounds.data_loader(
                num=10,
                out=-1,
                lag_n=lag_n,
                cv_round=self.cv_round,
                start=start_r,
                new_r_cut=None,
                min_traj_length=min_traj_length,
                only_finished=only_finished,
                time_series=True,
            )

            # get weights based on koopman theory. the CVs are binned with indicators
            weights = dlo.weights(
                koopman=True,
                indicator_CV=True,
                n_max=n_max,
                add_1=True,
            )

            fes_bias_tot = dlo._get_fes_bias_from_weights(
                weights=weights,
                cv=dlo.cv,
                n_grid=n_max,
                T=dlo.sti.T,
                collective_variable=dlo.collective_variable,
            )

        if plot:
            fold = str(self.rounds.path(c=self.cv_round))

            pf = []

            if use_prev_fs:
                pf.append(
                    bash_app_python(function=Bias.static_plot)(
                        bias=fesBias,
                        outputs=[File(f"{fold}/diff_FES_bias_{self.rnd}_inverted_{choice}.png")],
                        execution_folder=fold,
                        name=f"diff_FES_bias_{self.rnd}_inverted_{choice}.png",
                        inverted=True,
                        label="Free Energy [kJ/mol]",
                        stdout=f"diff_FES_bias_{self.rnd}_inverted_{choice}.stdout",
                        stderr=f"diff_FES_bias_{self.rnd}_inverted_{choice}.stderr",
                        margin=margin,
                        vmax=vmax,
                        **plot_kwargs,
                    ),
                )

                pf.append(
                    bash_app_python(function=Bias.static_plot)(
                        bias=fesBias,
                        outputs=[File(f"{fold}/diff_FES_bias_{self.rnd}_{choice}.png")],
                        name=f"diff_FES_bias_{self.rnd}_{choice}.png",
                        execution_folder=fold,
                        stdout=f"diff_FES_bias_{self.rnd}_{choice}.stdout",
                        stderr=f"diff_FES_bias_{self.rnd}_{choice}.stderr",
                        margin=margin,
                        vmax=vmax,
                        **plot_kwargs,
                    ),
                )

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
                    **plot_kwargs,
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
                    **plot_kwargs,
                ),
            )

            for f in pf:
                f.result()

        return fes_bias_tot
