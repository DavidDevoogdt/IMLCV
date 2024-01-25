import itertools
from functools import partial

import jax.numpy as jnp
import numpy as np
from IMLCV.base.bias import Bias
from IMLCV.base.bias import CompositeBias
from IMLCV.base.CV import CollectiveVariable
from IMLCV.base.CV import CV
from IMLCV.base.rounds import Rounds
from IMLCV.configs.bash_app_python import bash_app_python
from IMLCV.implementations.bias import GridBias
from IMLCV.implementations.bias import RbfBias
from molmod.units import kjmol
from molmod.units import picosecond
from parsl import File
from thermolib.thermodynamics.bias import BiasPotential2D
from thermolib.thermodynamics.fep import FreeEnergyHypersurfaceND
from thermolib.thermodynamics.histogram import HistogramND

# from IMLCV.base.bias import plot_app


class ThermoLIB:
    """class to convert data and CVs to different thermodynamic/ kinetic
    observables."""

    time_per_bin = 2 * picosecond

    def __init__(
        self,
        rounds: Rounds,
        rnd=None,
        cv_round: int | None = None,
        cv: CollectiveVariable | None = None,
    ) -> None:
        self.rounds = rounds

        if cv_round is None:
            self.cv_round = rounds.cv
        else:
            self.cv_round = cv_round

        if rnd is None:
            rnd = rounds.get_round(c=self.cv_round)

        self.rnd = rnd
        self.common_bias = self.rounds.get_bias(c=cv_round, r=self.rnd)

        if cv is None:
            b = self.rounds.get_bias(c=self.cv_round, r=self.rnd)

            self.collective_variable = b.collective_variable
        else:
            self.collective_variable = cv

    @staticmethod
    def _get_histos(
        bins,
        temp,
        trajs,
        biases: list[Bias],
        chunk_size=None,
        pmap=True,
    ):
        from IMLCV.base.CV import padded_pmap
        from IMLCV.base.bias import Bias
        import jax

        class _ThermoBiasND(BiasPotential2D):
            def __init__(self, bias: Bias, chunk_size=None, num=None) -> None:
                self.bias = bias
                self.chunk_size = chunk_size
                self.num = num

                super().__init__(f"IMLCV_bias_{num}")

            def __call__(self, *cv):
                print(".", end="")

                colvar = CV.combine(
                    *[
                        CV(
                            cv=jnp.array(
                                cvi.reshape(
                                    (-1, 1),
                                ),
                                dtype=jnp.float64,
                            ),
                        )
                        for cvi in cv
                    ],
                )

                def _get_bias(cvs):
                    out, _ = self.bias.compute_from_cv(
                        cvs=cvs,
                        diff=False,
                        chunk_size=chunk_size,
                    )
                    return out

                if pmap:
                    _get_bias = padded_pmap(_get_bias)

                out = _get_bias(colvar)

                return np.array(jnp.reshape(out, cv[0].shape), dtype=np.double)

            def print_pars(self, *pars_units):
                pass

        bias_wrapped = [_ThermoBiasND(bias=b, chunk_size=chunk_size, num=i) for i, b in enumerate(biases)]

        histo = HistogramND.from_wham(
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

        return histo

    def fes_nd_thermolib(
        self,
        plot=True,
        num_rnds=4,
        start_r=0,
        update_bounding_box=True,
        samples_per_bin=200,
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
    ):
        if temp is None:
            temp = self.rounds.T

        if directory is None:
            directory = self.rounds.path(c=self.cv_round, r=self.rnd)

        if dlo is None:
            dlo = self.rounds.data_loader(
                num=num_rnds,
                cv_round=self.cv_round,
                start=start_r,
                split_data=True,
                new_r_cut=None,
                min_traj_length=min_traj_length,
                chunk_size=chunk_size,
                get_bias_list=True,
                only_finished=only_finished,
            )

        trajs = dlo.cv
        biases = dlo.bias

        if plot:
            trajs_plot = self.rounds.data_loader(
                num=1,
                ignore_invalid=False,
                cv_round=self.cv_round,
                split_data=True,
                new_r_cut=None,
                min_traj_length=min_traj_length,
                only_finished=only_finished,
            ).cv

            bash_app_python(self.common_bias.plot, executors=["default"])(
                outputs=[File(f"{directory}/combined.png")],  # png because heavy file
                name="combined.png",
                execution_folder=directory,
                stdout="combined.stdout",
                stderr="combined.stderr",
                map=False,
                traj=trajs_plot,
                margin=margin,
            )

        c = CV.stack(*trajs)

        if update_bounding_box:
            bounding_box = [[c.cv[:, i].min(), c.cv[:, i].max()] for i in range(c.dim)]
        else:
            bounding_box = self.collective_variable.metric.bounding_box

        if n is None:
            n = int((c.batch_dim / samples_per_bin) ** (1 / c.dim))

        assert n >= 4, "sample more points"

        if n > n_max:
            print(f"truncating number of bins {n=} to {n_max=}")
            n = n_max

        bins = [np.linspace(mini, maxi, n, endpoint=True, dtype=np.double) for mini, maxi in bounding_box]

        histo = bash_app_python(ThermoLIB._get_histos, executors=["default"])(
            bins=bins,
            temp=temp,
            trajs=trajs,
            biases=biases,
            chunk_size=chunk_size,
            execution_folder=directory,
            pmap=pmap,
        ).result()

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

    def new_metric(self, plot=False, r=None):
        assert isinstance(self.rounds, Rounds)

        raise NotImplementedError

    def fes_bias(
        self,
        plot=True,
        max_bias=None,
        fes=None,
        choice="rbf",
        num_rnds=4,
        start_r=0,
        rbf_kernel="thin_plate_spline",
        rbf_degree=None,
        smoothing_threshold=5 * kjmol,
        samples_per_bin=200,
        chunk_size=None,
        resample_bias=True,
        update_bounding_box=True,  # make boudning box bigger for FES calculation
        n_max=60,
        min_traj_length=None,
        margin=0.5,
        grid=None,
        bounds=None,
        use_prev_fs=True,
        collective_variable=None,
        only_finished=True,
        **plot_kwargs,
    ):
        if fes is None:
            fes, grid, bounds = self.fes_nd_thermolib(
                plot=plot,
                start_r=start_r,
                samples_per_bin=samples_per_bin,
                num_rnds=num_rnds,
                chunk_size=chunk_size,
                update_bounding_box=update_bounding_box,
                n_max=n_max,
                min_traj_length=min_traj_length,
                margin=margin,
                only_finished=only_finished,
            )

        # fes is in 'xy'- indexing convention, convert to ij
        fs = np.transpose(fes.fs)

        # remove previous fs
        cv_grid = CV.stack(*list(zip(*grid))[1])

        if collective_variable is None:
            collective_variable = self.collective_variable

        if use_prev_fs:
            prev_fs = jnp.reshape(self.common_bias.compute_from_cv(cv_grid)[0], fs.shape)
            fs += np.array(prev_fs)

        # invert to use as bias
        mask = ~np.isnan(fs)
        fs[:] = -fs[:] + fs[mask].max()

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
            fes_bias_tot = fes_bias_tot.resample(cv_grid=cv_grid)

        if plot:
            fold = str(self.rounds.path(c=self.cv_round))

            pf = []

            if use_prev_fs:
                pf.append(
                    bash_app_python(fesBias.plot, executors=["default"])(
                        outputs=[File(f"{fold}/diff_FES_bias_{self.rnd}_inverted_{choice}.pdf")],
                        execution_folder=fold,
                        name=f"diff_FES_bias_{self.rnd}_inverted_{choice}.pdf",
                        inverted=True,
                        label="Free Energy [kJ/mol]",
                        stdout=f"diff_FES_bias_{self.rnd}_inverted_{choice}.stdout",
                        stderr=f"diff_FES_bias_{self.rnd}_inverted_{choice}.stderr",
                        margin=margin,
                        **plot_kwargs,
                    ),
                )

                pf.append(
                    bash_app_python(fesBias.plot, executors=["default"])(
                        outputs=[File(f"{fold}/diff_FES_bias_{self.rnd}_{choice}.pdf")],
                        name=f"diff_FES_bias_{self.rnd}_{choice}.pdf",
                        execution_folder=fold,
                        stdout=f"diff_FES_bias_{self.rnd}_{choice}.stdout",
                        stderr=f"diff_FES_bias_{self.rnd}_{choice}.stderr",
                        margin=margin,
                        **plot_kwargs,
                    ),
                )

            pf.append(
                bash_app_python(fes_bias_tot.plot, executors=["default"])(
                    outputs=[File(f"{fold}/FES_bias_{self.rnd}_inverted_{choice}.pdf")],
                    execution_folder=fold,
                    name=f"FES_bias_{self.rnd}_inverted_{choice}.pdf",
                    inverted=True,
                    label="Free Energy [kJ/mol]",
                    stdout=f"FES_bias_{self.rnd}_inverted_{choice}.stdout",
                    stderr=f"FES_bias_{self.rnd}_inverted_{choice}.stderr",
                    margin=margin,
                    **plot_kwargs,
                ),
            )

            pf.append(
                bash_app_python(fes_bias_tot.plot, executors=["default"])(
                    outputs=[File(f"{fold}/FES_bias_{self.rnd}_{choice}.pdf")],
                    execution_folder=fold,
                    name=f"FES_bias_{self.rnd}_{choice}.pdf",
                    stdout=f"FES_bias_{self.rnd}_{choice}.stdout",
                    stderr=f"FES_bias_{self.rnd}_{choice}.stderr",
                    margin=margin,
                    **plot_kwargs,
                ),
            )

            for f in pf:
                f.result()

        return fes_bias_tot
