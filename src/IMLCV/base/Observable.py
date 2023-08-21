import itertools
from functools import partial

import jax.numpy as jnp
import numpy as np
from IMLCV.base.bias import Bias
from IMLCV.base.bias import CompositeBias
from IMLCV.base.bias import plot_app
from IMLCV.base.CV import CollectiveVariable
from IMLCV.base.CV import CV
from IMLCV.base.rounds import Rounds
from IMLCV.configs.bash_app_python import bash_app_python
from IMLCV.implementations.bias import GridBias
from IMLCV.implementations.bias import RbfBias
from jax import jit
from molmod.units import kjmol
from molmod.units import picosecond
from parsl import File
from thermolib.thermodynamics.bias import BiasPotential2D
from thermolib.thermodynamics.fep import FreeEnergyHypersurfaceND
from thermolib.thermodynamics.histogram import HistogramND


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
            self.collective_variable = self.rounds.get_bias(c=self.cv_round, r=self.rnd).collective_variable
        else:
            self.collective_variable = cv

    def fes_nd_thermolib(
        self,
        plot=True,
        num_rnds=4,
        start_r=0,
        update_bounding_box=False,
        samples_per_bin=500,
        chunk_size=None,
    ):
        temp = self.rounds.T

        directory = self.rounds.path(c=self.cv_round, r=self.rnd)

        rnd = self.rnd

        trajs = []
        trajs_plot = []
        biases = []

        # TODO: change to rounds data loader
        for round, trajectory in self.rounds.iter(
            start=start_r,
            num=num_rnds,
            stop=self.rnd,
            c=self.cv_round,
        ):
            ti = trajectory.ti[trajectory.ti._t > round.tic.equilibration]

            if ti._cv is not None:
                cvs = ti.CV

            else:
                sp = ti.sp
                nl = sp.get_neighbour_list(
                    r_cut=round.tic.r_cut,
                    z_array=round.tic.atomic_numbers,
                )
                cvs, _ = self.collective_variable.compute_cv(sp=sp, nl=nl)

            if cvs.batch_dim <= 1:
                print("##############bdim {cvs.batch_dim} ignored\n")
                continue

            trajs.append(cvs)
            biases.append(File(str(self.rounds.path() / trajectory.name_bias)))
            if plot:
                if round.round == rnd:
                    trajs_plot.append(cvs)

        if plot:
            plot_app(
                bias=self.common_bias,
                outputs=[File(f"{directory}/combined.png")],  # png because heavy file
                execution_folder=directory,
                stdout="combined.stdout",
                stderr="combined.stderr",
                map=False,
                traj=trajs_plot,
            )

        # todo: take actual bounds instead of calculated bounds
        if update_bounding_box:
            bb = None
        else:
            bb = self.collective_variable.metric.bounding_box

        bounds, bins = self._FES_mg(
            cvs=trajs,
            bounding_box=bb,
            n=None,
            samples_per_bin=samples_per_bin,
        )

        @bash_app_python(executors=["default"])
        def get_histos(
            bins,
            temp,
            trajs,
            inputs=[],
            outputs=[],
        ):
            class ThermoBiasND(BiasPotential2D):
                def __init__(self, bias: Bias) -> None:
                    self.bias = bias

                    super().__init__("IMLCV_bias")

                def __call__(self, *cv):
                    shape = cv[0].shape

                    colvar = CV.combine(*[CV(cv=cvi.reshape((-1, 1))) for cvi in cv])
                    out, _ = self.bias.compute_from_cv(
                        cvs=colvar,
                        diff=False,
                        chunk_size=chunk_size,
                    )

                    # map meshgrids of cvs to IMLCV biases and evaluate
                    return np.array(jnp.reshape(out, shape), dtype=np.double)

                def print_pars(self, *pars_units):
                    pass

            biases = [ThermoBiasND(Bias.load(b.filepath)) for b in inputs]

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
                biasses=biases,
                temp=temp,
                verbosity="high",
            )

            return histo

        histo = get_histos(
            bins=bins,
            temp=temp,
            trajs=trajs,
            inputs=biases,
            execution_folder=directory,
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

        return fes, grid, bounds

    def new_metric(self, plot=False, r=None):
        assert isinstance(self.rounds, Rounds)

        # trans = []
        # cvs = None

        raise NotImplementedError

        # for run_data in self.rounds.iter(num=1, stop=r):
        #     bias = Bias.load(run_data["attr"]["name_bias"])
        #     if cvs is None:
        #         cvs = bias.collective_variable

        #     monitor = find_monitor(bias)
        #     assert monitor is not None

        #     trans.append(monitor.transitions)

        #     # this data should not be used anymore
        # self.rounds.invalidate_data(r=r)

        # transitions = jnp.vstack(trans)
        # if plot:
        #     fn = f"{self.folder}/round_{self.rnd}/"
        # else:
        #     fn = None

        # return cvs.metric.update_metric(transitions, fn=fn)

    def _FES_mg(self, cvs: list[CV], bounding_box, samples_per_bin=500, n=None):
        c = CV.stack(*cvs)

        if n is None:
            n = int((c.batch_dim / samples_per_bin) ** (1 / c.dim))

        assert n >= 4, "sample more points"

        if bounding_box is None:
            bounds = [[c.cv[:, i].min(), c.cv[:, i].max()] for i in range(c.dim)]
        else:
            bounds = bounding_box

        bins = [np.linspace(mini, maxi, n, endpoint=True, dtype=np.double) for mini, maxi in bounds]

        return bounds, bins

    def fes_bias(
        self,
        plot=True,
        max_bias=None,
        fs=None,
        choice="rbf",
        num_rnds=4,
        start_r=0,
        rbf_kernel="thin_plate_spline",
        rbf_degree=None,
        smoothing_threshold=5 * kjmol,
        samples_per_bin=100,
        chunk_size=None,
        **plot_kwargs,
    ):
        if fs is None:
            fes, grid, bounds = self.fes_nd_thermolib(
                plot=plot,
                start_r=start_r,
                samples_per_bin=samples_per_bin,
                num_rnds=num_rnds,
                chunk_size=chunk_size,
            )

        # fes is in 'xy'- indexing convention, convert to ij
        fs = np.transpose(fes.fs)

        # remove previous fs
        cv_grid = CV.stack(*list(zip(*grid))[1])
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

                fesBias = RbfBias(
                    cvs=self.collective_variable,
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
            fs[~mask] = 0.0
            fesBias = GridBias(cvs=self.collective_variable, vals=fs, bounds=bounds)
        else:
            raise ValueError

        fes_bias_tot = CompositeBias(biases=[self.common_bias, fesBias])

        if plot:
            fold = str(self.rounds.path(c=self.cv_round))

            pf = []

            pf.append(
                plot_app(
                    bias=fesBias,
                    outputs=[File(f"{fold}/diff_FES_bias_{self.rnd}_inverted_{choice}.pdf")],
                    inverted=True,
                    execution_folder=fold,
                    stdout=f"diff_FES_bias_{self.rnd}_inverted_{choice}.stdout",
                    stderr=f"diff_FES_bias_{self.rnd}_inverted_{choice}.stderr",
                    **plot_kwargs,
                ),
            )

            pf.append(
                plot_app(
                    bias=fesBias,
                    outputs=[File(f"{fold}/diff_FES_bias_{self.rnd}_{choice}.pdf")],
                    execution_folder=fold,
                    stdout=f"diff_FES_bias_{self.rnd}_{choice}.stdout",
                    stderr=f"diff_FES_bias_{self.rnd}_{choice}.stderr",
                    **plot_kwargs,
                ),
            )

            pf.append(
                plot_app(
                    bias=fes_bias_tot,
                    outputs=[File(f"{fold}/FES_bias_{self.rnd}_inverted_{choice}.pdf")],
                    inverted=True,
                    execution_folder=fold,
                    stdout=f"FES_bias_{self.rnd}_inverted_{choice}.stdout",
                    stderr=f"FES_bias_{self.rnd}_inverted_{choice}.stderr",
                    **plot_kwargs,
                ),
            )

            pf.append(
                plot_app(
                    bias=fes_bias_tot,
                    outputs=[File(f"{fold}/FES_bias_{self.rnd}_{choice}.pdf")],
                    execution_folder=fold,
                    stdout=f"FES_bias_{self.rnd}_{choice}.stdout",
                    stderr=f"FES_bias_{self.rnd}_{choice}.stderr",
                    **plot_kwargs,
                ),
            )

            for f in pf:
                f.result()

        return fes_bias_tot
