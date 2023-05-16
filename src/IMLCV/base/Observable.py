import itertools
from functools import partial

import jax.numpy as jnp
import numpy as np
from IMLCV.base.bias import Bias
from IMLCV.base.bias import plot_app
from IMLCV.base.CV import CV
from IMLCV.base.rounds import Rounds
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

    def __init__(self, rounds: Rounds, rnd=None) -> None:
        self.rounds = rounds
        if rnd is None:
            rnd = rounds.round
        self.rnd = rnd
        self.common_bias = self.rounds.get_bias(r=self.rnd)
        self.cvs = self.rounds.get_bias(r=self.rnd).collective_variable
        self.folder = rounds.folder

    def fes_nd_thermolib(
        self,
        plot=True,
        n=None,
        start_r=0,
        update_bounding_box=False,
        samples_per_bin=500,
    ):
        class ThermoBiasND(BiasPotential2D):
            def __init__(self, bias: Bias) -> None:
                self.bias = bias

                super().__init__("IMLCV_bias")

            def __call__(self, *cv):
                # map meshgrids of cvs to IMLCV biases and evaluate
                return np.array(self.f_par(jnp.array([*cv])), dtype=np.double)

            @partial(jit, static_argnums=(0,))
            def f_par(self, cvs):
                b, _ = jnp.apply_along_axis(
                    self.f,
                    axis=0,
                    arr=cvs,
                )
                return b

            @partial(jit, static_argnums=(0,))
            def f(self, point):
                return self.bias.compute_from_cv(
                    cvs=CV(cv=point),
                    diff=False,
                )

            def print_pars(self, *pars_units):
                pass

        temp = self.rounds.T

        directory = f"{self.folder}/round_{self.rnd}"

        rnd = self.rnd

        trajs = []
        trajs_plot = []
        biases = []

        cv = None

        for round, trajectory in self.rounds.iter(start=start_r, num=4, stop=self.rnd):
            bias = trajectory.get_bias()

            if cv is None:
                cv = bias.collective_variable

            ti = trajectory.ti[trajectory.ti._t > round.tic.equilibration]

            if ti._cv is not None:
                cvs = ti.CV

            else:
                sp = ti.sp
                nl = sp.get_neighbour_list(
                    r_cut=round.tic.r_cut,
                    z_array=round.tic.atomic_numbers,
                )
                cvs, _ = cv.compute_cv(sp=sp, nl=nl)

            if cvs.batch_dim <= 1:
                print("##############bdim {cvs.batch_dim} ignored\n")
                continue

            trajs.append(cvs)
            biases.append(ThermoBiasND(bias))
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
            bb = self.cvs.metric.bounding_box

        bounds, bins = self._FES_mg(
            cvs=trajs,
            bounding_box=bb,
            n=n,
            samples_per_bin=samples_per_bin,
        )

        # histo = bash_app_python(HistogramND.from_wham, executors=["model"])(
        histo = HistogramND.from_wham(
            bins=bins,
            trajectories=[
                np.array(
                    traj.cv,
                    dtype=np.double,
                )
                for traj in trajs
            ],
            error_estimate="mle_f",
            biasses=biases,
            temp=temp,
            verbosity="high",
            # execution_folder=directory,
        )  # .result()

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
        choice="gridbias",
        n=None,
        start_r=0,
        rbf_kernel="thin_plate_spline",
        rbf_degree=-1,
        smoothing_threshold=5 * kjmol,
        samples_per_bin=500,
        **plot_kwargs,
    ):
        if fs is None:
            fes, grid, bounds = self.fes_nd_thermolib(
                plot=plot,
                start_r=start_r,
                samples_per_bin=samples_per_bin,
            )

        # fes is in 'xy'- indexing convention, convert to ij
        fs = np.transpose(fes.fs)

        mask = ~np.isnan(fs)

        # invert to use as bias
        if max_bias is not None:
            fs[:] = -fs[:] + np.min([max_bias, fs[mask].max()])
        else:
            fs[:] = -fs[:] + fs[mask].max()

        fs[~mask] = 0.0

        fl = fes.flower.T
        fu = fes.fupper.T

        sigma = fu - fl
        sigma = (sigma) / smoothing_threshold

        if choice == "rbf":
            fslist = []
            smoothing_list = []
            cv: list[CV] = []

            for idx, cvi in grid:
                if not np.isnan(fs[idx]):
                    fslist.append(fs[idx])

                    cv += [cvi]

                    smoothing_list.append(sigma[idx])
            cv = CV.stack(*cv)

            fslist = jnp.array(fslist)
            bounds = jnp.array(bounds)

            def get_b(fact):
                eps = n / (bounds[:, 1] - bounds[:, 0]) * fact

                # 'cubic', 'thin_plate_spline', 'multiquadric', 'quintic', 'inverse_multiquadric', 'gaussian', 'inverse_quadratic', 'linear'

                fesBias = RbfBias(
                    cvs=self.cvs,
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
            fesBias = GridBias(cvs=self.cvs, vals=fs, bounds=bounds)

        else:
            raise ValueError

        if plot:
            plot_app(
                bias=fesBias,
                outputs=[
                    File(
                        f"{self.folder}/FES_thermolib_{self.rnd}_inverted_{choice}.pdf",
                    ),
                ],
                inverted=True,
                execution_folder=self.folder,
                stdout=f"FES_thermolib_{self.rnd}_inverted_{choice}.stdout",
                stderr=f"FES_thermolib_{self.rnd}_inverted_{choice}.stderr",
                **plot_kwargs,
            )

            plot_app(
                bias=fesBias,
                outputs=[File(f"{self.folder}/FES_bias_{self.rnd}_{choice}.pdf")],
                execution_folder=self.folder,
                stdout=f"FES_bias_{self.rnd}_{choice}.stdout",
                stderr=f"FES_bias_{self.rnd}_{choice}.stderr",
                **plot_kwargs,
            )

        return fesBias
