from __future__ import annotations

import itertools

import jax.numpy as jnp
import numpy as np
from jax import jit
from molmod.units import picosecond
from parsl import File

from IMLCV.base.bias import Bias, CompositeBias, CvMonitor, GridBias, RbfBias, plot_app
from IMLCV.base.CV import CV
from IMLCV.base.rounds import RoundsCV, RoundsMd
from thermolib.thermodynamics.bias import BiasPotential2D
from thermolib.thermodynamics.fep import FreeEnergyHypersurfaceND
from thermolib.thermodynamics.histogram import Histogram2D, HistogramND


class Observable:
    """class to convert data and CVs to different thermodynamic/ kinetic
    observables."""

    samples_per_bin = 40

    time_per_bin = 2 * picosecond

    def __init__(self, rounds: RoundsMd) -> None:
        self.rounds = rounds

        self.cvs = self.rounds.get_bias().collective_variable

        self.folder = rounds.folder

    def _fes_2d(self, plot=True, update_bounds=True):
        # fes = FreeEnergySurface2D.from_txt

        temp = self.rounds.T

        common_bias = self.rounds.get_bias()
        directory = f"{self.folder}/round_{self.rounds.round}"

        if isinstance(self.rounds, RoundsMd):

            trajs = []
            trajs_mapped = []
            biases = []

            time = 0
            cv = None

            for round, trajectory in self.rounds.iter(num=2):

                bias = trajectory.get_bias()

                if cv is None:
                    cv = bias.collective_variable
                sp = trajectory.ti.sp[trajectory.ti.t > round.tic.equilibration]

                cvs, _ = cv.compute_cv(sp)

                # arr = np.array(
                #     cvs.cv,
                #     dtype=np.double,
                # )

                trajs.append(cvs)

                biases.append(Observable._ThermoBias2D(bias))

            if plot:
                plot_app(
                    bias=common_bias,
                    outputs=[File(f"{directory}/combined_unmapped.pdf")],
                    map=False,
                    traj=trajs,
                    stdout=f"{directory}/combined_unmapped.stdout",
                    stderr=f"{directory}/combined_unmapped.stderr",
                )

                # plot_app(
                #     bias=common_bias,
                #     outputs=[File(f"{directory}/combined.pdf")],
                #     traj=trajs_mapped,
                # )

            trajs_mapped = [
                np.array(
                    traj.cv,
                    dtype=np.double,
                )
                for traj in trajs
            ]

            # todo: take actual bounds instead of calculated bounds
            bounds, bins = self._FES_mg(
                trajs=trajs, bounding_box=cv.metric.bounding_box
            )

            histo = HistogramND.from_wham(
                bins=bins,
                # pinit=pinit,
                trajectories=trajs_mapped,
                error_estimate="mle_f",
                biasses=biases,
                temp=temp,
                verbosity="high",
            )

        elif isinstance(self.rounds, RoundsCV):
            raise NotImplementedError("check this")
            trajs = []
            for dictionary in self.rounds.iter(num=np.Inf):
                pos = dictionary["positions"][:]

                if "cell" in dictionary:
                    cell = dictionary["cell"][:]
                    arr = np.array(
                        [
                            self.cvs.compute_cv(coordinates=x, cell=y)[0]
                            for (x, y) in zip(pos, cell)
                        ],
                        dtype=np.double,
                    )
                else:
                    arr = np.array(
                        [self.cvs.compute_cv(coordinates=p, cell=None)[0] for p in pos],
                        dtype=np.double,
                    )

                trajs.append(arr)

            bounds, bins = self._FES_mg(trajs=trajs, n=10)
            data = np.vstack(trajs)
            histo = Histogram2D.from_single_trajectory(
                data,
                bins,
                error_estimate="mle_f",
            )
        else:
            raise NotImplementedError

        fes = FreeEnergyHypersurfaceND.from_histogram(histo, temp)
        fes.set_ref()

        # construc list with centers CVs
        bin_centers = [0.5 * (x[:-1] + x[1:]) for x in bins]
        Ngrid = np.array([len(bi) for bi in bin_centers])
        grid = []
        for idx in itertools.product(*(range(x) for x in Ngrid)):
            center = [bin_centers[j][k] for j, k in enumerate(idx)]
            grid.append((idx, CV(cv=jnp.array(center))))

        return fes, grid, bounds

    def new_metric(self, plot=False, r=None):
        assert isinstance(self.rounds, RoundsMd)

        trans = []
        cvs = None

        def find_monitor(bias):

            if isinstance(bias, CvMonitor):
                return bias

            if isinstance(bias, CompositeBias):
                for b in bias.biases:
                    ret = find_monitor(b)
                    if ret is not None:
                        return ret

            return None

        for run_data in self.rounds.iter(num=1, r=r):
            bias = Bias.load(run_data["attr"]["name_bias"])
            if cvs is None:
                cvs = bias.collective_variable

            monitor = find_monitor(bias)
            assert monitor is not None

            trans.append(monitor.transitions)

            # this data should not be used anymore
        self.rounds.invalidate_data(r=r)

        transitions = jnp.vstack(trans)
        if plot:
            fn = f"{self.folder}/round_{self.rounds.round}/"
        else:
            fn = None

        return cvs.metric.update_metric(transitions, fn=fn)

    def _FES_mg(self, trajs: list[CV], bounding_box, n=None):

        if n is None:
            n = 0
            for t in trajs:
                n += t.cv.size

            # 20 points per bin on average
            n = int(n ** (1 / trajs[0].cv.ndim) / self.samples_per_bin)

        # if time is not None:
        #     bins_max = int((time/self.time_per_bin)**(1 / trajs[0].ndim))
        #     if bins_max > n:
        #         n = bins_max

        assert n >= 4, "sample more points"

        c = trajs[0]
        for t in trajs[1:]:
            c += t

        a = c.cv

        bounds = [[a[:, i].min(), a[:, i].max()] for i in range(a.shape[1])]
        bins = [
            np.linspace(mini, maxi, n, endpoint=True, dtype=np.double)
            for mini, maxi in bounds
        ]

        return bounds, bins

    class _ThermoBias2D(BiasPotential2D):
        def __init__(self, bias: Bias) -> None:
            self.bias = bias

            super().__init__("IMLCV_bias")

        def __call__(self, *cv):
            # CVs are already in mapped space

            cvs = jnp.array([*cv])

            @jit
            def f(point):
                return self.bias.compute_from_cv(
                    cvs=CV(cv=point),
                    diff=False,
                )

            b, _ = jnp.apply_along_axis(
                f,
                axis=0,
                arr=cvs,
            )

            b = np.array(b, dtype=np.double)
            # b[np.isnan(b)] = 0

            return b

        def print_pars(self, *pars_units):
            pass

    def fes_bias(
        self, kind="normal", plot=False, max_bias=None, fs=None, update_bounds=True
    ):
        if fs is None:
            fes, grid, bounds = self._fes_2d(plot=plot, update_bounds=True)

            if kind == "normal":
                fs = fes.fs
            elif kind == "fupper":
                fs = fes.fupper
            elif kind == "flower":
                fs = fes.flower
            else:
                raise ValueError

        # fes is in 'xy'- indexing convention, convert to ij
        fs = np.transpose(fs)

        # invert to use as bias
        if max_bias is not None:
            fs[:] = -fs[:] + np.min([max_bias, fs[~np.isnan(fs)].max()])
        else:
            fs[:] = -fs[:] + fs[~np.isnan(fs)].max()

        for choice in ["gridbias", "rbf"]:

            if choice == "rbf":

                fslist = []
                smoothing_list = []
                cv = None

                for idx, cvi in grid:

                    if not np.isnan(fs[idx]):
                        if cv is None:
                            cv = cvi
                        else:
                            cv += cvi
                        fslist.append(fs[idx])
                        smoothing_list.append(fes.fupper.T[idx] - fes.flower.T[idx])

                    # else:
                    #     fslist.append(0.0)

                fslist = jnp.array(fslist)
                sigmalist = jnp.array(smoothing_list)

                fesBias = RbfBias(
                    cvs=self.cvs,
                    vals=fslist,
                    cv=cv,
                    kernel="thin_plate_spline",
                    epsilon=1.0,
                    # smoothing=sigmalist,
                    degree=None,
                )
            elif choice == "gridbias":
                fesBias = GridBias(cvs=self.cvs, vals=fs, bounds=bounds)
            else:
                raise ValueError

            if plot:
                plot_app(
                    bias=fesBias,
                    outputs=[
                        File(
                            f"{self.folder}/FES_thermolib_{self.rounds.round}_inverted_{choice}.pdf"
                        )
                    ],
                    inverted=True,
                    stdout=f"{self.folder}/FES_thermolib_{self.rounds.round}_inverted_{choice}.stdout",
                    stderr=f"{self.folder}/FES_thermolib_{self.rounds.round}_inverted_{choice}.stderr",
                )

                plot_app(
                    bias=fesBias,
                    outputs=[
                        File(f"{self.folder}/FES_bias_{self.rounds.round}_{choice}.pdf")
                    ],
                )

        return fesBias
