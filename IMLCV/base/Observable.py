from __future__ import annotations

import itertools
from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import jit
from molmod.units import kjmol, picosecond
from parsl import File

from IMLCV.base.bias import Bias, CompositeBias, CvMonitor, GridBias, RbfBias, plot_app
from IMLCV.base.CV import CV
from IMLCV.base.rounds import RoundsMd
from IMLCV.external.parsl_conf.bash_app_python import bash_app_python
from thermolib.thermodynamics.bias import BiasPotential2D
from thermolib.thermodynamics.fep import FreeEnergyHypersurfaceND
from thermolib.thermodynamics.histogram import HistogramND


class Observable:
    """class to convert data and CVs to different thermodynamic/ kinetic
    observables."""

    samples_per_bin = 400
    time_per_bin = 2 * picosecond

    def __init__(self, rounds: RoundsMd) -> None:
        self.rounds = rounds

        self.cvs = self.rounds.get_bias().collective_variable

        self.folder = rounds.folder

    def _fes_2d(self, plot=True, n=8, start_r=0):
        # fes = FreeEnergySurface2D.from_txt

        temp = self.rounds.T

        common_bias = self.rounds.get_bias()
        directory = f"{self.folder}/round_{self.rounds.round}"

        trajs = []
        trajs_mapped = []
        biases = []

        time = 0
        cv = None

        for round, trajectory in self.rounds.iter(start=start_r, num=4):

            bias = trajectory.get_bias()

            if cv is None:
                cv = bias.collective_variable
            sp = trajectory.ti.sp[trajectory.ti.t > round.tic.equilibration]

            cvs, _ = cv.compute_cv(sp)
            trajs.append(cvs)
            biases.append(Observable._ThermoBias2D(bias))

        if plot:
            plot_app(
                bias=common_bias,
                outputs=[File(f"{directory}/combined.pdf")],
                map=False,
                traj=trajs,
                # stdout=f"{directory}/combined.stdout",
                # stderr=f"{directory}/combined.stderr",
            )

        trajs_mapped = [
            np.array(
                traj.cv,
                dtype=np.double,
            )
            for traj in trajs
        ]

        # todo: take actual bounds instead of calculated bounds
        bounds, bins = self._FES_mg(cvs=trajs, bounding_box=cv.metric.bounding_box, n=n)

        histo = bash_app_python(HistogramND.from_wham)(
            bins=bins,
            # pinit=pinit,
            trajectories=trajs_mapped,
            error_estimate="mle_f",
            biasses=biases,
            temp=temp,
            # verbosity="high",
            stdout=f"{directory}/histo.stdout",
            stderr=f"{directory}/histo.stderr",
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

        for run_data in self.rounds.iter(num=1, stop=r):
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

    def _FES_mg(self, cvs: list[CV], bounding_box, n=None):

        if n is None:
            n = 0
            for t in cvs:
                n += t.cv.size

            # 20 points per bin on average
            n = int((n / self.samples_per_bin) ** (1 / cvs[0].cv.ndim))

        # if time is not None:
        #     bins_max = int((time/self.time_per_bin)**(1 / trajs[0].ndim))
        #     if bins_max > n:
        #         n = bins_max

        assert n >= 4, "sample more points"

        c = CV.stack(*cvs)
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

    def fes_bias(
        self,
        plot=False,
        max_bias=None,
        fs=None,
        choice="rbf",
        n=None,
        start_r=0,
        rbf_kernel="gaussian",
        rbf_degree=-1,
        smoothing_threshold=5 * kjmol,
        **plot_kwargs,
    ):
        if fs is None:
            fes, grid, bounds = self._fes_2d(plot=plot, start_r=start_r)

        # fes is in 'xy'- indexing convention, convert to ij
        fs = np.transpose(fes.fs)

        mask = ~jnp.isnan(fs)

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

        # for choice in [
        #     "gridbias",
        #     "rbf",
        # ]:

        if choice == "rbf":

            min_err = jnp.inf
            min_eps = None

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
                        f"{self.folder}/FES_thermolib_{self.rounds.round}_inverted_{choice}.pdf"
                    )
                ],
                inverted=True,
                **plot_kwargs
                # margin=1,
                # stdout=f"{self.folder}/FES_thermolib_{self.rounds.round}_inverted_{choice}.stdout",
                # stderr=f"{self.folder}/FES_thermolib_{self.rounds.round}_inverted_{choice}.stderr",
            )

            plot_app(
                bias=fesBias,
                outputs=[
                    File(f"{self.folder}/FES_bias_{self.rounds.round}_{choice}.pdf")
                ],
                **plot_kwargs
                # margin=1.0,
            )

        return fesBias
