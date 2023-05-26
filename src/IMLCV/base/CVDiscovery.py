import itertools
from pathlib import Path
from typing import Iterator

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from hsluv import hsluv_to_rgb
from IMLCV.base.CV import CollectiveVariable
from IMLCV.base.CV import CV
from IMLCV.base.CV import CvFlow
from IMLCV.base.CV import CvMetric
from IMLCV.base.CV import CvTrans
from IMLCV.base.CV import NeighbourList
from IMLCV.base.CV import SystemParams
from IMLCV.base.MdEngine import StaticMdInfo
from IMLCV.base.rounds import Rounds
from IMLCV.implementations.CV import scale_cv_trans
from jax import Array
from jax import jacrev
from jax import vmap
from jax.random import choice
from jax.random import PRNGKey
from jax.random import split
from matplotlib import gridspec
from matplotlib.figure import Figure


class Transformer:
    def __init__(
        self,
        outdim,
        periodicity=None,
        bounding_box=None,
        descriptor=CvFlow,
    ) -> None:
        self.outdim = outdim

        if periodicity is None:
            periodicity = [False for _ in range(self.outdim)]

        self.periodicity = periodicity

        self.descriptor = descriptor

    def pre_fit(
        self,
        z: list[SystemParams],
        nl: list[NeighbourList] | None,
        chunk_size=None,
        scale=True,
    ) -> tuple[list[CV], CvFlow]:
        f = self.descriptor

        x: list[CV] = []

        for i, zi in enumerate(z):
            nli = nl[i] if nl is not None else None
            x.append(f.compute_cv_flow(zi, nli, chunk_size=chunk_size))

        if scale:
            g = scale_cv_trans(CV.stack(*x))
            x = [g.compute_cv_trans(xi)[0] for xi in x]

            f = f * g

        return x, f

    def fit(
        self,
        sp_list: list[SystemParams],
        nl_list: list[NeighbourList] | None,
        chunk_size=None,
        prescale=True,
        postscale=True,
        jac=jacrev,
        **fit_kwargs,
    ) -> tuple[CV, CollectiveVariable]:
        print("starting pre_fit")

        x, f = self.pre_fit(
            sp_list,
            nl_list,
            scale=prescale,
            chunk_size=chunk_size,
        )

        print("starting fit")
        y, g = self._fit(
            x,
            nl_list,
            chunk_size=chunk_size,
            **fit_kwargs,
        )

        print("starting post_fit")
        z, h = self.post_fit(y, scale=postscale)

        cv = CollectiveVariable(
            f=f * g * h,
            metric=CvMetric(
                periodicities=self.periodicity,
                bounding_box=jnp.vstack([jnp.min(z.cv, axis=0), jnp.max(z.cv, axis=0)]).T,
            ),
            jac=jac,
        )

        return z, cv

    def _fit(
        self,
        x: list[CV],
        nl: list[NeighbourList] | None,
        chunk_size=None,
        **kwargs,
    ) -> tuple[CV, CvTrans]:
        raise NotImplementedError

    def post_fit(self, y: list[CV], scale) -> tuple[CV, CvTrans]:
        y = CV.stack(*y)
        if not scale:
            return y, CvTrans.from_cv_function(lambda x, _: x)
        h = scale_cv_trans(y)
        return h.compute_cv_trans(y)[0], h


class CVDiscovery:
    """convert set of coordinates to good collective variables."""

    def __init__(self, transformer: Transformer) -> None:
        # self.rounds = rounds
        self.transformer = transformer

    @staticmethod
    def data_loader(
        rounds: Rounds,
        num=4,
        out=-1,
        split_data=False,
        new_r_cut=None,
        cv_round=None,
    ) -> tuple[list[SystemParams], list[NeighbourList] | None, CollectiveVariable, StaticMdInfo]:
        weights = []

        colvar = rounds.get_collective_variable()
        sti: StaticMdInfo | None = None
        sp: list[SystemParams] = []
        nl: list[NeighbourList] | None = [] if new_r_cut is not None else None

        for round, traj in rounds.iter(stop=None, num=num, c=cv_round):
            if sti is None:
                sti = round.tic

            sp0 = traj.ti.sp
            nl0 = (
                sp0.get_neighbour_list(
                    r_cut=new_r_cut,
                    z_array=round.tic.atomic_numbers,
                )
                if new_r_cut is not None
                else None
            )

            if (b0 := traj.ti.e_bias) is None:
                # map cvs
                bias = traj.get_bias()

                if new_r_cut != round.tic.r_cut:
                    nlr = (
                        sp0.get_neighbour_list(
                            r_cut=round.tic.r_cut,
                            z_array=round.tic.atomic_numbers,
                        )
                        if round.tic.r_cut is not None
                        else None
                    )
                else:
                    nlr = nl0

                if (cv0 := traj.ti.CV) is None:
                    if colvar is None:
                        colvar = bias.collective_variable

                    cv0, _ = bias.collective_variable.compute_cv(sp=sp0, nl=nlr)

                b0, _ = bias.compute_from_cv(cvs=cv0)

            sp.append(sp0)
            if nl is not None:
                assert nl0 is not None
                nl.append(nl0)

            beta = 1 / round.tic.T
            weight = jnp.exp(beta * b0)
            weights.append(weight)

        assert sti is not None
        assert len(sp) != 0
        if nl is not None:
            assert len(nl) == len(sp)

        def choose(key, probs: Array):
            key, key_return = split(key, 2)

            indices = choice(
                key=key,
                a=probs.shape[0],
                shape=(int(out),),
                # p=probs,
                replace=False,
            )

            return key_return, indices

        key = PRNGKey(0)

        out_sp: list[SystemParams] = []
        out_nl: list[NeighbourList] | None = [] if nl is not None else None

        if split_data:
            for n, wi in enumerate(weights):
                probs = wi / jnp.sum(wi)
                key, indices = choose(key, probs)

                out_sp.append(sp[n][indices])
                if nl is not None:
                    assert out_nl is not None
                    out_nl.append(nl[n][indices])

        else:
            probs = jnp.hstack(weights)
            probs = probs / jnp.sum(probs)

            key, indices = choose(key, probs)

            if len(sp) >= 1:
                out_sp.append(sum(sp[1:], sp[0])[indices])
                if nl is not None:
                    assert out_nl is not None
                    out_nl.append(sum(nl[1:], nl[0])[indices])

            else:
                out_sp.append(sp[0][indices])
                if nl is not None:
                    assert out_nl is not None
                    out_nl.append(nl[0][indices])

        return (out_sp, out_nl, colvar, sti)

    def compute(
        self,
        rounds: Rounds,
        num_rounds=4,
        samples=1e4,
        plot=True,
        new_r_cut=None,
        chunk_size=None,
        split_data=False,
        name=None,
        **kwargs,
    ) -> CollectiveVariable:
        (sp_list, nl_list, cv_old, sti) = self.data_loader(
            num=num_rounds,
            out=samples,
            rounds=rounds,
            new_r_cut=new_r_cut,
            split_data=split_data,
        )

        cvs_new, new_cv = self.transformer.fit(
            sp_list,
            nl_list,
            chunk_size=chunk_size,
            **kwargs,
        )

        if plot:
            sp = sum(sp_list[1:], sp_list[0])
            nl = sum(nl_list[1:], nl_list[0]) if nl_list is not None else None
            ind = np.random.choice(
                a=sp.shape[0],
                size=min(1000, sp.shape[0]),
                replace=False,
            )

            folder = rounds.path(c=rounds.cv, r=rounds.round)

            CVDiscovery.plot_app(
                name=str(folder / "cvdiscovery"),
                old_cv=cv_old,
                new_cv=new_cv,
                sps=sp[ind],
                nl=nl[ind] if nl is not None else None,
                chunk_size=chunk_size,
            )

        return new_cv

    @staticmethod
    def plot_app(
        sps: SystemParams,
        nl: NeighbourList,
        old_cv: CollectiveVariable,
        new_cv: CollectiveVariable,
        name,
        labels=None,
        chunk_size: int | None = None,
    ):
        """Plot the app for the CV discovery. all 1d and 2d plots are plotted directly, 3d or higher are plotted as 2d slices."""

        cv_data = []
        cv_data_mapped = []

        # collect the cvs if not provided
        cvs = [old_cv, new_cv]
        for cv in cvs:
            cvd = cv.compute_cv(sps, nl, chunk_size=chunk_size)[0].cv
            cvdm = vmap(cv.metric.map)(cvd)

            cv_data.append(np.array(cvd))
            cv_data_mapped.append(np.array(cvdm))

        # for z, data in enumerate([cv_data, cv_data_mapped]):
        for z, data in enumerate([cv_data]):
            # plot setting
            kwargs = {"s": 0.2}

            if labels is None:
                labels = [
                    ["cv in 1", "cv in 2", "cv in 3"],
                    ["cv out 1", "cv out 2", "cv out 3"],
                ]

                indim = cvs[0].n
                outdim = cvs[1].n

                fig = plt.figure()

                for in_out_color, ls, rs in CVDiscovery._grid_spec_iterator(fig=fig, indim=indim, outdim=outdim):
                    if in_out_color == 0:
                        dim = indim
                        color_data = data[0]
                    else:
                        dim = outdim
                        color_data = data[1]

                    max_val = jnp.max(color_data, axis=0)
                    min_val = jnp.min(color_data, axis=0)

                    data_col = (color_data - min_val) / (max_val - min_val)

                    # https://www.hsluv.org/
                    # hue 0-360 sat 0-100 lighness 0-1000

                    if dim == 1:  # skip luminance and set to 0.5. green/red = blue/yellow
                        lab = jnp.ones((data_col.shape[0], 3))
                        lab = lab.at[:, 0].set(data_col[:, 0] * 360)
                        lab = lab.at[:, 1].set(75)
                        lab = lab.at[:, 2].set(40)

                    if dim == 2:
                        lab = jnp.ones((data_col.shape[0], 3))
                        lab = lab.at[:, 0].set(data_col[:, 0] * 360)
                        lab = lab.at[:, 1].set(75)
                        lab = lab.at[:, 2].set(data_col[:, 1] * 100)

                    if dim == 3:
                        lab = jnp.ones((data_col.shape[0], 3))
                        lab = lab.at[:, 0].set(data_col[:, 0] * 360)
                        lab = lab.at[:, 1].set(data_col[:, 1] * 100)
                        lab = lab.at[:, 2].set(data_col[:, 2] * 100)

                    rgb = []

                    for s in lab:
                        rgb.append(hsluv_to_rgb(s))

                    rgb = jnp.array(rgb)

                    def do_plot(dim, in_out, axes):
                        data_proc = data[in_out]
                        if dim == 1:
                            data_proc = jnp.hstack([data_proc, rgb[:, [0]]])

                            f = CVDiscovery._plot_1d
                        elif dim == 2:
                            f = CVDiscovery._plot_2d
                        elif dim == 3:
                            f = CVDiscovery._plot_3d

                        f(fig, axes, data_proc, rgb, labels[0][0:dim], **kwargs)

                    do_plot(indim, 0, ls)
                    do_plot(outdim, 1, rs)

                n = Path(
                    f"{name}_{ 'mapped' if z==1 else ''}.pdf",
                )

                n.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(n)

    @staticmethod
    def _grid_spec_iterator(
        fig: Figure,
        indim,
        outdim,
    ) -> Iterator[tuple[list[int], int, list[plt.Axes], list[plt.Axes]]]:
        width_in = 1 if indim < 3 else 2
        width_out = 1 if outdim < 3 else 2

        spec = fig.add_gridspec(nrows=2, ncols=2, width_ratios=[width_in, width_out])
        yield 0, spec[0, 0], spec[0, 1]
        yield 1, spec[1, 0], spec[1, 1]

    @staticmethod
    def _plot_1d(fig: Figure, grid: gridspec, data, colors, labels, **scatter_kwargs):
        gs = grid.subgridspec(ncols=1, nrows=2, height_ratios=[1, 4])

        ax = fig.add_subplot(gs[1, 0])
        ax_histx = fig.add_subplot(gs[0, 0])

        # create inset
        ax.scatter(
            data[:, 0],
            data[:, 1],
            c=colors,
        )

        ax.set_xlabel(labels)
        ax.set_ylabel("count")

        ax_histx.hist(data[:, 0])

    @staticmethod
    def _plot_2d(fig: Figure, grid: gridspec, data, colors, labels, **scatter_kwargs):
        gs = grid.subgridspec(ncols=2, nrows=2, width_ratios=[4, 1], height_ratios=[1, 4])
        ax = fig.add_subplot(gs[1, 0])
        ax_histx = fig.add_subplot(gs[0, 0])
        ax_histy = fig.add_subplot(gs[1, 1])

        ax.scatter(
            *[data[:, l] for l in range(2)],
            c=colors,
            **scatter_kwargs,
        )
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])

        ax_histx.hist(data[:, 0])
        ax_histy.hist(data[:, 1], orientation="horizontal")

    @staticmethod
    def _plot_3d(fig: Figure, grid: gridspec, data, colors, labels, **scatter_kwargs):
        gs = grid.subgridspec(ncols=2, nrows=1, width_ratios=[1, 1], height_ratios=[1, 1])

        # ?https://matplotlib.org/3.2.1/gallery/axisartist/demo_floating_axes.html

        ax0 = fig.add_subplot(gs[0, 0], projection="3d")
        ax1 = fig.add_subplot(gs[0, 1], projection="3d")
        axes = [ax0, ax1]
        for ax in axes:
            ax.set_xlabel(labels[0])
            ax.set_ylabel(labels[1])
            ax.set_zlabel(labels[2])

            ax.view_init(elev=20, azim=45)

        # create  3d scatter plot with 2D histogram on side
        ax[0].scatter(
            data[:, 0],
            data[:, 1],
            data[:, 2],
            **scatter_kwargs,
            c=colors,
            zorder=1,
        )

        for a, b, z in [
            [0, 1, "z"],
            [0, 2, "y"],
            [1, 2, "x"],
        ]:
            Z, X, Y = np.histogram2d(data[:, a], data[:, b])

            X = (X[1:] + X[:-1]) / 2
            Y = (Y[1:] + Y[:-1]) / 2

            X, Y = np.meshgrid(X, Y)

            Z = (Z - Z.min()) / (Z.max() - Z.min())

            kw = {
                "facecolors": plt.cm.Greys(Z),
                "shade": True,
                "alpha": 1.0,
                "zorder": 0,
            }

            zz = np.zeros(X.shape) - 0.1

            if z == "z":
                ax[0].plot_surface(X, Y, zz, **kw)
            elif z == "y":
                ax[0].plot_surface(X, zz, Y, **kw)
            else:
                ax[0].plot_surface(zz, X, Y, **kw)

        # scatter 2d projections
        zz = np.zeros(data[:, 0].shape)
        for z in ["x", "y", "z"]:
            if z == "z":
                ax[1].scatter(
                    data[:, 0],
                    data[:, 1],
                    zz,
                    **scatter_kwargs,
                    zorder=1,
                    c=colors,
                )
            elif z == "y":
                ax[1].scatter(
                    data[:, 0],
                    zz,
                    data[:, 2],
                    **scatter_kwargs,
                    zorder=1,
                    c=colors,
                )
            else:
                ax[1].scatter(
                    zz,
                    data[:, 1],
                    data[:, 2],
                    **scatter_kwargs,
                    zorder=1,
                    c=colors,
                )
