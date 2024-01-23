from pathlib import Path
from typing import Iterator

import jax
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
from IMLCV.base.CV import padded_pmap
from IMLCV.base.CV import SystemParams
from IMLCV.base.rounds import Rounds
from IMLCV.implementations.CV import identity_trans
from IMLCV.implementations.CV import scale_cv_trans
from jax import pmap
from matplotlib import gridspec
from matplotlib.figure import Figure

# from jax_dataclasses import copy_and_mutate


class Transformer:
    def __init__(
        self,
        outdim,
        periodicity=None,
        bounding_box=None,
        descriptor=CvFlow,
        pre_scale=True,
        post_scale=True,
        **fit_kwargs,
    ) -> None:
        self.outdim = outdim

        if periodicity is None:
            periodicity = [False for _ in range(self.outdim)]

        self.periodicity = periodicity
        self.descriptor = descriptor
        self.pre_scale = pre_scale
        self.post_scale = post_scale

        self.fit_kwargs = fit_kwargs

    def pre_fit(
        self,
        z: list[SystemParams],
        nl: list[NeighbourList] | None,
        chunk_size=None,
        p_map=True,
    ) -> tuple[list[CV], CvFlow]:
        f = self.descriptor

        stack_dims = tuple([z_i.batch_dim for z_i in z])
        z = SystemParams.stack(*z)
        nl = NeighbourList.stack(*nl) if nl is not None else None

        def _f(sp, nl):
            return f.compute_cv_flow(sp, nl, chunk_size)[0]

        if p_map:
            _f = padded_pmap(_f)

        x: CV = _f(z, nl)
        x = x.replace(_stack_dims=stack_dims)

        if self.pre_scale:
            g = scale_cv_trans(x, lower=0, upper=1)

            x, _, _ = g.compute_cv_trans(x)

            f = f * g

        x = x.unstack()

        return x, f

    def fit(
        self,
        dlo: Rounds.data_loader_output,
        chunk_size=None,
        plot=True,
        plot_folder: str | Path | None = None,
        p_map=True,
        percentile=1,
        margin=0.05,
        jac=jax.jacrev,
        test=False,
    ) -> tuple[CV, CollectiveVariable]:
        if plot:
            assert plot_folder is not None, "plot_folder must be specified if plot=True"

        sp_list = dlo.sp
        nl_list = dlo.nl

        print("starting pre_fit")

        x, f = self.pre_fit(
            sp_list,
            nl_list,
            chunk_size=chunk_size,
            p_map=p_map,
        )

        if test:
            x_2, _ = f.compute_cv_flow(
                SystemParams.stack(*sp_list),
                NeighbourList.stack(*nl_list),
                chunk_size=chunk_size,
            )
            assert jnp.allclose(x_2.cv, CV.stack(*x).cv)

        print("starting fit")
        y, g = self._fit(
            x,
            dlo,
            chunk_size=chunk_size,
            **self.fit_kwargs,
        )

        print(f"{y.stack_dims=}")

        if y.stack_dims is None:
            print("faulty stack dims, replacing")
            y = y.replace(_stack_dim=[a.shape[0] for a in sp_list])

        if test:
            y_2, _, _ = g.compute_cv_trans(CV.stack(*x), NeighbourList.stack(*nl_list), chunk_size=chunk_size)
            assert jnp.allclose(y_2.cv, y.cv)

        # remove outliers from the data
        bounds = jnp.percentile(y.cv, jnp.array([percentile, 100 - percentile]), axis=0)

        # diff = bounds[1, :] - bounds[0, :]

        bounds_l = bounds[0, :]
        bounds_u = bounds[1, :]

        mask_l = jnp.all(y.cv > bounds_l, axis=1)
        mask_u = jnp.all(y.cv < bounds_u, axis=1)
        mask = mask_l & mask_u

        y_masked = y[mask]

        print("starting post_fit")
        z_masked, h = self.post_fit(y_masked)
        z, _, _ = h.compute_cv_trans(y)

        new_bounding_box = jnp.vstack(
            [jnp.min(z_masked.cv, axis=0), jnp.max(z_masked.cv, axis=0)],
        ).T

        new_collective_variable = CollectiveVariable(
            f=f * g * h,
            jac=jac,
            metric=CvMetric.create(
                periodicities=None,
                bounding_box=new_bounding_box,
            ),
        )

        if test:
            cv_full, _ = new_collective_variable.compute_cv(
                SystemParams.stack(*sp_list),
                NeighbourList.stack(*nl_list),
                chunk_size=chunk_size,
            )

            assert jnp.allclose(cv_full.cv, z.cv)

        if plot:
            # sp = SystemParams.stack(*sp_list)
            # nl = NeighbourList.stack(*nl_list) if nl_list is not None else None

            Transformer.plot_app(
                name=str(plot_folder / "cvdiscovery.pdf"),
                old_cv=dlo.collective_variable,
                new_cv=new_collective_variable,
                # sps=sp,
                # nl=nl if nl is not None else None,
                # chunk_size=chunk_size,
                # cv_data_old=CV.stack(*dlo.cv)[mask],
                # cv_data_new=z_masked,
                cv_data_old=CV.stack(*dlo.cv),
                cv_data_new=z,
                margin=0.1,
            )

        return z, new_collective_variable

    def _fit(
        self,
        x: list[CV],
        dlo: Rounds.data_loader_output,
        chunk_size=None,
        **fit_kwargs,
    ) -> tuple[CV, CvTrans]:
        raise NotImplementedError

    def post_fit(self, y: list[CV]) -> tuple[CV, CvTrans]:
        # y = CV.stack(*y)
        if not self.post_scale:
            return y, identity_trans
        h = scale_cv_trans(y)
        return h.compute_cv_trans(y)[0], h

    @staticmethod
    def plot_app(
        old_cv: CollectiveVariable | None = None,
        new_cv: CollectiveVariable | None = None,
        name: str | Path | None = None,
        labels=None,
        sps: SystemParams = None,
        nl: NeighbourList = None,
        chunk_size: int | None = None,
        cv_data_old: CV | None = None,
        cv_data_new: CV | None = None,
        cv_titles=None,
        data_titles=None,
        color_trajectories=False,
        margin=0.1,
    ):
        """Plot the app for the CV discovery. all 1d and 2d plots are plotted directly, 3d or higher are plotted as 2d slices."""

        cv_data: list[CV] = []

        if cv_data_old is None:
            cv_data.append(old_cv.compute_cv(sps, nl, chunk_size=chunk_size)[0])
        else:
            cv_data.append(cv_data_old)

        if cv_data_new is None:
            cv_data.append(new_cv.compute_cv(sps, nl, chunk_size=chunk_size)[0])
        else:
            cv_data.append(cv_data_new)

        if cv_titles is None:
            cv_titles = ["Old CV", "New CV"]

        # if data_titles is None:
        #     data_titles = ["Old Data", "New Data"]

        # plot setting
        kwargs = {
            "s": 0.3,
            "edgecolor": "none",
        }

        if labels is None:
            labels = [
                ["cv_1 [a.u.]", "cv_2 [a.u.]", "cv_3 [a.u.]"],
                ["cv_1 [a.u.]", "cv_2 [a.u.]", "cv_3 [a.u.]"],
            ]

        indim = cv_data[0].shape[1]
        outdim = cv_data[1].shape[1]

        plt.rc("text", usetex=False)
        plt.rc("font", family="DejaVu Sans", size=16)

        fig = plt.figure(figsize=(6, 6))

        for in_out_color, ls, rs in Transformer._grid_spec_iterator(
            fig=fig,
            indim=indim,
            outdim=outdim,
            cv_titles=cv_titles,
            data_titles=data_titles,
        ):
            if in_out_color == 0:
                rgb = Transformer._get_color_data(cv_data[0], indim, color_trajectories, margin=margin).cv
            else:
                rgb = Transformer._get_color_data(cv_data[1], outdim, color_trajectories, margin=margin).cv

            def do_plot(dim, in_out, axes):
                data_proc = cv_data[in_out].cv
                if dim == 1:
                    x = []
                    for i, ai in enumerate(cv_data[in_out].unstack()):
                        x.append(ai.cv * 0 + i)

                    data_proc = jnp.hstack([data_proc, jnp.vstack(x)])

                    f = Transformer._plot_1d
                elif dim == 2:
                    f = Transformer._plot_2d
                elif dim == 3:
                    f = Transformer._plot_3d

                f(fig, axes, data_proc, rgb, labels[in_out][0:dim], margin=margin, **kwargs)

            do_plot(indim, 0, ls)
            do_plot(outdim, 1, rs)

        if name is None:
            plt.show()
        else:
            name = Path(name)

            if (name.suffix != ".pdf") and (name.suffix != ".png"):
                print(f"{name.suffix} should be pdf or png, changing to pdf")

                name = Path(
                    f"{name}.pdf",
                )

            name.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(name)

    @staticmethod
    def _grid_spec_iterator(
        fig: Figure,
        indim,
        outdim,
        cv_titles,
        data_titles=None,
    ) -> Iterator[tuple[list[int], int, list[plt.Axes], list[plt.Axes]]]:
        width_in = 1 if indim < 3 else 2
        width_out = 1 if outdim < 3 else 2

        spec = fig.add_gridspec(
            nrows=3,
            ncols=3,
            width_ratios=[0.01, width_in, width_out],
            height_ratios=[0.01, 1, 1],
            wspace=0.3,
            hspace=0.2,
        )

        yield 0, spec[1, 1], spec[1, 2]
        yield 1, spec[2, 1], spec[2, 2]

        def add_ax(sp):
            ax = fig.add_subplot(sp)

            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_visible(False)
            return ax

        ax_left = add_ax(spec[0, 1])
        ax_left.set_title(cv_titles[0])

        ax_right = add_ax(spec[0, 2])
        ax_right.set_title(cv_titles[1])

        if data_titles is not None:
            ax_up = add_ax(spec[1, 0])
            ax_up.set_title(data_titles[0])

            ax_down = add_ax(spec[2, 0])
            ax_down.set_title(data_titles[1])

    @staticmethod
    def _plot_1d(fig: Figure, grid: gridspec, data, colors, labels, margin=None, **scatter_kwargs):
        gs = grid.subgridspec(
            ncols=1,
            nrows=2,
            height_ratios=[1, 4],
            wspace=0.05,
            hspace=0.05,
        )

        ax = fig.add_subplot(gs[1, 0])
        # ax.set_aspect('equal')
        ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)

        # create inset
        ax.scatter(
            data[:, 0],
            data[:, 1],
            c=colors,
            **scatter_kwargs,
        )

        in_xlim = jnp.logical_and(data[:, 0] > -margin, data[:, 0] < 1 + margin)
        n_points = jnp.sum(in_xlim)
        n_bins = 3 * int(1 + jnp.ceil(jnp.log2(n_points)))

        ax.set_xlabel(labels[0])
        ax.set_ylabel("trajectory")

        ax_histx.hist(data[:, 0], bins=n_bins, range=[-margin, 1 + margin])

        if margin is not None:
            ax.set_xlim(-margin, 1 + margin)
            ax_histx.set_xlim(-margin, 1 + margin)

    @staticmethod
    def _plot_2d(fig: Figure, grid: gridspec, data, colors, labels, margin=None, **scatter_kwargs):
        gs = grid.subgridspec(
            ncols=2,
            nrows=2,
            width_ratios=[4, 1],
            height_ratios=[1, 4],
            wspace=0.02,
            hspace=0.02,
        )
        ax = fig.add_subplot(gs[1, 0])

        ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
        ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

        ax.scatter(
            *[data[:, l] for l in range(2)],
            c=colors,
            **scatter_kwargs,
        )
        # ax.set_xlabel(labels[0])
        # ax.set_ylabel(labels[1])

        in_xlim = jnp.logical_and(data[:, 0] > -margin, data[:, 0] < 1 + margin)
        in_ylim = jnp.logical_and(data[:, 1] > -margin, data[:, 1] < 1 + margin)
        n_points = jnp.sum(jnp.logical_and(in_xlim, in_ylim))
        n_bins = 3 * int(1 + jnp.ceil(jnp.log2(n_points)))

        ax_histx.hist(data[:, 0], bins=n_bins, range=[-margin, 1 + margin])
        ax_histy.hist(data[:, 1], bins=n_bins, range=[-margin, 1 + margin], orientation="horizontal")
        ax_histy.tick_params(axis="x", rotation=-90)

        for b in [ax_histx, ax_histy]:
            b.spines["right"].set_visible(False)
            b.spines["top"].set_visible(False)
            b.spines["bottom"].set_visible(False)
            b.spines["left"].set_visible(False)

        ax_histx.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
        ax_histy.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)

        ax.locator_params(nbins=3)

        if margin is not None:
            ax.set_xlim(-margin, 1 + margin)
            # ax_histx.set_xlim(-margin, 1 + margin)

            ax.set_ylim(-margin, 1 + margin)
            # ax_histy.set_ylim(-margin, 1 + margin)

    @staticmethod
    def _plot_3d(fig: Figure, grid: gridspec, data, colors, labels, margin=None, **scatter_kwargs):
        gs = grid.subgridspec(
            ncols=2,
            nrows=1,
            width_ratios=[1, 1],
            height_ratios=[1],
        )

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
        ax0.scatter(
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

            zz = np.zeros(X.shape) - 1.1

            if z == "z":
                ax0.plot_surface(X, Y, zz, **kw)
            elif z == "y":
                ax0.plot_surface(X, zz, Y, **kw)
            else:
                ax0.plot_surface(zz, X, Y, **kw)

        # scatter 2d projections
        zz = np.zeros(data[:, 0].shape)
        for z in ["x", "y", "z"]:
            if z == "z":
                ax1.scatter(
                    data[:, 0],
                    data[:, 1],
                    zz,
                    **scatter_kwargs,
                    zorder=1,
                    c=colors,
                )
            elif z == "y":
                ax1.scatter(
                    data[:, 0],
                    zz,
                    data[:, 2],
                    **scatter_kwargs,
                    zorder=1,
                    c=colors,
                )
            else:
                ax1.scatter(
                    zz,
                    data[:, 1],
                    data[:, 2],
                    **scatter_kwargs,
                    zorder=1,
                    c=colors,
                )

    def _get_color_data(
        a: CV,
        dim: int,
        color_trajectories=True,
        color_1d=True,
        max_val=None,
        min_val=None,
        margin=None,
    ) -> CV:
        if dim == 1:
            if color_1d:
                x = []

                for i, ai in enumerate(a.unstack()):
                    x.append(ai.cv * 0 + i)

                a = a.replace(cv=jnp.hstack([a.cv, jnp.vstack(x)]))

            dim = 2

        if color_trajectories:
            a_out = []

            for ai in a.unstack():
                avg = jnp.mean(ai.cv, axis=0, keepdims=True)
                a_out.append(ai.replace(cv=ai.cv * 0 + avg))

            a = CV.stack(*a_out)

        color_data = a.cv

        if max_val is None:
            if margin is None:
                max_val = jnp.max(color_data, axis=0)
            else:
                max_val = jnp.full((dim,), -margin)

        if min_val is None:
            if margin is None:
                min_val = jnp.min(color_data, axis=0)
            else:
                min_val = jnp.full((dim,), 1 + margin)

        if (max_val == min_val).all():
            data_col = color_data

        else:
            data_col = (color_data - min_val) / (max_val - min_val)

        jnp.clip(data_col, 0.0, 1.0)

        # https://www.hsluv.org/
        # hue 0-360 sat 0-100 lighness 0-1000

        if dim == 1:  # skip luminance and set to 0.5. green/red = blue/yellow
            lab = jnp.ones((data_col.shape[0], 3))
            lab = lab.at[:, 0].set(data_col[:, 0] * 340)
            lab = lab.at[:, 1].set(75)
            lab = lab.at[:, 2].set(40)

        if dim == 2:
            lab = jnp.ones((data_col.shape[0], 3))
            lab = lab.at[:, 0].set(data_col[:, 0] * 340)
            lab = lab.at[:, 1].set(75)
            lab = lab.at[:, 2].set(data_col[:, 1] * 60 + 20)

        if dim == 3:
            lab = jnp.ones((data_col.shape[0], 3))
            lab = lab.at[:, 0].set(data_col[:, 0] * 340)
            lab = lab.at[:, 1].set(data_col[:, 1] * 60 + 20)
            lab = lab.at[:, 2].set(data_col[:, 2] * 60 + 20)

        rgb = []

        for s in lab:
            rgb.append(hsluv_to_rgb(s))

        rgb = jnp.array(rgb)

        return a.replace(cv=rgb)


class IdentityTransformer(Transformer):
    def _fit(
        self,
        x: list[CV],
        dlo: Rounds.data_loader_output,
        chunk_size=None,
        **fit_kwargs,
    ) -> tuple[CV, CvTrans]:
        cv = CV.stack(*x)

        return cv, identity_trans
