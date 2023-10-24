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
from IMLCV.base.CV import padded_pmap
from IMLCV.base.CV import SystemParams
from IMLCV.base.rounds import Rounds
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
            return f.compute_cv_flow(sp, nl, chunk_size)

        if p_map:
            _f = padded_pmap(_f)

        x = _f(z, nl)

        x_stacked = x.replace(_stack_dims=stack_dims)

        if self.pre_scale:
            g = scale_cv_trans(x_stacked, lower=0, upper=1)

            x_stacked, _ = g.compute_cv_trans(x_stacked)

            f = f * g

        x = x_stacked.unstack()

        return x, f

    def fit(
        self,
        dlo: Rounds.data_loader_output,
        chunk_size=None,
        plot=True,
        plot_folder: str | Path | None = None,
        p_map=True,
        percentile=5,
        margin=0.01,
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

        print("starting fit")
        y, g = self._fit(
            x,
            dlo,
            chunk_size=chunk_size,
            **self.fit_kwargs,
        )

        # remove outliers from the data
        bounds = jnp.percentile(y.cv, jnp.array([percentile, 100 - percentile]), axis=0)

        diff = bounds[1, :] - bounds[0, :]

        bounds_l = bounds[0, :] - margin * diff / (1 - 2 * percentile / 100) * (1 + margin)
        bounds_u = bounds[1, :] + margin * diff / (1 - 2 * percentile / 100) * (1 + margin)

        mask_l = jnp.all(y.cv > bounds_l, axis=1)
        mask_u = jnp.all(y.cv < bounds_u, axis=1)
        mask = mask_l & mask_u

        y_masked = y[mask]

        print("starting post_fit")
        z_masked, h = self.post_fit(y_masked)
        z, _ = h.compute_cv_trans(y)

        new_bounding_box = jnp.vstack(
            [jnp.min(z_masked.cv, axis=0), jnp.max(z_masked.cv, axis=0)],
        ).T

        # d = (new_bounding_box[:, 1] - new_bounding_box[:, 0]) * margin / 2

        # new_bounding_box = new_bounding_box.at[:, 0].set(new_bounding_box[:, 0] - d * margin / 2)
        # new_bounding_box = new_bounding_box.at[:, 1].set(new_bounding_box[:, 1] + d * margin / 2)

        new_collective_variable = CollectiveVariable(
            f=f * g * h,
            metric=CvMetric(
                periodicities=None,
                bounding_box=new_bounding_box,
            ),
        )

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
                cv_data_old=CV.stack(*dlo.cv)[mask],
                cv_data_new=z_masked,
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
            return y, CvTrans.from_cv_function(lambda x, _: x)
        h = scale_cv_trans(y)
        return h.compute_cv_trans(y)[0], h

    @staticmethod
    def plot_app(
        old_cv: CollectiveVariable,
        new_cv: CollectiveVariable,
        name: str | Path,
        labels=None,
        sps: SystemParams = None,
        nl: NeighbourList = None,
        chunk_size: int | None = None,
        cv_data_old: CV | None = None,
        cv_data_new: CV | None = None,
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

        # plot setting
        kwargs = {"s": 0.2}

        if labels is None:
            labels = [
                ["cv in 1", "cv in 2", "cv in 3"],
                ["cv out 1", "cv out 2", "cv out 3"],
            ]

        indim = cv_data[0].shape[1]
        outdim = cv_data[1].shape[1]

        fig = plt.figure()

        for in_out_color, ls, rs in Transformer._grid_spec_iterator(
            fig=fig,
            indim=indim,
            outdim=outdim,
        ):
            if in_out_color == 0:
                dim = indim
                color_data = cv_data[0].cv
            else:
                dim = outdim
                color_data = cv_data[1].cv

            max_val = jnp.max(color_data, axis=0)
            min_val = jnp.min(color_data, axis=0)

            if (max_val == min_val).all():
                data_col = color_data

            else:
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
                data_proc = cv_data[in_out].cv
                if dim == 1:
                    data_proc = jnp.hstack([data_proc, rgb[:, [0]]])

                    f = Transformer._plot_1d
                elif dim == 2:
                    f = Transformer._plot_2d
                elif dim == 3:
                    f = Transformer._plot_3d

                f(fig, axes, data_proc, rgb, labels[0][0:dim], **kwargs)

            do_plot(indim, 0, ls)
            do_plot(outdim, 1, rs)

        name = Path(name)
        if name.suffix != ".pdf":
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
        gs = grid.subgridspec(
            ncols=2,
            nrows=2,
            width_ratios=[4, 1],
            height_ratios=[1, 4],
        )
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
