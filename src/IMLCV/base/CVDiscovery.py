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
from IMLCV.base.CV import SystemParams
from IMLCV.base.rounds import Rounds
from IMLCV.implementations.CV import identity_trans
from IMLCV.implementations.CV import scale_cv_trans
from matplotlib import gridspec
from matplotlib.figure import Figure
from IMLCV.base.bias import NoneBias
from molmod.units import kjmol


class Transformer:
    def __init__(
        self,
        outdim,
        descriptor: CvFlow,
        pre_scale=True,
        post_scale=True,
        **fit_kwargs,
    ) -> None:
        self.outdim = outdim

        self.descriptor = descriptor
        self.pre_scale = pre_scale
        self.post_scale = post_scale

        self.fit_kwargs = fit_kwargs

    def pre_fit(
        self,
        dlo: Rounds.data_loader_output,
        chunk_size=None,
        p_map=True,
    ) -> tuple[list[CV], list[CV] | None, CvFlow]:
        f = self.descriptor

        x, x_t = dlo.apply_cv_flow(
            f,
            chunk_size=chunk_size,
            pmap=p_map,
        )

        if self.pre_scale:
            g = scale_cv_trans(CV.stack(*x), lower=0, upper=1)
            x, x_t = dlo.apply_cv_trans(
                g,
                x,
                x_t,
                chunk_size=chunk_size,
                pmap=p_map,
            )
            f = f * g

        return x, x_t, f

    def fit(
        self,
        dlo: Rounds.data_loader_output,
        chunk_size=None,
        plot=True,
        plot_folder: str | Path | None = None,
        p_map=True,
        percentile=5.0,
        margin=0.01,
        jac=jax.jacrev,
        test=False,
        check_nan=True,
        transform_FES=True,
        max_fes_bias=100 * kjmol,
        samples_per_bin=50,
        min_samples_per_bin=15,
    ) -> tuple[CV, CollectiveVariable]:
        if plot:
            assert plot_folder is not None, "plot_folder must be specified if plot=True"

        print("starting pre_fit")

        x, x_t, f = self.pre_fit(
            dlo,
            chunk_size=chunk_size,
            p_map=p_map,
        )

        if check_nan:
            print("checking pre_fit nans")
            dlo, x, x_t = dlo.filter_nans(x, x_t)

        if test:
            print("testing pre_fit")
            x_2, _ = f.compute_cv_flow(
                SystemParams.stack(*dlo.sp),
                NeighbourList.stack(*dlo.nl),
                chunk_size=chunk_size,
            )

            assert jnp.allclose(x_2.cv, CV.stack(*x).cv)

            if x_t is not None:
                x_t_2, _ = f.compute_cv_flow(
                    SystemParams.stack(*dlo.sp_t),
                    NeighbourList.stack(*dlo.nl_t),
                    chunk_size=chunk_size,
                )

                assert jnp.allclose(x_t_2.cv, CV.stack(*x_t).cv)

        print("starting fit")
        y, y_t, g = self._fit(
            x,
            x_t,
            dlo,
            chunk_size=chunk_size,
            **self.fit_kwargs,
        )

        if check_nan:
            print("checking fit nans")
            dlo, y, y_t = dlo.filter_nans(y, y_t)

        y = CV.stack(*y)
        if y_t is not None:
            y_t = CV.stack(*y_t)

            assert y_t.stack_dims == y.stack_dims, "y and y_t must have the same stack_dims"

        print(f"{y.stack_dims=}")

        if test:
            y_2, _, _ = g.compute_cv_trans(CV.stack(*x), NeighbourList.stack(*dlo.nl), chunk_size=chunk_size)
            assert jnp.allclose(y_2.cv, y.cv)

            if y_t is not None:
                y_t_2, _, _ = g.compute_cv_trans(
                    CV.stack(*x_t),
                    NeighbourList.stack(*dlo.nl_t),
                    chunk_size=chunk_size,
                )

        # remove outliers from the data
        _, mask = CvMetric.bounds_from_cv(y, percentile=percentile)
        y_masked = y[mask]

        print("starting post_fit")
        z_masked, h = self.post_fit(y_masked)
        z, _, _ = h.compute_cv_trans(y)

        mini = jnp.min(z_masked.cv, axis=0)
        maxi = jnp.max(z_masked.cv, axis=0)

        diff = maxi - mini

        mini_margin = mini - diff * (percentile / 200 + margin)
        maxi_margin = maxi + diff * (percentile / 200 + margin)

        new_bounding_box = jnp.vstack(
            [mini_margin, maxi_margin],
        ).T

        new_collective_variable = CollectiveVariable(
            f=f * g * h,
            jac=jac,
            metric=CvMetric.create(
                periodicities=None,
                bounding_box=new_bounding_box,
            ),
        )

        if transform_FES:
            print("transforming FES")
            bias = dlo.get_transformed_fes(
                # weights=dlo.weights(),
                new_cv=z,
                new_colvar=new_collective_variable,
                samples_per_bin=samples_per_bin,
                min_samples_per_bin=min_samples_per_bin,
                max_bias=max_fes_bias,
            )

            if plot:
                bias.plot(
                    name=str(plot_folder / "transformed_fes.pdf"),
                    margin=0.1,
                    inverted=True,
                )

        else:
            bias = NoneBias.create(new_collective_variable)

        if test:
            cv_full, _ = new_collective_variable.compute_cv(
                SystemParams.stack(*dlo.sp),
                NeighbourList.stack(*dlo.nl),
                chunk_size=chunk_size,
            )

            assert jnp.allclose(cv_full.cv, z.cv)

        if plot:
            Transformer.plot_app(
                name=str(plot_folder / "cvdiscovery.pdf"),
                collective_variables=[dlo.collective_variable, new_collective_variable],
                cv_data=[CV.stack(*dlo.cv), z],
                # weight=dlo.weights(),
                margin=0.1,
            )

        return z, new_collective_variable, bias

    def _fit(
        self,
        x: list[CV],
        x_t: list[CV] | None,
        dlo: Rounds.data_loader_output,
        chunk_size=None,
        **fit_kwargs,
    ) -> tuple[list[CV], list[CV] | None, CvTrans]:
        raise NotImplementedError

    def post_fit(self, y: list[CV]) -> tuple[CV, CvTrans]:
        # y = CV.stack(*y)
        if not self.post_scale:
            return y, identity_trans
        h = scale_cv_trans(y)
        return h.compute_cv_trans(y)[0], h

    @staticmethod
    def plot_app(
        collective_variables: list[CollectiveVariable],
        cv_data: list[CV] | list[list[CV]],
        weight: list[jax.Array] | list[list[jax.Array]] | None = None,
        duplicate_cv_data=True,
        name: str | Path | None = None,
        labels=None,
        cv_titles=None,
        data_titles=None,
        color_trajectories=False,
        margin=0.1,
        max_points=10000,
    ):
        """Plot the app for the CV discovery. all 1d and 2d plots are plotted directly, 3d or higher are plotted as 2d slices."""

        ncv = len(collective_variables)

        if duplicate_cv_data:
            cv_data = [cv_data] * ncv
            if weight is not None:
                weight = [weight] * ncv

        if weight is not None:
            weight = [jnp.hstack(w) for w in weight]

        metrics = [colvar.metric for colvar in collective_variables]

        if cv_titles is None:
            cv_titles = [f"cv_{i}" for i in range(ncv)]

        if data_titles is None:
            data_titles = [f"data_{i}" for i in range(ncv)]

        if labels is None:
            labels = [
                ["cv_1 [a.u.]", "cv_2 [a.u.]", "cv_3 [a.u.]"],
                ["cv_1 [a.u.]", "cv_2 [a.u.]", "cv_3 [a.u.]"],
            ]

        # if weight is not None:
        #     assert len(weight) == ncv
        #     weight = jnp.hstack(weight)

        inoutdims = [cv_data[n][n].shape[1] for n in range(ncv)]

        plt.rc("text", usetex=False)
        plt.rc("font", family="DejaVu Sans", size=16)

        fig = plt.figure(figsize=(6, 6))
        rgb_data = [
            Transformer._get_color_data(
                cv_data[n][n],
                inoutdims[n],
                color_trajectories,
                metric=metrics[n],
                margin=margin,
            ).cv
            for n in range(ncv)
        ]

        for data_in, in_out, axes in Transformer._grid_spec_iterator(
            fig=fig,
            dims=inoutdims,
            cv_titles=cv_titles,
            data_titles=data_titles,
        ):
            dim = inoutdims[in_out]

            data_proc = cv_data[data_in][in_out].cv
            if dim == 1:
                x = []
                for i, ai in enumerate(cv_data[data_in][in_out].unstack()):
                    x.append(ai.cv * 0 + i)

                data_proc = jnp.hstack([data_proc, jnp.vstack(x)])

                f = Transformer._plot_1d
            elif dim == 2:
                f = Transformer._plot_2d
            elif dim == 3:
                f = Transformer._plot_3d

            # plot setting
            kwargs = {
                "s": (300 / data_proc.shape[0]) ** (0.5),
                "edgecolor": "none",
            }

            f(
                fig,
                axes,
                data_proc,
                rgb_data[data_in],
                labels[0:dim],
                metric=metrics[in_out],
                weight=weight[data_in] if weight is not None else None,
                margin=margin,
                **kwargs,
            )

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
        dims,
        cv_titles,
        data_titles=None,
    ) -> Iterator[tuple[list[int], int, list[plt.Axes], list[plt.Axes]]]:
        widths = [1 if i < 3 else 2 for i in dims]

        spec = fig.add_gridspec(
            nrows=len(dims) + 1,
            ncols=len(dims) + 1,
            width_ratios=[0.3, *widths],
            height_ratios=[0, *[1 for _ in dims]],
            wspace=0.1,
            hspace=0.1,
        )

        for data_in in range(len(dims)):
            for cv_in in range(len(dims)):
                yield data_in, cv_in, spec[data_in + 1, cv_in + 1]

        for i in range(len(dims)):
            s = spec[0, i + 1]
            pos = s.get_position(fig)

            fig.text(
                x=(pos.x0 + pos.x1) / 2,
                y=(pos.y0 + pos.y1) / 2,
                s=cv_titles[i],
                ha="center",
                va="center",
            )

            if data_titles is not None:
                s = spec[i + 1, 0]
                pos = s.get_position(fig)

                fig.text(
                    x=(pos.x0 + pos.x1) / 2,
                    y=(pos.y0 + pos.y1) / 2,
                    s=data_titles[i],
                    ha="center",
                    va="center",
                    rotation=90,
                )

    @staticmethod
    def _plot_1d(
        fig: Figure,
        grid: gridspec,
        data,
        colors,
        labels,
        metric: CvMetric,
        weight=None,
        margin=None,
        **scatter_kwargs,
    ):
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
        if n_points != 0:
            n_bins = 3 * int(1 + jnp.ceil(jnp.log2(n_points)))
        else:
            n_bins = 10

        ax.set_xlabel(labels[0])
        ax.set_ylabel("trajectory")

        ax_histx.hist(data[:, 0], bins=n_bins, range=[-margin, 1 + margin])

        if margin is not None:
            ax.set_xlim(-margin, 1 + margin)
            ax_histx.set_xlim(-margin, 1 + margin)

    @staticmethod
    def _plot_2d(
        fig: Figure,
        grid: gridspec,
        data,
        colors,
        labels,
        metric: CvMetric,
        margin=None,
        weight=None,
        **scatter_kwargs,
    ):
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
            *[data[:, col] for col in range(2)],
            c=colors,
            **scatter_kwargs,
        )

        m = (metric.bounding_box[:, 1] - metric.bounding_box[:, 0]) * margin

        x_l = metric.bounding_box[0, :]
        m_x = m[0]
        y_l = metric.bounding_box[1, :]
        m_y = m[1]

        x_lim = [x_l[0] - m_x, x_l[1] + m_x]
        y_lim = [y_l[0] - m_y, y_l[1] + m_y]

        in_xlim = jnp.logical_and(data[:, 0] > x_lim[0], data[:, 0] < x_lim[1])
        in_ylim = jnp.logical_and(data[:, 1] > y_lim[0], data[:, 1] < y_lim[1])
        n_points = jnp.sum(jnp.logical_and(in_xlim, in_ylim))
        n_bins = 3 * int(1 + jnp.ceil(jnp.log2(n_points)))

        ax_histx.hist(data[:, 0], bins=n_bins, range=x_lim, weights=weight)
        ax_histy.hist(data[:, 1], bins=n_bins, range=y_lim, weights=weight, orientation="horizontal")
        ax_histy.tick_params(axis="x", rotation=-90)

        for b in [ax_histx, ax_histy]:
            b.spines["right"].set_visible(False)
            b.spines["top"].set_visible(False)
            b.spines["bottom"].set_visible(False)
            b.spines["left"].set_visible(False)

        ax_histx.tick_params(
            top=False,
            bottom=False,
            left=False,
            right=False,
            labelleft=False,
            labelbottom=False,
        )
        ax_histy.tick_params(
            top=False,
            bottom=False,
            left=False,
            right=False,
            labelleft=False,
            labelbottom=False,
        )

        ax.locator_params(nbins=4)

        ax.set_xticklabels([])
        ax.set_yticklabels([])

        if margin is not None:
            ax.set_xlim(*x_lim)
            ax.set_ylim(*y_lim)

    @staticmethod
    def _plot_3d(
        fig: Figure,
        grid: gridspec,
        data,
        colors,
        labels,
        metric: CvMetric,
        weight=None,
        margin=None,
        **scatter_kwargs,
    ):
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
        metric: CvMetric | None = None,
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

        if max_val is not None and min_val is not None:
            pass

        elif metric is not None:
            min_val = metric.bounding_box[:, 0]
            max_val = metric.bounding_box[:, 1]

        else:
            max_val = jnp.max(color_data, axis=0)
            min_val = jnp.min(color_data, axis=0)

        if margin is not None:
            max_val = max_val + (max_val - min_val) * margin
            min_val = min_val - (max_val - min_val) * margin

        if jnp.allclose(max_val, min_val):
            data_col = color_data
        else:
            data_col = (color_data - min_val) / (max_val - min_val)

        data_col = jnp.clip(data_col, 0.0, 1.0)

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

    def __mul__(self, other):
        # assert isinstance(other, Transformer), "can only multiply with another transformer"

        trans: list[Transformer] = []

        if isinstance(self, CombineTransformer):
            trans.extend(self.transformers)
        else:
            trans.append(self)

        if isinstance(other, CombineTransformer):
            trans.extend(other.transformers)
        else:
            trans.append(other)

        return CombineTransformer(trans)


class CombineTransformer(Transformer):
    def __init__(self, transformers: list[Transformer], **fit_kwargs) -> None:
        self.transformers = transformers

        self.outdim = transformers[-1].outdim
        self.descriptor = transformers[0].descriptor
        self.pre_scale = transformers[0].pre_scale
        self.post_scale = transformers[-1].post_scale
        self.fit_kwargs = fit_kwargs

    def _fit(
        self,
        x: list[CV],
        x_t: list[CV] | None,
        dlo: Rounds.data_loader_output,
        chunk_size=None,
        **fit_kwargs,
    ) -> tuple[list[CV], list[CV] | None, CvTrans]:
        trans = None

        for i, t in enumerate(self.transformers):
            print(f"fitting transformer {i+1}/{len(self.transformers)}")

            x, x_t, trans_t = t._fit(x, x_t, dlo, chunk_size=chunk_size, **t.fit_kwargs, **fit_kwargs)

            if trans is None:
                trans = trans_t
            else:
                trans *= trans_t

        return x, x_t, trans


class IdentityTransformer(Transformer):
    def _fit(
        self,
        x: list[CV],
        x_t: list[CV] | None,
        dlo: Rounds.data_loader_output,
        chunk_size=None,
        **fit_kwargs,
    ) -> tuple[list[CV], list[CV] | None, CvTrans]:
        return x, x_t, identity_trans
