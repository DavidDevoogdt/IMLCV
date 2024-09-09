from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Self

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.figure import Figure
from molmod.units import kjmol

from IMLCV.base.bias import Bias, NoneBias
from IMLCV.base.CV import CV, CollectiveVariable, CvFlow, CvMetric, CvTrans
from IMLCV.implementations.CV import _cv_slice, _scale_cv_trans, identity_trans, scale_cv_trans

if TYPE_CHECKING:
    from IMLCV.base.rounds import data_loader_output


class Transformer:
    def __init__(
        self,
        outdim,
        descriptor: CvFlow,
        pre_scale=True,
        post_scale=True,
        T_scale=10,
        **fit_kwargs,
    ) -> None:
        self.outdim = outdim

        self.descriptor = descriptor
        self.pre_scale = pre_scale
        self.post_scale = post_scale
        self.T_scale = T_scale

        self.fit_kwargs = fit_kwargs

    def pre_fit(
        self,
        dlo: data_loader_output,
        chunk_size=None,
        shmap=True,
        verbose=False,
        macro_chunk=10000,
    ) -> tuple[list[CV], list[CV] | None, CvFlow]:
        f = self.descriptor

        x, x_t = dlo.apply_cv(
            f,
            dlo.sp,
            dlo.sp_t,
            chunk_size=chunk_size,
            shmap=shmap,
            macro_chunk=macro_chunk,
            verbose=verbose,
        )

        if self.pre_scale:
            # todo: change
            g = scale_cv_trans(CV.stack(*x), lower=0, upper=1)
            x, x_t = dlo.apply_cv(
                g,
                x,
                x_t,
                chunk_size=chunk_size,
                shmap=shmap,
                verbose=verbose,
                macro_chunk=macro_chunk,
            )
            f = f * g

        return x, x_t, f

    @staticmethod
    def static_fit(transformer: Self, **kwargs):
        print("fit Transformer")

        return transformer.fit(**kwargs)

    def fit(
        self,
        dlo: data_loader_output | None,
        chunk_size=None,
        plot=True,
        plot_folder: str | Path | None = None,
        shmap=True,
        percentile=5.0,
        jac=jax.jacrev,
        transform_FES=True,
        max_fes_bias=100 * kjmol,
        n_max=60,
        samples_per_bin=50,
        min_samples_per_bin=5,
        verbose=True,
        cv_titles=None,
        vmax=100 * kjmol,
        macro_chunk=1000,
        **kwargs,
    ) -> tuple[CV, CollectiveVariable, Bias]:
        if plot:
            assert plot_folder is not None, "plot_folder must be specified if plot=True"

        print("getting weights")
        # koopman does not really make sense because it reinforces the current CV
        w = dlo.weights(
            chunk_size=chunk_size,
            macro_chunk=macro_chunk,
            koopman=False,
            verbose=verbose,
            samples_per_bin=samples_per_bin,
            n_max=n_max,
        )

        if plot:
            Transformer.plot_app(
                name=str(plot_folder / "cvdiscovery_pre.png"),
                collective_variables=[dlo.collective_variable],
                cv_data=[dlo.cv],
                weight=w,
                margin=0.1,
                T=dlo.sti.T,
                plot_FES=True,
                cv_titles=cv_titles,
                vmax=vmax,
                samples_per_bin=samples_per_bin,
                min_samples_per_bin=min_samples_per_bin,
            )

        print("starting pre_fit")

        x, x_t, f = self.pre_fit(
            dlo,
            chunk_size=chunk_size,
            shmap=shmap,
            verbose=verbose,
            macro_chunk=macro_chunk,
        )

        print("starting fit")
        x, x_t, g, w = self._fit(
            x,
            x_t,
            w,
            dlo,
            chunk_size=chunk_size,
            macro_chunk=macro_chunk,
            T_scale=self.T_scale,
            **self.fit_kwargs,
        )

        print("getting bounds")

        # remove outliers from the data
        bounds, mask, constants = CvMetric.bounds_from_cv(
            x,
            percentile=percentile,
            # weights=w,
            macro_chunk=macro_chunk,
            chunk_size=chunk_size,
        )

        assert not constants, "found constant collective variables"

        if self.post_scale:
            print("post scaling")
            trans = CvTrans.from_cv_function(
                _scale_cv_trans,
                upper=1,
                lower=0,
                mini=bounds[:, 0],
                diff=bounds[:, 1] - bounds[:, 0],
            )

            bounds = jnp.zeros_like(bounds)
            bounds = bounds.at[:, 1].set(1)

            x, x_t = dlo.apply_cv(
                trans,
                x,
                x_t,
                chunk_size=chunk_size,
                macro_chunk=macro_chunk,
                shmap=shmap,
                verbose=verbose,
            )

            g *= trans

        new_collective_variable = CollectiveVariable(
            f=f * g,
            jac=jac,
            metric=CvMetric.create(
                periodicities=None,
                bounding_box=bounds,
            ),
        )

        if transform_FES:
            print("transforming FES")
            from IMLCV.base.rounds import data_loader_output

            bias: Bias = data_loader_output._get_fes_bias_from_weights(
                dlo.sti.T,
                weights=w,
                collective_variable=new_collective_variable,
                cv=x,
                samples_per_bin=samples_per_bin,
                min_samples_per_bin=min_samples_per_bin,
                n_max=n_max,
                max_bias=max_fes_bias,
                macro_chunk=macro_chunk,
                chunk_size=chunk_size,
            )

            if plot:
                bias.plot(
                    name=str(plot_folder / "transformed_fes.png"),
                    margin=0.1,
                    inverted=True,
                    vmax=max_fes_bias,
                )

        else:
            bias = NoneBias.create(new_collective_variable)

        if plot:
            Transformer.plot_app(
                name=str(plot_folder / "cvdiscovery.png"),
                collective_variables=[dlo.collective_variable, new_collective_variable],
                cv_data=[dlo.cv, x],
                weight=w,
                margin=0.1,
                T=dlo.sti.T,
                plot_FES=True,
                cv_titles=cv_titles,
                vmax=vmax,
                samples_per_bin=samples_per_bin,
                min_samples_per_bin=min_samples_per_bin,
            )

        return x, new_collective_variable, bias

    def _fit(
        self,
        x: list[CV],
        x_t: list[CV] | None,
        w: list[jax.Array],
        dlo: data_loader_output,
        chunk_size=None,
        verbose=True,
        macro_chunk=1000,
        **fit_kwargs,
    ) -> tuple[list[CV], list[CV] | None, CvTrans, list[jax.Array] | None]:
        raise NotImplementedError

    # def post_fit(self, y: list[CV]) -> tuple[CV, CvTrans]:
    #     # y = CV.stack(*y)
    #     if not self.post_scale:
    #         return y, identity_trans
    #     h = scale_cv_trans(y)
    #     return h.compute_cv_trans(y)[0], h

    @staticmethod
    def plot_app(
        collective_variables: list[CollectiveVariable],
        cv_data: list[list[CV]] | list[list[list[CV]]],
        weight: list[list[jax.Array]] | list[list[list[jax.Array]]] | None = None,
        duplicate_cv_data=True,
        fes_biases: list[Bias] | None = None,
        name: str | Path | None = None,
        labels=None,
        cv_titles=None,
        data_titles=None,
        color_trajectories=False,
        margin=0.1,
        max_points=10000,
        plot_FES=False,
        T: float | None = None,
        vmax=100 * kjmol,
        samples_per_bin=50,
        min_samples_per_bin=5,
        dpi=300,
        n_max_fes=30,
        n_max_fep=100,
    ):
        """Plot the app for the CV discovery. all 1d and 2d plots are plotted directly, 3d or higher are plotted as 2d slices."""

        """data sorted according to data,then cv"""

        ncv = len(collective_variables)

        if duplicate_cv_data:
            cv_data = [cv_data] * ncv
            if weight is not None:
                weight = [weight] * ncv

        # if weight is not None:
        #     weight = [jnp.hstack(w) for w in weight]

        if plot_FES:
            assert weight is not None, "weight must be specified if plot_FES=True"
            assert T is not None, "T must be specified if plot_FES=True"

            from IMLCV.base.rounds import data_loader_output

            fesses = []

            margin_fesses = []

            for j, (wj, cvdj) in enumerate(zip(weight, cv_data)):
                fes_i = []
                margin_fes_i = []

                for i, (cvi, cvdata_i) in enumerate(zip(collective_variables, cvdj)):
                    if cvi.n != 2:
                        fes_i.append(None)
                        margin_fes_i.append(None)
                        continue

                    if fes_biases is not None and i == j:
                        fes_ij = fes_biases[j]
                    else:
                        fes_ij = data_loader_output._get_fes_bias_from_weights(
                            T=T,
                            collective_variable=cvi,
                            cv=cvdata_i,
                            weights=wj,
                            samples_per_bin=samples_per_bin,
                            min_samples_per_bin=min_samples_per_bin,
                            max_bias=vmax,
                            n_max=n_max_fes,
                        )

                    fes_i.append(fes_ij)

                    margin_fes_ij = []

                    for k in range(cvi.n):
                        cv_data_i_k = CV.stack(*cvdata_i)
                        cv_data_i_k = cv_data_i_k.replace(cv=cv_data_i_k.cv[:, k].reshape((-1, 1)))
                        cv_data_i_k = cv_data_i_k.unstack()

                        margin_fes_ij.append(
                            data_loader_output._get_fes_bias_from_weights(
                                T=T,
                                collective_variable=CollectiveVariable(
                                    f=cvi.f
                                    * CvTrans.from_cv_function(
                                        _cv_slice,
                                        indices=jnp.array([k]).reshape((-1,)),
                                    ),
                                    jac=cvi.jac,
                                    metric=CvMetric.create(
                                        periodicities=cvi.metric.periodicities[k : k + 1],
                                        bounding_box=cvi.metric.bounding_box[k : k + 1, :],
                                    ),
                                ),
                                cv=cv_data_i_k,
                                weights=wj,
                                samples_per_bin=samples_per_bin,
                                min_samples_per_bin=min_samples_per_bin,
                                max_bias=vmax,
                                n_max=n_max_fep,
                            )
                        )

                    margin_fes_i.append(margin_fes_ij)

                margin_fesses.append(margin_fes_i)
                fesses.append(fes_i)

        metrics = [colvar.metric for colvar in collective_variables]

        if cv_titles is None:
            cv_titles = [f"cv_{i}" for i in range(ncv)]

        if data_titles is None and not duplicate_cv_data:
            data_titles = [f"data_{i}" for i in range(ncv)]

        if labels is None:
            labels = [
                ["cv_1 [a.u.]", "cv_2 [a.u.]", "cv_3 [a.u.]"],
                ["cv_1 [a.u.]", "cv_2 [a.u.]", "cv_3 [a.u.]"],
            ]

        inoutdims = [cv_data[n][n][0].shape[1] for n in range(ncv)]

        print(f"{inoutdims=}")

        plt.rc("text", usetex=False)
        plt.rc("font", family="DejaVu Sans", size=16)

        fig = plt.figure(figsize=(6, 6))
        rgb_data = [
            Transformer._get_color_data(
                CV.stack(*cv_data[n][n]),
                inoutdims[n],
                color_trajectories,
                metric=metrics[n],
                margin=margin,
            ).cv
            for n in range(ncv)
        ]

        for data_in, in_out, axes, colorbar_spec in Transformer._grid_spec_iterator(
            fig=fig,
            dims=inoutdims,
            cv_titles=cv_titles,
            data_titles=data_titles,
        ):
            dim = inoutdims[in_out]

            data_proc = CV.stack(*cv_data[data_in][in_out]).cv
            if dim == 1:
                x = []
                for i, ai in enumerate(cv_data[data_in][in_out]):
                    x.append(ai.cv * 0 + i)

                data_proc = jnp.hstack([data_proc, jnp.vstack(x)])

                f = Transformer._plot_1d
            elif dim == 2:
                f = Transformer._plot_2d
            elif dim == 3:
                f = Transformer._plot_3d

            # plot setting
            kwargs = {
                "s": (100 / data_proc.shape[0]) ** (0.5),
                "edgecolor": "none",
            }

            f(
                fig=fig,
                grid=axes,
                data=data_proc,
                colors=rgb_data[data_in],
                labels=labels[0:dim],
                metric=metrics[in_out],
                weight=weight is not None,
                margin=margin,
                plot_FES=plot_FES and fesses[data_in][in_out] is not None,
                FES=fesses[data_in][in_out] if plot_FES else None,
                margin_fes=margin_fesses[data_in][in_out] if plot_FES else None,
                vmax=vmax,
                colorbar_spec=colorbar_spec,
                T=T,
                **kwargs,
            )

        if name is None:
            plt.show()
        else:
            name = Path(name)

            if (name.suffix != ".pdf") and (name.suffix != ".png"):
                print(f"{name.suffix} should be pdf or png, changing to pdf")

                name = Path(
                    f"{name}.png",
                )

            name.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(name, dpi=dpi)

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
            ncols=len(dims) + 2,
            width_ratios=[0.3, *widths, 0.5],
            height_ratios=[0.1, *[1 for _ in dims]],
            wspace=0.1,
            hspace=0.1,
        )

        for data_in in range(len(dims)):
            for cv_in in range(len(dims)):
                yield (
                    data_in,
                    cv_in,
                    spec[data_in + 1, cv_in + 1],
                    spec[1:, -1] if (data_in == 0 and cv_in == 0) else None,
                )

        # indicate CV data pair
        for i in range(1, len(dims) + 1):
            s = spec[i, i]
            pos = s.get_position(fig)

            fig.patches.extend(
                [
                    plt.Rectangle(
                        (pos.x0, pos.y0),
                        pos.x1 - pos.x0,
                        pos.y1 - pos.y0,
                        fill=True,
                        color="lightblue",
                        zorder=-10,
                        transform=fig.transFigure,
                        figure=fig,
                    )
                ]
            )

        # indicate discovered CV
        if data_titles is not None:
            for i in range(1, len(dims)):
                if not isinstance(data_titles[i], int):
                    continue

                if not isinstance(data_titles[i - 1], int):
                    continue

                if data_titles[i] == data_titles[i - 1] + 1:
                    s = spec[i, i + 1]
                    pos = s.get_position(fig)

                    fig.patches.extend(
                        [
                            plt.Rectangle(
                                (pos.x0, pos.y0),
                                pos.x1 - pos.x0,
                                pos.y1 - pos.y0,
                                fill=True,
                                color="lightgreen",
                                zorder=-10,
                                transform=fig.transFigure,
                                figure=fig,
                            )
                        ]
                    )

        offset = 0.05

        s0 = spec[0, 1]
        pos0 = s0.get_position(fig)
        s1 = spec[0, len(dims)]
        pos1 = s1.get_position(fig)

        fig.text(
            x=(pos0.x0 + pos1.x1) / 2,
            y=(pos0.y0 + pos1.y1) / 2 + offset,
            s="Collective Variables",
            ha="center",
            va="center",
        )

        if data_titles is not None:
            s0 = spec[1, 0]
            pos0 = s0.get_position(fig)
            s1 = spec[len(dims), 0]
            pos1 = s1.get_position(fig)

            fig.text(
                x=(pos0.x0 + pos1.x1) / 2 - offset,
                y=(pos0.y0 + pos1.y1) / 2,
                s="Data",
                ha="center",
                va="center",
                rotation=90,
            )

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
        grid: gridspec.GridSpec,
        data,
        colors,
        labels,
        metric: CvMetric,
        weight=None,
        margin=None,
        plot_FES=True,
        FES: Bias | None = None,
        margin_fes: list[Bias] | None = None,
        colorbar_spec=None,
        T=None,
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
        grid: gridspec.GridSpec,
        data,
        colors,
        labels,
        metric: CvMetric,
        margin=None,
        weight=False,
        plot_FES=True,
        FES: Bias | None = None,
        margin_fes: list[Bias] | None = None,
        vmin=0,
        vmax=100 * kjmol,
        colorbar_spec: gridspec.GridSpec | None = None,
        T=None,
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

        m = (metric.bounding_box[:, 1] - metric.bounding_box[:, 0]) * margin

        x_l = metric.bounding_box[0, :]
        m_x = m[0]
        y_l = metric.bounding_box[1, :]
        m_y = m[1]

        x_lim = [x_l[0] - m_x, x_l[1] + m_x]
        y_lim = [y_l[0] - m_y, y_l[1] + m_y]

        if plot_FES:
            assert FES is not None, "FES must be specified if plot_FES=True"

            bins, cv_grid, _, _ = metric.grid(
                n=80,
                endpoints=True,
                margin=margin,
                indexing="xy",
            )

            bias, _ = FES.compute_from_cv(cv_grid)

            bias = bias.reshape(len(bins[0]), len(bins[1]))

            bias -= jnp.max(bias)

            p = ax.imshow(
                -bias / (kjmol),
                cmap=plt.get_cmap("jet"),
                origin="lower",
                extent=[bins[0][0], bins[0][-1], bins[1][0], bins[1][-1]],
                vmin=vmin / kjmol,
                vmax=vmax / kjmol,
                aspect="auto",
            )

            if colorbar_spec is not None:
                cb_gs = colorbar_spec.subgridspec(
                    ncols=2,
                    nrows=3,
                    width_ratios=[1, 2],
                    height_ratios=[0.25, 0.5, 0.25],
                    wspace=0.02,
                    hspace=0.02,
                )

                ax_cbar = fig.add_subplot(cb_gs[1, 0])

                fig.colorbar(
                    p,
                    orientation="vertical",
                    cax=ax_cbar,
                    ticks=[vmin / kjmol, (vmin + vmax) / (2 * kjmol), vmax / kjmol],
                )
                ax_cbar.set_ylabel("Free Energy [kJ/mol]")

        ax.scatter(
            *[data[:, col] for col in range(2)],
            c=colors,
            **scatter_kwargs,
        )

        in_xlim = jnp.logical_and(data[:, 0] > x_lim[0], data[:, 0] < x_lim[1])
        in_ylim = jnp.logical_and(data[:, 1] > y_lim[0], data[:, 1] < y_lim[1])
        n_points = jnp.sum(jnp.logical_and(in_xlim, in_ylim))
        n_bins = n_points // 50

        x_bins = jnp.linspace(x_lim[0], x_lim[1], n_bins + 1)
        y_bins = jnp.linspace(y_lim[0], y_lim[1], n_bins + 1)

        bins_x_center = (x_bins[1:] + x_bins[:-1]) / 2
        bins_y_center = (y_bins[1:] + y_bins[:-1]) / 2

        if not weight:
            # raw number of points
            H, _ = jnp.histogramdd(data, bins=n_bins, range=[x_lim, y_lim])

            x_count = jnp.sum(H, axis=1)
            x_count /= jnp.sum(x_count)

            y_count = jnp.sum(H, axis=0)
            y_count /= jnp.sum(y_count)

            ax_histx.plot(bins_x_center, x_count, color="tab:blue")
            ax_histy.plot(y_count, bins_y_center, color="tab:blue")

        else:
            if margin_fes is not None:
                x_range = jnp.linspace(x_lim[0], x_lim[1], 500)
                y_range = jnp.linspace(y_lim[0], y_lim[1], 500)

                x_fes, _ = margin_fes[0].compute_from_cv(CV(cv=jnp.array(x_range).reshape((-1, 1))))
                y_fes, _ = margin_fes[1].compute_from_cv(CV(cv=jnp.array(y_range).reshape((-1, 1))))

                ax_histx.scatter(
                    x_range,
                    -x_fes / kjmol,
                    c=-x_fes / kjmol,
                    s=1,
                    vmin=vmin / kjmol,
                    vmax=vmax / kjmol,
                    cmap=plt.get_cmap("jet"),
                )

                ax_histx.set_xlim(x_lim[0], x_lim[1])
                ax_histx.set_ylim(vmin / kjmol, vmax / kjmol)

                ax_histy.scatter(
                    -y_fes / kjmol,
                    y_range,
                    c=-y_fes / kjmol,
                    s=1,
                    vmin=vmin / kjmol,
                    vmax=vmax / kjmol,
                    cmap=plt.get_cmap("jet"),
                )

                ax_histy.set_xlim(vmin / kjmol, vmax / kjmol)
                ax_histy.set_ylim(y_lim[0], y_lim[1])

            ax_histx.patch.set_alpha(0)
            ax_histy.patch.set_alpha(0)

            # ax_histx.set_ylim(0, 1)
            # ax_histy.set_xlim(0, 1)

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

        ax.tick_params(axis="both", length=1)

        if margin is not None:
            ax.set_xlim(*x_lim)
            ax.set_ylim(*y_lim)

    @staticmethod
    def _plot_3d(
        fig: Figure,
        grid: gridspec.GridSpec,
        data,
        colors,
        labels,
        metric: CvMetric,
        weight=None,
        margin=None,
        plot_FES=True,
        FES: Bias | None = None,
        margin_fes: list[Bias] | None = None,
        colorbar_spec: gridspec.GridSpec | None = None,
        T=None,
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
                "facecolors": plt.cm.jet(Z),
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

        from hsluv import hsluv_to_rgb

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
        w: list[jax.Array],
        dlo: data_loader_output,
        chunk_size=None,
        verbose=True,
        macro_chunk=1000,
        **fit_kwargs,
    ) -> tuple[list[CV], list[CV] | None, CvTrans]:
        trans = None

        for i, t in enumerate(self.transformers):
            print(f"fitting transformer {i+1}/{len(self.transformers)}")

            x, x_t, trans_t, _ = t._fit(
                x,
                x_t,
                dlo,
                chunk_size=chunk_size,
                verbose=verbose,
                macro_chunk=macro_chunk,
                **t.fit_kwargs,
                **fit_kwargs,
            )

            if trans is None:
                trans = trans_t
            else:
                trans *= trans_t

        return x, x_t, trans, w


class IdentityTransformer(Transformer):
    def _fit(
        self,
        x: list[CV],
        x_t: list[CV] | None,
        w: list[jax.Array],
        dlo: data_loader_output,
        chunk_size=None,
        verbose=True,
        macro_chunk=1000,
        **fit_kwargs,
    ) -> tuple[list[CV], list[CV] | None, CvTrans, list[jax.Array] | None]:
        return x, x_t, identity_trans, w
