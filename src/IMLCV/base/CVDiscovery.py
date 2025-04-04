from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Self

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.figure import Figure
from molmod.units import kjmol

from IMLCV.base.bias import Bias, NoneBias
from IMLCV.base.CV import CV, CollectiveVariable, CvFlow, CvMetric, CvTrans, ShmapKwargs
from IMLCV.external.hsluv import hsluv_to_rgb
from IMLCV.implementations.CV import _scale_cv_trans, identity_trans, scale_cv_trans

if TYPE_CHECKING:
    from IMLCV.base.rounds import DataLoaderOutput


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
        dlo: DataLoaderOutput,
        chunk_size=None,
        shmap=True,
        shmap_kwargs=ShmapKwargs.create(),
        verbose=False,
        macro_chunk=10000,
    ) -> tuple[list[CV], list[CV] | None, CvFlow]:
        f = self.descriptor

        x, x_t = dlo.apply_cv(
            f,
            dlo.sp,
            dlo.sp_t,
            dlo.nl,
            dlo.nl_t,
            chunk_size=chunk_size,
            shmap=shmap,
            macro_chunk=macro_chunk,
            verbose=verbose,
            print_every=1,
        )

        if self.pre_scale:
            # todo: change
            g = scale_cv_trans(CV.stack(*x), lower=0, upper=1)
            x, x_t = dlo.apply_cv(
                g,
                x,
                x_t,
                dlo.nl,
                dlo.nl_t,
                chunk_size=chunk_size,
                shmap=shmap,
                verbose=verbose,
                macro_chunk=macro_chunk,
                shmap_kwargs=shmap_kwargs,
            )
            f = f * g

        return x, x_t, f

    @staticmethod
    def static_fit(transformer: Self, **kwargs):
        print("fit Transformer")

        return transformer.fit(**kwargs)

    def fit(
        self,
        dlo: DataLoaderOutput | None,
        chunk_size=None,
        plot=True,
        plot_folder: str | Path | None = None,
        shmap=True,
        percentile=5.0,
        jac=jax.jacrev,
        transform_FES=True,
        max_fes_bias=100 * kjmol,
        n_max=1e5,
        samples_per_bin=20,
        min_samples_per_bin=3,
        verbose=True,
        cv_titles=None,
        vmax=100 * kjmol,
        macro_chunk=1000,
        shmap_kwargs=ShmapKwargs.create(),
        **kwargs,
    ) -> tuple[CV, CollectiveVariable, Bias]:
        if plot:
            assert plot_folder is not None, "plot_folder must be specified if plot=True"

        w = dlo._weights
        rho = dlo._rho

        from IMLCV.base.rounds import DataLoaderOutput

        bias: Bias = dlo.get_fes_bias_from_weights(
            samples_per_bin=samples_per_bin,
            min_samples_per_bin=min_samples_per_bin,
            n_max=n_max,
            max_bias=max_fes_bias,
            macro_chunk=macro_chunk,
            chunk_size=chunk_size,
        )

        if plot:
            Transformer.plot_app(
                name=str(plot_folder / "cvdiscovery_pre_bias.png"),
                collective_variables=[dlo.collective_variable],
                cv_data=None,
                biases=[dlo.ground_bias],
                margin=0.1,
                T=dlo.sti.T,
                plot_FES=True,
                cv_titles=cv_titles,
                vmax=max_fes_bias,
            )

            Transformer.plot_app(
                name=str(plot_folder / "cvdiscovery_pre_data_bias.png"),
                collective_variables=[dlo.collective_variable],
                cv_data=None,
                biases=[bias],
                margin=0.1,
                T=dlo.sti.T,
                plot_FES=True,
                cv_titles=cv_titles,
                vmax=max_fes_bias,
            )

            Transformer.plot_app(
                name=str(plot_folder / "cvdiscovery_pre_data.png"),
                collective_variables=[dlo.collective_variable],
                cv_data=[dlo.cv],
                margin=0.1,
                T=dlo.sti.T,
                plot_FES=True,
                cv_titles=cv_titles,
                vmax=max_fes_bias,
            )

        # koopman_weight

        (w,) = dlo.koopman_weight(
            verbose=verbose,
            n_max_koopman=n_max,
            samples_per_bin=samples_per_bin,
            chunk_size=chunk_size,
            correlation=False,
            koopman_eps=0,
            koopman_eps_pre=0,
            add_1=False,
        )

        bias_km: Bias = dlo._get_fes_bias_from_weights(
            weights=w,
            samples_per_bin=samples_per_bin,
            min_samples_per_bin=min_samples_per_bin,
            n_max=n_max,
            max_bias=max_fes_bias,
            macro_chunk=macro_chunk,
            chunk_size=chunk_size,
        )

        if plot:
            Transformer.plot_app(
                name=str(plot_folder / "cvdiscovery_pre_data_bias_km.png"),
                collective_variables=[dlo.collective_variable],
                cv_data=None,
                biases=[bias_km],
                margin=0.1,
                T=dlo.sti.T,
                plot_FES=True,
                cv_titles=cv_titles,
                vmax=max_fes_bias,
            )

        print("starting pre_fit")

        x, x_t, f = self.pre_fit(
            dlo,
            chunk_size=chunk_size,
            shmap=shmap,
            verbose=verbose,
            macro_chunk=macro_chunk,
            shmap_kwargs=shmap_kwargs,
        )

        print("starting fit")
        x, x_t, g, _ = self._fit(
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
                dlo.nl,
                dlo.nl_t,
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
            from IMLCV.base.rounds import DataLoaderOutput

            bias_new: Bias = DataLoaderOutput.get_fes_bias_from_weights(
                dlo.sti.T,
                weights=w,
                rho=rho,
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
                bias_new.plot(
                    name=str(plot_folder / "transformed_fes.png"),
                    margin=0.1,
                    inverted=False,
                    vmax=max_fes_bias,
                )

        else:
            bias_new = NoneBias.create(new_collective_variable)

        if plot:
            Transformer.plot_app(
                name=str(plot_folder / "cvdiscovery.png"),
                collective_variables=[dlo.collective_variable, new_collective_variable],
                cv_data=[
                    [dlo.cv, x],
                    [dlo.cv, x],
                ],
                biases=[
                    [bias, bias_new],
                    [bias, bias_new],
                ],
                margin=0.1,
                T=dlo.sti.T,
                plot_FES=True,
                cv_titles=cv_titles,
                duplicate_cv_data=False,
                vmax=max_fes_bias,
            )

            Transformer.plot_app(
                name=str(plot_folder / "cvdiscovery_data.png"),
                collective_variables=[dlo.collective_variable, new_collective_variable],
                cv_data=[
                    [dlo.cv, x],
                    [dlo.cv, x],
                ],
                margin=0.1,
                T=dlo.sti.T,
                plot_FES=True,
                cv_titles=cv_titles,
                duplicate_cv_data=False,
                vmax=max_fes_bias,
            )

        return x, new_collective_variable, bias_new

    def _fit(
        self,
        x: list[CV],
        x_t: list[CV] | None,
        w: list[jax.Array],
        dlo: DataLoaderOutput,
        chunk_size=None,
        verbose=True,
        macro_chunk=1000,
        **fit_kwargs,
    ) -> tuple[list[CV], list[CV] | None, CvTrans, list[jax.Array] | None]:
        raise NotImplementedError

    @staticmethod
    def plot_app(
        collective_variables: list[CollectiveVariable],
        cv_data: list[list[CV]] | list[list[list[CV]]] | None = None,
        biases: list[Bias] | list[list[Bias]] | None = None,
        indicate_plots: None | str | list[list[str]] = "lightblue",
        duplicate_cv_data=True,
        name: str | Path | None = None,
        labels=None,
        cv_titles=True,
        data_titles=None,
        color_trajectories=False,
        margin=0.1,
        plot_FES=False,
        T: float | None = None,
        vmin=0,
        vmax=100 * kjmol,
        dpi=300,
        n_max_bias=1e6,
        row_color=None,
        # indicate_cv_data=True,
        macro_chunk=10000,
        cmap="jet",
        offset=True,
        bar_label="FES [kJ/mol]",
    ):
        """Plot the app for the CV discovery. all 1d and 2d plots are plotted directly, 3d or higher are plotted as 2d slices."""

        """data sorted according to data,then cv"""

        assert not (cv_data is None and biases is None), "data or bias must be provided"

        if biases is None:
            plot_FES = False

        ncv = len(collective_variables)

        if duplicate_cv_data:
            if cv_data is not None:
                cv_data = [cv_data] * ncv

            if biases is not None:
                biases = [biases] * ncv

        assert (cv_data is not None) or (biases is not None), "data or bias must be provided"

        # do consistency checks
        ndata = len(cv_data) if cv_data is not None else len(biases)

        if cv_data is not None:
            assert len(cv_data) == ndata, "data must have the same length as cv"

        if biases is not None:
            assert len(biases) == ndata, "bias must have the same length as cv"

        for nd in range(ndata):
            if cv_data is not None:
                assert len(cv_data[nd]) == ncv, "data must have the same length as cv"

            if biases is not None:
                assert len(biases[nd]) == ncv, "bias must have the same length as cv"

        skip = jnp.full((ndata, ncv), False)

        # check if data and bias are None for
        for i in range(ndata):
            for j in range(ncv):
                has_cv_data = False
                if cv_data is not None:
                    if cv_data[i][j] is not None:
                        has_cv_data = True

                has_bias = False
                if biases is not None:
                    if biases[i][j] is not None:
                        has_bias = True

                if (not has_cv_data) and (not has_bias):
                    skip = skip.at[i, j].set(True)

        if plot_FES and biases is not None:
            # assert biases is not None, "bias must be provided if plot_FES=True"

            # assert weight is not None or biases is not None, "bias or weight must be provided if plot_FES=True"
            assert T is not None, "T must be provided if plot_FES=True"

            fesses = []

            for bi in biases:
                fesses_i = []

                for bij in bi:
                    bij: Bias

                    b_slice = (
                        bij.slice(
                            n_max_bias=n_max_bias,
                            T=T,
                            margin=margin,
                            macro_chunk=macro_chunk,
                            offset=offset,
                        )
                        if bij is not None
                        else None
                    )

                    fesses_i.append(b_slice)

                fesses.append(fesses_i)

        metrics = [colvar.metric for colvar in collective_variables]

        if cv_titles is True:
            cv_titles = [f"cv_{i}" for i in range(ncv)]

        if data_titles is None and not duplicate_cv_data:
            data_titles = [f"data_{i}" for i in range(ndata)]

        inoutdims = [collective_variables[n].n for n in range(ncv)]

        print(f"Plotting, dims: {inoutdims} {name if name is not None else ''}")

        plt.rc("text", usetex=False)
        plt.rc("font", family="DejaVu Sans", size=16)

        # change figsize depending on the number of CVs
        # fig = plt.figure(figsize=(6, 6))
        fig = plt.figure()

        if cv_data is not None:
            print("obtaining colors")

            rgb_data = []

            for n in range(ndata):
                if row_color is not None:
                    rc = row_color[n]
                    print(f"data column {n=}, cv row {rc=}")
                else:
                    rc = jnp.min(jnp.array([n, ncv - 1]))

                rgb_out = Transformer._get_color_data(
                    a=cv_data[n][rc],
                    dim=inoutdims[rc],
                    color_trajectories=color_trajectories,
                    metric=metrics[rc],
                    margin=margin,
                ).cv

                rgb_data.append(rgb_out)
            print("done")
        else:
            rgb_data = [None] * ndata

        for data_in, cv_in, axes in Transformer._grid_spec_iterator(
            fig=fig,
            dims=inoutdims,
            ncv=ncv,
            ndata=ndata,
            skip=skip,
            cv_titles=cv_titles,
            data_titles=data_titles,
            indicate_plots=indicate_plots,
            bar_label=bar_label,
            # indicate_cv_data=indicate_cv_data,
            cmap=plt.get_cmap(cmap),
            vmin=vmin,
            vmax=vmax,
        ):
            dim = inoutdims[cv_in]

            data_proc = None
            if cv_data is not None:
                if cv_data[data_in][cv_in] is not None:
                    data_proc = CV.stack(*cv_data[data_in][cv_in]).cv
                    if dim == 1:
                        x = []
                        for i, ai in enumerate(cv_data[data_in][cv_in]):
                            x.append(ai.cv * 0 + i)

                        data_proc = jnp.hstack([data_proc, jnp.vstack(x)])

            if dim == 1:
                f = Transformer._plot_1d
            elif dim == 2:
                f = Transformer._plot_2d
            elif dim == 3:
                f = Transformer._plot_3d
            else:
                print(f"cannot plot {dim=}, skipping")
                continue

            # plot setting
            if cv_data is not None:
                kwargs = {
                    "s": (2000 / data_proc.shape[0]) ** (0.5),
                    "edgecolor": "none",
                }
            else:
                kwargs = {
                    "s": 10,
                    "edgecolor": "none",
                }

            # print(f"{labels}")

            f(
                fig=fig,
                grid=axes,
                data=data_proc,
                colors=rgb_data[data_in] if rgb_data is not None else None,
                labels=(labels[cv_in][0:dim] if labels[cv_in] is not None else "xyzw")
                if labels is not None
                else "xyzw",
                collective_variable=collective_variables[cv_in],
                indices=tuple([i for i in range(dim)]),
                # weight=weight is not None,
                margin=margin,
                fesses=fesses[data_in][cv_in] if plot_FES else None,
                vmax=vmax,
                T=T,
                cmap=plt.get_cmap(cmap),
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
        ncv,
        ndata,
        skip,
        vmin,
        vmax,
        cmap,
        bar_label="FES [kJ/mol]",
        cv_titles=None,
        data_titles=None,
        indicate_plots=None,
        # indicate_cv_data=True,
    ) -> Iterator[tuple[list[int], int, list[plt.Axes], list[plt.Axes]]]:
        spaces = 1 if max(dims) < 3 else 2

        w_tot = [0.3, *[spaces] * ncv, 0.05, 0.1]  # extra 0.05 is to make the title fit
        h_tot = [0.1, *[spaces] * ndata]

        fig.set_figwidth(sum(w_tot) * 3, forward=True)
        fig.set_figheight(sum(h_tot) * 3, forward=True)

        w_space = 0.1
        h_space = 0.1

        spec = fig.add_gridspec(
            nrows=ndata + 1,
            ncols=ncv + 3,
            width_ratios=w_tot,
            height_ratios=h_tot,
            wspace=w_space,
            hspace=h_space,
        )

        norm = mpl.colors.Normalize(vmin=vmin / kjmol, vmax=vmax / kjmol)

        ax_cbar = fig.add_subplot(spec[1:, -2])

        fig.colorbar(
            mappable=mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=ax_cbar,
            orientation="vertical",
            ticks=[vmin / kjmol, (vmin + vmax) / (2 * kjmol), vmax / kjmol],
        )
        ax_cbar.set_ylabel(bar_label)

        for data_in in range(ndata):
            for cv_in in range(ncv):
                if skip[data_in, cv_in]:
                    continue

                yield (
                    data_in,
                    cv_in,
                    spec[data_in + 1, cv_in + 1],
                )

        if indicate_plots is not None:
            # indicate CV data pair
            for i in range(ndata):
                for j in range(ncv):
                    if isinstance(indicate_plots, str):
                        if i == j:
                            c_ij = indicate_plots
                        else:
                            c_ij = None
                    else:
                        c_ij = indicate_plots[i][j]

                    if c_ij is None:
                        continue

                    s = spec[i + 1, j + 1]
                    pos = s.get_position(fig)

                    nrows, ncols = spec.get_geometry()
                    subplot_params = spec.get_subplot_params(fig)
                    left = subplot_params.left
                    right = subplot_params.right
                    bottom = subplot_params.bottom
                    top = subplot_params.top
                    wspace = subplot_params.wspace
                    hspace = subplot_params.hspace
                    tot_width = right - left
                    tot_height = top - bottom

                    # calculate accumulated heights of columns
                    cell_h = tot_height / (nrows + hspace * (nrows - 1))
                    sep_h = hspace * cell_h

                    # # calculate accumulated widths of rows
                    cell_w = tot_width / (ncols + wspace * (ncols - 1))
                    sep_w = wspace * cell_w

                    fig.patches.extend(
                        [
                            plt.Rectangle(
                                (pos.x0 - sep_w, pos.y0 - sep_h),
                                pos.x1 - pos.x0 + sep_w,
                                pos.y1 - pos.y0 + sep_h,
                                fill=True,
                                color=c_ij,
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
            s1 = spec[ndata, 0]
            pos1 = s1.get_position(fig)

            fig.text(
                x=(pos0.x0 + pos1.x1) / 2 - offset,
                y=(pos0.y0 + pos1.y1) / 2,
                s="Data",
                ha="center",
                va="center",
                rotation=90,
            )

        for i in range(ncv):
            s = spec[0, i + 1]
            pos = s.get_position(fig)

            if cv_titles is not None:
                fig.text(
                    x=(pos.x0 + pos.x1) / 2,
                    y=(pos.y0 + pos.y1) / 2,
                    s=cv_titles[i],
                    ha="center",
                    va="center",
                )

        for i in range(ndata):
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
        collective_variable: CollectiveVariable,
        fesses: dict[int, dict[tuple, Bias]] | None = None,
        indices: tuple | None = None,
        margin=None,
        T=None,
        vmin=0,
        vmax=100 * kjmol,
        cmap=plt.get_cmap("jet"),
        **scatter_kwargs,
    ):
        # same as plot 2d but without y axis

        gs = grid.subgridspec(
            ncols=1,
            nrows=2,
            height_ratios=[1, 4],
            wspace=0.1,
            hspace=0.1,
        )

        ax = fig.add_subplot(gs[1, 0])

        # ax.set_aspect('equal')
        ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)

        metric = collective_variable.metric

        m = (metric.bounding_box[:, 1] - metric.bounding_box[:, 0]) * margin

        x_l = metric.bounding_box[0, :]
        m_x = m[0]

        x_lim = [x_l[0] - m_x, x_l[1] + m_x]

        if data is not None:
            y_lim = [jnp.min(data[:, 1]) - 0.5, jnp.max(data[:, 1]) + 0.5]
        else:
            y_lim = [0, 1]

        # create inset
        if data is not None:
            ax.scatter(
                data[:, 0],
                data[:, 1],
                c=colors,
                **scatter_kwargs,
            )

        ax.set_xticks([x_l[0], (x_l[0] + x_l[1]) / 2, x_l[1]])
        ax.set_xticklabels([])

        ax.set_yticks([])
        ax.set_yticklabels([])

        ax.patch.set_alpha(0)

        if fesses is None:
            if data is not None:
                in_xlim = jnp.logical_and(data[:, 0] > x_lim[0], data[:, 0] < x_lim[1])
                n_points = jnp.sum(in_xlim)
                n_bins = n_points // 50

                x_bins = jnp.linspace(x_lim[0], x_lim[1], n_bins + 1)

                bins_x_center = (x_bins[1:] + x_bins[:-1]) / 2

                # raw number of points
                H, _ = jnp.histogramdd(
                    data,
                    bins=n_bins,
                    range=[x_lim, y_lim],
                )

                x_count = jax.vmap(jnp.sum, in_axes=0)(H)
                x_count /= jnp.sum(x_count)

                ax_histx.plot(bins_x_center, x_count, color="tab:blue")

                ax_histx.patch.set_alpha(0)

        else:
            x_range = jnp.linspace(x_lim[0], x_lim[1], 500)

            x_fes, _ = fesses[1][(indices[0],)].compute_from_cv(CV(cv=jnp.array(x_range).reshape((-1, 1))))

            ax_histx.scatter(
                x_range,
                -x_fes / kjmol,
                c=-x_fes / kjmol,
                s=1,
                vmin=vmin / kjmol,
                vmax=vmax / kjmol,
                cmap=cmap,
            )

            ax_histx.set_xlim(x_lim[0], x_lim[1])
            ax_histx.set_ylim(vmin / kjmol, vmax / kjmol)

            ax_histx.patch.set_alpha(0)

        for b in [ax_histx]:
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

        if margin is not None:
            ax.set_xlim(*x_lim)

    @staticmethod
    def _plot_2d(
        fig: Figure,
        grid: gridspec.GridSpec,
        data,
        colors,
        labels,
        collective_variable: CollectiveVariable,
        fesses: dict[int, dict[tuple, Bias]] | None = None,
        indices: tuple | None = None,
        margin=None,
        vmin=0,
        vmax=100 * kjmol,
        T=None,
        print_labels=False,
        cmap=plt.get_cmap("jet"),
        plot_y=True,
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

        metric = collective_variable.metric

        m = (metric.bounding_box[:, 1] - metric.bounding_box[:, 0]) * margin

        x_l = metric.bounding_box[0, :]
        m_x = m[0]
        y_l = metric.bounding_box[1, :]
        m_y = m[1]

        x_lim = [x_l[0] - m_x, x_l[1] + m_x]
        y_lim = [y_l[0] - m_y, y_l[1] + m_y]

        if fesses is not None:
            assert indices is not None, "FES_indices must be provided if FES is provided"

            FES = fesses[2][indices]

        else:
            FES = None

        if FES is not None:
            # print("obtaining 2d fes")

            # cv grid is centered
            bins, _, cv_grid, _ = metric.grid(
                n=100,
                # endpoints=True,
                margin=margin,
                indexing="xy",
            )

            from IMLCV.base.rounds import DataLoaderOutput

            bias = DataLoaderOutput._apply_bias(
                x=[cv_grid],
                bias=FES,
                macro_chunk=1000,
                verbose=False,
                shmap=False,
            )[0]

            # print(f"fes: {bias=}")

            bias = bias.reshape(len(bins[0]) - 1, len(bins[1]) - 1)

            # bias -= jnp.max(bias)

            # imshow shows in middel of image
            _dx = (bins[0][1] - bins[0][0]) / 2
            _dy = (bins[1][1] - bins[1][0]) / 2

            p = ax.imshow(
                -bias / (kjmol),
                cmap=cmap,
                origin="lower",
                extent=[
                    bins[0][0] - _dx,
                    bins[0][-1] + _dx,
                    bins[1][0] - _dx,
                    bins[1][-1] + _dx,
                ],
                vmin=vmin / kjmol,
                vmax=vmax / kjmol,
                aspect="auto",
            )

        if data is not None:
            ax.scatter(
                data[:, 0],
                data[:, 1],
                c=colors,
                **scatter_kwargs,
            )

        if fesses is None:
            if data is not None:
                in_xlim = jnp.logical_and(data[:, 0] > x_lim[0], data[:, 0] < x_lim[1])
                in_ylim = jnp.logical_and(data[:, 1] > y_lim[0], data[:, 1] < y_lim[1])
                n_points = jnp.sum(jnp.logical_and(in_xlim, in_ylim))
                n_bins = n_points // 50

                x_bins = jnp.linspace(x_lim[0], x_lim[1], n_bins + 1)
                y_bins = jnp.linspace(y_lim[0], y_lim[1], n_bins + 1)

                bins_x_center = (x_bins[1:] + x_bins[:-1]) / 2
                bins_y_center = (y_bins[1:] + y_bins[:-1]) / 2

                # raw number of points
                H, _ = jnp.histogramdd(data, bins=n_bins, range=[x_lim, y_lim])

                x_count = jnp.sum(H, axis=1)
                x_count /= jnp.sum(x_count)

                y_count = jnp.sum(H, axis=0)
                y_count /= jnp.sum(y_count)

                ax_histx.plot(bins_x_center, x_count, color="tab:blue")
                ax_histy.plot(y_count, bins_y_center, color="tab:blue")

                ax_histx.patch.set_alpha(0)
                ax_histy.patch.set_alpha(0)

        else:
            x_range = jnp.linspace(x_lim[0], x_lim[1], 500)
            y_range = jnp.linspace(y_lim[0], y_lim[1], 500)

            x_fes, _ = fesses[1][(indices[0],)].compute_from_cv(CV(cv=jnp.array(x_range).reshape((-1, 1))))
            y_fes, _ = fesses[1][(indices[1],)].compute_from_cv(CV(cv=jnp.array(y_range).reshape((-1, 1))))

            ax_histx.scatter(
                x_range,
                -x_fes / kjmol,
                c=-x_fes / kjmol,
                s=1,
                vmin=vmin / kjmol,
                vmax=vmax / kjmol,
                cmap=cmap,
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
                cmap=cmap,
            )

            ax_histy.set_xlim(vmin / kjmol, vmax / kjmol)
            ax_histy.set_ylim(y_lim[0], y_lim[1])

            ax_histx.patch.set_alpha(0)
            ax_histy.patch.set_alpha(0)

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

        # ax.locator_params(nbins=4)
        ax.set_xticks([x_l[0], (x_l[0] + x_l[1]) / 2, x_l[1]])
        ax.set_xticks([y_l[0], (y_l[0] + y_l[1]) / 2, y_l[1]])

        ax.set_xticklabels([])
        ax.set_yticklabels([])

        ax.tick_params(axis="both", length=1)

        ax.patch.set_alpha(0)

        if print_labels and indices is not None and labels is not None:
            ax.set_xlabel(labels[indices[0]], labelpad=-1)
            ax.set_ylabel(labels[indices[1]], labelpad=-1)

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
        collective_variable: CollectiveVariable,
        fesses: dict[int, dict[tuple, Bias]] | None = None,
        indices: tuple | None = None,
        margin=None,
        vmin=0,
        vmax=100 * kjmol,
        T=None,
        cmap=plt.get_cmap("jet"),
        **scatter_kwargs,
    ):
        metric = collective_variable.metric

        gs = grid.subgridspec(
            ncols=2,
            nrows=2,
            wspace=0.01,
            hspace=0.01,
            width_ratios=[1, 1],
            height_ratios=[1, 1],
        )

        m = (metric.bounding_box[:, 1] - metric.bounding_box[:, 0]) * margin

        x_l = metric.bounding_box[0, :]
        m_x = m[0]
        y_l = metric.bounding_box[1, :]
        m_y = m[1]
        z_l = metric.bounding_box[2, :]
        m_z = m[2]

        x_lim = [x_l[0] - m_x, x_l[1] + m_x]
        y_lim = [y_l[0] - m_y, y_l[1] + m_y]
        z_lim = [z_l[0] - m_z, z_l[1] + m_z]

        # ?https://matplotlib.org/3.2.1/gallery/axisartist/demo_floating_axes.html

        ax0 = fig.add_subplot(gs[0, 1], projection="3d")
        ax0.view_init(elev=35, azim=225)

        # create  3d scatter plot with 2D histogram on side
        if data is not None:
            ax0.scatter(
                data[:, 0],
                data[:, 1],
                data[:, 2],
                **scatter_kwargs,
                c=colors,
                zorder=1,
            )

        # scatter grid data bases on bias

        if fesses is not None:
            bias = fesses[3][indices]

            # cmap = plt.get_cmap("jet")

            # print("obtaining 2d fes")

            n_grid = 30

            bins, cv_grid, cv_mid, _ = metric.grid(
                n=n_grid,
                margin=margin,
                indexing="ij",
            )

            from IMLCV.base.rounds import DataLoaderOutput

            bias = DataLoaderOutput._apply_bias(
                x=[cv_mid],
                bias=bias,
                macro_chunk=1000,
                verbose=False,
                shmap=False,
            )[0]

            # bias -= jnp.max(bias)

            normed_val = -(bias - vmin) / (vmax - vmin)

            normed_val = jnp.clip(normed_val, 0, 1)

            in_range = jnp.logical_and(normed_val >= 0, normed_val <= 1)

            color_bias = cmap(normed_val)

            # add alpha channel
            color_bias[:, 3] = jnp.exp(-3 * normed_val) / n_grid  # if all points in row are the same, the alpha is 1

            shm = (len(bins[0]) - 1, len(bins[1]) - 1, len(bins[2]) - 1)
            sh = (len(bins[0]), len(bins[1]), len(bins[2]))

            color_bias = color_bias.reshape((*shm, 4))

            method = "slices"

            # probably need to switch to pyvista for pure volumetric figure

            if method == "slices":
                XYZ = jnp.reshape(cv_mid.cv, (*shm, -1))
                import itertools

                # select 2 axis, figure out the third
                for i, j in itertools.combinations(range(3), 2):
                    xyz = [0, 1, 2]

                    y = xyz.pop(j)
                    x = xyz.pop(i)
                    z = xyz[0]

                    for z_i in range(shm[z]):
                        args = [None, None, None]

                        args[x] = XYZ[:, :, :, x].take(z_i, axis=z)
                        args[y] = XYZ[:, :, :, y].take(z_i, axis=z)
                        args[z] = XYZ[:, :, :, z].take(z_i, axis=z)

                        ax0.plot_surface(
                            *args,
                            rstride=1,
                            cstride=1,
                            facecolors=color_bias.take(z_i, axis=z),
                            edgecolor=None,
                            shade=False,
                        )

            elif method == "voxels":
                import types
                from collections import defaultdict

                import numpy as np

                # from matplotlib.cbook import _backports
                from mpl_toolkits.mplot3d import Axes3D, art3d  # NOQA

                def voxels(self, *args, **kwargs):
                    if len(args) >= 3:
                        # underscores indicate position only
                        def voxels(__x, __y, __z, filled, **kwargs):
                            return (__x, __y, __z), filled, kwargs
                    else:

                        def voxels(filled, **kwargs):
                            return None, filled, kwargs

                    xyz, filled, kwargs = voxels(*args, **kwargs)

                    # check dimensions
                    if filled.ndim != 3:
                        raise ValueError("Argument filled must be 3-dimensional")
                    size = np.array(filled.shape, dtype=np.intp)

                    # check xyz coordinates, which are one larger than the filled shape
                    coord_shape = tuple(size + 1)
                    if xyz is None:
                        x, y, z = np.indices(coord_shape)
                    else:
                        pass
                        x, y, z = (c for c in xyz)

                    def _broadcast_color_arg(color, name):
                        if np.ndim(color) in (0, 1):
                            # single color, like "red" or [1, 0, 0]
                            return color
                        elif np.ndim(color) in (3, 4):
                            # 3D array of strings, or 4D array with last axis rgb
                            if np.shape(color)[:3] != filled.shape:
                                raise ValueError(
                                    "When multidimensional, {} must match the shape of " "filled".format(name)
                                )
                            return color
                        else:
                            raise ValueError("Invalid {} argument".format(name))

                    # intercept the facecolors, handling defaults and broacasting
                    facecolors = kwargs.pop("facecolors", None)
                    if facecolors is None:
                        facecolors = self._get_patches_for_fill.get_next_color()
                    facecolors = _broadcast_color_arg(facecolors, "facecolors")

                    # # broadcast but no default on edgecolors
                    # edgecolors = kwargs.pop("edgecolors", None)
                    # edgecolors = _broadcast_color_arg(edgecolors, "edgecolors")

                    # include possibly occluded internal faces or not
                    internal_faces = kwargs.pop("internal_faces", False)

                    # always scale to the full array, even if the data is only in the center
                    self.auto_scale_xyz(x, y, z)

                    # points lying on corners of a square
                    square = np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=np.intp)

                    voxel_faces = defaultdict(list)

                    def permutation_matrices(n):
                        """Generator of cyclic permutation matices"""
                        mat = np.eye(n, dtype=np.intp)
                        for i in range(n):
                            yield mat
                            mat = np.roll(mat, 1, axis=0)

                    for permute in permutation_matrices(3):
                        pc, qc, rc = permute.T.dot(size)
                        pinds = np.arange(pc)
                        qinds = np.arange(qc)
                        rinds = np.arange(rc)

                        square_rot = square.dot(permute.T)

                        for p in pinds:
                            for q in qinds:
                                p0 = permute.dot([p, q, 0])
                                i0 = tuple(p0)
                                if filled[i0]:
                                    voxel_faces[i0].append(p0 + square_rot)

                                # draw middle faces
                                for r1, r2 in zip(rinds[:-1], rinds[1:]):
                                    p1 = permute.dot([p, q, r1])
                                    p2 = permute.dot([p, q, r2])
                                    i1 = tuple(p1)
                                    i2 = tuple(p2)
                                    if filled[i1] and (internal_faces or not filled[i2]):
                                        voxel_faces[i1].append(p2 + square_rot)
                                    elif (internal_faces or not filled[i1]) and filled[i2]:
                                        voxel_faces[i2].append(p2 + square_rot)

                                # draw upper faces
                                pk = permute.dot([p, q, rc - 1])
                                pk2 = permute.dot([p, q, rc])
                                ik = tuple(pk)
                                if filled[ik]:
                                    voxel_faces[ik].append(pk2 + square_rot)

                    # iterate over the faces, and generate a Poly3DCollection for each voxel
                    polygons = {}
                    for coord, faces_inds in voxel_faces.items():
                        # convert indices into 3D positions
                        if xyz is None:
                            faces = faces_inds
                        else:
                            faces = []
                            for face_inds in faces_inds:
                                ind = face_inds[:, 0], face_inds[:, 1], face_inds[:, 2]
                                face = np.empty(face_inds.shape)
                                face[:, 0] = x[ind]
                                face[:, 1] = y[ind]
                                face[:, 2] = z[ind]
                                faces.append(face)

                        poly = art3d.Poly3DCollection(
                            faces,
                            facecolors=facecolors[coord],
                            edgecolors=None,
                            **kwargs,
                        )
                        self.add_collection3d(poly)
                        polygons[coord] = poly

                    return polygons

                ax0.voxels = types.MethodType(voxels, ax0)

                ax0.voxels(
                    cv_grid.cv[:, 0].reshape(sh),
                    cv_grid.cv[:, 1].reshape(sh),
                    cv_grid.cv[:, 2].reshape(sh),
                    filled=in_range.reshape(shm),
                    facecolors=color_bias.reshape((*shm, 4)),
                    internal_faces=True,
                )

            elif method == "scatter":
                # this works but produces spheres instead of cubes
                ax0.scatter(
                    cv_grid.cv[in_range, 0],
                    cv_grid.cv[in_range, 1],
                    cv_grid.cv[in_range, 2],
                    s=100 * (bias.shape[0]) ** (-1 / 3),  # linear dimension
                    edgecolor="none",
                    c=color_bias,
                    zorder=-1,
                )
            else:
                raise ValueError(f"method {method} not recognized")

        ax0.grid(False)
        ax0.xaxis.pane.fill = False
        ax0.yaxis.pane.fill = False
        ax0.zaxis.pane.fill = False

        ax0.set_xlim(*x_lim)
        ax0.set_ylim(*y_lim)
        ax0.set_zlim(*z_lim)

        ax0.locator_params(nbins=4)

        # ax0.set_xticklabels([])
        # ax0.set_yticklabels([])
        # ax0.set_zticklabels([])

        ax0.tick_params(
            axis="x",
            label1On=False,
            label2On=False,
        )

        ax0.tick_params(
            axis="y",
            label1On=False,
            label2On=False,
        )
        ax0.tick_params(
            axis="z",
            label1On=False,
            label2On=False,
        )

        if labels is not None:
            ax0.set_xlabel(labels[indices[0]], labelpad=-15)
            ax0.set_ylabel(labels[indices[1]], labelpad=-15)
            ax0.set_zlabel(labels[indices[2]], labelpad=-15)

        # ax0.view_init(40, -30, 0)
        ax0.set_box_aspect(None, zoom=1.1)

        ax0.patch.set_alpha(0)

        # plot 2d fesses individually
        from itertools import combinations

        positions = [gs[0, 0], gs[1, 0], gs[1, 1]]

        for i, _indices in enumerate(combinations(indices, 2)):
            idx = jnp.array(_indices)

            # print(f"{idx=}")

            Transformer._plot_2d(
                fig=fig,
                grid=positions[i],
                data=data[:, _indices] if data is not None else None,
                colors=colors,
                labels=labels,
                collective_variable=collective_variable[idx],
                margin=margin,
                indices=_indices,
                fesses=fesses,
                vmin=vmin,
                vmax=vmax,
                print_labels=True,
                cmap=cmap,
                **scatter_kwargs,
            )

    def _get_color_data(
        a: list[CV],
        dim: int,
        color_trajectories=False,
        color_1d=True,
        metric: CvMetric | None = None,
        max_val=None,
        min_val=None,
        margin=None,
    ) -> CV:
        if metric is not None:
            bb = metric.bounding_box
            pp = metric.periodicities
        else:
            bb = None
            pp = None

        if dim == 1:
            if color_1d:
                x = []

                for i, ai in enumerate(a):
                    y = ai.cv * 0 + i

                    x.append(ai.replace(cv=jnp.hstack([ai.cv, jnp.vstack(y)])))

                a = x

                if bb is not None:
                    bb = jnp.vstack([bb, jnp.array([0, len(a)])])
                    pp = jnp.hstack([pp, jnp.array([False])])

            dim = 2

        # print(f"{dim=}")

        if color_trajectories:
            a_out = []

            for ai in a:
                avg = jnp.mean(ai.cv, axis=0, keepdims=True)
                a_out.append(ai.replace(cv=ai.cv * 0 + avg))

            a = a_out

        # color_data = a.cv

        if max_val is not None and min_val is not None:
            pass

        elif metric is not None:
            min_val = bb[:, 0]
            max_val = bb[:, 1]

        else:
            max_val = jnp.max(jnp.array([jnp.max(ai.cv, axis=0) for ai in a]), axis=0)
            min_val = jnp.min(jnp.array([jnp.min(ai.cv, axis=0) for ai in a]), axis=0)

        if margin is not None:
            _max_val = max_val + (max_val - min_val) * margin
            _min_val = min_val - (max_val - min_val) * margin

            if metric is not None:
                max_val = jax.vmap(lambda p, x, y: jnp.where(p, x, y))(pp, max_val, _max_val)
                min_val = jax.vmap(lambda p, x, y: jnp.where(p, x, y))(pp, min_val, _min_val)
            else:
                max_val = _max_val
                min_val = _min_val

        from IMLCV.base.CV import CvTrans

        close = bool(jnp.allclose(max_val, min_val))

        use_macro_chunk = True

        print(close)

        def _f(color_data, nl, c, shmap, shmap_kwargs, close, dim, pp):
            # print(f"{color_data=} {close=} {dim=}")
            if close:
                data_col = color_data.cv
            else:
                data_col = (color_data.cv - min_val) / (max_val - min_val)

            data_col = jnp.clip(data_col, 0.0, 1.0)

            periodicities = jnp.array([False, False, False])

            if pp is not None:
                periodicities = periodicities.at[: pp.shape[0]].set(pp)

            # https://www.hsluv.org/
            # hue 0-360 sat 0-100 lighness 0-100

            if dim == 1:
                lab = jnp.array([data_col[0] * 360, 75, 40])

                per = jnp.array([periodicities[0], False, False])

            elif dim == 2:
                lab = jnp.array([data_col[0] * 360, 75, data_col[1] * 100])
                per = jnp.array([periodicities[0], False, periodicities[1]])

            elif dim == 3:
                lab = jnp.array([data_col[0] * 360, data_col[1] * 100, data_col[2] * 100])

                per = jnp.array([periodicities[0], periodicities[1], periodicities[2]])

            # hue
            lab = lab.at[0].set(
                jnp.where(
                    per[0],
                    lab[0],  # data already periodic
                    lab[0] / 360 * 320 + 20,
                )
            )

            # sat
            lab = lab.at[1].set(
                jnp.where(
                    per[1],
                    60 * (jnp.sin(lab[1] / 100 * 2 * jnp.pi) + 1) / 2 + 20,
                    lab[1] * 0.6 + 20,
                )
            )

            # lightness
            lab = lab.at[2].set(
                jnp.where(
                    per[2],
                    60 * (jnp.sin(lab[2] / 100 * 2 * jnp.pi) + 1) / 2 + 20,
                    lab[2] * 0.6 + 20,
                )
            )

            rgb = hsluv_to_rgb(lab)

            out = color_data.replace(cv=rgb)

            return out

        if use_macro_chunk:
            from IMLCV.base.rounds import DataLoaderOutput

            # print(f"{a=}")

            out, _ = DataLoaderOutput.apply_cv(
                x=a,
                f=CvTrans.from_cv_function(
                    _f,
                    static_argnames=["close", "dim"],
                    close=close,
                    dim=dim,
                    pp=pp,
                ),
                verbose=False,
                macro_chunk=320,
                shmap=False,
            )

        else:
            from jax.tree_util import Partial

            _f = Partial(_f, nl=None, c=None, shmap=None, shmap_kwargs=None, close=close, dim=dim, pp=pp)
            _f = jax.vmap(_f)
            # _f = jax.jit(_f)

            out = []

            for i, ai in enumerate(a):
                out.append(_f(ai))
                if i % 100 == 0:
                    print(f"{i=}/{len(a)}")

        return CV.stack(*out)

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
        dlo: DataLoaderOutput,
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
        dlo: DataLoaderOutput,
        chunk_size=None,
        verbose=True,
        macro_chunk=1000,
        **fit_kwargs,
    ) -> tuple[list[CV], list[CV] | None, CvTrans, list[jax.Array] | None]:
        return x, x_t, identity_trans, w
