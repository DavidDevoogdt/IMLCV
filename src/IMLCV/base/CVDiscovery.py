from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
import matplotlib

# import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from IMLCV.base.bias import Bias, GridBias, NoneBias, StdBias
from IMLCV.base.CV import CV, CollectiveVariable, CvMetric, CvTrans, ShmapKwargs, SystemParams
from IMLCV.base.datastructures import MyPyTreeNode, Partial_decorator, vmap_decorator
from IMLCV.base.MdEngine import TrajectoryInfo
from IMLCV.base.UnitsConstants import kelvin, kjmol
from IMLCV.external.hsluv import hsluv_to_rgb
from IMLCV.implementations.CV import _scale_cv_trans, eigh_rot, identity_trans, scale_cv_trans

if TYPE_CHECKING:
    from IMLCV.base.rounds import DataLoaderOutput


class Transformer(MyPyTreeNode):
    descriptor: CvTrans | None = None
    pre_scale: bool = True
    post_scale: bool = True
    pass_trans: bool = False

    def pre_fit(
        self,
        dlo: DataLoaderOutput,
        chunk_size=None,
        shmap=True,
        shmap_kwargs=ShmapKwargs.create(),
        verbose=False,
        macro_chunk=10000,
    ):
        f = self.descriptor

        if f is not None:
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

        else:
            print("skipping pre fit")

            x, x_t, f = dlo.sp, dlo.sp_t, None

        return x, x_t, f

    @staticmethod
    def static_fit(transformer: Transformer, **kwargs):
        print("fit Transformer")

        return transformer.fit(**kwargs)

    def fit(
        self,
        dlo: DataLoaderOutput,
        chunk_size: int | None = None,
        plot=True,
        plot_folder: str | Path | None = None,
        shmap=True,
        percentile=1.0,
        margin=0.1,
        jac=jax.jacrev,
        transform_FES=True,
        koopman=True,
        max_fes_bias: float | None = None,
        n_max=1e5,
        samples_per_bin=5,
        min_samples_per_bin=1,
        verbose=True,
        cv_titles: list[str] | bool = True,
        vmax=100 * kjmol,
        macro_chunk=1000,
        shmap_kwargs=ShmapKwargs.create(),
        n_max_lin: int = 100,
        **kwargs,
    ) -> tuple[list[CV], CollectiveVariable, Bias, list[jax.Array]]:
        if plot:
            assert plot_folder is not None, "plot_folder must be specified if plot=True"

        assert dlo._weights is not None
        assert dlo._rho is not None

        # assert dlo._weights_t is not None
        # assert dlo._rho_t is not None

        w = dlo._weights
        rho = dlo._rho

        # w_t = dlo._weights_t
        # rho_t = dlo._rho_t

        print(f"{percentile=}, {margin=}")

        print("computing bias")

        from IMLCV.base.rounds import DataLoaderOutput

        if plot:
            bias, _, _, _ = dlo.get_fes_bias_from_weights(
                samples_per_bin=samples_per_bin,
                min_samples_per_bin=min_samples_per_bin,
                n_max=n_max,
                n_max_lin=n_max_lin,
                max_bias=max_fes_bias,
                macro_chunk=macro_chunk,
                chunk_size=chunk_size,
                recalc_bounds=False,
                smoothing=None,
            )

            Transformer.plot_app(
                name=str(plot_folder / "cvdiscovery_pre_data_bias.png"),  # type: ignore
                collective_variables=[dlo.collective_variable],
                cv_data=None,
                biases=[bias],
                margin=0.1,
                T=dlo.sti.T,
                plot_FES=True,
                cv_titles=cv_titles,
                vmax=vmax,
            )

            Transformer.plot_app(
                name=str(plot_folder / "cvdiscovery_pre_data.png"),  # type: ignore
                collective_variables=[dlo.collective_variable],
                cv_data=[dlo.cv],
                margin=0.1,
                T=dlo.sti.T,
                plot_FES=True,
                cv_titles=cv_titles,
                vmax=vmax,
            )

        # koopman_weight

        if koopman:
            w, wt, _, _ = dlo.koopman_weight(
                verbose=verbose,
                max_bins=n_max,
                samples_per_bin=samples_per_bin,
                chunk_size=chunk_size,
                # correlation=False,
                koopman_eps=0,
                koopman_eps_pre=0,
                # add_1=True,
            )  # type: ignore

            w: list[jax.Array]
            # w_t: list[jax.Array]

            bias_km, _, _, _ = dlo.get_fes_bias_from_weights(
                weights=w,
                samples_per_bin=samples_per_bin,
                min_samples_per_bin=min_samples_per_bin,
                n_max=n_max,
                n_max_lin=n_max_lin,
                max_bias=max_fes_bias,
                macro_chunk=macro_chunk,
                chunk_size=chunk_size,
                recalc_bounds=False,
                smoothing=None,
            )

            if plot:
                Transformer.plot_app(
                    name=str(plot_folder / "cvdiscovery_pre_data_bias_km.png"),  # type: ignore
                    collective_variables=[dlo.collective_variable],
                    cv_data=None,
                    biases=[bias_km],
                    margin=0.1,
                    T=dlo.sti.T,
                    plot_FES=True,
                    cv_titles=cv_titles,
                    vmax=vmax,
                )

        print("starting pre_fit")

        x, x_t, f = self.pre_fit(
            dlo,
            chunk_size=chunk_size,
            shmap=shmap,
            verbose=verbose,
            macro_chunk=macro_chunk,
            shmap_kwargs=shmap_kwargs,
        )  # type: ignore

        # x: list[CV]
        # x_t: list[CV]

        trans = f

        print("starting fit")
        x, x_t, g, w, extra_info, periodicities = self._fit(
            x=x,
            x_t=x_t,
            w=w,
            dlo=dlo,
            chunk_size=chunk_size,
            macro_chunk=macro_chunk,
        )  # type:ignore

        assert w is not None

        if trans is None:
            trans = g
        else:
            trans *= g

        print("getting bounds")

        # # first: rotate
        # tr_rot = eigh_rot(x, dlo._rho)

        # x, x_t = dlo.apply_cv(
        #     tr_rot,
        #     x,
        #     x_t,
        #     dlo.nl,
        #     dlo.nl_t,
        #     chunk_size=chunk_size,
        #     macro_chunk=macro_chunk,
        #     shmap=shmap,
        #     verbose=verbose,
        # )  # type: ignore

        # trans *= tr_rot

        # remove outliers from the data
        bounds, mask, constants = CvMetric.bounds_from_cv(
            x,
            percentile=percentile,
            margin=margin,
            # margin=None,
            weights=w,
            rho=rho,
            macro_chunk=macro_chunk,
            chunk_size=chunk_size,
        )

        if periodicities is not None:
            bounds = jnp.where(
                jnp.array(periodicities)[:, None],
                jnp.array([[-jnp.pi, jnp.pi]] * len(periodicities)),
                bounds,
            )

        print(f"pre {bounds=} {periodicities=}")

        assert not jnp.any(constants), "found constant collective variables"

        if self.post_scale:
            print("post scaling")
            s_trans = CvTrans.from_cv_function(
                _scale_cv_trans,
                upper=1.0,
                lower=0.0,
                mini=jnp.where(periodicities, 0.0, bounds[:, 0]) if periodicities is not None else bounds[:, 0],
                diff=jnp.where(periodicities, 1.0, bounds[:, 1] - bounds[:, 0])
                if periodicities is not None
                else bounds[:, 1] - bounds[:, 0],
            )

            bounds_unit = jnp.zeros_like(bounds)
            bounds_unit = bounds_unit.at[:, 1].set(1)

            if periodicities is not None:
                bounds = jnp.where(
                    jnp.array(periodicities)[:, None],
                    jnp.array([[-jnp.pi, jnp.pi]] * len(periodicities)),
                    bounds_unit,
                )
            else:
                bounds = bounds_unit

            x, x_t = dlo.apply_cv(
                s_trans,
                x,
                x_t,
                dlo.nl,
                dlo.nl_t,
                chunk_size=chunk_size,
                macro_chunk=macro_chunk,
                shmap=shmap,
                verbose=verbose,
            )  # type: ignore

            trans *= s_trans

        print(f"post {bounds=}")

        new_collective_variable = CollectiveVariable(
            f=trans,
            jac=jac,
            metric=CvMetric.create(
                periodicities=periodicities,
                bounding_box=bounds,
            ),
            name=cv_titles[1] if isinstance(cv_titles, list) else "",
            extra_info=extra_info,
        )

        print(f"{periodicities=}")

        if transform_FES:
            print("transforming FES")
            from IMLCV.base.rounds import DataLoaderOutput

            bias_new, _, _, _ = DataLoaderOutput._get_fes_bias_from_weights(
                dlo.sti.T,
                weights=w,
                rho=rho,
                collective_variable=new_collective_variable,
                cv=x,
                samples_per_bin=samples_per_bin,
                min_samples_per_bin=min_samples_per_bin,
                n_max=n_max,
                n_max_lin=n_max_lin,
                max_bias=max_fes_bias,
                macro_chunk=macro_chunk,
                chunk_size=chunk_size,
                # recalc_bounds=True,
                bounds=bounds,
                smoothing=None,
            )

            if plot:
                bias_new.plot(
                    name=str(plot_folder / "transformed_fes.png"),  # type: ignore
                    margin=0.1,
                    inverted=False,
                    vmax=vmax,
                    cv_title=cv_titles[1] if isinstance(cv_titles, list) else cv_titles,
                    data_title=False,
                )

        else:
            bias_new = NoneBias.create(new_collective_variable)

        if plot:
            Transformer.plot_app(
                name=str(plot_folder / "cvdiscovery.png"),  # type: ignore
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
                data_titles=None,
                duplicate_cv_data=False,
                vmax=vmax,
            )

            Transformer.plot_app(
                name=str(plot_folder / "cvdiscovery_data.png"),  # type: ignore
                collective_variables=[dlo.collective_variable, new_collective_variable],
                cv_data=[
                    [dlo.cv, x],
                    [dlo.cv, x],
                ],
                margin=0.1,
                T=dlo.sti.T,
                plot_FES=True,
                cv_titles=cv_titles,
                data_titles=None,
                duplicate_cv_data=False,
                vmax=vmax,
            )

        return x, new_collective_variable, bias_new, w

    def _fit(
        self,
        x: list[CV] | list[SystemParams],
        x_t: list[CV] | list[SystemParams] | None,
        w: list[jax.Array],
        # w_t: list[jax.Array],
        dlo: DataLoaderOutput,
        chunk_size: int | None = None,
        verbose=True,
        macro_chunk=1000,
        # **fit_kwargs,
    ) -> tuple[list[CV], list[CV], CvTrans, list[jax.Array] | None, tuple[str, ...] | None, jax.Array | None]:
        raise NotImplementedError

    @staticmethod
    def plot_app(
        collective_variables: list[CollectiveVariable],
        cv_data: list[list[CV]] | list[list[list[CV]]] | None = None,
        biases: list[Bias] | list[list[Bias]] | None = None,
        indicate_plots: None | str | list[list[str | None]] = "lightblue",
        duplicate_cv_data=True,
        name: str | Path | None = None,
        labels=None,
        cv_titles: bool | list[str] = True,
        data_titles: list[str] | None = None,
        color_trajectories=False,
        margin=0.1,
        plot_FES=False,
        T: float | None = None,
        vmin: float | list[float] = 0,
        vmax: float | list[float] = 100 * kjmol,
        dpi=300,
        n_max_bias=1e6,
        row_color: list[int] | jax.Array | None = None,
        plot_std: bool = False,
        # indicate_cv_data=True,
        macro_chunk=10000,
        cmap="viridis",
        offset=True,
        bar_label="FES [kJ/mol]",
        title="",
    ):
        """Plot the app for the CV discovery. all 1d and 2d plots are plotted directly, 3d or higher are plotted as 2d slices."""

        """data sorted according to data,then cv"""

        assert not (cv_data is None and biases is None), "data or bias must be provided"

        if biases is None:
            plot_FES = False

        ncv = len(collective_variables)

        if duplicate_cv_data:
            if cv_data is not None:
                cv_data = [cv_data] * ncv  # type:ignore

            if biases is not None:
                biases = [biases] * ncv  # type:ignore

        assert (cv_data is not None) or (biases is not None), "data or bias must be provided"

        # do consistency checks
        ndata = len(cv_data) if cv_data is not None else len(biases)  # type:ignore

        if cv_data is not None:
            assert len(cv_data) == ndata, "data must have the same length as cv"

        if biases is not None:
            assert len(biases) == ndata, "bias must have the same length as cv"

        for nd in range(ndata):
            if cv_data is not None:
                assert len(cv_data[nd]) == ncv, "data must have the same length as cv"

            if biases is not None:
                assert len(biases[nd]) == ncv, "bias must have the same length as cv"  # type:ignore

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
                    if biases[i][j] is not None:  # type:ignore
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

                for bij in bi:  # type: ignore
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
            cv_titles = [colvar.name for colvar in collective_variables]

        # if data_titles is None and not duplicate_cv_data:
        #     data_titles = [f"data_{i}" for i in range(ndata)]

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
                    a=cv_data[n][rc],  # type: ignore
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
            plot_FES=plot_FES,
            title=title,
        ):
            dim = inoutdims[cv_in]

            if isinstance(vmin, list):
                _vmin = vmin[data_in]
            else:
                _vmin = vmin

            if isinstance(vmax, list):
                _vmax = vmax[data_in]
            else:
                _vmax = vmax

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
                assert data_proc is not None
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

            if labels is None and collective_variables[cv_in].cvs_name is not None:
                _labels = collective_variables[cv_in].cvs_name
                print(f"using cv names as labels: {labels}")
            elif labels is not None:
                _labels = labels[cv_in]
            else:
                _labels = None

            f(
                fig=fig,
                grid=axes,  # type: ignore
                data=data_proc,
                colors=rgb_data[data_in] if rgb_data is not None else None,
                labels=_labels,
                collective_variable=collective_variables[cv_in],
                indices=tuple([i for i in range(dim)]),
                # weight=weight is not None,
                margin=margin,
                fesses=fesses[data_in][cv_in] if plot_FES else None,
                vmax=_vmax,
                vmin=_vmin,
                T=T,
                cmap=plt.get_cmap(cmap),
                plot_std=plot_std,
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
    def plot_CV(
        collective_variable_projection: CollectiveVariable,
        collective_variables: list[CollectiveVariable],
        cv_data: list[list[CV]],
        sp_data: list[list[SystemParams]],
        weights: list[list[jax.Array]] | None = None,
        name: str | Path | None = None,
        timescales: list[list[float]] | None = None,
        projection_cv_title: str | None = None,
        cv_titles: list[str] | None = None,
        margin=0.1,
        plot_FES=False,
        T=300 * kelvin,
        vmin: float | list[float] = 0,
        vmax: float | list[float] = 100 * kjmol,
        dpi=300,
        n_max_bias=1e6,
        macro_chunk=1000,
        cmap_fes="viridis",
        cmap_dens="hot",
        cmap_data="cividis",
        vmax_dens=3,
        vmin_dens=-1,
        offset=True,
        bar_label="Free Energy [kJ/mol]",
        title="Collective Variables",
        fontsize_small=16,
        fontsize_large=20,
        grid_bias_order=0,
        plot_density=True,
        rbf_bias=False,
        extra_info_title: str | None = None,
        get_fes_bias_kwargs={},
    ):
        # data_converted = []

        from IMLCV.base.rounds import DataLoaderOutput
        from IMLCV.implementations.CV import _cv_slice

        dim_map = [
            [(0,)],
            [(0, 1)],
            [(0, 1), (1, 2), (0, 2)],
            [(0, 1), (2, 3)],
        ]

        collective_variables = [a.replace(cvs_name=[f"$q_{i}$" for i in range(a.n)]) for a in collective_variables]

        assert len(collective_variables) == len(cv_data), "cv_data must have the same length as collective_variables"

        proj_plots = dim_map[collective_variable_projection.n - 1]
        n_proj_plots = len(proj_plots) + 1  # +1: always add small space to put cv infomration
        n_colvars = [colvar.n for colvar in collective_variables]
        # n_colvars = [len(cvp) for cvp in colvars_plots]
        max_n_colvar = max(n_colvars)

        # figure out size on right hand side to plot original FES and density

        n_needed = 1

        for i in range(len(collective_variables)):
            ni = dim_map[collective_variables[i].n - 1]
            a = jnp.ceil(len(ni) / (n_proj_plots - 1))
            if a > n_needed:
                n_needed = int(a)

        print(f"{n_needed=}")

        size_square = 3

        nrows = len(collective_variables) * n_proj_plots + 2
        ncols = max_n_colvar + 6 + n_needed  # one for title, 2 for FES and dens,  1 for colorbar
        width_rows = [size_square] * 2 + [0.5] + [size_square] * (max_n_colvar) + [2.0] * 3 + [size_square] * (n_needed)
        height_rows = [0.5] + [0.1] + ([0.1] + [size_square] * (n_proj_plots - 1)) * len(collective_variables)

        rhs = max_n_colvar + 6

        # make figure and gridspeciterator
        fig = plt.figure(
            figsize=(
                sum(width_rows),
                sum(height_rows),
            )
        )

        wspace = 0.3
        hspace = 0.3

        gs = gridspec.GridSpec(
            nrows=nrows,
            ncols=ncols,
            figure=fig,
            width_ratios=width_rows,
            height_ratios=height_rows,
            wspace=wspace,
            hspace=hspace,
        )

        print(f"{gs[0, 0].get_position(fig)=} , {gs[0, 1].get_position(fig)=} {gs[1, 0].get_position(fig)=}")

        wspace = gs[0, 1].get_position(fig).x0 - gs[0, 0].get_position(fig).x1
        hspace = gs[0, 0].get_position(fig).y1 - gs[1, 0].get_position(fig).y0

        print(f"{wspace=}, {hspace=}")

        # raise

        # plot all the projected data

        for i in range(len(collective_variables)):
            print(f"{i=}")

            cv_data_i = cv_data[i]
            col_var_i = collective_variables[i]
            sp_data_i = sp_data[i]

            # print(f"{cv_data_i=}, {col_var_i=}")

            if weights is not None:
                weights_i = weights[i]
            else:
                weights_i = [jnp.ones(len(ci.cv)) for ci in cv_data_i]

            cv_proj, _ = DataLoaderOutput.apply_cv(
                f=collective_variable_projection.f,
                x=sp_data_i,
                macro_chunk=macro_chunk,
                verbose=True,
            )

            # print(f"{cv_proj=}")

            for i_n, n_proj_idx in enumerate(proj_plots):
                print(f"{i_n=} {n_proj_idx} ")

                cv_proj_idx, _ = DataLoaderOutput.apply_cv(
                    f=CvTrans.from_cv_function(
                        _cv_slice,
                        indices=jnp.array(n_proj_idx),
                    ),
                    x=cv_proj,
                    macro_chunk=macro_chunk,
                )

                # print(f"{n_proj_idx=}")

                col_var_idx = collective_variable_projection[n_proj_idx]

                # print(f"{col_var_idx.metric=}")

                colors = []

                for j in range(col_var_i.n):
                    indices = (j,)

                    print(f"{j=} ")

                    colvar_i_slice = col_var_i[indices]

                    cv_slice, _ = DataLoaderOutput.apply_cv(
                        f=CvTrans.from_cv_function(
                            _cv_slice,
                            indices=jnp.array(indices),
                        ),
                        x=cv_data_i,
                    )

                    colors.append(cv_slice)

                if len(colors) == 1:
                    color = colors[0]
                else:
                    color = list(map(lambda x: CV.combine(*x), zip(*colors)))

                # print(f"{cv_proj_idx[0].shape=}")

                fes, _, dens, color = DataLoaderOutput._get_fes_bias_from_weights(
                    T=T,
                    weights=weights_i,
                    rho=[jnp.ones_like(w) / w.shape[0] for w in weights_i],
                    collective_variable=col_var_idx,
                    cv=cv_proj_idx,
                    samples_per_bin=5,
                    min_samples_per_bin=1,
                    n_max=n_max_bias,
                    max_bias=None,
                    macro_chunk=macro_chunk,
                    chunk_size=None,
                    recalc_bounds=False,
                    output_density_bias=True,
                    rbf_bias=rbf_bias,
                    observable=color,
                    grid_bias_order=grid_bias_order,
                    smoothing=1,
                )

                # print(f"{T=} {vmin/kjmol=} {vmax/kjmol=}")

                # print(f"{len(n_proj_idx)=} {fes=}")

                if len(n_proj_idx) == 1:
                    plot_f = Transformer._plot_1d
                else:
                    plot_f = Transformer._plot_2d

                plot_f(
                    fig=fig,
                    grid=gs[
                        3 + n_proj_plots * i + i_n,
                        0,
                    ],  # type: ignore
                    fesses=fes.slice(T=T),
                    indices=(0, 1),
                    margin=margin,
                    collective_variable=col_var_idx,
                    labels=col_var_idx.cvs_name,
                    print_labels=True,
                    cmap=plt.get_cmap(cmap_fes),
                    fontsize=fontsize_small,
                    vmax=vmax,
                    vmin=vmin,
                )

                assert dens is not None

                plot_f(
                    fig=fig,
                    grid=gs[
                        3 + n_proj_plots * i + i_n,
                        1,
                    ],  # type: ignore
                    fesses=dens.slice(T=T),
                    indices=(0, 1),
                    margin=margin,
                    cmap=plt.get_cmap(cmap_dens),
                    collective_variable=col_var_idx,
                    labels=col_var_idx.cvs_name,
                    vmax=vmax_dens,
                    vmin=vmin_dens,
                    print_labels=True,
                    fontsize=fontsize_small,
                    # show_1d_marginals=False,
                )

                for j in range(col_var_i.n):
                    print(f"{j=}  ")

                    # print(f"{color["vals"].shape=}")

                    color_j_bias = GridBias(
                        collective_variable=col_var_idx,
                        n=color["n"],
                        bounds=color["bounds"],
                        vals=jnp.apply_along_axis(lambda x: x[j], -1, color["vals"]),  # type: ignore
                        order=grid_bias_order,
                    )

                    # change to vmin  and vmax of cv metric
                    plot_f(
                        fig=fig,
                        grid=gs[
                            3 + n_proj_plots * i + i_n,
                            3 + j,
                        ],  # type: ignore
                        fesses=color_j_bias.slice(T=T),
                        indices=(0, 1),
                        margin=margin,
                        cmap=plt.get_cmap(cmap_data),
                        collective_variable=col_var_idx,
                        labels=col_var_idx.cvs_name,
                        vmax=1.0,
                        print_labels=True,
                        show_1d_marginals=False,
                        fontsize=fontsize_small,
                    )

                #     indices = (j,)

                #     print(f"{j=}  {indices=} ")

                #     colvar_i_slice = col_var_i[indices]

                #     cv_slice, _ = DataLoaderOutput.apply_cv(
                #         f=CvTrans.from_cv_function(
                #             _cv_slice,
                #             indices=jnp.array(indices),
                #         ),
                #         x=cv_data_i,
                #     )

                #     colors = Transformer._get_color_data(
                #         a=cv_slice,
                #         dim=len(indices),
                #         color_trajectories=False,
                #         metric=colvar_i_slice.metric,
                #         margin=margin,
                #     ).cv

                #     Transformer._plot_2d(
                #         fig=fig,
                #         grid=gs[
                #             1 + n_proj_plots * i + i_n,
                #             3 + j,
                #         ],  # type: ignore
                #         data=CV.stack(*cv_proj_idx).cv,
                #         colors=colors,
                #         collective_variable=col_var_idx,
                #         print_labels=True,
                #         labels=col_var_idx.cvs_name,
                #         **{"s": 1, "edgecolor": "none"},
                #     )

        # plot all the original data for reference

        idx = jnp.array(jnp.meshgrid(jnp.arange(n_needed), jnp.arange(n_proj_plots - 1))).reshape(2, -1)

        for i, (cv_data_i, colvar_i) in enumerate(zip(cv_data, collective_variables)):
            if weights is not None:
                weights_i = weights[i]
            else:
                weights_i = [jnp.ones(len(ci.cv)) for ci in cv_data_i]

            for j, indices in enumerate(dim_map[colvar_i.n - 1]):
                print(f"plotting original FES for cv {i}, dim {indices}")

                colvar_i_slice = colvar_i[indices]

                cv_data_i_slice, _ = DataLoaderOutput.apply_cv(
                    f=CvTrans.from_cv_function(
                        _cv_slice,
                        indices=jnp.array(indices),
                    ),
                    x=cv_data_i,
                )

                fes, _, dens, _ = DataLoaderOutput._get_fes_bias_from_weights(
                    T=T,
                    weights=weights_i,
                    rho=[jnp.ones_like(w) / w.shape[0] for w in weights_i],
                    collective_variable=colvar_i_slice,
                    cv=cv_data_i_slice,
                    samples_per_bin=5,
                    min_samples_per_bin=1,
                    n_max=n_max_bias,
                    max_bias=None,
                    macro_chunk=macro_chunk,
                    chunk_size=None,
                    recalc_bounds=False,
                    output_density_bias=False,
                    rbf_bias=rbf_bias,
                    grid_bias_order=grid_bias_order,
                    smoothing=1,
                )

                if len(indices) == 1:
                    plot_f = Transformer._plot_1d
                else:
                    plot_f = Transformer._plot_2d

                plot_f(
                    fig=fig,
                    grid=gs[
                        3 + n_proj_plots * i + idx[1, j],
                        rhs + idx[0, j],
                    ],  # type: ignore
                    fesses=fes.slice(T=T),
                    indices=(0, 1),
                    margin=margin,
                    collective_variable=colvar_i_slice,
                    labels=colvar_i_slice.cvs_name,
                    print_labels=True,
                    cmap=plt.get_cmap(cmap_fes),
                    fontsize=fontsize_small,
                    vmax=vmax,
                    vmin=vmin,
                )

                # Transformer._plot_2d(
                #     fig=fig,
                #     grid=gs[
                #         2 + n_proj_plots * i + idx[1, j],
                #         rhs + n_needed + idx[0, j],
                #     ],  # type: ignore
                #     fesses=dens.slice(T=T),
                #     indices=(0, 1),
                #     margin=margin,
                #     cmap=plt.get_cmap(cmap_dens),
                #     collective_variable=colvar_i_slice,
                #     labels=colvar_i_slice.cvs_name,
                #     vmax=vmax_dens,
                #     vmin=vmin_dens,
                #     print_labels=True,
                #     # show_1d_marginals=False,
                # )

        # titles, lines etc

        def draw_hline(i, j1, j2):
            left_cell = gs[i, j1].get_position(fig)
            right_cell = gs[i, j2].get_position(fig)

            x0 = left_cell.x0
            x1 = right_cell.x1
            y = left_cell.y1

            fig.add_artist(
                Line2D(
                    xdata=[x0, x1],
                    ydata=[y, y],
                    color="black",
                    linewidth=2,
                )
            )

        for i in range(1, nrows - 1, n_proj_plots):
            draw_hline(i + 1, 0, max_n_colvar + 2)
            draw_hline(i + 1, rhs, ncols - 1)

        for i in range(len(collective_variables)):
            col_var_i = collective_variables[i]
            cv_data_i = cv_data[i]

        def draw_vline(i1, i2, j):
            left_cell_top = gs[i1, j].get_position(fig)
            left_cell_bottom = gs[i2, j].get_position(fig)

            x = left_cell_top.x1 + wspace / 2.0
            y0 = left_cell_bottom.y0
            y1 = left_cell_top.y1

            fig.add_artist(
                Line2D(
                    xdata=[x, x],
                    ydata=[y0, y1],
                    color="black",
                    linewidth=2,
                )
            )

        draw_vline(1, nrows - 1, 1)
        draw_vline(1, nrows - 1, max_n_colvar - 2)

        def set_text(i1, i2, j1, j2, t, rotate=False, **kwargs):
            s = gs[i1, j1]
            pos = s.get_position(fig)

            s2 = gs[i2, j2]
            pos2 = s2.get_position(fig)
            fig.text(
                (pos.x0 + pos2.x1) / 2 - wspace / 2,
                (pos.y0 + pos2.y1) / 2 - hspace / 2,
                s=t,
                horizontalalignment="center",
                verticalalignment="center",
                rotation=90 if rotate else 0,
                fontsize=fontsize_large,
                **kwargs,
            )

        if projection_cv_title is None:
            projection_cv_title = collective_variable_projection.name

        set_text(0, 0, 0, 2, "Projected", fontweight="bold")
        set_text(0, 0, 3, rhs - 4, "Projected collective variable", fontweight="bold")

        set_text(0, 0, rhs, -1, "Original", fontweight="bold")

        set_text(1, 1, 0, 0, "Free Energy")
        set_text(1, 1, 1, 1, "Density")
        set_text(1, 1, 2, 2, "mode:")

        set_text(1, 1, rhs, rhs + n_needed - 1, "Free Energy")
        # set_text(1, 1, rhs + n_needed, -1, "Density")

        for j in range(max_n_colvar):
            set_text(1, 1, 3 + j, 3 + j, f"$q_{j}$")

        if cv_titles is None:
            cv_titles = [colvar.name for colvar in collective_variables]

        # label each block of rows on the left with the corresponding collective variable name
        for i in range(len(collective_variables)):
            set_text(
                2 + n_proj_plots * i + 1,
                2 + n_proj_plots * (i + 1) - 1,
                2,
                2,
                f"CV {collective_variables[i].name}",
                rotate=True,
                # fontsize=20,
                fontweight="bold",
            )

        def _add_vert_colorbar(col_idx: int, cmap, label, vmin, vmax, exp=False):
            cb_width = 0.015
            top_pos = gs[2, col_idx].get_position(fig)
            bottom_pos = gs[nrows - 1, col_idx].get_position(fig)
            y0 = bottom_pos.y0
            y1 = top_pos.y1
            height = y1 - y0
            pos_width = top_pos.x1 - top_pos.x0
            x = top_pos.x0 + (pos_width - cb_width) / 2.0

            ax_cb = fig.add_axes(rect=(x, y0, cb_width, height))

            norm = Normalize(vmin=vmin, vmax=vmax)
            m = ScalarMappable(norm=norm, cmap=cmap)
            m.set_array([])
            cbar = fig.colorbar(
                m,
                cax=ax_cb,
                orientation="vertical",
                location="left",
            )

            if exp:
                assert norm.vmax is not None

                ticks = jnp.arange(
                    norm.vmin,
                    norm.vmax + 1e-5,
                )
                cbar.set_ticks(ticks)  # type: ignore
                cbar.set_ticklabels([f"$10^{{{int(-t)}}}$" for t in ticks])

            cbar.ax.tick_params(labelsize=fontsize_small)

            cbar.set_label(label, fontsize=fontsize_large)
            return cbar

        # add colorbars for FES (second-last column) and Density (last column)

        _add_vert_colorbar(max_n_colvar + 3, plt.get_cmap(cmap_fes), bar_label, vmin=vmin / kjmol, vmax=vmax / kjmol)
        _add_vert_colorbar(max_n_colvar + 4, plt.get_cmap(cmap_dens), "Density", vmin=vmin_dens, vmax=3, exp=True)
        _add_vert_colorbar(max_n_colvar + 5, plt.get_cmap(cmap_data), "CV [a.u.]", vmin=0, vmax=1)

        if timescales is not None:
            for i, ts in enumerate(timescales):
                print(f"timescales: {ts}")
                for j, t in enumerate(ts):
                    set_text(2 + n_proj_plots * i, 2 + n_proj_plots * i, 3 + j, 3 + j, f"{t:.1f} ns")

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
    def plot_CV_corr(
        collective_variable_projection: CollectiveVariable,
        collective_variables: list[CollectiveVariable],
        ti: list[list[TrajectoryInfo]],
        # cv_data: list[list[CV]],
        # sp_data: list[list[SystemParams]],
        # weights: list[list[jax.Array]] | None = None,
        # std: list[list[jax.Array]] | None = None,
        name: str | Path | None = None,
        timescales: list[list[float]] | None = None,
        projection_cv_title: str | None = None,
        cv_titles: list[str] | None = None,
        margin=0.1,
        plot_FES=False,
        T=300 * kelvin,
        vmin=0,
        vmax=100 * kjmol,
        dpi=300,
        n_max_bias=1e6,
        macro_chunk=1000,
        cmap_fes="viridis",
        cmap_dens="hot",
        cmap_data="cividis",
        vmax_dens=3,
        vmin_dens=-1,
        offset=True,
        bar_label=None,
        title="Collective Variables",
        fontsize_small=16,
        fontsize_large=20,
        grid_bias_order=0,
        plot_density=True,
        rbf_bias=False,
        smoothing=None,
        overlay_mask=True,
        extra_info_title: str | None = None,
        get_fes_bias_kwargs: dict = {},
        plot_std=False,
    ):
        # data_converted = []

        from IMLCV.base.rounds import DataLoaderOutput
        from IMLCV.implementations.CV import _cv_slice

        if bar_label is None:
            if plot_std:
                bar_label = "Free Energy Std Dev [kJ/mol]"
            else:
                bar_label = "Free Energy [kJ/mol]"

        dim_map = [
            [(0,)],  # 1
            [(0, 1)],  # 2
            [(0, 1), (1, 2), (0, 2)],  # 3
            [(0, 1), (2, 3)],  # 4
            [(0, 1), (2, 3), (4)],  # 5
            [(0, 1), (2, 3), (4, 5)],  # 6
            [(0, 1), (2, 3), (4, 5), (6)],  # 7
            [(0, 1), (2, 3), (4, 5), (6, 7)],  # 8
            [
                (0, 1),
                (2, 3),
                (4, 5),
                (6, 7),
                (8),
            ],  # 9
            [
                (0, 1),
                (2, 3),
                (4, 5),
                (6, 7),
                (8, 9),
            ],  # 10
            [
                (0, 1),
                (2, 3),
                (4, 5),
                (6, 7),
                (8, 9),
                (10),
            ],  # 11
            [
                (0, 1),
                (2, 3),
                (4, 5),
                (6, 7),
                (8, 9),
                (10, 11),
            ],  # 12
            [
                (0, 1),
                (2, 3),
                (4, 5),
                (6, 7),
                (8, 9),
                (10, 11),
                (12),
            ],  # 13
            [
                (0, 1),
                (2, 3),
                (4, 5),
                (6, 7),
                (8, 9),
                (10, 11),
                (12, 13),
            ],  # 14
            [
                (0, 1),
                (2, 3),
                (4, 5),
                (6, 7),
                (8, 9),
                (10, 11),
                (12, 13),
                (14),
            ],  # 15
            [
                (0, 1),
                (2, 3),
                (4, 5),
                (6, 7),
                (8, 9),
                (10, 11),
                (12, 13),
                (14, 15),
            ],  # 16
            [
                (0, 1),
                (2, 3),
                (4, 5),
                (6, 7),
                (8, 9),
                (10, 11),
                (12, 13),
                (14, 15),
                (16,),
            ],  # 17
            [
                (0, 1),
                (2, 3),
                (4, 5),
                (6, 7),
                (8, 9),
                (10, 11),
                (12, 13),
                (14, 15),
                (16, 17),
            ],  # 17
        ]

        if isinstance(cmap_fes, str):
            cmap_fes = plt.get_cmap(cmap_fes)

        cmap_fes.set_over(alpha=0)
        cmap_fes.set_under(alpha=0)

        collective_variables = [a.replace(cvs_name=[f"$q_{i}$" for i in range(a.n)]) for a in collective_variables]
        if collective_variable_projection.cvs_name is None:
            collective_variable_projection = collective_variable_projection.replace(
                cvs_name=[f"$z_{i}$" for i in range(collective_variable_projection.n)]
            )

        ncv = len(collective_variables)

        dim_1 = collective_variable_projection.n
        dim_2_r = [colvar_i.n for colvar_i in collective_variables]

        max_dim_2 = max(dim_2_r)

        print(f"{dim_1=} {dim_2_r=} {max_dim_2=} {ncv=}")

        size_square = 3.0

        width_cols = [size_square] + [0.5] + [size_square] * max_dim_2
        height_rows = [0.5] * 2 + ([0.5] + [size_square] * dim_1) * ncv + [0.1]

        n_needed = 1

        for i in range(len(collective_variables)):
            ni = dim_map[collective_variables[i].n - 1]
            a = jnp.ceil(len(ni) / (dim_1))
            if a > n_needed:
                n_needed = int(a)

        print(f"{n_needed=}")

        width_cols += [size_square] * n_needed

        # colorbar
        width_cols += [2.5]

        start_col = 2
        start_row = 2

        wspace = 0.3
        hspace = 0.3

        nrows = len(height_rows)
        ncols = len(width_cols)

        print(f"{height_rows=}, {width_cols=}")

        # make figure and gridspeciterator
        fig = plt.figure(
            figsize=(
                sum(width_cols) + wspace * (len(width_cols)),
                sum(height_rows) + hspace * (len(height_rows)),
            )
        )

        gs = gridspec.GridSpec(
            nrows=nrows,
            ncols=ncols,
            figure=fig,
            width_ratios=width_cols,
            height_ratios=height_rows,
            wspace=wspace,
            hspace=hspace,
        )

        print(f"{gs[0, 0].get_position(fig)=} , {gs[0, 1].get_position(fig)=} {gs[1, 0].get_position(fig)=}")

        wspace = gs[0, 1].get_position(fig).x0 - gs[0, 0].get_position(fig).x1
        hspace = gs[0, 0].get_position(fig).y0 - gs[1, 0].get_position(fig).y1
        print(f"{wspace=}, {hspace=}")

        for i in range(len(collective_variables)):
            print(f"{i=}")

            ti_i = ti[i]

            col_var_i = collective_variables[i]

            # print(f"{col_var_i=} {i=} ")
            cv_data_i = [a.CV for a in ti_i]
            sp_data_i = [a.sp for a in ti_i]
            weights_i = [a.w for a in ti_i]

            assert cv_data_i is not None, "cv_data cannot be None when using TrajectoryInfo"

            if weights_i is None:
                weights_i = [jnp.ones(len(ci.cv)) for ci in cv_data_i]

            sigma_i = [a.sigma for a in ti_i] if ti_i[0].sigma is not None else None

            cv_proj, _ = DataLoaderOutput.apply_cv(
                f=collective_variable_projection.f,
                x=sp_data_i,
                macro_chunk=macro_chunk,
                verbose=True,
            )

            # print(f"{cv_proj=}")

            import itertools

            # print(f"################### plotting for CV set {i} #########################")

            print(f"############################## FES plots for projection CV alone")
            for j, ab in enumerate(dim_map[dim_1 - 1]):
                # print(f"################## {ab} #########################")
                print(f"{ab=}")

                cv_proj_ab, _ = DataLoaderOutput.apply_cv(
                    f=CvTrans.from_cv_function(
                        _cv_slice,
                        indices=jnp.array(ab),
                    ),
                    x=cv_proj,
                    macro_chunk=macro_chunk,
                )

                colvar_ab = collective_variable_projection[ab]

                kw = dict(
                    T=T,
                    weights=weights_i,
                    rho=[jnp.ones_like(w) / w.shape[0] for w in weights_i],
                    collective_variable=colvar_ab,
                    cv=cv_proj_ab,
                    samples_per_bin=5,
                    min_samples_per_bin=1,
                    n_max=n_max_bias,
                    max_bias=None,
                    macro_chunk=macro_chunk,
                    chunk_size=None,
                    recalc_bounds=False,
                    output_density_bias=True,
                    rbf_bias=rbf_bias,
                    grid_bias_order=grid_bias_order,
                    smoothing=smoothing,
                    set_outer_border=False,
                    overlay_mask=overlay_mask,
                    std_bias=plot_std,
                    weights_std=sigma_i,
                )

                kw.update(get_fes_bias_kwargs)

                fes, std_bias, dens, _ = DataLoaderOutput._get_fes_bias_from_weights(**kw)  # type: ignore

                if colvar_ab.n == 1:
                    plot_f = Transformer._plot_1d

                else:
                    plot_f = Transformer._plot_2d

                if plot_std:
                    f = std_bias
                else:
                    f = fes

                plot_f(
                    fig=fig,
                    grid=gs[
                        1 + i * (dim_1 + 1) + start_row + j,
                        0,
                    ],  # type: ignore
                    fesses=f.slice(T=T),
                    indices=(0, 1),
                    margin=margin,
                    collective_variable=colvar_ab,
                    labels=colvar_ab.cvs_name,
                    print_labels=True,
                    cmap=cmap_fes,
                    fontsize=fontsize_small,
                    vmax=vmax,
                    vmin=vmin,
                )

            print(f"############################## FES plots for projection CV vs new CV")
            # FES plots for one CV refernce vs one CV new
            for a, b in itertools.product(jnp.arange(dim_1), jnp.arange(dim_2_r[i])):
                # print(f"################## {(a,b)=} #########################")
                print(f"{a=} {b=} ")

                cv_proj_a, _ = DataLoaderOutput.apply_cv(
                    f=CvTrans.from_cv_function(
                        _cv_slice,
                        indices=jnp.array([a]),
                    ),
                    x=cv_proj,
                    macro_chunk=macro_chunk,
                )

                cv_i_b, _ = DataLoaderOutput.apply_cv(
                    f=CvTrans.from_cv_function(
                        _cv_slice,
                        indices=jnp.array([b]),
                    ),
                    x=cv_data_i,
                    macro_chunk=macro_chunk,
                )

                cv_ab = [CV.combine(cv_a, cv_b) for cv_a, cv_b in zip(cv_proj_a, cv_i_b)]

                colvar_a = collective_variable_projection[(int(a),)]
                colvar_b = col_var_i[(int(b),)]

                assert colvar_a.cvs_name is not None
                assert colvar_b.cvs_name is not None

                colvar_ab = CollectiveVariable(
                    f=colvar_a.f + colvar_b.f,
                    metric=CvMetric.create(
                        bounding_box=jnp.vstack([colvar_a.metric.bounding_box, colvar_b.metric.bounding_box]),
                        periodicities=jnp.hstack([colvar_a.metric.periodicities, colvar_b.metric.periodicities]),
                        extensible=jnp.hstack([colvar_a.metric.extensible, colvar_b.metric.extensible]),
                    ),
                    name="",
                    cvs_name=tuple([*colvar_a.cvs_name, *colvar_b.cvs_name]),
                )

                # print(f"{cv_proj_idx[0].shape=}")

                kw = dict(
                    T=T,
                    weights=weights_i,
                    rho=[jnp.ones_like(w) / w.shape[0] for w in weights_i],
                    collective_variable=colvar_ab,
                    cv=cv_ab,
                    samples_per_bin=5,
                    min_samples_per_bin=1,
                    n_max=n_max_bias,
                    max_bias=None,
                    macro_chunk=macro_chunk,
                    chunk_size=None,
                    recalc_bounds=False,
                    output_density_bias=True,
                    rbf_bias=rbf_bias,
                    grid_bias_order=grid_bias_order,
                    smoothing=smoothing,
                    set_outer_border=False,
                    overlay_mask=overlay_mask,
                    std_bias=plot_std,
                    weights_std=sigma_i,
                )

                kw.update(get_fes_bias_kwargs)

                fes, std_bias, dens, _ = DataLoaderOutput._get_fes_bias_from_weights(**kw)  # type: ignore

                plot_f = Transformer._plot_2d

                if plot_std:
                    f = std_bias
                else:
                    f = fes
                plot_f(
                    fig=fig,
                    grid=gs[
                        1 + i * (dim_1 + 1) + start_row + a,
                        start_col + b,
                    ],  # type: ignore
                    fesses=f.slice(T=T),
                    indices=(0, 1),
                    margin=margin,
                    collective_variable=colvar_ab,
                    labels=colvar_ab.cvs_name,
                    print_labels=True,
                    cmap=cmap_fes,
                    fontsize=fontsize_small,
                    vmax=vmax,
                    vmin=vmin,
                )

            # FES plot for new CVs
            idx = jnp.array(jnp.meshgrid(jnp.arange(n_needed), jnp.arange(max_dim_2))).reshape(2, -1)

            print(f"############################## FES plots for new CV alone")

            for j, indices in enumerate(dim_map[col_var_i.n - 1]):
                # print(f"################## {indices} #########################")
                # print(f"plotting original FES for cv {i}, dim {indices} {idx=}  {idx.shape=} {max_dim_2=}")

                colvar_i_slice = col_var_i[indices]

                cv_data_i_slice, _ = DataLoaderOutput.apply_cv(
                    f=CvTrans.from_cv_function(
                        _cv_slice,
                        indices=jnp.array(indices),
                    ),
                    x=cv_data_i,
                )

                kw = dict(
                    T=T,
                    weights=weights_i,
                    rho=[jnp.ones_like(w) / w.shape[0] for w in weights_i],
                    collective_variable=colvar_i_slice,
                    cv=cv_data_i_slice,
                    samples_per_bin=5,
                    min_samples_per_bin=1,
                    n_max=n_max_bias,
                    max_bias=None,
                    macro_chunk=macro_chunk,
                    chunk_size=None,
                    recalc_bounds=False,
                    output_density_bias=False,
                    rbf_bias=rbf_bias,
                    grid_bias_order=grid_bias_order,
                    smoothing=smoothing,
                    set_outer_border=False,
                    overlay_mask=overlay_mask,
                    std_bias=plot_std,
                    weights_std=sigma_i,
                )

                kw.update(get_fes_bias_kwargs)

                fes, std_bias, dens, _ = DataLoaderOutput._get_fes_bias_from_weights(**kw)

                if len(indices) == 1:
                    plot_f = Transformer._plot_1d
                else:
                    plot_f = Transformer._plot_2d

                if plot_std:
                    f = std_bias
                else:
                    f = fes

                plot_f(
                    fig=fig,
                    grid=gs[
                        1 + start_row + (dim_1 + 1) * i + idx[1, j],
                        start_col + max_dim_2 + idx[0, j],
                    ],  # type: ignore
                    fesses=f.slice(T=T),
                    indices=(0, 1),
                    margin=margin,
                    collective_variable=colvar_i_slice,
                    labels=colvar_i_slice.cvs_name,
                    print_labels=True,
                    cmap=cmap_fes,
                    fontsize=fontsize_small,
                    vmax=vmax,
                    vmin=vmin,
                )

        def draw_hline(i, j1, j2):
            left_cell = gs[i, j1].get_position(fig)
            right_cell = gs[i, j2].get_position(fig)

            x0 = left_cell.x0
            x1 = right_cell.x1
            y = left_cell.y1

            fig.add_artist(
                Line2D(
                    xdata=[x0, x1],
                    ydata=[y, y],
                    color="black",
                    linewidth=2,
                )
            )

        # if dim_1 != 1:
        for i in range(start_row, start_row + ncv * (dim_1 + 1) + 1, dim_1 + 1):
            draw_hline(i, 0, ncols - 2)

        draw_hline(start_row - 1, 0, ncols - 2)

        def draw_vline(i1, i2, j):
            left_cell_top = gs[i1, j].get_position(fig)
            left_cell_bottom = gs[i2, j].get_position(fig)

            x = left_cell_top.x1  # + wspace / 2.0
            y0 = left_cell_bottom.y0
            y1 = left_cell_top.y1

            fig.add_artist(
                Line2D(
                    xdata=[x, x],
                    ydata=[y0, y1],
                    color="black",
                    linewidth=2,
                )
            )

        draw_vline(start_row, nrows - 1, 0)
        draw_vline(start_row, nrows - 1, 1 + max_dim_2)

        def set_text(i1, i2, j1, j2, t, rotate=False, fontsize=fontsize_large, **kwargs):
            s = gs[i1, j1]
            pos = s.get_position(fig)

            s2 = gs[i2, j2]
            pos2 = s2.get_position(fig)
            fig.text(
                (pos.x0 + pos2.x1) / 2 - wspace / 2,
                (pos.y0 + pos2.y1) / 2 - hspace / 2,
                s=t,
                horizontalalignment="center",
                verticalalignment="center",
                rotation=90 if rotate else 0,
                fontsize=fontsize,
                **kwargs,
            )

        if projection_cv_title is None:
            projection_cv_title = collective_variable_projection.name

        set_text(1, 1, start_col - 1, start_col - 1, f"mode:", fontsize=fontsize_small)
        for j in range(max_dim_2):
            set_text(1, 1, start_col + j, start_col + j, f"$q_{j}$", fontsize=fontsize_small)

        if cv_titles is None:
            cv_titles = [colvar.name for colvar in collective_variables]

        # label each block of rows on the left with the corresponding collective variable name
        for i in range(len(collective_variables)):
            set_text(
                start_row + (dim_1 + 1) * i + 1,
                start_row + (dim_1 + 1) * (i + 1) - 1,
                start_col - 1,
                start_col - 1,
                f"CV {collective_variables[i].name}",
                rotate=True,
                # fontsize=20,
                fontweight="bold",
            )

            if collective_variables[i].extra_info is not None:
                if extra_info_title is not None:
                    set_text(
                        start_row + (dim_1 + 1) * i,
                        start_row + (dim_1 + 1) * i,
                        start_col - 1,
                        start_col - 1,
                        extra_info_title,
                        fontsize=fontsize_small,
                    )

                for j in range(collective_variables[i].n):
                    set_text(
                        start_row + (dim_1 + 1) * i,
                        start_row + (dim_1 + 1) * i,
                        start_col + j,
                        start_col + j,
                        collective_variables[i].extra_info[j],
                        fontsize=fontsize_small,
                    )

        set_text(0, 0, 0, 0, "Reference CV", fontweight="bold")
        set_text(0, 0, start_col - 1, start_col + max_dim_2 - 1, "Mixed CV", fontweight="bold")
        set_text(0, 0, start_col + max_dim_2, start_col + max_dim_2 + n_needed - 1, "Learned CV", fontweight="bold")

        def _add_vert_colorbar(col_idx: int, cmap, label, vmin, vmax, exp=False):
            cb_width = 0.015
            top_pos = gs[2, col_idx].get_position(fig)
            bottom_pos = gs[nrows - 1, col_idx].get_position(fig)
            y0 = bottom_pos.y0
            y1 = top_pos.y1
            height = y1 - y0
            pos_width = top_pos.x1 - top_pos.x0
            x = top_pos.x0 + (pos_width - cb_width) / 2.0

            ax_cb = fig.add_axes((x, y0, cb_width, height))
            norm = Normalize(vmin=vmin, vmax=vmax)
            m = ScalarMappable(norm=norm, cmap=cmap)
            m.set_array([])
            cbar = fig.colorbar(
                m,
                cax=ax_cb,
                orientation="vertical",
                location="left",
            )

            if exp:
                ticks = jnp.arange(
                    norm.vmin,
                    norm.vmax + 1e-5,
                )
                cbar.set_ticks(ticks)
                cbar.set_ticklabels([f"$10^{{{int(-t)}}}$" for t in ticks])

            cbar.ax.tick_params(labelsize=fontsize_small)

            cbar.set_label(label, fontsize=fontsize_large)
            return cbar

        # add colorbars for FES (second-last column) and Density (last column)

        _add_vert_colorbar(
            2 + max_dim_2 + n_needed, plt.get_cmap(cmap_fes), bar_label, vmin=vmin / kjmol, vmax=vmax / kjmol
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
        vmin: float | list[float],
        vmax: float | list[float],
        cmap,
        bar_label: str | list[str] = "FES [kJ/mol]",
        cv_titles=None,
        data_titles=None,
        indicate_plots=None,
        plot_FES=False,
        title="Collective Variables",
        # indicate_cv_data=True,
    ):
        spaces = 1 if max(dims) < 3 else 2

        w_tot = [
            0.3,
            *[spaces] * ncv,
        ]  # extra 0.05 is to make the title fit
        if plot_FES:
            w_tot.append(0.05)
        w_tot.append(0.1)
        h_tot = [0.1, *[spaces] * ndata]  # 0.1 for title

        fig.set_figwidth(sum(w_tot) * 3, forward=True)
        fig.set_figheight(sum(h_tot) * 3, forward=True)

        w_space = 0.1
        h_space = 0.1

        spec = fig.add_gridspec(
            nrows=len(h_tot),
            ncols=len(w_tot),
            width_ratios=w_tot,
            height_ratios=h_tot,
            wspace=w_space,
            hspace=h_space,
        )

        def plot_colorbar(vmin, vmax, ystart, yend, l):
            norm = Normalize(vmin=vmin / kjmol, vmax=vmax / kjmol)  # type: ignore

            ax_cbar = fig.add_subplot(spec[ystart:yend, -2])

            fig.colorbar(
                mappable=ScalarMappable(norm=norm, cmap=cmap),  # type: ignore
                cax=ax_cbar,  # type: ignore
                orientation="vertical",
                ticks=[vmin / kjmol, (vmin + vmax) / (2 * kjmol), vmax / kjmol],
            )
            ax_cbar.set_ylabel(l)

        if plot_FES:
            if isinstance(vmin, list) or isinstance(vmax, list):
                for i in range(ndata):
                    vmin_i = vmin[i] if isinstance(vmin, list) else vmin
                    vmax_i = vmax[i] if isinstance(vmax, list) else vmax
                    _l = bar_label[i] if isinstance(bar_label, list) else bar_label
                    plot_colorbar(vmin_i, vmax_i, i + 1, i + 2, _l)

            else:
                plot_colorbar(vmin, vmax, 1, ndata + 1, bar_label)

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
                            plt.Rectangle(  # type: ignore
                                (pos.x0 - sep_w, pos.y0 - sep_h),
                                pos.x1 - pos.x0 + sep_w,
                                pos.y1 - pos.y0 + sep_h,
                                fill=True,
                                color=c_ij,
                                zorder=-10,  # type: ignore
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
            s=title,
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
        collective_variable: CollectiveVariable,
        data: jax.Array | None = None,
        colors: jax.Array | None = None,
        labels: tuple[str, ...] | str | None = None,
        fesses: dict[int, dict[tuple, Bias]] | None = None,
        indices: tuple | None = None,
        margin: float = 0.2,
        T=None,
        vmin=0,
        vmax=100 * kjmol,
        cmap=plt.get_cmap("viridis"),
        print_labels=True,
        show_1d_marginals=True,
        fontsize=None,
        plot_std=False,
        **scatter_kwargs,
    ):
        # print(f"{vmin/kjmol=} {vmax / kjmol=} {cmap=}")

        gs = grid.subgridspec(  # type: ignore
            ncols=3,
            nrows=3,
            width_ratios=[0.5, 4, 1.0],  # make it consistent with 2D
            height_ratios=[1, 4, 0.5],
            wspace=0.1,
            hspace=0.1,
        )

        # data in main square, FES/ histogram on top
        if data is None:
            ax_histx = fig.add_subplot(gs[1, 1])
        else:
            ax = fig.add_subplot(gs[1, 1])
            ax_histx = fig.add_subplot(gs[0, 1], sharex=ax)

        metric = collective_variable.metric

        # print(f"{metric=}")

        m = (metric.bounding_box[:, 1] - metric.bounding_box[:, 0]) * margin
        x_l = metric.bounding_box[0, :]
        m_x = m[0]
        x_lim = [x_l[0] - m_x, x_l[1] + m_x]

        # print(f"{x_lim=}")

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

                n_bins = int(4 * jnp.round(n_points ** (1 / 3)))

                # print(f"{n_points=} {n_bins=} ")

                x_bins = jnp.linspace(x_lim[0], x_lim[1], n_bins + 1)

                bins_x_center = (x_bins[1:] + x_bins[:-1]) / 2

                # raw number of points
                H, _ = jnp.histogramdd(
                    data,
                    bins=n_bins,
                    range=[x_lim, y_lim],
                )

                x_count = vmap_decorator(jnp.sum, in_axes=0)(H)
                x_count /= jnp.sum(x_count)

                ax_histx.plot(bins_x_center, x_count, color="tab:blue")

                ax_histx.patch.set_alpha(0)

        else:
            assert indices is not None

            # print(f"sampling 2000 points")

            x_range = jnp.linspace(x_lim[0], x_lim[1], 2000)

            # print(f"{fesses=}")
            bias: Bias | StdBias = fesses[1][(indices[0],)]

            # if plot_std:
            #     if isinstance(FES, StdBias):
            #         print(f"plotting std FES")

            #         bias = bias.std_bias
            #     else:
            #         raise ValueError("plot_std is True but FES is not StdBias")

            x_fes, _ = bias.compute_from_cv(CV(cv=jnp.array(x_range).reshape((-1, 1))))

            ax_histx.scatter(
                x_range,
                x_fes / kjmol if plot_std else -x_fes / kjmol,
                c=-x_fes / kjmol,
                s=1,
                vmin=vmin / kjmol,
                vmax=vmax / kjmol,
                cmap=cmap,
            )

            # if isinstance(bias, StdBias):
            #     print(f"plotting std shading")

            #     x_std, _ = bias.compute_std_from_cv(CV(cv=jnp.array(x_range).reshape((-1, 1))))

            #     ax_histx.fill_between(
            #         x_range,
            #         (-x_fes - x_std) / kjmol,
            #         (-x_fes + x_std) / kjmol,
            #         color="gray",
            #         alpha=0.5,
            #     )

            #     # ax_histx.fill_between(
            #     #     x_range,
            #     #     (-x_fes - x_std) / kjmol,
            #     #     (-x_fes + x_std) / kjmol,
            #     #     color="gray",
            #     #     alpha=0.5,
            #     # )

            ax_histx.set_xlim(x_lim[0], x_lim[1])  # type: ignore
            ax_histx.set_ylim(vmin / kjmol, vmax / kjmol)

            ax_histx.patch.set_alpha(0)

        if data is not None:
            for b in [ax_histx]:
                b.spines["right"].set_visible(False)
                b.spines["top"].set_visible(False)
                b.spines["bottom"].set_visible(False)
                b.spines["left"].set_visible(False)

            ax_histx.tick_params(
                top=False,
                bottom=True,
                left=False,
                right=False,
                labelleft=False,
                labelbottom=False,
            )

        else:
            ax_histx.spines["right"].set_visible(False)
            ax_histx.spines["top"].set_visible(False)
            ax_histx.spines["left"].set_visible(False)

            ax_histx.yaxis.set_ticks_position("none")
            ax_histx.tick_params(
                top=False,
                bottom=True,
                left=False,
                right=False,
                labelleft=False,
                labelbottom=False,
            )

        ax_histx.set_xticks([x_l[0], (x_l[0] + x_l[1]) / 2, x_l[1]])

        if print_labels and indices is not None and labels is not None:
            if data is not None:
                ax.set_xlabel(labels[indices[0]], fontsize=fontsize)
            else:
                ax_histx.set_xlabel(labels[indices[0]], labelpad=-1, fontsize=fontsize)

        # ax.set_ylabel("FES [kJ/mol]", fontsize=fontsize)

        if data is not None:
            if margin is not None:
                ax.set_xlim(*x_lim)  # type: ignore

    @staticmethod
    def _plot_2d(
        fig: Figure,
        grid: gridspec.GridSpec,
        collective_variable: CollectiveVariable,
        data: jax.Array | None = None,
        colors: jax.Array | None = None,
        labels: tuple[str, ...] | str | None = None,
        fesses: dict[int, dict[tuple, Bias]] | None = None,
        indices: tuple | None = None,
        margin: float = 0.1,
        vmin=0,
        vmax=100 * kjmol,
        T=None,
        print_labels=True,
        cmap=plt.get_cmap("viridis"),
        plot_y=True,
        show_1d_marginals=True,
        fontsize=None,
        plot_std=False,
        **scatter_kwargs,
    ):
        cmap.set_over(alpha=0)
        cmap.set_under(alpha=0)
        gs = grid.subgridspec(  # type: ignore
            ncols=3,
            nrows=3,
            width_ratios=[0.5, 4, 1],  # first col: y label
            height_ratios=[1, 4, 0.5],  # extra row for x label
            wspace=0.02,
            hspace=0.02,
        )
        ax = fig.add_subplot(gs[1, 1])

        ax_histx = fig.add_subplot(gs[0, 1], sharex=ax)
        ax_histy = fig.add_subplot(gs[1, 2], sharey=ax)

        metric = collective_variable.metric

        # print(f"inside plot 2d, got {metric=}")

        if margin is None:
            margin = 0.0

        m = (metric.bounding_box[:, 1] - metric.bounding_box[:, 0]) * margin

        x_l = metric.bounding_box[0, :]
        m_x = m[0]
        y_l = metric.bounding_box[1, :]
        m_y = m[1]

        x_lim = jnp.array([x_l[0] - m_x, x_l[1] + m_x])
        y_lim = jnp.array([y_l[0] - m_y, y_l[1] + m_y])

        if fesses is not None:
            assert indices is not None, "FES_indices must be provided if FES is provided"

            FES = fesses[2][indices]

        else:
            FES = None

        if FES is not None:
            # print("obtaining 2d fes")

            # cv grid is centered
            bins, _, _, cv_grid, _ = metric.grid(
                n=100,
                # endpoints=True,
                margin=margin,
                indexing="xy",
            )

            from IMLCV.base.rounds import DataLoaderOutput

            # if plot_std:
            #     if isinstance(FES, StdBias):
            #         print(f"plotting std FES")

            #         FES = FES.std_bias
            #     else:
            #         raise ValueError("plot_std is True but FES is not StdBias")

            bias = DataLoaderOutput._apply_bias(
                x=[cv_grid],
                bias=FES,
                macro_chunk=1000,
                verbose=False,
                shmap=False,
            )[0]

            # if isinstance(bias, StdBias):
            # print(f"{bias[bias>0]/kjmol=}")

            # print(f"fes: {bias=}")

            bias = bias.reshape(len(bins[0]) - 1, len(bins[1]) - 1)

            print(f"{jnp.nanmin(bias)/kjmol=}, {jnp.nanmax(bias)/kjmol=}")

            # if plot_std:
            #     bias = -bias

            # bias -= jnp.max(bias)

            # imshow shows in middel of image
            _dx = (bins[0][1] - bins[0][0]) / 2
            _dy = (bins[1][1] - bins[1][0]) / 2

            _ = ax.imshow(
                -bias / (kjmol),
                cmap=cmap,
                origin="lower",
                extent=[
                    bins[0][0] - _dx,  # type: ignore
                    bins[0][-1] + _dx,  # type: ignore
                    bins[1][0] - _dy,  # type: ignore
                    bins[1][-1] + _dy,  # type: ignore
                ],
                aspect="auto",
                vmin=vmin / kjmol,
                vmax=vmax / kjmol,
            )

        if data is not None:
            ax.scatter(
                data[:, 0],
                data[:, 1],
                c=colors,
                **scatter_kwargs,
            )

        if show_1d_marginals:
            if fesses is None:
                if data is not None:
                    in_xlim = jnp.logical_and(data[:, 0] > x_lim[0], data[:, 0] < x_lim[1])
                    in_ylim = jnp.logical_and(data[:, 1] > y_lim[0], data[:, 1] < y_lim[1])
                    n_points = jnp.sum(jnp.logical_and(in_xlim, in_ylim))

                    n_bins = int(4 * jnp.round(n_points ** (1 / 3)))

                    # print(f"{n_points=} {n_bins=} ")
                    # n_bins = n_points // 50

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
                assert indices is not None, "indices must be provided for data scatter"

                x_range = jnp.linspace(x_lim[0], x_lim[1], 500)
                y_range = jnp.linspace(y_lim[0], y_lim[1], 500)

                x_bias = fesses[1][(indices[0],)]
                y_bias = fesses[1][(indices[1],)]

                x_fes, _ = x_bias.compute_from_cv(CV(cv=jnp.array(x_range).reshape((-1, 1))))
                y_fes, _ = y_bias.compute_from_cv(CV(cv=jnp.array(y_range).reshape((-1, 1))))

                # if plot_std:
                #     x_fes = -x_fes
                #     y_fes = -y_fes

                # print(f"{x_fes[x_fes<0]/kjmol=}, {y_fes[y_fes<0]/kjmol=}")

                ax_histx.scatter(
                    x_range,
                    -x_fes / kjmol,
                    c=-x_fes / kjmol,
                    s=1,
                    vmin=vmin / kjmol,
                    vmax=vmax / kjmol,
                    cmap=cmap,
                )

                ax_histx.set_xlim(x_lim[0], x_lim[1])  # type: ignore
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
                ax_histy.set_ylim(y_lim[0], y_lim[1])  # type: ignore

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
            labelsize=fontsize,
        )
        ax_histy.tick_params(
            top=False,
            bottom=False,
            left=False,
            right=False,
            labelleft=False,
            labelbottom=False,
            labelsize=fontsize,
        )

        # ax.locator_params(nbins=4)
        ax.set_xticks([x_l[0], (x_l[0] + x_l[1]) / 2, x_l[1]])
        ax.set_yticks([y_l[0], (y_l[0] + y_l[1]) / 2, y_l[1]])

        ax.set_xticklabels([])
        ax.set_yticklabels([])

        ax.tick_params(axis="both", length=1)

        ax.patch.set_alpha(0)

        if print_labels and indices is not None and labels is not None:
            ax.set_xlabel(labels[indices[0]], labelpad=-1, fontsize=fontsize)
            ax.set_ylabel(labels[indices[1]], labelpad=-1, fontsize=fontsize)

        if margin is not None:
            ax.set_xlim(*x_lim)  # type: ignore
            ax.set_ylim(*y_lim)  # type: ignore

    @staticmethod
    def _plot_3d(
        fig: Figure,
        grid: gridspec.GridSpec,
        data,
        colors,
        labels,
        collective_variable: CollectiveVariable,
        fesses: dict[int, dict[tuple, Bias]] | None = None,
        indices: tuple[int, ...] | None = None,
        margin: float = 0.1,
        vmin=0,
        vmax=100 * kjmol,
        T=None,
        print_labels=True,
        cmap=plt.get_cmap("viridis"),
        plot_std=False,
        **scatter_kwargs,
    ):
        metric = collective_variable.metric

        gs = grid.subgridspec(  # type: ignore
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
        ax0.view_init(elev=35, azim=225)  # type: ignore

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
            assert indices is not None
            bias = fesses[3][indices]

            # cmap = plt.get_cmap("jet")

            # print("obtaining 2d fes")

            n_grid = 30

            bins, _, cv_grid, cv_mid, _ = metric.grid(
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

            color_bias = jnp.array(cmap(normed_val))

            # add alpha channel
            color_bias = color_bias.at[:, 3].set(
                jnp.exp(-3 * normed_val) / n_grid
            )  # if all points in row are the same, the alpha is 1

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
                        import numpy as onp

                        args: list[np.ndarray | None] = [None, None, None]

                        args[x] = XYZ[:, :, :, x].take(z_i, axis=z).__array__()
                        args[y] = XYZ[:, :, :, y].take(z_i, axis=z).__array__()
                        args[z] = XYZ[:, :, :, z].take(z_i, axis=z).__array__()

                        ax0.plot_surface(  # type: ignore
                            *args,
                            rstride=1,
                            cstride=1,
                            facecolors=color_bias.take(z_i, axis=z).__array__(),
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
                        def voxels(__x, __y, __z, filled, **kwargs):  # type: ignore
                            return (__x, __y, __z), filled, kwargs
                    else:

                        def voxels(filled, **kwargs):  # type: ignore
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
                                    "When multidimensional, {} must match the shape of filled".format(name)
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
                        pinds = np.arange(pc)  # type: ignore
                        qinds = np.arange(qc)  # type: ignore
                        rinds = np.arange(rc)  # type: ignore

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

                ax0.voxels = types.MethodType(voxels, ax0)  # type: ignore

                ax0.voxels(  # type: ignore
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
                    s=100 * (bias.shape[0]) ** (-1 / 3),  # linear dimension #type: ignore
                    edgecolor="none",
                    c=color_bias,
                    zorder=-1,
                )
            else:
                raise ValueError(f"method {method} not recognized")

        ax0.grid(False)
        ax0.xaxis.pane.fill = False  # type: ignore
        ax0.yaxis.pane.fill = False  # type: ignore
        ax0.zaxis.pane.fill = False  # type: ignore

        ax0.set_xlim(*x_lim)  # type: ignore
        ax0.set_ylim(*y_lim)  # type: ignore
        ax0.set_zlim(*z_lim)  # type: ignore

        ax0.locator_params(nbins=4)

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
            axis="z",  # type: ignore
            label1On=False,
            label2On=False,
        )

        assert indices is not None

        if labels is not None:
            ax0.set_xlabel(labels[indices[0]], labelpad=-15)
            ax0.set_ylabel(labels[indices[1]], labelpad=-15)
            ax0.set_zlabel(labels[indices[2]], labelpad=-15)  # type: ignore

        # ax0.view_init(40, -30, 0)
        ax0.set_box_aspect(None, zoom=1.1)  # type: ignore

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
                collective_variable=collective_variable[_indices],
                margin=margin,
                indices=_indices,
                fesses=fesses,
                vmin=vmin,
                vmax=vmax,
                print_labels=print_labels,
                cmap=cmap,
                plot_std=plot_std,
                **scatter_kwargs,
            )

    @staticmethod
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

                if bb is not None and pp is not None:
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
            assert bb is not None
            min_val = bb[:, 0]
            max_val = bb[:, 1]

        else:
            max_val = jnp.max(jnp.array([jnp.max(ai.cv, axis=0) for ai in a]), axis=0)
            min_val = jnp.min(jnp.array([jnp.min(ai.cv, axis=0) for ai in a]), axis=0)

        if margin is not None:
            _max_val = max_val + (max_val - min_val) * margin
            _min_val = min_val - (max_val - min_val) * margin

            if metric is not None:
                max_val = vmap_decorator(lambda p, x, y: jnp.where(p, x, y))(pp, max_val, _max_val)
                min_val = vmap_decorator(lambda p, x, y: jnp.where(p, x, y))(pp, min_val, _min_val)
            else:
                max_val = _max_val
                min_val = _min_val

        from IMLCV.base.CV import CvTrans

        close = bool(jnp.allclose(max_val, min_val))

        use_macro_chunk = True

        def _f(color_data, nl, shmap, shmap_kwargs, close, dim, pp) -> jax.Array:
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
                verbose=True,
                macro_chunk=10000,
                shmap=False,
                jit_f=True,
            )

        else:
            _f2 = vmap_decorator(
                Partial_decorator(
                    _f,
                    nl=None,
                    shmap=None,
                    shmap_kwargs=None,
                    close=close,
                    dim=dim,
                    pp=pp,
                )
            )

            # _f = jit_decorator(_f)

            out = []

            for i, ai in enumerate(a):
                out.append(_f2(ai))
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

        return CombineTransformer.create(transformers=trans)


class CombineTransformer(Transformer):
    transformers: list[Transformer]

    @staticmethod
    def create(transformers: list[Transformer]) -> CombineTransformer:
        return CombineTransformer(
            transformers=transformers,
            # outdim=transformers[-1].outdim,
            descriptor=transformers[0].descriptor,
            pre_scale=transformers[0].pre_scale,
            post_scale=transformers[-1].post_scale,
        )

    def _fit(
        self,
        x: list[CV] | list[SystemParams],
        x_t: list[CV] | list[SystemParams] | None,
        w: list[jax.Array],
        # w_t: list[jax.Array],
        dlo: DataLoaderOutput,
        chunk_size=None,
        verbose=True,
        macro_chunk=1000,
        # **fit_kwargs,
    ) -> tuple[list[CV], list[CV], CvTrans, list[jax.Array] | None, tuple[str, ...] | None, jax.Array | None]:
        trans = None

        assert len(self.transformers) > 0, "No transformers to fit"

        # periods = []

        pass_trans = None

        for i, t in enumerate(self.transformers):
            print(f"fitting transformer {i + 1}/{len(self.transformers)}")

            if t.pass_trans:
                _, _, trans_t, _, _, _ = t._fit(
                    x,
                    x_t,
                    w,
                    dlo,
                    chunk_size=chunk_size,
                    verbose=verbose,
                    macro_chunk=macro_chunk,
                )

                if pass_trans is None:
                    pass_trans = trans_t
                else:
                    pass_trans *= trans

            else:
                x, x_t, trans_t, w, _, _ = t._fit(
                    x,
                    x_t,
                    w,
                    dlo,
                    chunk_size=chunk_size,
                    verbose=verbose,
                    macro_chunk=macro_chunk,
                    trans=pass_trans,
                )
                pass_trans = None

                if trans is None:
                    trans = trans_t
                else:
                    trans *= trans_t

            # periods = [per]

        assert trans is not None

        x = cast(list[CV], x)
        x_t = cast(list[CV], x_t)

        return x, x_t, trans, w, None, None


class IdentityTransformer(Transformer):
    def _fit(
        self,
        x: list[CV] | list[SystemParams],
        x_t: list[CV] | list[SystemParams] | None,
        w: list[jax.Array],
        # w_t: list[jax.Array],
        dlo: DataLoaderOutput,
        chunk_size=None,
        verbose=True,
        macro_chunk=1000,
        **fit_kwargs,
    ) -> tuple[list[CV], list[CV] | None, CvTrans, list[jax.Array] | None, jax.Array | None]:
        assert isinstance(x, list) and isinstance(x_t, list), "x and x_t must be lists"

        assert isinstance(x[0], CV), "x must be a list of CV objects"
        assert isinstance(x_t[0], CV), "x_t must be a list of CV objects"

        x = cast(list[CV], x)
        x_t = cast(list[CV], x_t)

        return x, x_t, identity_trans, w, None, None


class CvTransTransformer(Transformer):
    trans: CvTrans

    def _fit(
        self,
        x: list[CV] | list[SystemParams],
        x_t: list[CV] | list[SystemParams] | None,
        w: list[jax.Array],
        # w_t: list[jax.Array],
        dlo: DataLoaderOutput,
        chunk_size: int | None = None,
        verbose=True,
        macro_chunk=1000,
        # **fit_kwargs,
    ) -> tuple[list[CV], list[CV], CvTrans, list[jax.Array] | None, jax.Array | None]:
        x, x_t = dlo.apply_cv(
            x=x,
            x_t=x_t,
            f=self.trans,
            chunk_size=chunk_size,
            verbose=verbose,
            macro_chunk=macro_chunk,
            jit_f=True,
        )

        return x, x_t, self.trans, w, None, None
