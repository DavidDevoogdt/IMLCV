from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp

# import matplotlib as mpl
from IMLCV.base.bias import Bias, NoneBias
from IMLCV.base.CV import CollectiveVariable, CvTrans
from IMLCV.base.dataobjects import CV, CvMetric, ShmapKwargs, SystemParams
from IMLCV.base.datastructures import MyPyTreeNode
from IMLCV.base.plot import plot_app
from IMLCV.base.UnitsConstants import kjmol
from IMLCV.implementations.CV import _scale_cv_trans, identity_trans, scale_cv_trans

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
        samples_per_bin=20,
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
                smoothing=0.1 / (kjmol**2),
            )

            plot_app(
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

            plot_app(
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
                smoothing=0.1 / (kjmol**2),
            )

            if plot:
                plot_app(
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
                smoothing=0.1 / (kjmol**2),
                weights_std=dlo._weights_std,
            )
            print(f"done transforming FES")

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
            plot_app(
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

            plot_app(
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

        # pass_trans = None

        for i, t in enumerate(self.transformers):
            print(f"fitting transformer {i + 1}/{len(self.transformers)}")

            # if t.pass_trans:
            #     _, _, trans_t, _, _, _ = t._fit(
            #         x,
            #         x_t,
            #         w,
            #         dlo,
            #         chunk_size=chunk_size,
            #         verbose=verbose,
            #         macro_chunk=macro_chunk,
            #     )

            #     if pass_trans is None:
            #         pass_trans = trans_t
            #     else:
            #         pass_trans *= trans

            # else:
            x, x_t, trans_t, w, _, _ = t._fit(
                x,
                x_t,
                w,
                dlo,
                chunk_size=chunk_size,
                verbose=verbose,
                macro_chunk=macro_chunk,
                # trans=pass_trans,
            )
            # pass_trans = None

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
