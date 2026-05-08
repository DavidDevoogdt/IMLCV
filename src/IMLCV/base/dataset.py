from __future__ import annotations

from functools import partial
from typing import Callable, TypeVar, cast

import jax
import jax.numpy as jnp
from jax import Array

from IMLCV.base.bias import Bias, BiasModify, CompositeBias, GridBias, StdBias
from IMLCV.base.CV import (
    CollectiveVariable,
    CvTrans,
)
from IMLCV.base.dataobjects import (
    CV,
    CvMetric,
    NeighbourList,
    NeighbourListInfo,
    NeighbourListUpdate,
    ShmapKwargs,
    StaticMdInfo,
    SystemParams,
    TrajectoryInfo,
    macro_chunk_map,
    macro_chunk_map_fun,
    padded_shard_map,
    padded_vmap,
)
from IMLCV.base.decoratros import MyPyTreeNode, Partial_decorator, jit_decorator, vmap_decorator
from IMLCV.base.koopman import KoopmanModel
from IMLCV.base.reweight import _solve_wham
from IMLCV.base.UnitsConstants import (
    boltzmann,
    kelvin,
    kjmol,
)
from IMLCV.implementations.bias import GridMaskBias, RbfBias, _clip

T = TypeVar("T")
X = TypeVar("X", "CV", "SystemParams", "NeighbourList")
X2 = TypeVar("X2", "CV", "SystemParams", "NeighbourList")


class WeightOutput(MyPyTreeNode):
    weight: list[Array]  # e^( beta U - F_i ), ammount of bias
    p_select: list[Array]  # selection probability, due to sampling
    F: list[Array]  # free energy, if available
    density: list[Array]  # density of bin ik
    n_bin: list[Array]  # number of points in bin_ik
    n_eff_bin: list[Array] | None = None  # effective number of points in bin_ik
    grid_nums: list[Array] | None = None  # grid numbers of each point

    p_std: list[Array] | None = (
        None  # standard deviation of p_select. standerd deviation for bin is obtained by sum_i  p_i**2 p_std_i**2/ (sum_i p_i)**2
    )

    FES_bias: Bias | None = None
    FES_bias_std: Bias | None = None

    labels: list[int] | None = None

    frac_full: float

    bounds: jax.Array | None = None
    n_bins: int | None = None  # total number of bins
    n_hist: int | None = None  # number of bins per dimension

    N_samples_eff: int | None = None


class DataLoaderOutput(MyPyTreeNode):
    sti: StaticMdInfo
    ti: list[TrajectoryInfo]
    collective_variable: CollectiveVariable
    labels: list[Array] | None = None
    nl: list[NeighbourList] | NeighbourList | None = None
    nl_t: list[NeighbourList] | NeighbourList | None = None
    time_series: bool = False
    tau: float | None = None
    bias: list[Bias] | None = None
    ground_bias: Bias | None = None

    scaled_tau: bool = False

    frac_full: float = 1.0
    bounds: jax.Array | None = None
    n_hist: int | None = None

    @property
    def sp(self) -> list[SystemParams] | None:
        return [ti.sp for ti in self.ti]

    @property
    def sp_t(self) -> list[SystemParams] | None:
        return [ti.sp_t for ti in self.ti] if self.ti[0].sp_t is not None else None

    @property
    def cv(self) -> list[CV] | None:
        return [ti.CV for ti in self.ti] if self.ti[0].CV is not None else None

    @property
    def cv_t(self) -> list[CV] | None:
        return [ti.CV_t for ti in self.ti] if self.ti[0].CV_t is not None else None

    @property
    def _weights(self) -> list[Array] | None:
        return [ti.w for ti in self.ti] if self.ti[0].w is not None else None

    @property
    def _rho(self) -> list[Array] | None:
        return [ti.rho for ti in self.ti] if self.ti[0].rho is not None else None

    @property
    def _rho_t(self) -> list[Array] | None:
        return [ti.rho_t for ti in self.ti] if self.ti[0].rho_t is not None else None

    @property
    def _weights_t(self) -> list[Array] | None:
        return [ti.w_t for ti in self.ti] if self.ti[0].w_t is not None else None

    @property
    def e_pot(self) -> list[Array] | None:
        return [ti.e_pot for ti in self.ti] if self.ti[0].e_pot is not None else None

    @property
    def _weights_std(self) -> list[Array] | None:
        return [ti.sigma for ti in self.ti] if self.ti[0].sigma is not None else None

    @property
    def _w_dyn(self) -> list[Array] | None:
        return [ti.w_dyn for ti in self.ti] if self.ti[0].w_dyn is not None else None

    @staticmethod
    def get_histo(
        data_nums: list[CV],
        weights: None | list[Array] = None,
        log_w: bool = False,
        macro_chunk: int | None = 320,
        verbose: bool = False,
        shape_mask: Array | None = None,
        nn: int = -1,
        f_func: Callable | None = None,
        observable: list[CV] | None = None,
    ) -> jax.Array:
        if f_func is None:

            def _f_func(x, nl):
                return x
        else:
            _f_func = f_func

        if shape_mask is not None:
            nn = int(jnp.sum(shape_mask)) + 1  # +1 for bin -1

        nn_range = jnp.arange(nn)

        if shape_mask is not None:
            nn_range = nn_range.at[-1].set(-1)

        h = jnp.zeros((nn,))

        def _get_histo(
            x: tuple[Array | None, Array, Array | None],
            cv_i: CV,
            obs: CV | None,
            weights: Array | None,
            _weights_t=None,
            _dweights_t=None,
        ) -> tuple[Array | None, Array, Array | None]:
            alpha_factors, h, obs_array = x

            cvi = cv_i.cv[:, 0]

            if not log_w:
                wi = jnp.ones_like(cvi) if weights is None else weights
                h = h.at[cvi].add(wi)

                if obs is not None:
                    # print(f"{obs_array.shape=} {cvi.shape=} {obs.cv.shape=} {wi.shape=}")

                    u = obs.cv * wi[:, None]

                    obs_array = obs_array.at[cvi, :].add(u)  # type: ignore

                return (alpha_factors, h, obs_array)

            if alpha_factors is None:
                alpha_factors = jnp.full_like(h, -jnp.inf)

            # this log of hist. all calculations are done in log space to avoid numerical issues
            w_log = jnp.zeros_like(cvi) if weights is None else weights

            @partial(vmap_decorator, in_axes=(0, None, None, 0))
            def get_max(ref_num, data_num, log_w, alpha):
                val = jnp.where(data_num == ref_num, log_w, -jnp.inf)

                max_new = jnp.max(val)  # type: ignore
                max_tot = jnp.max(jnp.array([alpha, max_new]))

                return max_tot, max_new > alpha, max_tot - alpha

            alpha_factors, alpha_factors_changed, d_alpha = get_max(nn_range, cvi, w_log, alpha_factors)

            # shift all probs by the change in alpha
            h: Array = jnp.where(alpha_factors_changed, h - d_alpha, h)  # type:ignore

            if obs is not None:
                obs_array: jax.Array = jnp.where(alpha_factors_changed[:, None], obs_array - d_alpha, obs_array)  # type:ignore

            m = alpha_factors != -jnp.inf

            p_new = jnp.zeros_like(h)
            p_new = p_new.at[cvi].add(jnp.exp(w_log - alpha_factors[cvi]))

            h: jax.Array = jnp.where(m, jnp.log(jnp.exp(h) + p_new), h)

            if obs is not None:
                obs_array = obs_array.at[cvi].add(obs.cv * jnp.exp(w_log - alpha_factors[cvi]))  # type: ignore

            return (alpha_factors, h, obs_array)

        (alpha_factors, h, obs) = macro_chunk_map_fun(
            f=_f_func,
            # op=data_nums[0].stack,
            y=data_nums,
            y_t=None if observable is None else observable,
            macro_chunk=macro_chunk,
            verbose=verbose,
            chunk_func=_get_histo,
            chunk_func_init_args=(
                None,
                h,
                jnp.zeros((nn, *observable[0].shape[1:])) if observable is not None else None,
            ),
            w=weights,
            jit_f=True,
        )

        if log_w:
            assert alpha_factors is not None
            h += alpha_factors

            if obs is not None:
                obs += alpha_factors[:, None]

        if shape_mask is not None:
            # print(f"{h[-1]=}")
            h = h[:-1]

            if observable is not None:
                assert obs is not None
                obs = obs[:-1]

        if observable is not None:
            return h, obs

        return h

    @staticmethod
    def _histogram(
        metric: CvMetric,
        n_grid=40,
        grid_bounds=None,
        chunk_size=None,
        chunk_size_mid=1,
    ):
        bins, _, _, cv_mid, bounds = metric.grid(n=n_grid, bounds=grid_bounds)

        # print(f"{bounds=} {grid_bounds=}")

        @partial(CvTrans.from_cv_function, mid=cv_mid, bounds=bounds)
        def closest_trans(cv: CV, _nl, shmap, shmap_kwargs, mid: CV, bounds: jax.Array):
            m = jnp.argmin(jnp.sum((mid.cv - cv.cv) ** 2, axis=1), keepdims=True)
            m = jnp.where(jnp.all(jnp.logical_and(cv.cv > bounds[:, 0], cv.cv < bounds[:, 1])), m, -1)

            return cv.replace(cv=m)

        nums = closest_trans.compute_cv(cv_mid, chunk_size=chunk_size)[0].cv

        return (
            cv_mid,
            nums,
            bins,
            closest_trans,
            Partial_decorator(DataLoaderOutput.get_histo, nn=cv_mid.shape[0]),
        )

    @staticmethod
    def _unstack_weights(stack_dims, weights: Array) -> list[Array]:
        return [
            d.cv.reshape((-1,))
            for d in CV(
                cv=jnp.expand_dims(weights, 1),
                _stack_dims=stack_dims,
            ).unstack()
        ]

    @staticmethod
    def norm_w(w_stacked: jax.Array, ret_n=False) -> jax.Array | tuple[jax.Array, jax.Array]:
        n = jnp.sum(w_stacked)

        if ret_n:
            return w_stacked / n, n

        return w_stacked / n

    @staticmethod
    def check_w(w_stacked: jax.Array) -> jax.Array:  # type:ignore
        if jnp.any(jnp.isnan(w_stacked)):
            print(f"WARNING: w_stacked has {jnp.sum(jnp.isnan(w_stacked))} nan values")
            w_stacked: jax.Array = jnp.where(jnp.isnan(w_stacked), 0, w_stacked)

        if jnp.any(jnp.isinf(w_stacked)):
            print(f"WARNING: w_stacked has {jnp.sum(jnp.isinf(w_stacked))} inf values")
            w_stacked = jnp.where(jnp.isinf(w_stacked), 0, w_stacked)

        if jnp.any(w_stacked < 0):
            print(f"WARNING: w_stacked has {jnp.sum(w_stacked < 0)} neg values")
            w_stacked = jnp.where(w_stacked < 0, 0, w_stacked)

        if jnp.sum(w_stacked) < 1e-16:
            print("WARNING: all w_Stacked values are zero")
            raise

        if len(w_stacked) == 0:
            print("WARNING: len w_Stacked is zero")

        return w_stacked

    def koopman_weight(
        self,
        w: list[Array] | None = None,
        w_t: list[Array] | None = None,
        dw: list[Array] | None = None,
        samples_per_bin: int = 50,
        max_bins: int | float = 1e5,
        out_dim: int = -1,
        chunk_size: int | None = None,
        indicator_CV: bool = True,
        koopman_eps: float = 0,
        koopman_eps_pre: float = 0,
        cv_0: list[CV] | None = None,
        cv_t: list[CV] | None = None,
        macro_chunk: int = 1000,
        verbose: bool = False,
        max_features_koopman: int = 5000,
        margin: float = 0.1,
        add_1=True,
        only_diag: bool = False,
        calc_pi: bool = True,
        sparse: bool = False,
        output_w_corr: bool = False,
        correlation: bool = True,
        return_km: bool = False,
        labels: jax.Array | list[int] | None = None,
        koopman_kwargs: dict = {},
        shrink=False,
        shrinkage_method="bidiag",
    ) -> tuple[list[Array], list[Array] | None, list[Array] | None, list[KoopmanModel] | None]:
        if cv_0 is None:
            cv_0 = self.cv

        if add_1 is None:
            add_1 = not indicator_CV

        sd = [a.shape[0] for a in cv_0]

        ndim = cv_0[0].shape[1]

        if cv_t is None:
            assert self.cv_t is not None
            cv_t = self.cv_t

        def unstack_w(w_stacked, stack_dims=None):
            if stack_dims is None:
                stack_dims = sd
            return self._unstack_weights(stack_dims, w_stacked)

        if w is None:
            assert self._weights is not None
            w = self._weights

        if w_t is None:
            assert self._weights_t is not None
            w_t = self._weights_t

        assert self._weights_t is not None

        if dw is None:
            dw = self._w_dyn

        assert dw is not None

        if verbose:
            print("koopman weights")

        if labels is None:
            labels = jnp.array(self.labels) if self.labels is not None else jnp.ones((len(cv_0),))
        else:
            labels = jnp.array(labels)

        unique_labels = jnp.unique(labels)

        if len(unique_labels) > 1:
            print(f"runninbounds_from_cvg koopman weights for {len(unique_labels)} disconnected regions")

        if indicator_CV:
            print("getting bounds")
            grid_bounds, _, constants = CvMetric.bounds_from_cv(
                cv_0,
                margin=margin,
                # chunk_size=chunk_size,
                # n=20,
            )

            if constants:
                print("not performing koopman weighing because of constants in cv")
                # koopman = False

                out_0 = w
                # out_1 = w_t
                out_2 = None
                out_3 = None
                out_4 = None

                return out_0, out_2, out_3, out_4

            tot_samples = sum(sd)

            n_hist = CvMetric.get_n(
                samples_per_bin=samples_per_bin,
                samples=tot_samples,
                n_dims=ndim,
                max_bins=max_bins,
            )

            if verbose:
                print(f"using {n_hist=}")

            cv_mid, nums, bins, closest, get_histo = DataLoaderOutput._histogram(
                metric=self.collective_variable.metric,
                n_grid=n_hist,
                grid_bounds=grid_bounds,
                chunk_size=chunk_size,
            )

            grid_nums, grid_nums_t = self.apply_cv(closest, cv_0, cv_t, chunk_size=chunk_size, macro_chunk=macro_chunk)
            assert grid_nums_t is not None

        _sd = sd
        _cv_0 = cv_0
        _cv_t = cv_t
        _weights = w
        _weights_t = w_t
        _dynamic_weights = dw

        if indicator_CV:
            _grid_nums = grid_nums
            _grid_nums_t = grid_nums_t

        _rho = self._rho
        _rho_t = self._rho_t
        assert _rho is not None
        assert _rho_t is not None

        w_out: list[jax.Array] = [jnp.array(0.0)] * len(sd)
        w_corr_out: list[jax.Array] = [jnp.array(0.0)] * len(sd)

        w_out_t: list[jax.Array] = [jnp.array(0.0)] * len(sd)
        w_corr_out_t: list[jax.Array] = [jnp.array(0.0)] * len(sd)

        km_out: list[KoopmanModel] = []

        for label in unique_labels:
            print(f"calculating koopman weights for {label=}")

            label_mask = [int(a) for a in jnp.argwhere(labels == label).reshape((-1,))]

            # idx_inv = jnp.arange()

            sd = []
            cv_0 = []
            cv_t = []
            weights = []
            weights_t = []
            weights_dyn = []

            if indicator_CV:
                grid_nums = []
                grid_nums_t = []
            rho = []
            rho_t = []

            for idx in label_mask:
                sd.append(_sd[idx])
                cv_0.append(_cv_0[idx])
                cv_t.append(_cv_t[idx])
                weights.append(_weights[idx])
                weights_t.append(_weights_t[idx])
                weights_dyn.append(_dynamic_weights[idx])

                if indicator_CV:
                    grid_nums.append(_grid_nums[idx])
                    grid_nums_t.append(_grid_nums_t[idx])
                rho.append(_rho[idx])
                rho_t.append(_rho_t[idx])

            if indicator_CV:
                print("getting bounds")

                w_pos = [(a > 0) * 1.0 for a in weights]

                hist = get_histo(grid_nums, w_pos)
                hist_t = get_histo(grid_nums_t, w_pos)

                mask = jnp.argwhere(jnp.logical_and(hist > 0, hist_t > 0)).reshape(-1)

                # print(f"{label=} {mask=} {hist.shape=}")

                @partial(CvTrans.from_cv_function, mask=mask)
                def get_indicator(cv: CV, nl, shmap, shmap_kwargs, mask):
                    out = jnp.zeros((hist.shape[0],))  # type: ignore
                    out = out.at[cv.cv].set(1)
                    out = jnp.take(out, mask)

                    print(f"{out=}")

                    return cv.replace(cv=out)

                cv_km = grid_nums
                cv_km_t = grid_nums_t

                tr = get_indicator

            else:
                cv_km = cv_0
                cv_km_t = cv_t

                tr = None

            if verbose:
                print("constructing koopman model")

            kpn_kw = dict(
                cv_0=cv_km,
                cv_t=cv_km_t,
                nl=None,
                w=weights,
                w_t=weights_t,
                weights_dyn=weights_dyn,
                dynamic_weights=weights_dyn,
                rho=rho,
                rho_t=rho_t,
                add_1=add_1,
                # method="tcca",
                symmetric=False,
                chunk_size=chunk_size,
                macro_chunk=macro_chunk,
                verbose=verbose,
                eps=koopman_eps,
                eps_pre=koopman_eps_pre,
                trans=tr,
                max_features=max_features_koopman,
                max_features_pre=max_features_koopman,
                only_diag=only_diag,
                calc_pi=True,
                scaled_tau=self.scaled_tau,
                sparse=sparse,
                out_dim=out_dim,
                correlation=correlation,
                shrink=shrink,
                shrinkage_method=shrinkage_method,
            )
            kpn_kw |= koopman_kwargs

            km = self.koopman_model(**kpn_kw)  # type: ignore

            try:
                w, w_t, w_corr_t, w_corr_t, b = km.koopman_weight(
                    chunk_size=chunk_size,
                    macro_chunk=macro_chunk,
                    verbose=verbose,
                )
            except Exception as e:
                print(f"koopman reweighing failed for {label=}")
                print(e)

                w_unstacked = weights
                w_unstacked_t = weights_t
                w_corr = [jnp.ones_like(a) for a in weights]
                b = False

            if not b:
                print(f"koopman reweighing failed for {label=}")

            def process(w_corr, w_unstacked=None):
                print(f"{jnp.sum(jnp.hstack(w_corr) < 0)=}  {jnp.sum(~jnp.isfinite(jnp.hstack(w_corr)))=}")

                w_stacked_log = jnp.log(jnp.hstack(w_corr))

                if w_unstacked is not None:
                    w_stacked_log += jnp.log(jnp.hstack(w_unstacked))

                w_stacked_log -= jnp.max(w_stacked_log)
                w_stacked = jnp.exp(w_stacked_log)

                w_stacked = self.check_w(w_stacked)
                w_stacked: jax.Array = self.norm_w(w_stacked)  # type:ignore
                w_unstacked = self._unstack_weights(sd, w_stacked)

                return w_unstacked

            w_unstacked = process(w_corr, weights)
            # print(f"{w_corr_t=} {weights_t=}")
            w_unstacked_t = process(w_corr_t, weights_t)

            if w_corr is not None and output_w_corr:
                w_corr = process(w_corr)

            for n, idx in enumerate(label_mask):
                w_out[idx] = w_unstacked[n]
                w_out_t[idx] = w_unstacked_t[n]
                if w_corr is not None and output_w_corr:
                    w_corr_out[idx] = w_corr[n]

            if return_km:
                km_out.append(None)

        sd = _sd

        out_0 = w_out
        out_1 = w_out_t
        out_2 = None
        out_3 = None

        if output_w_corr:
            out_2 = w_corr_out

        if return_km:
            out_3 = km_out

        return out_0, out_1, out_2, out_3

    def rescale_rho_T(self, factor, rho=None):
        if rho is None:
            rho = self._rho

        if factor == 1:
            return self._rho

        assert factor > 0, "factor must be >0"

        energies = [ti_i.e_pot for ti_i in self.ti]

        beta = 1 / (self.sti.T * boltzmann)

        new_rho = []

        for rho_i, e_i in zip(rho, energies):
            rho_new_i = jnp.exp(jnp.log(rho_i) + beta * (1 - 1 / factor) * e_i)
            new_rho.append(rho_new_i)

        return new_rho

    @staticmethod
    def wham_weight(
        self,
        samples_per_bin: int = 10,
        min_samples_per_bin: int = 1,
        n_max: float | int = 1e5,
        n_max_lin: int | None = None,
        chunk_size: int | None = None,
        biases: list[Bias] = None,
        cv_0: list[CV] | None = None,
        cv_t: list[CV] | None = None,
        macro_chunk: int | None = 1000,
        verbose: bool = False,
        margin: float = 0.0,
        compute_std: bool = False,
        sparse_inverse: bool = True,
        recalc_bounds=True,
        correlation_method: str | None = None,
        blav_pair=False,
        n_subgrids: int = 10,
        n_hist: int | None = None,
        get_e_wall=True,
    ) -> WeightOutput:
        if cv_0 is None:
            cv_0 = self.cv

        if cv_t is None:
            cv_t = self.cv_t

        # TODO:https://pubs.acs.org/doi/pdf/10.1021/acs.jctc.9b00867

        # get raw rescaling
        u_unstacked = []
        e_unstacked = []

        beta = 1 / (self.sti.T * boltzmann)

        for ti_i in self.ti:
            e = ti_i.e_bias

            if e is None:
                if verbose:
                    print("WARNING: no bias enerrgy found")
                e = jnp.zeros((ti_i.shape,))

            # e -= jnp.min(e)

            u = beta * e

            u_unstacked.append(u)

            e_unstacked.append(ti_i.e_pot * beta)

        (
            frac_full,
            labels,
            x_labels,
            num_labels,
            grid_nums_mask,
            get_histo,
            bins,
            cv_mid,
            hist_mask,
            _,
            bounds,
            tau_i,
            _,
            n_hist,
        ) = self.get_bincount(
            cv_0,
            n_max=n_max,
            n_max_lin=n_max_lin,
            n_hist=n_hist,
            margin=margin,
            chunk_size=chunk_size,
            samples_per_bin=samples_per_bin,
            min_samples_per_bin=min_samples_per_bin,
            macro_chunk=macro_chunk,
            verbose=verbose,
            recalc_bounds=recalc_bounds,
            correlation_method=correlation_method,
        )

        if correlation_method is not None:
            print(f"{tau_i=}")

        len_i = len(u_unstacked)
        len_k = jnp.sum(hist_mask)
        log_U_ik = jnp.full((len_i, len_k), jnp.inf)
        H_ik = jnp.zeros((len_i, len_k))

        label_i = []

        print("generating centered offsets")

        # 2. Generate centered offsets
        def get_centered_offsets(a):
            w = a[1] - a[0]
            step = w / n_subgrids
            # Start at half a step, end at w minus half a step, then shift to be zero-centered
            return jnp.linspace(step / 2, w - step / 2, n_subgrids) - (w / 2)

        dx = jnp.array(jnp.meshgrid(*[get_centered_offsets(a) for a in bins], indexing="ij"))
        dx = CV(cv=dx.reshape((dx.shape[0], -1)).T)
        n_elem = dx.cv.shape[0]

        print(f"{dx.shape=}")

        @jax.jit
        @partial(vmap_decorator, in_axes=(0, None), out_axes=0)
        def get_b(center: CV, bias_i: Bias):
            biases, _ = bias_i.compute_from_cv(dx.replace(cv=dx.cv + center.cv))

            u = -beta * biases
            m = jnp.max(u)

            return jnp.log(jnp.sum(jnp.exp(u - m))) + m - jnp.log(n_elem)

        integral_tlog = False
        print(f"using {'integral' if integral_tlog else 'direct'} method to compute log_U_ik")

        for i in range(len_i):
            if verbose:
                print(".", end="", flush=True)
                if (i + 1) % 100 == 0:
                    print("")

            t_log_H_ik: jax.Array = get_histo(
                [grid_nums_mask[i]],
                None,
                log_w=True,
                shape_mask=hist_mask,
                macro_chunk=macro_chunk,
            )  # type:ignore

            # 2 possible ways to compute log_U_ik, the integral is more stable, but the direct way samples the actual bias and density.
            if integral_tlog:
                # e^(-t_log_U_ik) = int e^(-u) dx/ int dx for x in bin
                t_log_U_ik = -get_b(cv_mid[hist_mask], biases[i])

            else:
                # e^(-t_log_U_ik) = < e^(-u) >_b = < e^u e^(-1) >/< e^u>

                t_log_U_ik: jax.Array = (
                    get_histo(
                        [grid_nums_mask[i]],
                        [u_unstacked[i]],
                        log_w=True,
                        shape_mask=hist_mask,
                        macro_chunk=macro_chunk,
                    )
                    - t_log_H_ik
                )

                t_log_U_ik = jnp.where(jnp.isfinite(t_log_U_ik), t_log_U_ik, jnp.inf)

            # print(f"{t_log_U_ik=} {t_log_U_ik_2=} {t_log_U_ik - t_log_U_ik_2  =}")

            H_ik = H_ik.at[i, :].set(jnp.exp(t_log_H_ik))
            log_U_ik = log_U_ik.at[i, :].set(t_log_U_ik)

            labels_i = jnp.sum(x_labels[:, grid_nums_mask[i].cv.reshape(-1)], axis=1)
            label_i.append(jnp.argmax(labels_i))

        w_ik = jnp.full((log_U_ik.shape[0], log_U_ik.shape[1]), 0.0)
        ps_ik = jnp.full((log_U_ik.shape[0], log_U_ik.shape[1]), 0.0)
        dens_ik = jnp.full((log_U_ik.shape[0], log_U_ik.shape[1]), 0.0)
        mask_ik = jnp.full((log_U_ik.shape[0], log_U_ik.shape[1]), False)

        F_k = jnp.full((log_U_ik.shape[1],), jnp.inf)
        F_i = jnp.full((log_U_ik.shape[0],), jnp.inf)

        if compute_std:
            sigma_Fi = jnp.full((log_U_ik.shape[0],), jnp.inf)
            sigma_Fk = jnp.full((log_U_ik.shape[1],), jnp.inf)

        tau_i = tau_i

        label_i = jnp.array(label_i).reshape((-1))

        print(f"{jnp.max(log_U_ik)=}")

        for nl in range(0, num_labels):
            mk = labels == nl
            nk = int(jnp.sum(mk))
            arg_mk = jnp.argwhere(mk).reshape((-1,))

            mi = label_i == nl
            ni = jnp.sum(mi)
            arg_mi = jnp.argwhere(mi).reshape((-1,))

            if nk == 0 or ni == 0:
                print(f"skipping label {nl} with {nk} bins and {ni} trajectories")
                continue

            print(f"running wham with {nk} bins and {ni} trajectories")

            # # this is just a meshgrid

            @partial(vmap_decorator, in_axes=(0, None))
            @partial(vmap_decorator, in_axes=(None, 0))
            def _get_arg_ik(i, k):
                return jnp.array([i, k])

            arg_ik = _get_arg_ik(arg_mi, arg_mk)

            H_ik_nl = H_ik[arg_ik[:, :, 0], arg_ik[:, :, 1]]
            log_U_ik_nl = log_U_ik[arg_ik[:, :, 0], arg_ik[:, :, 1]]
            tau_i_nl = tau_i[arg_mi]

            (
                log_w_ik_nl,
                log_ps_ik_nl,
                F_i_nl,
                log_dens_ik_nl,
                mask_ik_nl,
                F_k_new,
                # sigma_Fi_nl,
                sigma_Fk_nl,
            ) = _solve_wham(
                log_U_ik_nl,
                H_ik_nl / tau_i_nl[:, None],
                verbose=verbose,
                compute_std=compute_std,
            )

            H_ik_nl = jnp.where(mask_ik_nl, H_ik_nl, 0.0)

            mask_ik = mask_ik.at[arg_ik[:, :, 0], arg_ik[:, :, 1]].set(mask_ik_nl)
            w_ik = w_ik.at[arg_ik[:, :, 0], arg_ik[:, :, 1]].set(jnp.exp(log_w_ik_nl))
            ps_ik = ps_ik.at[arg_ik[:, :, 0], arg_ik[:, :, 1]].set(jnp.exp(log_ps_ik_nl))
            dens_ik = dens_ik.at[arg_ik[:, :, 0], arg_ik[:, :, 1]].set(jnp.exp(log_dens_ik_nl))
            H_ik = H_ik.at[arg_ik[:, :, 0], arg_ik[:, :, 1]].set(H_ik_nl)

            F_k = F_k.at[arg_mk].set(F_k_new)
            F_i = F_i.at[arg_mi].set(F_i_nl)

            if compute_std:
                sigma_Fk = sigma_Fk.at[arg_mk].set(sigma_Fk_nl)
                # sigma_Fi = sigma_Fi.at[arg_mi].set(sigma_Fi_nl)

        N_i_tau = jnp.sum(H_ik, axis=1) / tau_i
        # print(f"{N_i_tau=}")

        F_k /= beta
        # F_i /= beta
        log_U_ik /= beta
        if compute_std:
            sigma_Fk /= beta
            # sigma_Fi /= beta

        print(f"{jnp.max(F_k)/kjmol=} {jnp.min(F_k)/kjmol=}  {F_k/kjmol=}")

        if compute_std:
            print(f"{jnp.max(sigma_Fk)/kjmol=} {jnp.min(sigma_Fk)/kjmol=}  {sigma_Fk/kjmol=}")
            # print(f"{jnp.max(sigma_Fi)/kjmol=} {jnp.min(sigma_Fi)/kjmol=}  {sigma_Fi/kjmol=}")

        # print(f"valid i: {jnp.any(mask_ik, axis=1)=} valid_k: {jnp.any(mask_ik, axis=0)=}")

        s = 0.0

        s_ps_k = jnp.zeros((len_k + 1,))

        w_out = []
        ps_out = []
        dens_out = []
        F_k_out = []
        n_bin_out = []
        n_eff_out = []
        grid_nums_out = []
        sigma_out = []

        # E_k = jnp.zeros((len_k,))

        dw_mins = []
        dw_maxs = []

        log_w_tot_sq = jnp.full((len_k + 1,), -jnp.inf)
        log_w_tot_k = jnp.full((len_k + 1,), -jnp.inf)

        H_ik_req = jnp.zeros((len_i, len_k + 1))

        k_count = jnp.array((len_k,))

        def _log_add(a, b, k):
            _lw = a.at[k].max(b)

            _lw = jnp.where(jnp.isfinite(_lw), _lw, 0.0)

            out = jnp.log(jnp.exp(a - _lw).at[k].add(jnp.exp(b - _lw[k]))) + _lw

            # print(f"{normal - fancy =}")

            return out

        # print(f"{w_ik=}")

        for i in range(len_i):
            k_arr = grid_nums_mask[i].cv.reshape(-1)

            H_ik_req = H_ik_req.at[i, k_arr].add(1)

            mask = jnp.logical_or(jnp.logical_not(mask_ik[i, k_arr]), k_arr == -1)

            # print(f"{i=} {k_arr=} ")
            # print(f"{jnp.sum(mask)=}")

            grid_nums_out.append(k_arr)

            # get difference between binned and actual bias

            dw = jnp.where(mask, 0.0, jnp.exp(u_unstacked[i] - log_U_ik[i, k_arr] * beta))
            dw_inv = jnp.where(dw > 0, 1 / dw, 0)
            # print(f"{1/dw=}")

            # dw_mins.append(jnp.min(dw))
            # dw_maxs.append(jnp.max(dw))

            w = jnp.where(mask, 0.0, w_ik[i, k_arr]) * dw

            # print(f"{w=} ")

            ps = jnp.where(mask, 0.0, ps_ik[i, k_arr])
            dens = jnp.where(mask, 0.0, dens_ik[i, k_arr]) * dw_inv
            n_bin = jnp.where(mask, 0.0, H_ik[i, k_arr])
            n_eff = jnp.where(mask, 0.0, H_ik[i, k_arr] / tau_i[i])

            # https://pubs.acs.org/doi/pdf/10.1021/acs.jctc.3c01423?ref=article_openPDF

            w_out.append(w)
            ps_out.append(ps)
            dens_out.append(dens)
            n_bin_out.append(n_bin)
            n_eff_out.append(n_eff)
            F_k_out.append(jnp.where(mask, jnp.inf, F_k[k_arr]))

            s += jnp.sum(jnp.where(mask, 0.0, jnp.exp(jnp.log(w) + jnp.log(dens) + jnp.log(ps) - jnp.log(n_bin))))

            log_w_tot_k = _log_add(
                log_w_tot_k,
                jnp.log(w) + jnp.log(dens) - jnp.log(n_bin) + jnp.log(ps),
                k_arr,
            )

            log_w_tot_sq = _log_add(
                log_w_tot_sq,
                2 * (jnp.log(w) + jnp.log(dens) - jnp.log(n_bin) + jnp.log(ps)),
                k_arr,
            )

            s_ps_k = s_ps_k.at[k_arr].add(jnp.exp(jnp.log(ps) - jnp.log(n_bin)))

        assert jnp.allclose(s_ps_k[:-1], 1, rtol=1e-04, atol=1e-04), (
            f"ps sum to {s_ps_k[:-1]} instead of 1, check your wham solution and histogram binning"
        )

        print(f"{s=}")

        # if compute_std:

        # new value of sigma
        sigma_ind = jnp.full((len_k + 1,), jnp.inf)
        sigma_ind = sigma_ind.at[:-1].set(sigma_Fk)
        sigma_ind = sigma_ind / jnp.exp(log_w_tot_sq / 2 - log_w_tot_k)

        if compute_std:
            sigma_out = []
            log_simga_sq_k_test = jnp.full((len_k + 1,), -jnp.inf)

            for i in range(len_i):
                k_arr = grid_nums_mask[i].cv.reshape(-1)

                # w = w_out[i] * dens_out[i]

                sigma = sigma_ind[k_arr]

                sigma_out.append(sigma)

                log_simga_sq_k_test = _log_add(
                    log_simga_sq_k_test,
                    2
                    * (
                        jnp.log(sigma)
                        + jnp.log(ps_out[i])
                        - jnp.log(n_bin_out[i])
                        + jnp.log(dens_out[i])
                        + jnp.log(w_out[i])
                    ),
                    k_arr,
                )

            sigma_k_test = jnp.exp(0.5 * log_simga_sq_k_test - log_w_tot_k)

            print(f"{( jnp.linalg.norm(sigma_k_test[:-1] -sigma_Fk)/kjmol)=}   ")

        mask_k = jnp.any(mask_ik, axis=0)

        # print(f"{E_k/kjmol=}")

        # print(f"{N_i_tau=}")

        output_weight_kwargs = {
            "weight": w_out,
            "p_select": ps_out,
            "n_bin": n_bin_out,
            "density": dens_out,
            "F": F_k_out,
            "grid_nums": grid_nums_out,
            "labels": label_i,
            "frac_full": frac_full,
            "p_std": sigma_out if compute_std else None,
            "bounds": bounds,
            "n_bins": len_k,
            "n_hist": n_hist,
            "n_eff_bin": n_eff_out,
            "N_samples_eff": jnp.sum(N_i_tau),
        }

        return WeightOutput(**output_weight_kwargs)

    def bincount_weight(
        self,
        samples_per_bin: int = 10,
        n_max: float | int = 1e5,
        n_max_lin: float | int = 50,
        n_hist: int | None = None,
        chunk_size: int | None = None,
        cv_0: list[CV] | None = None,
        cv_t: list[CV] | None = None,
        macro_chunk: int | None = 1000,
        verbose: bool = False,
        margin: float = 0.1,
        min_samples: int = 3,
        recalc_bounds=True,
        compute_labels=False,
        use_energies: bool = False,
    ) -> WeightOutput:
        if cv_0 is None:
            cv_0 = self.cv

        if cv_t is None:
            cv_t = self.cv_t

        # TODO:https://pubs.acs.org/doi/pdf/10.1021/acs.jctc.9b00867

        beta = 1 / (self.sti.T * boltzmann)

        # len = [c.shape[0] for c in cv_0]
        len_i = len(cv_0)

        (
            frac_full,
            labels,
            x_labels,
            num_labels,
            grid_nums_mask,
            get_histo,
            bins,
            cv_mid,
            hist_mask,
            n_bin,
            # w_hist,
            _,
            corr,
            _log_w,
            n_hist,
        ) = self.get_bincount(
            cv_0,
            n_max=n_max,
            n_hist=n_hist,
            n_max_lin=n_max_lin,
            margin=margin,
            chunk_size=chunk_size,
            samples_per_bin=samples_per_bin,
            min_samples_per_bin=min_samples,
            macro_chunk=macro_chunk,
            verbose=verbose,
            recalc_bounds=recalc_bounds,
            compute_labels=compute_labels,
            use_energies=use_energies,
        )

        # print(f"{n_hist=} {w_hist=}")

        w_out = []
        ps_out = []
        dens_out = []
        F_k_out = []
        n_bin_out = []

        H_ik = hist_mask

        print(f"{H_ik.shape=}")

        F_k = jnp.zeros_like(n_hist / beta)

        # E_k = jnp.zeros((len_k,))

        # mask_k = jnp.any(mask_ik, axis=0)

        label_i = []

        for i in range(len_i):
            gi = grid_nums_mask[i].cv.reshape(-1)

            if compute_labels:
                labels_i = jnp.sum(x_labels[:, grid_nums_mask[i].cv.reshape(-1)], axis=1)
                label_i.append(jnp.argmax(labels_i))
            else:
                label_i.append(0)

            n_b = n_hist[gi]

            w = jnp.ones_like(n_b)
            ps = jnp.exp(_log_w[i])

            dens = jnp.ones_like(n_b)
            n_bin = n_b

            w_out.append(w)
            ps_out.append(ps)
            dens_out.append(dens)
            n_bin_out.append(n_bin)
            F_k_out.append(F_k[gi])

            # E_k = E_k.at[gi].add(e_unstacked[i][gi] * ps / n_bin)

        output_weight_kwargs = {
            "weight": w_out,
            "p_select": ps_out,
            "n_bin": n_bin_out,
            "density": dens_out,
            "F": F_k_out,
            # "grid_nums": grid_nums_mask,
            "labels": label_i,
            "frac_full": frac_full,
            "N_samples_eff": jnp.sum(jnp.array([a.shape[0] / b for a, b in zip(cv_0, corr)])),
            "n_bins": H_ik.shape[0],
            "n_hist": n_hist,
        }

        return WeightOutput(**output_weight_kwargs)

    @staticmethod
    def _correlation_time(
        cv_0: list["CV"],
        verbose: bool = True,
        method: str = "acf",
        const_acf: float = 5.0,
        periodicities: jax.Array | None = None,
        find_cutoff: bool = False,
        cutoff_steps: int = 20,
    ):
        # https://juser.fz-juelich.de/record/152532/files/FZJ-2014-02136.pdf
        # block-averaging estimate of integrated correlation times for diagnostics

        # print(f"starting block-averaging estimation of integrated correlation times")

        g_list: list[float] = []
        cutoff_list: list[int] = []  # NEW: Store calculated cutoff indices
        TE_list: list[float] = []

        if periodicities is not None:
            print("applying periodicity to cv for correlation time estimation")
            cv_0, _ = DataLoaderOutput.apply_cv(
                CvTrans.from_cv_function(PeriodicKoopmanModel._exp_periodic, periodicities=periodicities),
                cv_0,
            )

        @partial(jax.jit, static_argnames=["norm"])
        def _acf(x: jax.Array, idx_0=None, norm=True):
            T = x.shape[0]

            if idx_0 is not None:
                # Create mask
                mask = jnp.arange(T)[:, None] >= idx_0
                active_count = T - idx_0

                # Calculate mean ONLY of the active part
                # If we just use jnp.mean(x), the zeros drag the mean down!
                sum_active = jnp.sum(jnp.where(mask, x, 0.0), axis=0)
                mu_active = sum_active / active_count

                # Center only the active data, leave zeros as zeros
                x_centered = jnp.where(mask, x - mu_active, 0.0)
            else:
                x_centered = x - jnp.mean(x, axis=0)
                active_count = T

            @partial(vmap_decorator, in_axes=1)
            def f(feat_x):
                # 1. FFT-based correlation is much faster for JIT
                n_fft = 2 * T
                X_f = jnp.fft.fft(feat_x, n=n_fft)

                # ACF = IFFT( conj(X) * X )
                # This gives the correlation: sum( conj(x[t]) * x[t+lag] )
                corr_full = jnp.fft.ifft(jnp.conj(X_f) * X_f, n=n_fft)

                # Extract lags and take the REAL part
                # (The ACF of any signal at lag 0 is its variance, which is real)
                corr = corr_full[:T].real
                # Extract the valid lags
                corr = corr_full[:T]

                # 2. Correct normalization for zero-padded data
                # We divide by the number of actual overlapping pairs
                lags = jnp.arange(T, 0, -1)
                if idx_0 is not None:
                    # The number of pairs decreases as we move toward T
                    # But it starts at 'active_count'
                    lags = jnp.clip(lags - (T - active_count), min=1)

                corr = corr / lags

                if norm:
                    # Normalize by variance (lag 0)
                    corr = jnp.where(corr[0] == 0, corr, corr / corr[0])

                # integrated autocorrelation time (use lags 1..M)
                def hilbert(x, N=None, axis=-1):
                    N = x.shape[axis]

                    Xf = jax.numpy.fft.fft(x, N, axis=axis)
                    h = jnp.zeros(N, dtype=Xf.dtype)
                    if N % 2 == 0:
                        h = h.at[0].set(1)
                        h = h.at[N // 2].set(1)
                        h = h.at[1 : N // 2].set(2)
                    else:
                        h = h.at[0].set(1)
                        h = h.at[1 : (N + 1) // 2].set(2)

                    if x.ndim > 1:
                        ind = [jnp.newaxis] * x.ndim
                        ind[axis] = slice(None)
                        h = h[tuple(ind)]
                    x = jax.numpy.fft.ifft(Xf * h, axis=axis)
                    return x

                # (Inside hilbert, ensure you use the active_count for the slice)
                envelope = jnp.abs(hilbert(corr))
                # Important: only fit the envelope up to active_count/2
                valid_len = active_count // 2

                # Define loss on truncated envelope
                def loss_fn(tau, y):
                    # y is the envelope
                    # We must only compare the valid (non-zero) part of the envelope
                    t_axis = jnp.arange(T)
                    y_pred = jnp.exp(-t_axis / tau)

                    # Mask the loss so we don't fit the zero-padded tail
                    loss_mask = t_axis < valid_len
                    res = jnp.where(loss_mask, (y - y_pred) ** 2, 0.0)
                    return jnp.sum(res)

                from jaxopt import GradientDescent

                solver = GradientDescent(fun=loss_fn)
                result = solver.run(jnp.array([1.0]), envelope)

                tau = result.params
                return (1 + jnp.exp(-1 / tau)) / (1 - jnp.exp(-1 / tau))

            # We need to reshape/mask the final scales if the features were 0
            time_scales = f(x_centered)
            return jnp.max(time_scales)

        # REFACTORED: Now takes a jax.Array instead of CV object
        @jax.jit
        def _blav(data: jax.Array, idx_0=None):
            data = data.reshape((data.shape[0], -1))

            if idx_0 is not None:
                raise NotImplementedError("idx_0 slicing not implemented for block-averaging method yet")

            errors = []
            block_sizes = []

            for bs in range(1, data.shape[0] // 10 + 1):
                n_blocks = data.shape[0] // bs
                block_means = jnp.mean(data[: n_blocks * bs, :].reshape((n_blocks, bs, -1)), axis=1)
                block_var_naive = jnp.var(block_means, axis=0, ddof=1) / n_blocks

                errors.append(jnp.sqrt(block_var_naive))
                block_sizes.append(bs)

            errors = jnp.array(errors)

            from jaxopt import GaussNewton

            def fit_func(x, B, err):
                t_int, TE = x
                TE = jnp.exp(TE)
                t_int = jnp.exp(t_int) + 1
                return TE * jnp.sqrt(B / (B + t_int - 1)) - err

            @partial(jax.vmap, in_axes=(None, 1), out_axes=0)
            def solve(block_sizes, errors):
                solver = GaussNewton(residual_fun=fit_func, tol=1e-6)
                out = solver.run((0.0, errors[0]), block_sizes, errors)
                log_t_int_est, log_TE_est = out.params
                t_int_est = jnp.exp(log_t_int_est) + 1
                TE_est = jnp.exp(log_TE_est)
                return t_int_est, TE_est

            t_int, TE = solve(jnp.array(block_sizes), errors)

            t_int = jnp.where(jnp.isfinite(t_int), t_int, jnp.inf)
            t_int = jnp.where(t_int > data.shape[0] / 2, data.shape[0] / 2, t_int)
            t_int = jnp.where(t_int < 1.0, 1.0, t_int)

            # Fix: Return max across features to decouple from the outer loop index 'i'
            return jnp.max(t_int)

        if method == "blav":
            _f = _blav
        elif method == "acf":
            _f = _acf
        else:
            raise ValueError(f"method {method=} unknown, choose from 'acf','blav'")

        for i, cv_i in enumerate(cv_0):
            # if verbose:
            #     print(".", end="", flush=True)
            #     if (i + 1) % 100 == 0:
            #         print("")

            cv_data_array = cv_i.cv
            T = cv_data_array.shape[0]

            if find_cutoff:
                # g = _f(cv_data_array)

                T = cv_data_array.shape[0]
                t0_grid = jnp.linspace(0, T // 2, cutoff_steps, dtype=int)

                g = jax.vmap(_f, in_axes=(None, 0))(cv_data_array, t0_grid)  # type: ignore

                g_idx = jnp.argmax(g / (T - t0_grid))

                g = g[g_idx]

                cutoff_list.append(g_idx)
            else:
                g = _f(cv_data_array)

            g_list.append(g)

        g_i = jnp.array(g_list)

        if find_cutoff:
            cutoffs = jnp.array(cutoff_list)
            return cutoffs, g_i

        return g_i

    def get_bincount(
        self,
        cv_0: list[CV] | None = None,
        energies: list[jax.Array] | None = None,
        temp_energies=None,
        use_energies=False,
        n_max=1e5,
        n_max_lin: int | None = 150,
        margin=0.0,
        chunk_size=None,
        samples_per_bin=20,
        min_samples_per_bin=1,
        macro_chunk: int | None = 1000,
        verbose=False,
        compute_labels=True,
        recalc_bounds=True,
        n_hist: int | None = None,
        bounds: jax.Array | None = None,
        correlation_method: str | None = None,
        tau_i: jax.Array | None = None,
    ):
        if cv_0 is None:
            cv_0 = self.cv

        if n_hist is None:
            n_hist = self.n_hist

        if bounds is None:
            bounds = self.bounds

        if use_energies:
            if energies is None:
                energies = self.e_pot
            if temp_energies is None:
                temp_energies = self.sti.T

        return DataLoaderOutput._get_bincount(
            cv_0=cv_0,
            n_max=n_max,
            margin=margin,
            chunk_size=chunk_size,
            samples_per_bin=samples_per_bin,
            min_samples_per_bin=min_samples_per_bin,
            macro_chunk=macro_chunk,
            verbose=verbose,
            metric=self.collective_variable.metric,
            compute_labels=compute_labels,
            recalc_bounds=recalc_bounds,
            n_hist=n_hist,
            bounds=bounds,
            correlation_method=correlation_method,
            n_max_lin=n_max_lin,
            use_energies=use_energies,
            energies=energies,
            temp_energies=temp_energies,
            tau_i=tau_i,
        )

    @staticmethod
    def _get_bincount(
        cv_0: list[CV],
        metric: CvMetric,
        use_energies=False,
        energies: list[jax.Array] | None = None,
        temp_energies=300 * kelvin,
        n_max: float | int = 1e5,
        n_hist: int | None = None,
        n_max_lin: int | None = None,
        bounds: jax.Array | None = None,
        margin=0.0,
        chunk_size=None,
        samples_per_bin=20,
        min_samples_per_bin=1,
        macro_chunk: int | None = 1000,
        verbose=False,
        compute_labels=True,
        frac_full: float | None = 1.0,
        recalc_bounds=True,
        correlation_method: str | None = None,
        tau_i: jax.Array | None = None,
    ):
        # helper method

        ndim = cv_0[0].shape[1]

        print(f"{bounds=}{metric=} {recalc_bounds=}")

        if bounds is not None:
            print(f"using provided {bounds=}")
            grid_bounds = bounds

        else:
            print("getting bounds")

            if recalc_bounds:
                bounds, _, constants = CvMetric.bounds_from_cv(
                    cv_0,
                    margin=0,
                    # chunk_size=chunk_size,
                    macro_chunk=macro_chunk,
                    n=30,
                    verbose=verbose,
                )

            else:
                bounds = metric.bounding_box

            bounds_margin = (bounds[:, 1] - bounds[:, 0]) * margin
            # bounds_margin = jnp.where(   self.periodicities, )
            bounds = bounds.at[:, 0].set(bounds[:, 0] - bounds_margin)
            bounds = bounds.at[:, 1].set(bounds[:, 1] + bounds_margin)

            grid_bounds = jax.vmap(lambda x, y: jnp.where(metric.extensible, y, x), in_axes=(1, 1), out_axes=(1))(
                metric.bounding_box,
                bounds,
            )

        print(f"{grid_bounds=}")

        if correlation_method is not None:
            if tau_i is not None:
                print("using provided correlation times")

                assert tau_i.shape[0] == len(cv_0), "tau_i must have the same length as cv_0"

            else:
                print("correlating data to get effective sample size")
                tau_i = DataLoaderOutput._correlation_time(
                    cv_0,
                    verbose=verbose,
                    method=correlation_method,
                    periodicities=metric.periodicities,
                )

                sd = [int(jnp.ceil(a.shape[0] / t)) for a, t in zip(cv_0, tau_i)]
                tot_samples = sum(sd)
                print(f"estimated effective sample size after correlation: {sd} {tot_samples=}")

        else:
            sd = [a.shape[0] for a in cv_0]
            tot_samples = sum(sd)

        # prepare histo

        def get_hist(n_hist):
            if verbose:
                print(f"using {n_hist=}")

            if n_hist <= 3:
                print(f"WARNING: {n_hist=}, adjusting to 3")
                n_hist = 3

            print("getting histo")
            cv_mid, nums, bins, closest, get_histo = DataLoaderOutput._histogram(
                metric=metric,
                n_grid=n_hist,
                grid_bounds=grid_bounds,
                chunk_size=chunk_size,
            )

            grid_nums, _ = DataLoaderOutput.apply_cv(
                closest,
                cv_0,
                chunk_size=chunk_size,
                macro_chunk=macro_chunk,
                verbose=verbose,
            )

            if verbose:
                print("getting histo")

            _log_n = [jnp.full(c.shape[0], 0) for c in cv_0]

            log_hist_k = get_histo(
                grid_nums,
                _log_n,
                log_w=True,
                macro_chunk=macro_chunk,
                verbose=verbose,
            )

            # print

            print(f"{log_hist_k=}")

            if use_energies:
                assert energies is not None, "energies must be provided when use_energies is True"
                assert temp_energies is not None, "temp_energies must be provided when use_energies is True"

                print(f"using provided energies to rescale histo")
                beta = 1 / (temp_energies * boltzmann)  # temp does not matter
                _log_w = [-beta * e for e in energies]

                log_e_pot_ik = jnp.zeros((len(cv_0), log_hist_k.shape[0]))
                log_hist_ik = jnp.zeros((len(cv_0), log_hist_k.shape[0]))

                for i, (_log_wi, gi) in enumerate(zip(_log_w, grid_nums)):
                    _log_hist_w_ik = get_histo(
                        [gi],
                        [_log_wi],
                        log_w=True,
                        macro_chunk=macro_chunk,
                        verbose=False,
                    )

                    log_e_pot_ik = log_e_pot_ik.at[i, :].set(_log_hist_w_ik)

                    print(".", end="", flush=True)
                    if (i + 1) % 100 == 0:
                        print("")

                print("done")

                x_max_k = jnp.max(log_e_pot_ik, axis=0, keepdims=True)

                log_e_pot_k = jnp.log(jnp.sum(jnp.exp(log_e_pot_ik - x_max_k), axis=0)) + x_max_k.reshape(-1)

                log_w_ik = log_e_pot_ik - log_e_pot_k.reshape(1, -1)  # + log_hist_ik

                _log_w = [log_w_ik[i, gi.cv.reshape((-1,))] for i, gi in enumerate(grid_nums)]

            else:
                _log_w = [jnp.zeros_like(c.cv.reshape(-1)) for c in cv_0]

            # print(f"{grid_nums=}")

            # this is chosen such that if you sampling according to w/nb, you get weight 1 per bin on average.

            print(f"{min_samples_per_bin=}")

            # print(f"{jnp.exp(hist)=}")
            hist_mask = log_hist_k >= jnp.log(min_samples_per_bin)  # at least 1 sample
            # hist_mask = log_hist_k > -jnp.inf

            print(f"{jnp.sum(hist_mask)=}")

            return (
                cv_mid,
                # nums,
                bins,
                # closest,
                get_histo,
                grid_nums,
                jnp.exp(log_hist_k[hist_mask]),
                hist_mask,
                _log_w,
            )

        if n_hist is None:
            if frac_full is None:
                print(f"getting frac full")

                # if n_max > 1e3:
                #     n_hist = CvMetric.get_n(
                #         samples_per_bin=samples_per_bin,
                #         samples=tot_samples,
                #         n_dims=ndim,
                #         max_bins=jnp.min(jnp.array([1e3, n_max])),  # 10 000 test grid points
                #     )

                #     # pre test to get empty fraction
                #     (
                #         cv_mid,
                #         # nums,
                #         bins,
                #         # closest,
                #         get_histo,
                #         grid_nums,
                #         _n_hist_mask,
                #         hist_mask,
                #         _,
                #     ) = get_hist(n_hist)

                #     frac_full = jnp.sum(hist_mask) / hist_mask.shape[0]

                #     print(f"{frac_full=}")
                # else:
                #     frac_full = 1.0

            n_hist = CvMetric.get_n(
                samples_per_bin=samples_per_bin,
                samples=tot_samples,
                n_dims=ndim,
                max_bins=n_max,  # compensate for empty spaces
            )

            print(f"{n_hist=} {frac_full=} {n_max=}")
        else:
            print(f"using provided {n_hist=}")

        if n_max_lin is not None:
            if n_hist > n_max_lin:
                print(f"capping n_hist from {n_hist} to {n_max_lin}")
                n_hist = n_max_lin

        (
            cv_mid,
            # nums,
            bins,
            # closest,
            get_histo,
            grid_nums,
            _n_hist_mask,
            hist_mask,
            _log_w,
        ) = get_hist(n_hist)

        frac_full = jnp.sum(hist_mask) / hist_mask.shape[0]

        print(f"{   jnp.astype(_n_hist_mask,jnp.int_)=}")

        idx_inv = jnp.full(hist_mask.shape[0], -1)
        idx_inv = idx_inv.at[hist_mask].set(jnp.arange(jnp.sum(hist_mask)))

        # print(f"{jnp.argwhere(hist_mask)=}")

        grid_nums_mask = [g.replace(cv=idx_inv[g.cv]) for g in grid_nums]

        # print(f"{grid_nums=}")
        # print(f"{grid_nums_mask=}")

        n_hist_mask = jnp.sum(hist_mask)
        # print(f"{_n_hist_mask=}")

        if compute_labels:
            # x start with False for each gridpoint
            # rows are different regions, columns are grid points.
            # either there is already one or more regions that includes one of the grid point
            #  if so add new k to that region and merge regions if necessary
            # add new row for remaining k if no region includes it
            x = jnp.full((0, n_hist_mask), False)

            for fni, gn in enumerate(grid_nums_mask):
                b = jnp.full(n_hist_mask + 1, False)
                b = b.at[gn.cv.reshape(-1)].set(True)
                b = b[:-1]

                if jnp.sum(b) == 0:
                    # b doesn't include any grid point, continue. this either means that the trajectory is empty or that all points are in bins with insufficient points
                    continue

                in_rows, new_rows = vmap_decorator(
                    lambda u, v: (jnp.logical_and(u, v).any(), jnp.logical_or(u, v)), in_axes=(None, 0)
                )(b, x)

                if in_rows.any():
                    n = jnp.sum(in_rows)

                    rr = jnp.argwhere(in_rows).reshape(-1)

                    if n == 1:  # add visited parts to row
                        x = x.at[rr[0], :].set(new_rows[rr[0], :])

                    else:  # merge connected parts
                        rows = jnp.vstack([x[rr, :], b])
                        new_row = jnp.any(rows, axis=0)

                        x = x.at[rr[0], :].set(new_row)

                        print(f" merging {rr.shape[0]} rows into {rr[0]} {x.shape=}")
                        x = jnp.delete(x, rr[1:], axis=0)
                        print(f" after merging  {x.shape=}")

                else:  # create row
                    x = jnp.vstack([x, b])

            # print(f"{x.shape=}")

            # print(f"{jnp.sum(x, axis=1)=}")
            # print(f"{jnp.sum(x, axis=0)=}")

            # remove rows that are completely false
            x = x[jnp.any(x, axis=1), :]

            # there is one row which is True, others should be false
            labels = vmap_decorator(lambda x: jnp.argwhere(x, size=1).reshape(()), in_axes=1)(x)
            unique_vals, _ = jnp.unique(labels, return_inverse=True)

            print(f"{unique_vals=}")

            num_labels = unique_vals.shape[0]

            if num_labels > 1:
                print(f"found {num_labels} different regions {labels=}")

            x_labels = jnp.hstack([x, jnp.full((x.shape[0], 1), False)])
        else:
            labels, x_labels, num_labels = None, None, None

        return (
            frac_full,
            labels,
            x_labels,
            num_labels,
            grid_nums_mask,
            get_histo,
            bins,
            cv_mid,
            hist_mask,
            _n_hist_mask,
            # w_hist_mask,
            grid_bounds,
            tau_i if correlation_method else jnp.ones((len(cv_0),)),
            _log_w,
            n_hist,
        )

    @staticmethod
    def _transform(
        cv,
        nl,
        shmap,
        shmap_kwargs,
        argmask: jax.Array | None = None,
        pi: jax.Array | None = None,
        add_1: bool = False,
        add_1_pre: bool = False,
        q: jax.Array | None = None,
        l: jax.Array | None = None,
    ) -> jax.Array:
        x = cv.cv

        # print(f"inside {x.shape=} {q=} {argmask=} ")

        if argmask is not None:
            x = x[argmask]

        if pi is not None:
            x = x - pi

        if add_1_pre:
            x = jnp.hstack([x, jnp.array([1])])

        if q is not None:
            x = x @ q

        if l is not None:
            x = x * l

        if add_1:
            x = jnp.hstack([x, jnp.array([1])])

        return cv.replace(cv=x, _combine_dims=None)

    def koopman_model(
        self,
        cv_0: list[CV] | list[SystemParams] | None = None,
        cv_t: list[CV] | list[SystemParams] | None = None,
        nl: list[NeighbourList] | NeighbourList | None = None,
        nl_t: list[NeighbourList] | NeighbourList | None = None,
        # method="tcca",
        only_return_weights=False,
        symmetric=False,
        rho: list[jax.Array] | None = None,
        w: list[jax.Array] | None = None,
        rho_t: list[jax.Array] | None = None,
        w_t: list[jax.Array] | None = None,
        dynamic_weights: list[jax.Array] | None = None,
        eps: float = 0.0,
        eps_pre: float = 0.0,
        eps_shrink: float = 1e-3,
        max_features=5000,
        max_features_pre=5000,
        out_dim=-1,
        add_1=True,
        chunk_size=None,
        macro_chunk=1000,
        verbose=False,
        trans=None,
        T_scale: float = 1.0,
        only_diag=False,
        calc_pi=False,
        scaled_tau=None,
        sparse=True,
        correlation=True,
        auto_cov_threshold: float | None = None,
        shrink=True,
        shrinkage_method="bidiag",
        generator=False,
        use_w=True,
        periodicities: jax.Array | None = None,
        iters_nonlin: int = 10000,
        epochs: int = 2000,
        init_learnable_params=True,
        entropy_reg: float = 0.0,
        target_smoothness: float | None = None,
        alpha_smooth: float = 0.0,
        batch_size: int = 1024,
        batch_chunk_size: int = 16,
        beta_timecon: float | None = None,
    ) -> "KoopmanModel":
        # TODO: https://www.mdpi.com/2079-3197/6/1/22

        # assert method in ["tica", "tcca"]

        if scaled_tau is None:
            scaled_tau = self.scaled_tau

        if cv_0 is None:
            cv_0 = self.cv

        if nl is None:
            nl = self.nl

        if rho is None:
            if self._rho is not None:
                rho = self._rho
            else:
                print("W koopman model not given, will use uniform weights")

                t = sum([cvi.shape[0] for cvi in cv_0])
                rho = [jnp.ones((cvi.shape[0],)) / t for cvi in cv_0]

        if T_scale != 1.0:
            print(f"rescaling rho for T_scale {T_scale=}")

            rho = self.rescale_rho_T(T_scale, rho=rho)

        if w is None:
            if self._weights is not None:
                w = self._weights
            else:
                print("W koopman model not given, will use uniform weights")

                t = sum([cvi.shape[0] for cvi in cv_0])
                w = [jnp.ones((cvi.shape[0],)) / t for cvi in cv_0]

        if not generator:
            if cv_t is None:
                cv_t = self.cv_t

            if nl_t is None:
                nl_t = self.nl_t

            assert cv_t is not None

            if rho_t is None:
                if self._rho_t is not None:
                    rho_t = self._rho_t
                else:
                    print("W_t koopman model not given, will use w")

                    rho_t = rho

            if T_scale != 1.0:
                rho_t = self.rescale_rho_T(T_scale, rho=rho_t)

            if w_t is None:
                if self._weights_t is not None:
                    w_t = self._weights_t
                else:
                    print("W_t koopman model not given, will use w")

                    w_t = w

        if dynamic_weights is None:
            # print("dynamic weights not given, will use uniform weights")

            if self._w_dyn is not None:
                print("using provided dynamic weights for koopman model")
                dynamic_weights = self._w_dyn
            else:
                dynamic_weights = [jnp.ones((cvi.shape[0],)) for cvi in cv_0]

            # print(f"{dynamic_weights=} ")

        return KoopmanModel.create(
            w=w,
            w_t=w_t,
            dynamic_weights=dynamic_weights,
            rho=rho,
            rho_t=rho_t,
            cv_0=cv_0,
            cv_t=cv_t,
            nl=nl,
            nl_t=nl_t,
            add_1=add_1,
            eps=eps,
            eps_pre=eps_pre,
            # method=method,
            symmetric=symmetric,
            out_dim=out_dim,
            eps_shrink=eps_shrink,
            # koopman_weight=koopman_weight,
            max_features=max_features,
            max_features_pre=max_features_pre,
            tau=self.tau,
            chunk_size=chunk_size,
            macro_chunk=macro_chunk,
            verbose=verbose,
            trans=trans,
            # T_scale=T_scale,
            only_diag=only_diag,
            calc_pi=calc_pi,
            scaled_tau=scaled_tau,
            sparse=sparse,
            only_return_weights=only_return_weights,
            correlation_whiten=correlation,
            auto_cov_threshold=auto_cov_threshold,
            shrink=shrink,
            shrinkage_method=shrinkage_method,
            generator=generator,
            use_w=use_w,
            periodicities=periodicities,
            iters_nonlin=iters_nonlin,
            epochs=epochs,
            batch_size=batch_size,
            init_learnable_params=init_learnable_params,
            entropy_reg=entropy_reg,
            target_smoothness=target_smoothness,
            alpha_smooth=alpha_smooth,
            batch_chunk_size=batch_chunk_size,
            beta_timecon=beta_timecon,
        )

    def filter_nans(
        self,
        x: list[CV] | None = None,
        x_t: list[CV] | None = None,
        macro_chunk=1000,
    ):
        if x is None:
            x = self.cv

        if x_t is None:
            x_t = self.cv_t

        def _check_nan(x: CV, nl, cv, shmap):
            return x.replace(cv=jnp.isnan(x.cv) | jnp.isinf(x.cv))

        check_nan = CvTrans.from_cv_function(_check_nan)

        nan_x, nan_x_t = self.apply_cv(
            check_nan,
            x,
            x_t,
            macro_chunk=macro_chunk,
            shmap=False,
            verbose=True,
        )

        nan = CV.stack(*nan_x).cv

        if nan_x_t is not None:
            nan = jnp.logical_or(nan, CV.stack(*nan_x_t).cv)

        assert not jnp.any(nan), f"found {jnp.sum(nan)}/{len(nan)} nans or infs in the cv data"

        if jnp.any(nan):
            raise

    @staticmethod
    def apply_cv(
        f: CvTrans,
        x: list[CV] | list[SystemParams],
        x_t: list[CV] | list[SystemParams] | None = None,
        nl: list[NeighbourList] | NeighbourList | None = None,
        nl_t: list[NeighbourList] | NeighbourList | None = None,
        chunk_size: int | None = None,
        macro_chunk: int | None = 1000,
        shmap: bool = False,
        shmap_kwargs=ShmapKwargs.create(),
        verbose: bool = False,
        print_every: int = 10,
        jit_f: bool = True,
        debug_print: bool = False,
    ) -> tuple[list[CV], list[CV] | None]:
        _f = f.compute_cv

        if x_t is not None:
            if not isinstance(x_t[0], x[0].__class__):
                print(
                    f"WARNING: x_t is of type {x_t[0].__class__}, but x is of type {x[0].__class__}, this may lead to errors"
                )

        def __f(x: X, nl: NeighbourList | None) -> CV:
            r, _ = _f(
                x,
                nl,
                chunk_size=chunk_size,
                shmap=shmap,
            )

            return r

        out = DataLoaderOutput._apply(
            x=x,
            x_t=x_t,
            nl=nl,
            nl_t=nl_t,
            f=__f,
            macro_chunk=macro_chunk,
            verbose=verbose,
            jit_f=jit_f,
            print_every=print_every,
            debug_print=debug_print,
        )

        return out

    @staticmethod
    def _apply(
        x: list[X],
        f: Callable[[X, NeighbourList | None], X2],
        x_t: list[X] | None = None,
        nl: list[NeighbourList] | NeighbourList | None = None,
        nl_t: list[NeighbourList] | NeighbourList | None = None,
        macro_chunk: int | None = 1000,
        verbose: bool = False,
        jit_f: bool = True,
        print_every: int = 10,
        debug_print: bool = False,
    ) -> tuple[list[X2], list[X2] | None]:
        if verbose:
            print("inside _apply")

        out = macro_chunk_map(
            f=f,
            # op=x[0].__class__.stack,
            y=x,
            y_t=x_t,
            nl=nl,
            nl_t=nl_t,
            macro_chunk=macro_chunk,
            verbose=verbose,
            jit_f=jit_f,
            print_every=print_every,
            debug_print=False,
        )

        if verbose:
            print("outside _apply")
        # jax.debug.inspect_array_sharding(out[0], callback=print)

        if x_t is not None:
            z, z_t = out

        else:
            z = out
            z_t = None

        z = cast(list[X2], z)
        if z_t is not None:
            z_t = cast(list[X2], z_t)

        for i in range(len(z)):
            assert z[i].shape[0] == x[i].shape[0], (
                f" shapes do not match {[zi.shape[0] for zi in z]} != {[xi.shape[0] for xi in x]}"
            )

            if z_t is not None:
                assert z[i].shape[0] == z_t[i].shape[0]

        return z, z_t

    @staticmethod
    def _apply_bias(
        x: list[CV],
        bias: Bias,
        chunk_size=None,
        macro_chunk=1000,
        verbose=False,
        shmap=True,
        shmap_kwargs=ShmapKwargs.create(),
        jit_f=True,
    ) -> list[jax.Array]:
        def f(x: CV, _):
            b = bias.compute_from_cv(
                x,
                chunk_size=chunk_size,
                shmap=False,
            )[0]

            return x.replace(cv=b.reshape((-1, 1)))

        out, _ = DataLoaderOutput._apply(
            x=x,
            f=f,
            macro_chunk=macro_chunk,
            verbose=verbose,
            jit_f=jit_f,
        )

        return [a.cv[:, 0] for a in out]

    def apply_bias(self, bias: Bias, chunk_size=None, macro_chunk=1000, verbose=False, shmap=False) -> list[Array]:
        return DataLoaderOutput._apply_bias(
            x=self.cv,
            bias=bias,
            chunk_size=chunk_size,
            macro_chunk=macro_chunk,
            verbose=verbose,
            shmap=shmap,
        )

    def get_point(
        self,
        point: CV,
        sigma=0.01,
        key=jax.random.PRNGKey(0),
    ) -> tuple[CV, SystemParams, NeighbourList | NeighbourListInfo | None, TrajectoryInfo]:
        import jax.numpy as jnp

        from IMLCV.implementations.bias import HarmonicBias

        # print(f"{self.collective_variable=}")

        mu = self.collective_variable.metric.bounding_box[:, 1] - self.collective_variable.metric.bounding_box[:, 0]
        k = self.sti.T * boltzmann / (mu**2 * 2 * sigma**2)

        center_bias = HarmonicBias.create(cvs=self.collective_variable, k=k, q0=point)

        center = self.apply_bias(
            bias=center_bias,
        )

        p = jnp.exp(-jnp.hstack(center) / (self.sti.T * boltzmann))
        p /= jnp.sum(p)

        n = []

        for i, x in enumerate(self.cv):
            n.append(jnp.stack([jnp.full(x.shape[0], i), jnp.arange(x.shape[0])], axis=1))

        n = jnp.vstack(n)

        idx = jax.random.choice(key, a=n, p=p, shape=(), replace=True)

        return (
            self.cv[idx[0]][idx[1]],
            self.sp[idx[0]][idx[1]],
            self.nl[idx[0]][idx[1]] if self.nl is not None else None,
            self.ti[idx[0]],
        )

    def get_fes_bias_from_weights(
        self,
        cv: list[CV] | None = None,
        weights: list[Array] | None = None,
        rho: list[Array] | None = None,
        weights_std: list[Array] | None = None,
        samples_per_bin=100,
        min_samples_per_bin: int | None = 1,
        n_max=1e5,
        n_max_lin=1000,
        max_bias=None,
        chunk_size=None,
        macro_chunk=1000,
        max_bias_margin=0.2,
        rbf_bias=True,
        kernel="multiquadric",
        collective_variable: CollectiveVariable | None = None,
        set_outer_border=True,
        rbf_degree: int | None = None,
        smoothing=0.1 / (kjmol**2),
        frac_full=1.0,
        recalc_bounds=True,
        output_density_bias=False,
        observable: list[CV] | None = None,
        overlay_mask=False,
        std_bias=True,
        margin=0.3,
        n_hist: int | None = None,
        bounds: jax.Array | None = None,
    ):
        if cv is None:
            cv = self.cv

        if weights is None:
            weights = self._weights

        assert weights is not None

        if rho is None:
            rho = self._rho

        assert rho is not None

        if weights_std is None:
            weights_std = self._weights_std

        if collective_variable is None:
            collective_variable = self.collective_variable

        if frac_full is None:
            frac_full = self.frac_full

        if n_hist is None:
            if self.n_hist is not None:
                print("using stored n_hist")
            n_hist = self.n_hist

        if bounds is None:
            if self.bounds is not None:
                print("using stored bounds")

            bounds = self.bounds

        return DataLoaderOutput._get_fes_bias_from_weights(
            T=self.sti.T,
            weights=weights,
            rho=rho,
            weights_std=weights_std,
            collective_variable=collective_variable,
            cv=cv,
            samples_per_bin=samples_per_bin,
            min_samples_per_bin=min_samples_per_bin,
            n_max_lin=n_max_lin,
            n_max=n_max,
            max_bias=max_bias,
            chunk_size=chunk_size,
            macro_chunk=macro_chunk,
            max_bias_margin=max_bias_margin,
            rbf_bias=rbf_bias,
            kernel=kernel,
            set_outer_border=set_outer_border,
            rbf_degree=rbf_degree,
            smoothing=smoothing,
            frac_full=frac_full,
            recalc_bounds=recalc_bounds,
            output_density_bias=output_density_bias,
            observable=observable,
            overlay_mask=overlay_mask,
            std_bias=std_bias,
            margin=margin,
            bounds=bounds,
            n_hist=n_hist,
        )

    @staticmethod
    def _get_fes_bias_from_weights(
        T,
        weights: list[jax.Array],
        rho: list[jax.Array],
        collective_variable: CollectiveVariable,
        cv: list[CV],
        samples_per_bin: int = 1000,
        min_samples_per_bin: int = 5,
        n_max: int | float = 1e5,
        n_max_lin: int | float = 100,
        max_bias: float | None = None,
        chunk_size: int | None = None,
        macro_chunk: int | None = 1000,
        max_bias_margin: float = 0.2,
        rbf_bias: bool = True,
        kernel: str = "multiquadric",
        set_outer_border: bool = True,
        rbf_degree: int | None = None,
        smoothing=0.1 / (kjmol**2),
        frac_full=1.0,
        compute_frac_full=False,
        grid_bias_order: int = 0,
        verbose=False,
        recalc_bounds=True,
        margin=0.3,
        output_density_bias=False,
        observable: list[CV] | None = None,
        overlay_mask=False,
        std_bias=True,
        weights_std: list[jax.Array] | None = None,
        n_hist: int | None = None,
        bounds: jax.Array | None = None,
    ) -> tuple[Bias, StdBias | None, Bias | None, dict]:
        beta = 1 / (T * boltzmann)

        (
            _,
            _,
            _,
            _,
            grid_nums_mask,
            get_histo,
            bins,
            cv_mid,
            hist_mask,
            n_hist_mask,
            bounds,
            _,
            _log_w,
            n_bin,
        ) = DataLoaderOutput._get_bincount(
            cv,
            metric=collective_variable.metric,
            n_hist=n_hist,
            n_max_lin=n_max_lin,
            bounds=bounds,
            n_max=n_max,
            margin=margin,
            chunk_size=chunk_size,
            samples_per_bin=samples_per_bin,
            min_samples_per_bin=min_samples_per_bin,
            macro_chunk=macro_chunk,
            verbose=verbose,
            compute_labels=False,
            frac_full=frac_full,
            recalc_bounds=recalc_bounds,
        )

        def safe_log_sum(*x):
            p_out = None

            for xi in x:
                p = jnp.where(xi > 0, jnp.log(xi), jnp.full_like(xi, -jnp.inf))

                if p_out is None:
                    p_out = p
                else:
                    p_out += p

            return p_out

        w_log = [safe_log_sum(wi, rhoi) for wi, rhoi in zip(weights, rho)]

        log_w_rho_grid_mask = get_histo(
            grid_nums_mask,
            w_log,
            macro_chunk=macro_chunk,
            log_w=True,
            shape_mask=hist_mask,
        )

        # mask = hist_mask

        # mask = jnp.logical_and(
        #     hist_mask,
        #     jnp.isfinite(log_w_rho_grid_mask),
        # )

        fes_grid = jnp.full(hist_mask.shape, jnp.inf)
        fes_grid = fes_grid.at[hist_mask].set(-log_w_rho_grid_mask / beta)
        fes_grid -= jnp.nanmin(fes_grid)

        mask_tot = jnp.isfinite(fes_grid)

        if std_bias and weights_std is None:
            print("no weights std given, cannot compute std bias")
            std_bias = False

        if std_bias:
            log_w_sigma_sq = get_histo(
                grid_nums_mask,
                [safe_log_sum(wi, wi, wstdi, wstdi, rhoi, rhoi) for wi, rhoi, wstdi in zip(weights, rho, weights_std)],
                macro_chunk=macro_chunk,
                log_w=True,
                shape_mask=hist_mask,
            )

            # print(f"{log_w_sigma_sq=}")

            log_w = get_histo(
                grid_nums_mask,
                [safe_log_sum(wi, rhoi) for wi, rhoi in zip(weights, rho)],
                macro_chunk=macro_chunk,
                log_w=True,
                shape_mask=hist_mask,
            )

            log_w_sigma_grid = jnp.full(hist_mask.shape, -jnp.inf)
            log_w_sigma_grid = log_w_sigma_grid.at[hist_mask].set(log_w_sigma_sq) / 2
            log_w_grid = log_w_grid = jnp.full(hist_mask.shape, -jnp.inf)
            log_w_grid = log_w_grid.at[hist_mask].set(log_w)

            # print(f"{log_w_grid[mask_tot]=} {log_w_sigma_grid[mask_tot]=}")

            std_grid = jnp.exp(log_w_sigma_grid - log_w_grid)

            print(f"{std_grid[mask_tot]/kjmol=}")

            mask_std = jnp.isfinite(std_grid) & (std_grid >= 0)
            if (n := jnp.sum(~mask_std[mask_tot])) > 0:
                print(f"WARNING: found {n} grid points with non finite or negative std, setting to 5 kjmo")
                std_grid = std_grid.at[~mask_std].set(5 * kjmol)

        bounds_gb = jnp.array([[(a[0] + a[1]) / 2, (a[-1] + a[-2]) / 2] for a in bins])
        shape = tuple([len(a) - 1 for a in bins])

        print(f"{fes_grid.shape=} {shape=} ")

        bounds_adjusted = GridBias.adjust_bounds(bounds=bounds_gb, n=shape[0])

        if rbf_bias:
            fes_grid_selection = fes_grid[mask_tot]
            cv_selection = cv_mid[mask_tot]

            range_frac = jnp.array([b[1] - b[0] for b in bins])

            print(f"{range_frac=}")

            # epsilon = 1 / (range_frac * n_bin ** (1 / cv_selection.shape[1]))

            dv = jnp.array([b[1] - b[0] for b in bins])

            epsilon = 1 / (2.5 * dv)
            print(f"{epsilon=}")

            print(f"{fes_grid_selection/kjmol=}")

            if std_bias:
                sigma_grid_selection = std_grid[mask_tot]
                print(f"{sigma_grid_selection/kjmol=}")

            else:
                sigma_grid_selection = None

            # dv = 1
            # print(f"{dv=}")

            bias = RbfBias.create(
                cvs=collective_variable,
                cv=cv_selection,
                kernel=kernel,
                vals=-fes_grid_selection,
                epsilon=epsilon,
                degree=rbf_degree,
                smoothing=smoothing,
                sigma=sigma_grid_selection,
                dv=dv,
                grid_full=cv_mid,
            )

            if overlay_mask:
                grid_mask_bias = GridMaskBias(
                    collective_variable=collective_variable,
                    n=shape[0],
                    bounds=bounds_adjusted,
                    vals=fes_grid.reshape(shape) * 0.0 + 1.0,
                    order=grid_bias_order,
                )

                bias = CompositeBias.create(biases=[bias, grid_mask_bias], fun=jnp.prod)

        else:
            bias = GridBias(
                collective_variable=collective_variable,
                n=shape[0],
                bounds=bounds_adjusted,
                vals=-fes_grid.reshape(shape),
                order=grid_bias_order,
            )

        if std_bias:
            print(f"creating std bias with {std_grid[mask_tot]/kjmol=}")

            std_bias = StdBias.create(
                bias=GridBias(
                    collective_variable=collective_variable,
                    n=shape[0],
                    bounds=bounds_adjusted,
                    vals=log_w_grid.reshape(shape) / beta,
                    order=grid_bias_order,
                ),
                log_exp_sigma=GridBias(
                    collective_variable=collective_variable,
                    n=shape[0],
                    bounds=bounds_adjusted,
                    vals=log_w_sigma_grid.reshape(shape),
                    order=grid_bias_order,
                ),
                # dv=jnp.array(jnp.array([b[1] - b[0] for b in bins])),
            )
        else:
            std_bias = None

        if output_density_bias:
            shape = tuple([len(a) - 1 for a in bins])
            dv = jnp.prod(jnp.array([(a[1] - a[0]) for a in bins]))
            # v = jnp.prod(jnp.array(n_hist_mask.shape)) * dv

            dens = n_hist_mask / jnp.sum(n_hist_mask * dv)

            dens_grid = jnp.full(hist_mask.shape, jnp.inf)
            dens_grid = dens_grid.at[hist_mask].set(-jnp.log10(dens))

            dens_bias = GridBias(
                collective_variable=collective_variable,
                n=shape[0] + 1,
                bounds=jnp.array([[(a[0] + a[1]) / 2, (a[-1] + a[-2]) / 2] for a in bins]),
                vals=-dens_grid.reshape(shape),
                order=grid_bias_order,
            )
        else:
            dens_bias = None

        if observable is not None:
            print(f"reweighting observable")

            shape = tuple([len(a) - 1 for a in bins])

            # print(f"{observable=} {grid_nums_mask=}")
            x = get_histo(
                grid_nums_mask,
                [jnp.exp(a) for a in w_log],
                observable=observable,
                log_w=False,
                macro_chunk=macro_chunk,
                verbose=verbose,
                shape_mask=hist_mask,
            )

            _, obs = x

            obs = jax.vmap(lambda x: x / jnp.exp(log_w_rho_grid_mask), in_axes=1, out_axes=1)(obs)

            print(f"{hist_mask.shape=} {obs.shape=}")

            obs_grid = jnp.full((hist_mask.shape[0], *obs.shape[1:]), jnp.nan)
            obs_grid = obs_grid.at[hist_mask, :].set(obs)

            obs = dict(
                n=shape[0] + 1,
                bounds=jnp.array([[(a[0] + a[1]) / 2, (a[-1] + a[-2]) / 2] for a in bins]),
                vals=obs_grid.reshape((*shape, -1)),
                order=grid_bias_order,
            )

        else:
            obs = None

        return bias, std_bias, dens_bias, obs

    def get_transformed_fes(
        self,
        new_cv: list[CV],
        new_colvar: CollectiveVariable,
        samples_per_bin=5,
        min_samples_per_bin: int = 1,
        chunk_size=1,
        smoothing=0.1 / (kjmol**2),
        max_bias=None,
        shmap=False,
        n_grid_old=50,
        n_grid_new=30,
    ) -> RbfBias:
        old_cv = self.cv
        old_cv_stack = CV.stack(*self.cv)
        new_cv_stack = CV.stack(*new_cv)

        # get bins for new CV
        grid_bounds_new, _, _ = CvMetric.bounds_from_cv(new_cv, margin=0.1)
        cv_mid_new, nums_new, _, closest_new, get_histo_new = DataLoaderOutput._histogram(
            n_grid=n_grid_new,
            grid_bounds=grid_bounds_new,
            metric=new_colvar.metric,
        )
        grid_nums_new = closest_new.compute_cv(new_cv_stack, chunk_size=chunk_size)[0].cv

        # get bins for old CV
        grid_bounds_old, _, _ = CvMetric.bounds_from_cv(old_cv, margin=0.1)
        cv_mid_old, nums_old, _, closest_old, get_histo_old = DataLoaderOutput._histogram(
            n_grid=n_grid_old,
            grid_bounds=grid_bounds_old,
            metric=self.collective_variable.metric,
        )
        grid_nums_old = closest_old.compute_cv(old_cv_stack, chunk_size=chunk_size)[0].cv

        assert self.ground_bias is not None

        # get old FES weights
        beta = 1 / (self.sti.T * boltzmann)
        p_grid_old = jnp.exp(
            beta
            * self.ground_bias.compute_from_cv(
                cvs=cv_mid_old,
                chunk_size=chunk_size,
            )[0]
        )
        p_grid_old /= jnp.sum(p_grid_old)

        @partial(vmap_decorator, in_axes=(None, 0))
        def f(b0, old_num):
            b = jnp.logical_and(grid_nums_old == old_num, b0)
            return jnp.sum(b)

        def prob(new_num, grid_nums_new, nums_old, p_grid_old):
            b = grid_nums_new == new_num
            bins = f(b, nums_old)

            bins_sum = jnp.sum(bins)
            p_bins = bins * jnp.where(bins_sum == 0, 0, 1 / bins_sum)

            return bins_sum, jnp.sum(p_bins * p_grid_old)

        # takes a lot of memory somehow

        prob = Partial_decorator(prob, grid_nums_new=grid_nums_new, nums_old=nums_old, p_grid_old=p_grid_old)
        prob = padded_vmap(prob, chunk_size=chunk_size)

        if shmap:
            prob = padded_shard_map(prob)

        num_grid_new, p_grid_new = prob(nums_new)  # type:ignore

        # mask = num_grid_new >= min_samples_per_bin

        # print(f"{jnp.sum(mask)}/{mask.shape[0]} bins with samples")

        # p_grid_new = p_grid_new  # [mask]

        fes_grid = -jnp.log(p_grid_new) / beta
        fes_grid -= jnp.min(fes_grid)

        raise NotImplementedError("determine eps")

        bias = RbfBias.create(
            cvs=new_colvar,
            cv=cv_mid_new,  # [mask],
            kernel="multiquadric",
            vals=-fes_grid,
            smoothing=smoothing,
        )

        if max_bias is None:
            max_bias = jnp.max(fes_grid)

        bias = BiasModify.create(
            bias=bias,
            fun=_clip,
            kwargs={"a_min": -max_bias, "a_max": 0.0},
        )

        return bias

    def transform_FES(
        self,
        trans: CvTrans,
        T: float | None = None,
        max_bias=100 * kjmol,
        n_grid=25,
    ):
        raise NotImplementedError("determine eps")

        if T is None:
            T = self.sti.T

        _, _, cv_grid, _, _ = self.collective_variable.metric.grid(n=n_grid)
        new_cv_grid, _, log_det = trans.compute_cv(cv_grid, log_Jf=True)

        FES_bias_vals, _ = self.ground_bias.compute_from_cv(cv_grid)

        new_FES_bias_vals = FES_bias_vals + T * boltzmann * log_det
        new_FES_bias_vals -= jnp.max(new_FES_bias_vals)

        # weight = jnp.exp(new_FES_bias_vals / (T * boltzmann))
        # weight /= jnp.sum(weight)

        bounds, _, _ = self.collective_variable.metric.bounds_from_cv(
            new_cv_grid,
            weights=self._weight,
            rho=self._rho,
            percentile=1e-5,
            # margin=0.1,
        )

        print(f"{bounds=}")

        new_collective_variable = CollectiveVariable(
            f=self.collective_variable.f * trans,
            metric=CvMetric.create(
                bounding_box=bounds,
                periodicities=self.collective_variable.metric.periodicities,
            ),
        )

        new_FES_bias = RbfBias.create(
            cvs=new_collective_variable,
            vals=new_FES_bias_vals,
            cv=new_cv_grid,
            kernel="multiquadric",
        )

        if max_bias is None:
            max_bias = -jnp.min(new_FES_bias_vals)

        return BiasModify.create(
            bias=new_FES_bias,
            fun=_clip,
            kwargs={"a_min": -max_bias, "a_max": 0.0},
        )

    def recalc(
        self, chunk_size: int | None = None, macro_chunk: int | None = 1000, shmap: bool = False, verbose: bool = False
    ):
        x, x_t = self.apply_cv(
            self.collective_variable.f,
            self.cv,
            self.cv_t,
            self.nl,
            self.nl_t,
            chunk_size=chunk_size,
            shmap=shmap,
            macro_chunk=macro_chunk,
            verbose=verbose,
        )

        self.cv = x
        self.cv_t = x_t

    def calc_neighbours(
        self,
        r_cut,
        chunk_size=None,
        macro_chunk=1000,
        verbose=False,
        only_update=False,
        chunk_size_inner=10,
        max=(2, 2, 2),
    ):
        if self.time_series:
            assert self.sp_t is not None
            y = [*self.sp, *self.sp_t]
        else:
            y = [*self.sp]

        from IMLCV.base.UnitsConstants import angstrom

        print(f"nl info ")

        nl_info = NeighbourListInfo.create(
            r_cut=r_cut,
            r_skin=0 * angstrom,
            z_array=self.sti.atomic_numbers,
        )

        # @partial(jit_decorator, static_argnames=["update"])
        def _f(sp, update):
            return sp._get_neighbour_list(
                info=nl_info,
                chunk_size=chunk_size,
                chunk_size_inner=chunk_size_inner,
                shmap=False,
                only_update=only_update,
                update=update,
            )

        if only_update:

            def f0(nl_update: NeighbourListUpdate | None, sp: SystemParams, *_):
                if nl_update is None:
                    b, nn, n_xyz, _ = sp._get_neighbour_list(
                        info=nl_info,
                        chunk_size=chunk_size,
                        chunk_size_inner=chunk_size_inner,
                        shmap=False,
                        only_update=only_update,
                    )

                    return NeighbourListUpdate.create(
                        nxyz=n_xyz,
                        num_neighs=int(nn),
                        stack_dims=None,
                    )

                b, new_nn, new_xyz, _ = _f(sp, nl_update)

                if not jnp.all(b):
                    n_xyz, nn = nl_update.nxyz, nl_update.num_neighs

                    assert n_xyz is not None

                    nl_update = NeighbourListUpdate.create(
                        nxyz=tuple([max(a, int(b)) for a, b in zip(n_xyz, new_xyz)]),
                        num_neighs=max(nn, int(new_nn)),
                        stack_dims=None,
                    )

                    print(f"updating neighbour list {nl_update=}")

                return nl_update

            def f_inner(x: SystemParams, nl: NeighbourList | None):
                return x

            # print(f"test")

            nl_update = macro_chunk_map_fun(
                f=f_inner,
                y=y,
                nl=None,
                macro_chunk=macro_chunk,
                verbose=verbose,
                jit_f=False,
                chunk_func=f0,
                w=None,
            )

            assert nl_update is not None

            self.nl = NeighbourList(
                info=nl_info,
                update=nl_update,
                sp_orig=None,
            )

            self.nl_t = self.nl if self.time_series else None

            return

        def f(sp: SystemParams, _: NeighbourList | None) -> NeighbourList:
            b, _, _, xnl = sp._get_neighbour_list(
                info=nl_info,
                chunk_size=chunk_size,
                chunk_size_inner=chunk_size_inner,
                shmap=False,
                only_update=only_update,
            )  # type:ignore

            assert jnp.all(b)
            assert xnl is not None

            return xnl

        nl = macro_chunk_map(
            f=f,
            # op=SystemParams.stack,
            y=y,
            nl=None,
            macro_chunk=macro_chunk,
            verbose=verbose,
            jit_f=True,
        )  # type:ignore

        if self.time_series:
            nl, nl_t = nl[0 : len(self.sp)], nl[len(self.sp) :]
        else:
            nl_t = None

        nl = cast(list[NeighbourList], nl)
        if nl_t is not None:
            nl_t = cast(list[NeighbourList], nl_t)

        self.nl = nl
        self.nl_t = nl_t
