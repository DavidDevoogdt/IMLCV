from __future__ import annotations

from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array

from IMLCV.base.CV import (
    CvTrans,
)
from IMLCV.base.dataobjects import (
    CV,
    CvMetric,
    macro_chunk_map_fun,
)
from IMLCV.base.decoratros import Partial_decorator, vmap_decorator


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
        Partial_decorator(get_histo, nn=cv_mid.shape[0]),
    )


from typing import Callable
import jax
import jax.numpy as jnp
from jax.tree_util import Partial as partial


def _build_trans_idx_fn(unique_packed: jax.Array, max_val: int):
    """
    Returns a JIT-able function that maps pairs to their 1D index.
    Transitions not in the unique set are mapped to -1.
    """

    @jax.jit
    def trans_idx_fn(from_states, to_states):
        # Pack the query transitions
        query_packed = from_states * max_val + to_states

        # Use searchsorted to find the index in the unique list
        indices = jnp.searchsorted(unique_packed, query_packed)

        # print(f"{from_states=} {to_states=} {query_packed=} {indices=}")

        # indices = jnp.reshape(indices, (-1,))

        # searchsorted returns the insertion point; we must verify the match
        # Handle OOB indices (where query > max unique_packed)
        valid_idx = jnp.minimum(indices, len(unique_packed) - 1)
        match = unique_packed[valid_idx] == query_packed

        return jnp.where(match, valid_idx, -1)

    return trans_idx_fn


def _pack_transitions(from_s: jax.Array, to_s: jax.Array, max_val: int):
    return from_s * max_val + to_s


def _unpack_transitions(packed: jax.Array, max_val: int):
    from_s = packed // max_val
    to_s = packed % max_val
    return jnp.vstack([from_s, to_s]).T


def _get_unique_transition_map(data_nums: list[CV], lag: int = 1):
    # 1. Collect all transitions across all trajectories
    all_from = []
    all_to = []
    for d in data_nums:
        cvi = d.cv[:, 0]
        all_from.append(cvi[:-lag])
        all_to.append(cvi[lag:])

    from_s = jnp.concatenate(all_from)
    to_s = jnp.concatenate(all_to)

    # 2. Pack pairs into a single int64 for unique sorting
    # We use a large multiplier (max_state + 1) to ensure uniqueness
    max_val = jnp.max(from_s) + 1
    packed_transitions = _pack_transitions(from_s, to_s, max_val)

    # 3. Find unique packed values
    unique_packed = jnp.unique(packed_transitions)

    unique_unpacked = _unpack_transitions(unique_packed, max_val)

    return unique_packed, unique_unpacked, max_val


def get_transition_counts(
    data_nums: list["CV"],
    lag: int = 1,
    weights: None | list[jax.Array] = None,
    log_w: bool = False,
    macro_chunk: int | None = 320,
    verbose: bool = False,
    # trans_idx_fn: Callable[[jax.Array, jax.Array], jax.Array] | None = None,
    nn_trans: int = -1,
    shape_mask: jax.Array | None = None,
    f_func: Callable | None = None,
) -> jax.Array:
    """
    Computes truncated transition counts using a 1D mapping.

    trans_idx_fn: A JAX-jittable function that takes `from_states` and `to_states`
                    and returns an integer index [0, nn_trans - 1].
                    Should return -1 for transitions that are truncated/ignored.
    """

    unique_p, unique_unpacked, m_val = _get_unique_transition_map(data_nums, lag=1)
    trans_idx_fn = _build_trans_idx_fn(unique_p, m_val)

    print(f"{unique_unpacked.shape=} {m_val=}")

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

    def _get_trans(
        x: tuple[jax.Array | None, jax.Array, jax.Array | None],
        cv_i: CV,
        cv_it: CV | None,
        weights: jax.Array | None,
        _weights_t=None,
        _dweights_t=None,
    ) -> tuple[jax.Array | None, jax.Array, jax.Array | None]:
        alpha_factors, h, obs_array = x

        # cvi = cv_i.cv[:, 0]

        # Slice trajectories by lag to get from -> to pairs
        # from_s = cvi[:-lag]
        # to_s = cvi[lag:]

        from_s = cv_i.cv
        to_s = cv_it.cv

        # print(f"{from_s.shape=} {to_s.shape=}")

        # Map 2D transitions to 1D tracked indices
        trans_idx = trans_idx_fn(from_s, to_s)

        if not log_w:
            wi = jnp.ones_like(from_s) if weights is None else weights
            h = h.at[trans_idx].add(wi)

            # if obs is not None:
            #     # Align observable to the start state of the transition
            #     u = obs.cv[:-lag] * wi[:, None]
            #     obs_array = obs_array.at[trans_idx, :].add(u)  # type: ignore

            return (alpha_factors, h, obs_array)

        if alpha_factors is None:
            alpha_factors = jnp.full_like(h, -jnp.inf)

        # Log space calculations
        w_log = jnp.zeros_like(from_s) if weights is None else weights

        # Use your exact vmapped log-sum-exp trick, applied to transition indices
        @partial(vmap_decorator, in_axes=(0, None, None, 0))
        def get_max(ref_num, data_num, log_w, alpha):
            val = jnp.where(data_num == ref_num, log_w, -jnp.inf)
            max_new = jnp.max(val)  # type: ignore
            max_tot = jnp.max(jnp.array([alpha, max_new]))
            return max_tot, max_new > alpha, max_tot - alpha

        alpha_factors, alpha_factors_changed, d_alpha = get_max(nn_range, trans_idx, w_log, alpha_factors)

        # shift all probs by the change in alpha
        h = jnp.where(alpha_factors_changed, h - d_alpha, h)  # type:ignore

        m = alpha_factors != -jnp.inf

        print(f"{w_log=} {alpha_factors=} {alpha_factors_changed=} {d_alpha=} {trans_idx=} ")

        trans_idx = trans_idx.reshape(-1)

        p_new = jnp.zeros_like(h)
        p_new = p_new.at[trans_idx].add(jnp.exp(w_log - alpha_factors[trans_idx]))  # type: ignore

        h = jnp.where(m, jnp.log(jnp.exp(h) + p_new), h)

        return (alpha_factors, h, obs_array)

    (alpha_factors, h, obs) = macro_chunk_map_fun(
        f=_f_func,
        y=[d[:-lag] for d in data_nums],
        y_t=[d[lag:] for d in data_nums],
        macro_chunk=macro_chunk,
        verbose=verbose,
        chunk_func=_get_trans,
        chunk_func_init_args=(
            None,
            h,
            None,
        ),
        w=weights,
        jit_f=True,
    )

    if log_w:
        assert alpha_factors is not None
        h += alpha_factors

    h = h[:-1]

    return h, unique_unpacked
