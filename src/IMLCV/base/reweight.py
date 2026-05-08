from __future__ import annotations


import jax
import jax.numpy as jnp
from jax import Array


from collections import defaultdict
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jaxopt
import numpy as np


from IMLCV.base.decoratros import jit_decorator, vmap_decorator


def _solve_wham(log_U_ik_nl, H_ik_nl, verbose=False, compute_std=False):
    # print(f"{tau_i_nl=}")

    def log_sum_exp_safe(*x: Array, min_val=None):
        _x: Array = jnp.nansum(jnp.stack(x, axis=0), axis=0)  # type:ignore

        x_max = jnp.nanmax(_x)

        x_max = jnp.where(jnp.isfinite(x_max), x_max, 0.0)

        out = jnp.log(jnp.nansum(jnp.exp(_x - x_max))) + x_max

        if min_val is not None:
            print(f"using {min_val=}")
            out: Array = jnp.where(out < min_val, min_val, out)  # type:ignore

        return out + 1e-10

    def get_N_H(mask_ik: jax.Array, H_ik: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        # print(f"{mask_i.shape=}  {H_ik.shape=} ")

        _H_ik = jnp.where(mask_ik, H_ik, 0)  # type:ignore

        _H_k = jnp.sum(_H_ik, axis=(0))
        _N_i = jnp.sum(_H_ik, axis=(1))

        return _N_i, _H_k, _H_ik

    def get_F_i(F_k: jax.Array, mask_ik: jax.Array, log_x: tuple[Array, Array]):
        log_U_ik, _ = log_x

        log_U_ik = jnp.where(mask_ik, log_U_ik, jnp.inf)

        F_i = -vmap_decorator(log_sum_exp_safe, in_axes=(None, 0))(-F_k, -log_U_ik)

        return F_i

    def norm_F_k(F_k: Array):
        # mask_k = jnp.isfinite(F_k)

        F_k = jnp.where(jnp.isfinite(F_k), F_k, jnp.inf)

        F_k_norm = log_sum_exp_safe(-F_k)

        F_k += F_k_norm

        return F_k

    def get_F_k(
        F_i: Array,
        mask_ik: Array,
        log_x: tuple[Array, Array],
    ):
        log_U_ik, H_ik = log_x

        # mask_i = jnp.isfinite(F_i)

        N_i, H_k, H_ik = get_N_H(mask_ik, H_ik)

        def _s(*x: Array):
            return jnp.nansum(jnp.array(x))

        # odds of using sample k
        log_ps_ik = vmap_decorator(
            vmap_decorator(
                _s,
                in_axes=(None, 0, None, 0),
            ),  # k
            in_axes=(0, 0, 0, None),
        )(
            # -jnp.log(H_ik),
            jnp.log(N_i),
            -log_U_ik,
            +F_i,
            -vmap_decorator(log_sum_exp_safe, in_axes=(None, None, 1))(
                jnp.log(N_i),
                F_i,
                -log_U_ik,
            ),
        )

        # log_rho_ik = jnp.where(log_rho_ik < -10 * jnp.log(10), -jnp.inf, log_rho_ik)

        log_w_ik = vmap_decorator(
            vmap_decorator(
                _s,
                in_axes=(0, None),
            ),  # k
            in_axes=(0, 0),  # i
        )(
            log_U_ik,
            -F_i,
        )

        log_dens_ik = jax.vmap(jnp.add, in_axes=(0, 0))(
            jnp.log(H_ik),
            -jnp.log(N_i),
        )

        mask_ik = jnp.logical_and(
            mask_ik,
            jnp.logical_and(jnp.isfinite(log_ps_ik), jnp.isfinite(log_w_ik)),
        )

        # mask_ik = jnp.logical_and(
        #     mask_ik,
        #     jnp.logical_and(
        #         jnp.isfinite(log_ps_ik),
        #         log_ps_ik > jnp.log(1e-10),
        #     ),
        # )

        log_ps_ik = jnp.where(mask_ik, log_ps_ik, -jnp.inf)
        log_w_ik = jnp.where(mask_ik, log_w_ik, -jnp.inf)
        log_dens_ik = jnp.where(mask_ik, log_dens_ik, -jnp.inf)

        log_ps_ik = jnp.where(jnp.isfinite(log_ps_ik), log_ps_ik, -jnp.inf)
        log_w_ik = jnp.where(jnp.isfinite(log_w_ik), log_w_ik, -jnp.inf)
        log_dens_ik = jnp.where(jnp.isfinite(log_dens_ik), log_dens_ik, -jnp.inf)

        F_k = -vmap_decorator(log_sum_exp_safe, in_axes=(1, 1, 1))(
            log_dens_ik,
            log_w_ik,
            log_ps_ik,
        )  # sum over i

        F_k = jnp.where(jnp.isfinite(F_k), F_k, -jnp.inf)

        F_k = norm_F_k(F_k)
        return F_k, (mask_ik, log_dens_ik, log_w_ik, log_ps_ik)

    @jit_decorator
    def T(x: tuple[Array, Array], log_x: tuple[Array, Array]):
        F_k, mask_ik = x
        mask_ik = jnp.where(mask_ik == 1.0, True, False)
        # F_k = -jnp.log(a_k)/

        F_i = get_F_i(F_k, mask_ik, log_x)
        F_k, (mask_ik, log_dens_ik, log_w_ik, log_ps_ik) = get_F_k(F_i, mask_ik, log_x)

        return (F_k, jnp.where(mask_ik, 1.0, 0.0)), (F_i, log_w_ik, log_ps_ik, log_dens_ik)

    @jit_decorator
    def norm(x: tuple[Array, Array], log_x: tuple[Array, Array]):
        F_k, mask_ik = x
        (F_k_p, _), _ = T((F_k, mask_ik), log_x)

        a_k = jnp.exp(-F_k)
        a_k_p = jnp.exp(-F_k_p)

        return 0.5 * jnp.sum((a_k - a_k_p) ** 2)

    @jit_decorator
    def kl_div(x: tuple[Array, Array], log_x):
        F_k, mask_ik = x
        (F_k_p, _), _ = T(x, log_x)

        print("inside kl div")

        a_k = jnp.exp(-F_k)

        kl_div_k = a_k * (-F_k + F_k_p)

        return kl_div_k  # , jnp.sum(kl_div_k)

    import jaxopt

    log_x_mask = (log_U_ik_nl, H_ik_nl)

    loss_f = True

    #
    mask_ik_float = jnp.full(H_ik_nl.shape, 1.0)
    mask_ik = jnp.where(mask_ik_float == 1.0, True, False)
    N_i, H_k, H_ik = get_N_H(mask_ik, H_ik_nl)

    # this optimizes F_i. much faster
    @jit_decorator
    def loss(F_i, N_i, log_U_ik_nl, H_k):
        # F_i -= jnp.min(F_i)

        loss = jnp.sum(N_i * F_i) - jnp.sum(
            H_k * jax.vmap(log_sum_exp_safe, in_axes=(None, None, 1))(jnp.log(N_i), F_i, -log_U_ik_nl)
        )

        # F_i+C is also a solution for any C, so we add a small regularization to fix the scale
        # reg = 1e-10 * jnp.sum(F_i**2) / F_i.shape[0]

        return -loss  # + reg

    if loss_f:
        solver = jaxopt.LBFGS(
            fun=loss,
            maxiter=10000,
            tol=1e-10,
            verbose=True,
        )
        out = solver.run(
            jnp.zeros(N_i.shape, dtype=jnp.float_),
            N_i=N_i,
            log_U_ik_nl=log_U_ik_nl,
            H_k=H_k,
        )
        F_i_nl = out.params
        # get F_k and norm
        F_k_nl, (mask_ik_nl, log_dens_ik_nl, log_w_ik_nl, log_ps_ik_nl) = get_F_k(F_i_nl, mask_ik, log_x_mask)
        F_i_nl = get_F_i(F_k_nl, mask_ik, log_x_mask)  # get F_i from normed F_k to ensure consistency
        F_k_nl, (mask_ik_nl, log_dens_ik_nl, log_w_ik_nl, log_ps_ik_nl) = get_F_k(F_i_nl, mask_ik, log_x_mask)

        mask_ik_nl = jnp.where(mask_ik_nl == 1.0, True, False)

    else:
        from jaxopt import base
        from jaxopt._src.fixed_point_iteration import FixedPointState

        class FP(jaxopt.FixedPointIteration):
            def update(self, params, state: FixedPointState, *args, **kwargs) -> base.OptStep:
                next_params, aux = self._fun(params, *args, **kwargs)

                F_k, _ = params
                F_k_p, _ = next_params

                error = jnp.sqrt(jnp.mean((F_k - F_k_p) ** 2))

                # a_k = jnp.exp(-F_k)
                # error = jnp.sum(a_k * (-F_k + F_k_p))

                next_state = FixedPointState(
                    iter_num=state.iter_num + 1,
                    error=error,  # type:ignore
                    aux=aux,
                    num_fun_eval=state.num_fun_eval + 1,
                )

                if self.verbose:
                    self.log_info(next_state, error_name="Distance btw Iterates")

                jax.lax.cond(
                    state.iter_num % 100 == 0,
                    lambda: jax.debug.print("iter {iter_num}, error {error:.2e}", iter_num=state.iter_num, error=error),  # type:ignore
                    lambda: None,
                )

                return base.OptStep(params=next_params, state=next_state)

        solver = FP(
            fixed_point_fun=T,
            # history_size=5,
            tol=1e-6,
            implicit_diff=True,
            has_aux=True,
            maxiter=2000,
        )

        out = solver.run(
            (
                jnp.full((log_U_ik_nl.shape[1],), 0.0),
                jnp.full(H_ik_nl.shape, 1.0),
            ),
            log_x=log_x_mask,
        )

        F_k_nl, mask_ik_float = out.params
        mask_ik_nl = jnp.where(mask_ik_float == 1.0, True, False)

        assert out.state.aux is not None
        F_i_nl, log_w_ik_nl, log_ps_ik_nl, log_dens_ik_nl = out.state.aux

        # print(f"{jnp.exp(log_w_ik_nl)=}")

        if verbose:
            print(f" {out.state.iter_num=} {out.state.error=} ")

    if jnp.isnan(out.state.error):
        print(f"error is nan {jnp.sum(mask_ik_nl)=} {out.state.iter_num=}")
        raise

        # print(f"wham done! {out.state.error=} ")

    if verbose:
        n, k = norm((F_k_nl, mask_ik_float), log_x_mask), kl_div((F_k_nl, mask_ik_float), log_x_mask)
        print(f"wham err={n}, kl divergence={jnp.sum(k)}  ")

    if compute_std:
        method = "thermolib"

        len_i, len_k = H_ik.shape

        H_ik_tau = H_ik

        N_i_tau = jnp.sum(H_ik_tau, axis=1)

        mask_i = jnp.any(mask_ik_nl, axis=1)
        mask_k = jnp.any(mask_ik_nl, axis=0)

        # print(f"{mask_i=} {mask_k=}")

        H_ik_mask = H_ik[mask_i, :][:, mask_k]
        log_U_ik_mask = log_U_ik_nl[mask_i, :][:, mask_k]
        F_i_mask = F_i_nl[mask_i]
        F_k_mask = F_k_nl[mask_k]
        N_i_tau_mask = N_i_tau[mask_i]
        H_k_mask = H_k[mask_k]
        H_ik_tau_mask = H_ik_tau[mask_i, :][:, mask_k]

        log_N_i_tau_mask = jnp.log(N_i_tau_mask)
        # print(f"{N_i_tau_mask=}")

        log_N_tot_tau_mask = jnp.log(jnp.sum(H_ik_tau_mask))

        nk = F_k_mask.shape[0]
        ni = log_N_i_tau_mask.shape[0]

        if not jnp.all(mask_i):
            print(f"WARNING: some trajectories have no valid bins {jnp.argwhere(jnp.logical_not(mask_i))=}")

        if not jnp.all(mask_k):
            print(f"WARNING: some bins have no valid samples {jnp.argwhere(jnp.logical_not(mask_k))=}")

        # see thermolib derivation

        print(f"test: {nk=} {ni=}")

        def _s(*x):
            return jnp.sum(jnp.stack(x, axis=0), axis=0)

        # this gives same result
        # def lagrangian_full(F_k, F_i, lambda_i, mu):
        #     # Logic to unpack z into F_i, F_k, lambdas, mu
        #     # Then return the loss value

        #     # F_i, F_k, lambda_i, mu = unpack_all(z)

        #     log_arg = F_i[:, None] - F_k[None, :] - log_U_ik_mask

        #     term_1 = jnp.sum(H_ik_tau_mask * log_arg)
        #     term_2 = jnp.exp(log_N_tot_tau_mask) * mu * (1 - jnp.sum(jnp.exp(-F_k)))
        #     term_3 = jnp.sum(N_i_tau * lambda_i * (1 - jnp.sum(jnp.exp(log_arg), axis=1)))

        #     err = term_1 + term_2 + term_3

        #     return -err

        A_FF_diag = vmap_decorator(
            lambda log_b_i, F_k_mask: jnp.sum(jnp.exp(-F_k_mask + log_N_i_tau_mask + F_i_mask - log_b_i)),
            in_axes=(1, 0),
        )(log_U_ik_mask, F_k_mask)

        A_FG = -jnp.exp(
            vmap_decorator(
                vmap_decorator(_s, in_axes=(0, None, None, 0)),  # k
                in_axes=(None, 0, 0, 0),  # i
            )(-F_k_mask, log_N_i_tau_mask, F_i_mask, -log_U_ik_mask)
        ).T

        A_Fl = -jnp.exp(
            vmap_decorator(
                vmap_decorator(_s, in_axes=(0, None, None, 0)),  # k
                in_axes=(None, 0, 0, 0),  # i
            )(-F_k_mask, log_N_i_tau_mask, F_i_mask, -log_U_ik_mask)
        ).T

        A_Fm = -jnp.exp(-F_k_mask + log_N_tot_tau_mask).reshape((nk, 1))

        A_GG_diag = jnp.exp(log_N_i_tau_mask)

        A_Gl_diag = jnp.exp(log_N_i_tau_mask)

        # make shur complement

        print(f"using precond")

        @jit_decorator
        def matvec(x):
            # Split x into the components corresponding to F, G, l, m
            x_F = x[:nk]
            x_G = x[nk : nk + ni]
            x_l = x[nk + ni : nk + 2 * ni]
            x_m = x[nk + 2 * ni :]

            # Perform block-multiplication using your individual blocks
            y_F = A_FF_diag * x_F + A_FG @ x_G + A_Fl @ x_l + A_Fm @ x_m
            y_G = A_FG.T @ x_F + A_GG_diag * x_G + A_Gl_diag * x_l
            y_l = A_Fl.T @ x_F + A_Gl_diag * x_G
            y_m = A_Fm.T @ x_F

            return jnp.concatenate([y_F, y_G, y_l, y_m])

        total_dim = 2 * ni + nk + 1

        print(
            f"sparsity = {(jnp.size(A_FF_diag) + jnp.size(A_FG) + jnp.size(A_Fl) + jnp.size(A_Fm) + jnp.size(A_GG_diag) + jnp.size(A_Gl_diag)) / (total_dim**2):.2e} "
        )

        F_diag = jnp.hstack([A_FF_diag, A_GG_diag, jnp.zeros(ni), jnp.zeros(1)])
        F_diag_inv = jnp.where(F_diag == 0, 1.0, F_diag ** (-1))

        if jnp.isnan(F_diag).any():
            print(f"WARNING: F_diag contains nan values {F_k_mask=} {log_N_i_tau_mask=} {F_i_mask=} {log_U_ik_mask=}")

        def preconditioner(x):
            return x * F_diag_inv

        def hutchinson_diag_inv(matvec_fn, total_dim, num_samples=32, key=None):
            """
            Estimates the diagonal of the inverse of a matrix.

            Args:
                matvec_fn: Function that computes F @ x.
                total_dim: The dimension of the matrix (e.g., 10000).
                num_samples: Number of random vectors (higher = more accurate).
                key: JAX PRNG key.
            """
            if key is None:
                key = jax.random.PRNGKey(0)

            # 1. Generate Rademacher random vectors (values are -1 or 1)
            # These are the "probes"
            v = jax.random.choice(key, jnp.array([-1.0, 1.0]), shape=(num_samples, total_dim))

            # 2. Define the solve for a single random vector
            @jit_decorator
            def solve_inverse_probe(vi):
                jax.debug.print(".")
                # We solve F * xi = vi  =>  xi = F^-1 * vi
                # We use GMRES because of the saddle-point (zero diagonal) structure
                xi, _ = jax.scipy.sparse.linalg.gmres(
                    matvec_fn,
                    vi,
                    tol=1e-3,
                    restart=30,
                    maxiter=10,
                    M=preconditioner,
                )
                return xi

            # 3. Use vmap to solve all probes in parallel (leveraging GPU)src/IMLCV/new_yaff
            # x_probes shape: (num_samples, total_dim)
            x_probes = jax.vmap(solve_inverse_probe)(v)

            # 4. Compute the element-wise product and average
            # (v * (F^-1 * v)) summed over the samples
            diag_estimates = v * x_probes

            mean_diag = jnp.nanmean(diag_estimates, axis=0)

            return mean_diag

        # Use vmap to parallelize the solves for the first 'nk' elements
        # do this with regular map for big matrices?
        # sigma_Fk_mask = jax.vmap(get_diag_element)(jnp.arange(nk))
        # sigma_Fk_mask = jax.lax.map(get_diag_element, jnp.arange(nk), batch_size=100)
        sigma_Fk_mask = hutchinson_diag_inv(matvec, total_dim, num_samples=128, key=jax.random.PRNGKey(42))[:nk]

        print(f"{sigma_Fk_mask=}")

        sigma_Fk_mask: Array = jnp.where(sigma_Fk_mask < 0, 0, jnp.sqrt(sigma_Fk_mask))  # type:ignore
        sigma_Fk = jnp.full((len_k,), jnp.inf)
        sigma_Fk = sigma_Fk.at[mask_k].set(sigma_Fk_mask)

    else:
        sigma_Fk = None

    return log_w_ik_nl, log_ps_ik_nl, F_i_nl, log_dens_ik_nl, mask_ik_nl, F_k_nl, sigma_Fk


def _norm_log_vec_masked(log_v: Array, mask: Array) -> Array:
    """Row-normalise a log-probability vector over valid (mask=True) entries."""
    log_v = jnp.where(mask, log_v, -jnp.inf)
    log_Z = jax.scipy.special.logsumexp(log_v)
    return jnp.where(mask, log_v - log_Z, -jnp.inf)


class DHAMSparseResult(NamedTuple):
    F_k: Array  # (n_bins,)              unbiased free energies
    pi_k: Array  # (n_bins,)              unbiased stationary distribution
    log_T_kn: Array  # (n_bins, n_neigh_max)  log transition probs (sparse)
    neigh_kn: Array  # (n_bins, n_neigh_max)  neighbor bin indices, -1=pad
    log_w_ik: Array  # (n_traj, n_bins)       reweighting log-weights
    #   same semantics as WHAM log_w_ik_nl
    converged: bool
    n_iter: int
    final_loss: float


def _dham_build_sparse_transitions(
    dtraj_list: list[Array],  # discrete trajectories, one per window
    window_ids: list[int],  # window index per trajectory
    n_windows: int,
    n_bins: int,
    n_neigh_max: int,
    lagtime: int = 1,
) -> tuple[Array, Array]:
    """
    Build sparse transition arrays from discrete trajectories.

    Neighbors are chosen as the top-n_neigh_max most visited destinations
    pooled over all windows. This ensures the neighbor set is shared between
    C_ikn and T_kn.

    Returns
    -------
    neigh_kn : (n_bins, n_neigh_max) int32
        Destination bin index for each source bin k and neighbor slot n.
        Padded with -1 where fewer than n_neigh_max neighbors exist.
    C_ikn : (n_windows, n_bins, n_neigh_max) float32
        Transition counts. C_ikn[i, k, n] = count of transitions from bin k
        to bin neigh_kn[k, n] in window i.
    """
    # ── pass 1: accumulate raw counts per (window, src, dst) ─────────────────
    # counts_k[k] = {dst: total_count_pooled_over_windows}
    counts_pooled: list[dict[int, int]] = [defaultdict(int) for _ in range(n_bins)]
    # counts_ik[i][k] = {dst: count}
    counts_ik: list[list[dict[int, int]]] = [[defaultdict(int) for _ in range(n_bins)] for _ in range(n_windows)]

    for dtraj, win_id in zip(dtraj_list, window_ids):
        arr = jnp.asarray(dtraj, dtype=jnp.int32)
        src = arr[:-lagtime]
        dst = arr[lagtime:]
        valid = (src >= 0) & (src < n_bins) & (dst >= 0) & (dst < n_bins)
        for s, d in zip(src[valid], dst[valid]):
            counts_pooled[s][d] += 1
            counts_ik[win_id][s][d] += 1

    # ── pass 2: select top-n_neigh_max neighbors per source bin ──────────────
    neigh_kn = jnp.full((n_bins, n_neigh_max), -1, dtype=jnp.int32)

    for k in range(n_bins):
        neighbors = counts_pooled[k]
        if not neighbors:
            continue
        top = sorted(neighbors.keys(), key=lambda d: -neighbors[d])[:n_neigh_max]
        for n, dst in enumerate(top):
            neigh_kn[k, n] = dst

    # ── pass 3: fill C_ikn using the fixed neighbor slots ────────────────────
    # build reverse lookup: for each source bin k, dst → slot n
    dst_to_slot: list[dict[int, int]] = [{} for _ in range(n_bins)]
    for k in range(n_bins):
        for n in range(n_neigh_max):
            d = neigh_kn[k, n]
            if d >= 0:
                dst_to_slot[k][d] = n

    C_ikn = jnp.zeros((n_windows, n_bins, n_neigh_max), dtype=jnp.float32)
    for i in range(n_windows):
        for k in range(n_bins):
            for dst, cnt in counts_ik[i][k].items():
                slot = dst_to_slot[k].get(dst, None)
                if slot is not None:
                    C_ikn[i, k, slot] += cnt
                # transitions to non-neighbor bins are dropped (should be rare
                # if n_neigh_max is large enough — check with diagnostics below)

    return neigh_kn, C_ikn


def _dham_neighbor_diagnostics(
    dtraj_list: list[Array],
    window_ids: list[int],
    n_windows: int,
    n_bins: int,
    lagtime: int = 1,
) -> None:
    """
    Print statistics on the number of distinct neighbors per source bin.
    Use this to choose n_neigh_max before building the full sparse structure.
    """
    counts_pooled: list[dict[int, int]] = [defaultdict(int) for _ in range(n_bins)]
    for dtraj, win_id in zip(dtraj_list, window_ids):
        arr = jnp.asarray(dtraj, dtype=jnp.int32)
        src = arr[:-lagtime]
        dst = arr[lagtime:]
        valid = (src >= 0) & (src < n_bins) & (dst >= 0) & (dst < n_bins)
        for s, d in zip(src[valid], dst[valid]):
            counts_pooled[s][d] += 1

    n_neigh = jnp.array([len(counts_pooled[k]) for k in range(n_bins)])
    n_neigh = n_neigh[n_neigh > 0]  # only visited bins

    print(f"Neighbor count statistics over visited bins:")
    print(f"  mean:   {n_neigh.mean():.1f}")
    print(f"  median: {jnp.median(n_neigh):.1f}")
    print(f"  90th %: {jnp.percentile(n_neigh, 90):.1f}")
    print(f"  99th %: {jnp.percentile(n_neigh, 99):.1f}")
    print(f"  max:    {n_neigh.max()}")
    print(f"Suggested n_neigh_max: {int(jnp.percentile(n_neigh, 99)) + 2}")


def _dham_solve_dham_sparse(
    log_U_ik: Array,  # (n_traj, n_bins)
    H_ik: Array,  # (n_traj, n_bins)  stationary counts
    C_ikn: Array,  # (n_traj, n_bins, n_neigh_max) transition counts
    neigh_kn: Array,  # (n_bins, n_neigh_max) destination bin indices
    *,
    maxiter: int = 5000,
    tol: float = 1e-9,
    verbose: bool = False,
) -> DHAMSparseResult:
    """
    Solve DHAM with sparse (n_bins, n_neigh_max) storage for both C and T.

    Parameters
    ----------
    log_U_ik:
        Log bias factor. Same sign convention as WHAM:
            log_U_ik[i, k] = -beta * U_i(x_k)
        Unvisited bins in window i: set to -inf (or very large negative).
    H_ik:
        Stationary histogram counts per window per bin (same as WHAM).
    C_ikn:
        Sparse transition counts. C_ikn[i, k, n] = count of transitions
        from bin k to bin neigh_kn[k, n] in window i.
    neigh_kn:
        Neighbor index array from build_sparse_transitions. -1 = padding.
    """
    n_traj, n_bins = H_ik.shape
    _, _, n_neigh_max = C_ikn.shape

    neigh_kn = jnp.asarray(neigh_kn, dtype=jnp.int32)
    C_ikn = jnp.asarray(C_ikn, dtype=jnp.float32)
    log_U_ik = jnp.asarray(log_U_ik, dtype=jnp.float32)
    H_ik = jnp.asarray(H_ik, dtype=jnp.float32)

    # ── masks ─────────────────────────────────────────────────────────────────
    mask_ik = H_ik > 0  # (n_traj, n_bins)
    mask_k = jnp.any(mask_ik, axis=0)  # (n_bins,)
    mask_kn = neigh_kn >= 0  # (n_bins, n_neigh_max)

    # safe neighbor indices for gather (replace -1 with 0 — masked out later)
    neigh_safe = jnp.where(mask_kn, neigh_kn, 0)  # (n_bins, n_neigh_max)

    log_U_ik = jnp.where(mask_ik, log_U_ik, -500.0)

    # ── parameterisation ──────────────────────────────────────────────────────
    # Free parameters:
    #   log_pi_raw : (n_bins,)              — unnormalised log stationary dist
    #   log_T_raw  : (n_bins, n_neigh_max)  — unnormalised log transition probs
    #
    # Both are normalised inside the loss.

    n_params_pi = n_bins
    n_params_T = n_bins * n_neigh_max

    def unpack(theta: Array):
        log_pi_raw = theta[:n_params_pi]
        log_T_raw = theta[n_params_pi:].reshape(n_bins, n_neigh_max)

        # normalise π over valid bins
        log_pi_k = _norm_log_vec_masked(log_pi_raw, mask_k)  # (n_bins,)

        # row-normalise T over valid neighbor slots
        log_T_kn = jax.vmap(_norm_log_vec_masked)(log_T_raw, mask_kn)  # (n_bins, n_neigh_max)

        return log_pi_k, log_T_kn

    # ── negative log-likelihood (sparse) ──────────────────────────────────────

    @jax.jit
    def neg_log_likelihood(theta: Array) -> Array:
        log_pi_k, log_T_kn = unpack(theta)

        # log Z_i = log Σ_k π_k · exp(log_U_ik[i,k])   shape: (n_traj,)
        log_Z_i = jax.scipy.special.logsumexp(log_pi_k[None, :] + log_U_ik, axis=1)

        # gather log T for each (src=k, dst=neigh_kn[k,n])
        # log_T_kn[k, n] is already the right value — no gather needed for T
        # We need log_pi and log_U at the source bin k:
        #   shape broadcast: (n_traj, n_bins, n_neigh_max)

        log_p_ikn = (
            log_pi_k[None, :, None]  # (1, n_bins, 1)   π at source k
            + log_U_ik[:, :, None]  # (n_traj, n_bins, 1) bias at k
            + log_T_kn[None, :, :]  # (1, n_bins, n_neigh_max) T_{k,n}
            - log_Z_i[:, None, None]  # (n_traj, 1, 1)   normalisation
        )

        # valid mask: bin k must be visited in window i AND neighbor slot is real
        valid = (
            mask_ik[:, :, None]  # (n_traj, n_bins, 1)
            & mask_kn[None, :, :]  # (1, n_bins, n_neigh_max)
            & (C_ikn > 0)  # (n_traj, n_bins, n_neigh_max)
        )

        C_safe = jnp.where(valid, C_ikn, 0.0)
        log_p_safe = jnp.where(valid, log_p_ikn, 0.0)

        return -jnp.sum(C_safe * log_p_safe)

    # ── initialisation ────────────────────────────────────────────────────────
    # log π: proportional to pooled stationary counts
    log_H_k = jnp.log(jnp.sum(H_ik, axis=0) + 1.0)
    log_pi0 = _norm_log_vec_masked(log_H_k, mask_k)  # (n_bins,)

    # log T: proportional to pooled transition counts per neighbor slot
    C_kn_pooled = jnp.sum(C_ikn, axis=0) + 1e-10  # (n_bins, n_neigh_max)
    log_T0 = jax.vmap(_norm_log_vec_masked)(jnp.log(C_kn_pooled), mask_kn)  # (n_bins, n_neigh_max)

    theta0 = jnp.concatenate([log_pi0, log_T0.ravel()])

    # ── optimise ──────────────────────────────────────────────────────────────
    solver = jaxopt.LBFGS(
        fun=neg_log_likelihood,
        maxiter=maxiter,
        tol=tol,
        verbose=verbose,
    )
    result = solver.run(theta0)
    theta_opt = result.params

    # ── extract results ───────────────────────────────────────────────────────
    log_pi_k, log_T_kn = unpack(theta_opt)

    pi_k = jnp.exp(log_pi_k)

    # Free energies (same convention as WHAM F_k)
    F_k = -log_pi_k
    F_k = F_k - jnp.min(jnp.where(mask_k, F_k, jnp.inf))

    # Per-window reweighting log-weights (same semantics as WHAM log_w_ik_nl)
    log_w_ik = log_pi_k[None, :] - log_U_ik
    log_w_ik = jnp.where(mask_ik, log_w_ik, -jnp.inf)

    converged = bool(result.state.error < tol)
    n_iter = int(result.state.iter_num)
    final_loss = float(neg_log_likelihood(theta_opt))

    if verbose:
        print(f"DHAM sparse done: {n_iter=}, error={result.state.error:.2e}, {converged=}")
        print(f"  π_k range: [{float(pi_k.min()):.2e}, {float(pi_k.max()):.2e}]")
        # fraction of transition counts captured by neighbor structure
        total_C = float(jnp.sum(C_ikn))
        print(f"  total transitions captured: {total_C:.0f}")

    return DHAMSparseResult(
        F_k=F_k,
        pi_k=pi_k,
        log_T_kn=log_T_kn,
        neigh_kn=neigh_kn,
        log_w_ik=log_w_ik,
        converged=converged,
        n_iter=n_iter,
        final_loss=final_loss,
    )


def _dham_reconstruct_T_dense(
    result: DHAMSparseResult,
    n_bins: int,
) -> Array:
    """
    Reconstruct a dense (n_bins, n_bins) transition matrix from sparse result.
    Only use for analysis / small n_bins — will OOM for large systems.
    """
    T = jnp.zeros((n_bins, n_bins), dtype=jnp.float32)
    log_T_kn = jnp.array(result.log_T_kn)
    neigh_kn = jnp.array(result.neigh_kn)

    for k in range(n_bins):
        for n in range(neigh_kn.shape[1]):
            dst = neigh_kn[k, n]
            if dst >= 0:
                T[k, dst] = jnp.exp(log_T_kn[k, n])
    return T
