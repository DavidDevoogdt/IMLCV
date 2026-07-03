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


def _s(*x: Array):
    return jnp.nansum(jnp.array(x))


def safe_log(x: Array) -> Array:
    """log(x) that returns -inf (with a *zero* gradient, not nan) for x <= 0.

    jnp.log(0) -> -inf with an inf/nan gradient; differentiating through it
    (which happens here inside LBFGS / implicit-diff) poisons the optimizer.
    The double-where keeps the backward pass finite.
    """
    pos = x > 0
    return jnp.where(pos, jnp.log(jnp.where(pos, x, 1.0)), -jnp.inf)


def log_sum_exp_safe(*x: Array, min_val=None):
    _x: Array = jnp.nansum(jnp.stack(x, axis=0), axis=0)  # type:ignore

    # nansum can still produce nan from (+inf) + (-inf) when a collapsed bin
    # meets a masked entry. Treat that as an empty (-inf) contribution.
    _x = jnp.where(jnp.isnan(_x), -jnp.inf, _x)

    x_max = jnp.nanmax(_x)
    finite_max = jnp.isfinite(x_max)
    x_max_s = jnp.where(finite_max, x_max, 0.0)

    # exclude non-finite entries from the sum instead of relying on nansum
    sumexp = jnp.sum(jnp.where(jnp.isfinite(_x), jnp.exp(_x - x_max_s), 0.0))

    # if every contribution was -inf (empty), return -inf cleanly (no nan)
    out: Array = jnp.where(finite_max, jnp.log(sumexp) + x_max_s, -jnp.inf)  # type:ignore

    if min_val is not None:
        out = jnp.where(out < min_val, min_val, out)  # type:ignore

    return out


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


def norm_F_k(F_k: Array, max_spread: float = 700.0):
    # mask_k = jnp.isfinite(F_k)

    finite = jnp.isfinite(F_k)
    F_k = jnp.where(finite, F_k, jnp.inf)

    F_k_norm = log_sum_exp_safe(-F_k)

    F_k += F_k_norm

    return F_k


def get_F_k(
    F_i: Array,
    mask_ik: Array,
    log_x: tuple[Array, Array],
    overlap_logU_cap: float | None = None,
):
    log_U_ik, H_ik = log_x

    # mask_i = jnp.isfinite(F_i)

    N_i, H_k, H_ik = get_N_H(mask_ik, H_ik)

    if overlap_logU_cap is not None:
        # only keep samles wihtin 30 kT of bias minimum.
        logU_min_i = jnp.min(jnp.where(mask_ik, log_U_ik, jnp.inf), axis=1, keepdims=True)
        excess_ik = log_U_ik - logU_min_i  # excess bias per (i, k)
        min_excess_k = jnp.min(jnp.where(mask_ik, excess_ik, jnp.inf), axis=0)
        supported_k = min_excess_k < overlap_logU_cap  # reached by some window within cap
        mask_ik = jnp.logical_and(mask_ik, supported_k[None, :])
        N_i, H_k, H_ik = get_N_H(mask_ik, H_ik)

    # odds of using sample k
    log_ps_ik = vmap_decorator(
        vmap_decorator(
            _s,
            in_axes=(None, 0, None, 0),
        ),  # k
        in_axes=(0, 0, 0, None),
    )(
        # -jnp.log(H_ik),
        safe_log(N_i),
        -log_U_ik,
        +F_i,
        -vmap_decorator(log_sum_exp_safe, in_axes=(None, None, 1))(
            safe_log(N_i),
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
        safe_log(H_ik),
        -safe_log(N_i),
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


def solve_wham(log_U_ik_nl, H_ik_nl, verbose=False, compute_std=False):
    # print(f"{tau_i_nl=}")

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
            H_k * jax.vmap(log_sum_exp_safe, in_axes=(None, None, 1))(safe_log(N_i), F_i, -log_U_ik_nl)
        )

        # F_i+C is also a solution for any C, so we add a small regularization to fix the scale
        # reg = 1e-10 * jnp.sum(F_i**2) / F_i.shape[0]
        reg = 1e-3 * (jnp.sum(F_i)) ** 2

        return -loss + reg

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


def solve_dham(
    T_il: jax.Array,  # (n_sims, max_nn_trans) N_ij^alpha
    unique_pairs_ikt: jax.Array,  # (n_sims, max_nn_trans, 2) [from, to] indices, -1 for padding
    log_U_ik: jax.Array,  # (n_sims, n_states) Bias potentials beta * u_i^alpha
    max_iter: int = 5000,
    tol: float = 1e-8,
):
    """
    DHAM solver for trajectory-specific counts with padding.
    Derived from the Likelihood in Equation 6 of image_3a3392.png.
    """

    print(f"{T_il=} {unique_pairs_ikt.shape}")

    n_sims, max_nn_trans, _ = unique_pairs_ikt.shape
    n_states = log_U_ik.shape[1]

    # print(f"DHAM: n_sims={n_sims}, max_nn_trans={max_nn_trans}, n_states={n_states}")
    # --- 1. Precompute Constants & Masks ---
    # mask_trans = unique_pairs_ikt[:, :, 0] != -1
    # from_idx = jnp.where(mask_trans, unique_pairs_ikt[:, :, 0], 0)
    # to_idx = jnp.where(mask_trans, unique_pairs_ikt[:, :, 1], 0)

    from_idx = unique_pairs_ikt[:, :, 0]
    to_idx = unique_pairs_ikt[:, :, 1]

    # N_ij^alpha (Observed counts)
    N_ij_alpha = T_il

    # N_ji^alpha (Reverse counts for symmetry)
    # Note: In a pair-based setup, we treat the pair (i,j) and (j,i) together.
    # If your data only provides one direction, you'd need to ensure C_ij includes both.
    # For Equation 7: C_ij^alpha = N_ij^alpha + N_ji^alpha
    # We'll assume the input unique_pairs contains the directed transitions.

    # Residence times t_i^alpha (sum of outgoing counts)
    sim_range = jnp.arange(n_sims)[:, jnp.newaxis]

    t_alpha_i = jnp.zeros((n_sims, n_states)).at[sim_range, from_idx].add(N_ij_alpha)

    print(f"{unique_pairs_ikt=}")

    mask_l = jnp.logical_not(jnp.logical_or(from_idx == -1, to_idx == -1))
    mask_l = jnp.logical_and(mask_l, from_idx != to_idx)  # Exclude self-transitions if they are not valid
    mask_k = jnp.full((n_states,), True)

    print(f"Initial mask_l sum: {jnp.sum(mask_l)} {jnp.sum(~mask_l)}")

    while True:
        N_ij_alpha_valid = jnp.where(mask_l, N_ij_alpha, 0.0)

        # Total incoming counts to state k: Sum_{alpha, i} N_ik^alpha
        total_incoming_k = jnp.zeros(n_states).at[to_idx].add(N_ij_alpha_valid)
        total_outgoing_k = jnp.zeros(n_states).at[from_idx].add(N_ij_alpha_valid)

        if (total_incoming_k == 0).any() or (total_outgoing_k == 0).any():
            print(f"WARNING: Some states have zero incoming or outgoing counts. Iteratively refining masks.")
        else:
            break

        mask_k = jnp.logical_and(total_incoming_k > 0, total_outgoing_k > 0)
        mask_l = mask_l & mask_k[from_idx] & mask_k[to_idx]

    print(f"{total_incoming_k=} {total_outgoing_k=}")

    # Precompute bias factors: B_i^alpha = t_i^alpha * exp(beta * u_i^alpha)
    log_B_alpha_ik = jnp.log(t_alpha_i + 1e-12) + log_U_ik
    log_B_alpha_ik = jnp.where(mask_k, log_B_alpha_ik, -1e10)
    log_B_alpha_ik = jnp.where(jnp.isfinite(log_B_alpha_ik), log_B_alpha_ik, -1e10)

    # # --- 2. Fixed-Point Iteration (Pair-Based) ---

    def body_fun(p):
        # 1. Map current p to transitions (only for valid states)
        # Using 1e-12 to prevent division by zero in x_ij calculation
        p_i = p[from_idx]
        p_j = p[to_idx]

        # 2. Bias factors B = t_i * exp(u_i)
        # We use exp(log_B) here; if log_B is very large, consider log-space logic
        B_i = jnp.exp(log_B_alpha_ik[sim_range, from_idx])
        B_j = jnp.exp(log_B_alpha_ik[sim_range, to_idx])

        # 3. Equation 7: Calculate local flux x_ij^alpha
        # denom = B_i/p_i + B_j/p_j
        denom_x = (B_i / (p_i + 1e-12)) + (B_j / (p_j + 1e-12))

        # Only calculate x for valid transitions
        x_ij_alpha = jnp.where(mask_l, N_ij_alpha / (denom_x + 1e-12), 0.0)

        # 4. Equation 8: p_k update (Numerator)
        # B_k for the target state
        B_k = jnp.exp(log_B_alpha_ik[sim_range, to_idx])
        term = x_ij_alpha * B_k

        # Sum up contributions to each state k
        numerator_k = jnp.zeros(n_states).at[to_idx].add(term)

        # 5. Finalize p_new
        # If state is masked, set to 0. Otherwise, normalize by total incoming flux.
        p_new = jnp.where(mask_k, numerator_k / (total_incoming_k + 1e-12), 0.0)

        # Re-normalize to 1.0 across the active set
        return p_new / (jnp.sum(p_new) + 1e-12)

    # --- 3. Solver Execution ---
    init_p = jnp.ones(n_states) / n_states

    def cond_fun(state):
        p_prev, p_curr, i = state
        return (i < max_iter) & (jnp.max(jnp.abs(p_curr - p_prev)) > tol)

    def step_fun(state):
        _, p_curr, i = state
        return p_curr, body_fun(p_curr), i + 1

    _, final_p, _ = jax.lax.while_loop(cond_fun, step_fun, (init_p, body_fun(init_p), 0))

    print(f"{final_p=}")

    # --- 4. Final WHAM-Compatible Outputs ---
    F_k = -jnp.log(final_p + 1e-12)

    # # [NEW] Acceleration Factors for each point (state) k:
    # # Defined as the average bias weight <exp(beta * u_k)> over simulations
    # log_acc_num = jax.vmap(log_sum_exp_safe, in_axes=(None, 0))(jnp.log(t_alpha_i + 1e-12), log_U_ik)
    # log_acc_den = jnp.log(jnp.sum(t_alpha_i, axis=0) + 1e-12)
    # log_acceleration_k = log_acc_num - log_acc_den

    return log_w_ik_nl, log_ps_ik_nl, F_i_nl, log_dens_ik_nl, mask_ik_nl, F_k, sigma_Fk
