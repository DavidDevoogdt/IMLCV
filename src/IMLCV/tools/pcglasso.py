from functools import partial

import jax
import jax.numpy as jnp

from IMLCV.base.datastructures import jit_decorator


def f(d, A):
    # Example objective: f(d) = sum(log(d)) + d^T A d
    # Replace with your actual objective function
    return 0.5 * d @ A @ d - jnp.sum(jnp.log(d))


@partial(jit_decorator, static_argnames=("f", "max_ls_iter"))
def line_search(d, Delta, f, A, eta_min, max_ls_iter=20):
    # Backtracking line search with Wolfe condition
    eta = 1.0
    c1 = 1e-4

    f0 = f(d, A)
    grad_phi0 = -jnp.dot(Delta, jax.grad(f, 0)(d, A))

    def body(eta_b):
        eta, _, it = eta_b
        d_new = d - eta * Delta

        def d_pos(d_new):
            f1 = f(d_new, A)

            b = f1 <= f0 + c1 * eta * grad_phi0

            return b

        def d_neg(d_new):
            return False

        b = jax.lax.cond(jnp.all(d_new > 0), d_pos, d_neg, operand=d_new)

        eta = jnp.where(b, eta, eta * 0.5)

        return eta, b, it + 1

    def cond(eta_b):
        eta, b, it = eta_b
        return jnp.logical_and(~jnp.logical_and(jnp.logical_not(b), eta > eta_min), it < max_ls_iter)

    eta, b, it = jax.lax.while_loop(cond, body, (eta, False, 0))

    return jnp.where(eta < eta_min, eta_min, eta)


@partial(jit_decorator, static_argnames=("k", "verbose"))
def diag_newton(A, d=None, k=100, eta_min=1e-8, tol=1e-6, verbose=0):
    p = A.shape[0]
    a = jnp.diag(A)
    if d is None:
        d = jnp.ones(p)
    f_old = jnp.inf

    def body(carry):
        d, f_old, _, it = carry
        g = A @ d - 1.0 / d
        h = a + 1.0 / (d**2)
        Delta = g / h

        # phi(eta) = f(d - eta * Delta)
        eta = line_search(d, Delta, f, A, eta_min)
        d = d - eta * Delta
        d = jnp.maximum(d, 1e-12)
        f_new = f(d, A)
        f_delta = f_old - f_new

        return d, f_new, f_delta, it + 1

    def cond(carry):
        _, _, f_delta, it = carry
        if verbose >= 2:
            jax.debug.print("iter {it}: f_delta={f_delta:.2E}", it=it, f_delta=f_delta)
        return jnp.logical_and(f_delta > tol, it < k)

    d, f_old, f_delta, _it = jax.lax.while_loop(cond, body, (d, f_old, jnp.inf, 0))

    if verbose >= 1:
        jax.debug.print("diag_newton converged in {it} iterations with f_delta={f_delta:.2E}", it=_it, f_delta=f_delta)

    return d


def soft_threshold(x, lmbda):
    return jnp.sign(x) * jnp.maximum(jnp.abs(x) - lmbda, 0.0)


@partial(jit_decorator, static_argnames=("max_iter", "lasso_iter_max", "verbose", "par", "fixed_diag"))
def glasso_cd(
    S,
    lmbda,
    R=None,
    W=None,
    tau=1e-8,
    delta_max_eps=1e-6,
    lasso_iter_max=100,
    max_iter=100,
    verbose=0,
    par=False,
    fixed_diag=False,
):
    """
    Coordinate Descent for graphical lasso (Algorithm 2).
    Args:
        S: (p, p) sample covariance matrix (symmetric, positive semidefinite)
        lmbda: regularization parameter (float)
        tau: convergence threshold (float)
        max_iter: maximum number of outer iterations
        verbose: print progress
    Returns:
        R: (p, p) precision matrix estimate
        W: (p, p) inverse of R
    """
    p = S.shape[0]

    if R is None:
        R = jnp.zeros((p, p))
    else:
        R = -R
        R = R.at[jnp.diag_indices(p)].set(0.0)

    if W is None:
        W = S

    dj = jnp.diag(S) + lmbda

    @jax.jit
    def lasso_update(S, Rj, W, j, lmbda, tau):
        v = W @ Rj

        def lasso_body(carry):
            Rj, v, _, it = carry

            delta_max = 0.0

            def update_i(i, val):
                Rj, v, delta_max = val

                if fixed_diag:

                    def update_R_i(i, Rj, v, delta_max):
                        c = soft_threshold(S[i, j] - v[i] + W[i, i] * Rj[i], lmbda) / W[i, i]
                        delta = c - Rj[i]
                        Rj = Rj.at[i].set(c)
                        v = v + delta * W[:, i]
                        delta_max = jnp.maximum(delta_max, jnp.abs(delta))

                        return Rj, v, delta_max

                    return jax.lax.cond(
                        i == j,
                        lambda _: (Rj, v, delta_max),
                        lambda _: update_R_i(i, Rj, v, delta_max),
                        operand=None,
                    )
                else:

                    def update_R_i(i, Rj, v, delta_max):
                        c = soft_threshold(S[i, j] - v[i] + dj[i] * Rj[i], lmbda) / dj[i]
                        delta = c - Rj[i]

                        Rj = Rj.at[i].set(c)
                        v = v + delta * W[:, i]
                        delta_max = jnp.maximum(delta_max, jnp.abs(delta))

                        return Rj, v, delta_max

                    return update_R_i(i, Rj, v, delta_max)

            Rj, v, delta_max = jax.lax.fori_loop(0, p, update_i, (Rj, v, delta_max))
            return (Rj, v, delta_max, it + 1)

        def lasso_cond(carry):
            _, _, delta_max, it = carry
            if verbose >= 3:
                jax.debug.print("  Lasso iter {it}: delta_max={delta_max:.2E}", it=it, delta_max=delta_max)
            return jnp.logical_and(delta_max > delta_max_eps, it < lasso_iter_max)

        (Rj, v, delta_max, it) = jax.lax.while_loop(lasso_cond, lasso_body, (Rj, v, jnp.inf, 0))
        if verbose >= 2:
            jax.debug.print("  Lasso converged in {it} iterations with {delta_max:.2E}", it=it, delta_max=delta_max)

        Wj = v

        if fixed_diag:
            Wj = Wj.at[j].set(1.0 + v.T @ Rj)
        else:
            Wj = Wj.at[j].set(dj[j])

        return Rj, Wj

    @jax.jit
    def glasso_body(carry):
        R, W, _, it = carry

        if not par:
            Delta_max = 0.0

            def update_j(j, val):
                R, W, Delta_max = val

                Rj, Wj = lasso_update(S, R[:, j], W, j, lmbda, tau)

                Wj_old = W[:, j]

                W = W.at[:, j].set(Wj)
                W = W.at[j, :].set(Wj)
                R = R.at[:, j].set(Rj)

                Delta_max = jnp.maximum(Delta_max, jnp.linalg.norm(Wj_old - Wj, ord=1))

                return R, W, Delta_max

            R, W, Delta_max = jax.lax.fori_loop(0, p, update_j, (R, W, Delta_max))
        else:

            @partial(jax.vmap, in_axes=(0, None, None), out_axes=(1))
            def update_j_parallel(j, R, W):
                Rj, Wj = lasso_update(S, R[:, j], W, j, lmbda, tau)
                return Rj, Wj

            R_new, W_new = update_j_parallel(jnp.arange(p), R, W)

            W_new = (W_new + W_new.T) / 2.0
            R_new = (R_new + R_new.T) / 2.0

            Delta_max = jnp.max(jnp.linalg.norm(W_new - W, ord=1, axis=0))

            W = W_new
            R = R_new

        return (R, W, Delta_max, it + 1)

    @jax.jit
    def glasso_cond(carry):
        _, _, Delta_max, it = carry

        if verbose >= 1:
            jax.debug.print("glassp Iter {it}: Delta_max={Delta_max:.2E}", it=it, Delta_max=Delta_max)

        return jnp.logical_and(Delta_max > tau, it < max_iter)

    R, W, Delta_max, _it = jax.lax.while_loop(glasso_cond, glasso_body, (R, W, jnp.inf, 0))

    if verbose >= 1:
        jax.debug.print("glasso_cd converged in {it} iterations with {Delta_max:.2E}", it=_it, Delta_max=Delta_max)

    R = -R

    if fixed_diag:
        for i in range(p):
            R = R.at[i, i].set(1.0)
    else:
        for i in range(p):
            a = jnp.sum(W[:, i] * R[:, i])
            R = R.at[:, i].set(R[:, i] / a)

    R = (R + R.T) / 2.0
    return R, W


@partial(jit_decorator, static_argnames=("max_iter", "max_iter_glasso", "verbose", "par"))
def pcglasso(
    S,
    lmbda=1e-1,
    tol_newton=1e-5,
    tol_glasso=1e-5,
    tol_tot=1e-5,
    max_iter=100,
    max_iter_glasso=100,
    alpha=0.0,
    verbose=1,
    par=True,
):
    diag_S = jnp.diag(S)

    p = S.shape[0]

    C = jnp.einsum("i,ij,j->ij", 1 / diag_S ** (1 / 2), S, 1 / diag_S ** (1 / 2))
    R = jnp.eye(p)
    d_opt = jnp.ones(p)
    W = R

    def body(carry):
        d_opt, R, W, it, _ = carry
        d_opt_new = diag_newton(
            (R * C) / (1 - alpha),
            d_opt,
            k=100,
            tol=tol_newton,
            verbose=verbose - 1,
        )

        n1 = jnp.linalg.norm(d_opt - d_opt_new, ord=1)
        d_opt = d_opt_new

        # print(f"{it}: optimizing R,W")
        R_new, W_new = glasso_cd(
            S=jnp.diag(d_opt) @ C @ jnp.diag(d_opt),
            R=R,
            W=W,
            lmbda=lmbda,
            tau=tol_glasso,
            max_iter=max_iter_glasso,
            verbose=verbose - 1,
            par=par,
        )

        n2 = jnp.linalg.norm(R - R_new, ord=1)
        R = R_new

        n3 = jnp.linalg.norm(W - W_new, ord=1) if W is not None else jnp.inf
        W = W_new

        # print(f"{it}: {n1=}, {n2=}, {n3=}")

        return d_opt, R, W, it + 1, (n1, n2, n3)

    def cond(carry):
        _, _, _, it, (n1, n2, n3) = carry

        if verbose >= 1:
            jax.debug.print(
                "outer iter {it}: {n1:.2E}, {n2:.2E}, {n3:.2E} {tol_tot} ", it=it, n1=n1, n2=n2, n3=n3, tol_tot=tol_tot
            )

        return jnp.logical_and(
            it < max_iter, ~jnp.logical_and(jnp.logical_and(n1 < tol_tot, n2 < tol_tot), n3 < tol_tot)
        )

    d_opt, R, W, it, _ = jax.lax.while_loop(cond, body, (d_opt, R, W, 0, (jnp.inf, jnp.inf, jnp.inf)))

    S_inv = jnp.einsum("ij,i,j->ij", R, diag_S ** (-1 / 2) * d_opt, diag_S ** (-1 / 2) * d_opt)
    S_reg = jnp.einsum("ij,i,j->ij", W, diag_S ** (1 / 2) / d_opt, diag_S ** (1 / 2) / d_opt)

    jax.debug.print("pcglasso converged in {it} iterations ", it=it)
    return S_inv, S_reg
