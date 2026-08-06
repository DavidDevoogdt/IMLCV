from __future__ import annotations

import time
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from IMLCV.base.covariances import Covariances, BendParams
from IMLCV.base.CV import CvTrans
from IMLCV.base.dataobjects import CV, NeighbourList, SystemParams, chunk_selector
from IMLCV.base.decoratros import MyPyTreeNode, Partial_decorator, jit_decorator, vmap_decorator, field
from IMLCV.base.UnitsConstants import boltzmann, kjmol, nanosecond, amu, angstrom, femtosecond, picosecond

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from jax.scipy.linalg import solve_triangular


class KoopmanModel(MyPyTreeNode):
    s: jax.Array

    cov: Covariances
    W0: jax.Array
    W1: jax.Array

    argmask: jax.Array | None
    argmask_s: jax.Array | None

    shape: int

    cv_0: list[CV] | list[SystemParams]
    cv_t: list[CV] | list[SystemParams]
    nl: NeighbourList | None
    nl_t: NeighbourList | None

    w: list[jax.Array] | None = None
    rho: list[jax.Array] | None = None

    w_t: list[jax.Array] | None = None
    rho_t: list[jax.Array] | None = None

    periodicities: jax.Array | bool | None = None

    use_w: bool = True

    dynamic_weights: list[jax.Array] | None = None

    shrink: bool = False
    shrinkage_method: str = "bidiag"

    eps: float = 1e-10
    eps_pre: float | None = None

    only_diag: bool = False
    calc_pi: bool = True

    scaled_tau: bool = False

    add_1: bool = True
    max_features: int = 5000
    max_features_pre: int = 5000
    out_dim: int | None = None
    # method: str = "tcca"
    correlation_whiten: bool = True
    verbose: bool = True

    tau: float | None = None
    # T_scale: float = 1.0
    eps_shrink: float = 1e-3

    generator: bool = False

    trans: CvTrans | CvTrans | None = None

    constant_threshold: float = 1e-14
    entropy_reg: float = 0.0
    target_smoothness: float = 20 * kjmol / (boltzmann * 300)
    alpha_smooth: float = 1.0
    beta_timecon: float | None = None
    tau_thermo: float | None = None
    halve_even_odd: bool = False

    pair_info: tuple[jax.Array, jax.Array] | None = None

    @staticmethod
    def create(
        w: list[jax.Array] | None,
        rho: list[jax.Array] | None,
        w_t: list[jax.Array] | None,
        rho_t: list[jax.Array] | None,
        cv_0: list[CV] | list[SystemParams],
        cv_t: list[CV] | list[SystemParams] | None,
        dynamic_weights: list[jax.Array] | None = None,
        nl: NeighbourList | None = None,
        nl_t: NeighbourList | None = None,
        add_1=True,
        eps: float = 1e-14,
        eps_pre: float = 0,
        # method="tcca",
        symmetric=False,
        out_dim=-1,  # maximum dimension for koopman model
        max_features=5000,
        max_features_pre=5000,
        tau=None,
        tau_thermo: float = None,
        macro_chunk=128,
        chunk_size=None,
        verbose=True,
        trans: CvTrans | None = None,
        # T_scale: float = 1.0,
        only_diag=False,
        calc_pi=False,
        use_scipy=False,
        auto_cov_threshold=None,
        sparse=True,
        # n_modes=10,
        scaled_tau=False,
        only_return_weights=False,
        correlation_whiten=False,
        out_eps=None,
        constant_threshold: float = 1e-14,
        shrink=True,
        shrinkage_method="bidiag",
        eps_shrink=1e-3,
        glasso=True,
        generator: bool = False,
        use_w=True,
        # periodicities: jax.Array | None = None,
        exp_period=True,
        vamp_r=2,
        iters_nonlin=int(1e4),
        print_nonlin_every=10,
        epochs=2000,
        batch_size=1024,
        batch_chunk_size=16,
        init_learnable_params=False,
        min_std=1e-2,  # prevents collapse into singularities
        entropy_reg=0.0,
        target_smoothness=20 * kjmol / (boltzmann * 300),
        alpha_smooth=1.0,
        T_mult=1.0,
        beta_timecon: float | None = None,
        use_girsanov=False,
        curvature_radius=1.0,
        tau_baro=500 * picosecond,
        hessian=True,
        hessian_steps=0.1,
        halve_even_odd=True,
        periodicities=None,
        key=jax.random.PRNGKey(42),
    ):
        #  see Optimal Data-Driven Estimation of Generalized Markov State Models
        if verbose:
            print(f"getting covariancesm {generator=}")

        epochs = int(epochs)

        print(f"{out_dim=} {eps=} {eps_pre=}")
        print(f"{calc_pi=} {add_1=}   ")
        print(f"{use_w=}")

        time_series = cv_t is not None

        if time_series:
            assert len(w) == len(w_t)
            assert len(rho) == len(rho_t)

        if nl is not None:
            print(f"using neighbour lists in koopman model {nl=}")

        print(f"new tot w")

        print(f"using  {alpha_smooth=}")

        def tot_w(w, rho):
            # wi = exp(-beta F)
            # rhoi = exp( beta F + beta U_bias ) * n_samples/n_selected

            w_log = [jnp.log(wi) * alpha_smooth + jnp.log(rhoi) for wi, rhoi in zip(w, rho)]
            z = jnp.hstack(w_log)

            z_max = jnp.nanmax(z)
            norm = jnp.log(jnp.nansum(jnp.exp(z - z_max))) + z_max

            w_tot = [jnp.exp(w_log_i - norm) for w_log_i in w_log]

            s = 0
            for wi in w_tot:
                s += jnp.nansum(wi)

            print(f"{s=}")

            return w_tot

        w_tot = tot_w(w, rho)

        if use_girsanov:
            w_tot_dyn = tot_w(w_tot, dynamic_weights) if dynamic_weights is not None else None
        else:
            w_tot_dyn = None

        if time_series:
            assert cv_t is not None
            assert w_t is not None
            assert rho_t is not None

            w_tot_t = tot_w(w_t, rho_t)
        else:
            w_tot_t = None

        # if periodicities is not None:
        #     _tr_per = CvTrans.from_cv_function(PeriodicKoopmanModel._exp_periodic, periodicities=periodicities)

        #     if trans is not None:
        #         trans *= _tr_per
        #     else:
        #         trans = _tr_per

        if periodicities is not None:
            if isinstance(cv_0, CV):
                x0 = cv_0[0][0]

                if trans is not None:
                    x0, _ = trans.compute_cv(x0, nl=nl)

            else:
                assert trans is not None, "trans must be provided for SystemParams input"

                x0, _ = trans.compute_cv(cv_0[0][0], nl=nl)

            assert x0.combine_dims is not None

            idx_per = jnp.argwhere(periodicities)
            idx_non_per = jnp.argwhere(jnp.logical_not(periodicities))
            n_per = idx_per.shape[0]
            n_non_per = idx_non_per.shape[0]

            key, subkey = jax.random.split(key)
            w_per = jax.random.normal(subkey, (n_per, 2, 2))
            pi_per = jax.random.normal(subkey, (n_per, 2))
            w_non_per = jax.random.normal(subkey, (n_non_per, 2))

            _tr_per = CvTrans.from_cv_function(
                KoopmanModel._periodic_trans_pre,
                idx_per=idx_per,
                idx_nonper=idx_non_per,
                learnable_argnames=("w_per", "w_nonper", "mu_per"),
                w_per=w_per,
                w_nonper=w_non_per,
                mu_per=pi_per,
            )

            trans *= _tr_per

        variational = False
        if trans is not None:
            if trans.num_learnable_params > 0:
                variational = True

        # if variational and add_1:
        #     add_1 = False

        if add_1:
            print("adding 1 to  basis set")

            from IMLCV.implementations.CV import append_trans

            _add_1 = append_trans(v=jnp.array([1]))

            if trans is None:
                trans = _add_1
            else:
                trans *= _add_1

        # print(f" {calc_pi=}")

        # print(f"{dynamic_weights=}")

        if variational:
            # batch_size = 1e4
            # epochs = 50

            print(f"{macro_chunk=} {batch_chunk_size=}")

            assert trans is not None
            print(f"using variational approach {trans.num_learnable_params=} ")

            trans.learnable_params_shape

            if init_learnable_params:
                print("initializing learnable params")
                key, subkey = jax.random.split(key)
                learable_params = trans.init_learnable_params(subkey)
            # trans = trans.apply_learnable_params(learable_params)
            else:
                print("using default learnable params")
                learable_params = trans.get_learnable_params()

            import optax

            # assert not generator

            print(f"{tau=} {tau_thermo=}")

            tau = tau if tau is not None else 1.0
            tau_thermo = tau_thermo if tau_thermo is not None else 1.0

            init_steps = 100

            cv_out_shape, _ = jax.eval_shape(trans.compute_cv, cv_0[0], nl)
            dim = cv_out_shape.cv.shape[1]

            N = cv_out_shape.cv.shape[0]

            W_baro = (
                tau_baro
                / tau_thermo
                * (3 * N + 3)
                / (beta_timecon if beta_timecon is not None else 1 / (300 * boltzmann))
            )

            print(f"{W_baro=}   ")

            # scaling for cell params

            # sigma_silverman = (4 / (N * (dim + 2))) ** (1 / (dim + 4))

            # print(f"{N=} {dim=} {sigma_silverman=}")

            print(f"{cv_out_shape.cv.shape=} {curvature_radius=} ")

            # # if batch_size is None:
            # grid_size = int(2000 ** (1 / dim))
            # # else:
            # #     grid_size = int(min(batch_size, 1000) ** (1 / dim))
            # print(f"entropy grid size per dim: {grid_size}")

            print("Using AdamW optimizer for mini-batch optimization")

            # 1. Define a Schedule: Warmup + Cosine Decay
            # A CV needs to settle in before you hit it with high rates
            # total_steps = 1000  # Adjust based on your total epochs/batches
            # warmup_steps = max(epochs // 5, 100)  # Warmup for the first 20% of training or at least 100 steps

            lr_schedule = optax.warmup_cosine_decay_schedule(
                init_value=1e-3,
                peak_value=1e-2,
                warmup_steps=200 if epochs > 200 else epochs // 5,
                decay_steps=epochs,
                end_value=1e-3,
            )

            # 2. Add Gradient Clipping
            # This is the "shield" against those massive spikes in your plot
            optimizer = optax.chain(
                optax.adaptive_grad_clip(clipping=0.1),  # clipping relative to recent gradient norms
                optax.adamw(
                    lr_schedule,
                    weight_decay=1e-4,
                ),
            )

            optimizer = optax.apply_if_finite(
                optimizer,
                max_consecutive_errors=5,
            )

            opt_state = optimizer.init(learable_params)

            prefetch = False

            angle = halve_even_odd

            if batch_size is None:
                print("Using full-batch optimization")

                if isinstance(cv_0[0], CV):
                    x0_chunk = CV.stack(*cv_0)
                else:
                    x0_chunk = SystemParams.stack(*cv_0)

                if cv_t is not None:
                    if isinstance(cv_t[0], CV):
                        xt_chunk = CV.stack(*cv_t)
                    else:
                        xt_chunk = SystemParams.stack(*cv_t)

                w_chunk = jnp.hstack(w_tot) if use_w else None

                lbfgs = True

            else:

                def stack_incremental(group):
                    """list of CV/SystemParams/array -> one contiguous pytree.
                    Avoids the mega-concat: writes each chunk via dynamic_update_slice."""
                    sizes = [p.shape[0] for p in group]
                    total = sum(sizes)

                    # one empty buffer per leaf, same class (CV/SystemParams) preserved by tree.map
                    buf = jax.tree.map(lambda a: jnp.empty((total, *a.shape[1:]), a.dtype), group[0])

                    start = 0
                    for p, n in zip(group, sizes):
                        buf = jax.tree.map(
                            lambda b, src: jax.lax.dynamic_update_slice_in_dim(b, src, start, axis=0),
                            buf,
                            p,
                        )
                        start += n
                    return buf

                @partial(jax.jit, static_argnames=("target_size",))
                def gather(idx_source_weights, pools, target_size):
                    """One shared global index, applied to every stacked group."""
                    key, weights = idx_source_weights
                    total = jax.tree_util.tree_leaves(next(g for g in pools if g is not None))[0].shape[0]
                    idx = jax.random.choice(key, total, (target_size,), p=weights, replace=True)
                    return [None if g is None else jax.tree.map(lambda a: a[idx], g) for g in pools]

                w_subsample = w_tot_dyn if use_girsanov else w_tot
                w_subsample_stack = jnp.hstack(w_subsample) if w_subsample is not None else None

                groups = [cv_0, cv_t, w_tot, w_tot_t, dynamic_weights, w_subsample]
                pools = [None if g is None else stack_incremental(g) for g in groups]

                def _get_batch(key):
                    subsample = True
                    key, subkey = jax.random.split(key)

                    # if subsample:
                    # each epoch:
                    key, subkey = jax.random.split(key)
                    # w = jnp.hstack(w_tot_dyn) if use_girsanov else (jnp.hstack(w_tot) if use_w else None)
                    x0_chunk, xt_chunk, w_chunk, wt_chunk, wdyn_chunk, wtot_chunk = gather(
                        (subkey, w_subsample_stack), pools, batch_size
                    )

                    w_chunk /= wtot_chunk
                    if time_series:
                        wt_chunk /= wtot_chunk

                    batch = dict(
                        x0_i=x0_chunk,
                        xt_i=xt_chunk,
                        w_i=w_chunk,
                        wt_i=wt_chunk,
                        wdyn_i=wdyn_chunk if dynamic_weights is not None else None,
                        # nl_i=nl_chunk,
                    )

                    return batch, key

                def get_batch(key):
                    batch, key = _get_batch(key)
                    return key, (batch["x0_i"], batch["xt_i"], batch["w_i"], batch["wt_i"], batch["wdyn_i"])

                # --------------------------------------------------------------------------- #
                # Hyperparameters (all python scalars -> hashable -> safe as jit static args)  #
                # --------------------------------------------------------------------------- #
                class LossHP(MyPyTreeNode):
                    eta: float = 0.1  # soft mode cutoff: modes >1/eta faster than
                    #   the slowest retained one are ~ignored
                    R0: float = 3.0  # curvature length scale (mass-weighted, e.g.
                    #   10*sqrt(kB T / m w^2)); the ONLY balance knob
                    rate_floor: float = 1e-6  # hard floor on rates (caps timescales, clamps
                    #   negative numerical eigenvalues)
                    rel_jitter: float = 1e-6  # Cholesky jitter as a fraction of trace(C00)/N
                    ortho_weight: float = 1.0  # fixed multiplier on the ||C00 - I||^2 constraint
                    n_skip: int = field(pytree_node=False, default=0)  # drop this many lowest (stationary) modes before
                    #   the loss. Set to 1 if a constant/stationary
                    #   mode (rate ~ 0) is present in the spectrum,
                    #   else it captures all the slow-mode weight.
                    #   G^{-1} metric (multiply curvature by t_k).
                    #   Set False if C_bend is already metricised.
                    curvature_mode: str = field(
                        pytree_node=False, default="trace"
                    )  # "trace" (degeneracy-safe, default) | "per_mode"
                    reward_mode: str = field(
                        pytree_node=False, default="saturated"
                    )  # "saturated" (couples modes via one denominator)
                    #   | "additive" (linear penalty, needs additive_mu)
                    additive_mu: float = 10.0  # penalty weight when reward_mode == "additive"
                    hinge_beta: float = 2.0  # sharpness of hinge when reward_mode == "hinge"

                    grad_floor: float = (
                        1e-3  # floors the ||grad phi||^2 in the curvature penalty to prevent singularities;
                    )

                    w_ortho: float = 1e-1
                    w_periodic: float = 1
                    w_grad: float = 1e-2
                    w_improper: float = 1.0
                    w_conformal: float = 1.0
                    w_origin: float = 1.0  # new term to penalise mean of periodic components
                    w_radial: float = 1.0  # new term to penalise radial deviation of periodic components

                # --------------------------------------------------------------------------- #
                # Linear-algebra kernels (verified against explicit inverse / per-mode sum)    #
                # --------------------------------------------------------------------------- #
                def _sym(A):
                    # jnp.conj(A.T) or A.conj().T enforces Hermitian symmetry A = A^H
                    return 0.5 * (A + jnp.conj(A.T))

                def _cholesky_jittered(C00, rel_jitter):
                    C = _sym(C00)
                    N = C.shape[0]
                    jit = rel_jitter * jnp.trace(C) / N
                    return jnp.linalg.cholesky(C + jit * jnp.eye(N))

                def _whiten(L, A):
                    """Return L^{-1} A L^{-H} via two triangular solves (Hermitian symmetrised)."""
                    # 1. Y = L^{-1} A   (shape: N x N)
                    Y = solve_triangular(L, A, lower=True)

                    # 2. M = Y L^{-H} = (L^{-1} A) L^{-H}
                    # Taking conjugate transpose inside solve_triangular:
                    # We solve L M^H = Y^H  --> M^H = L^{-1} Y^H  --> M = (L^{-1} Y^H)^H
                    M = solve_triangular(L, Y.conj().T, lower=True).conj().T

                    # 3. Enforce numerical Hermitian symmetry M = M^H
                    return _sym(M)

                def _filter_matrix(U, phi_vals):
                    """phi(G_tilde) restricted to the columns of U:  U diag(phi_vals) U^T.

                    Smooth function of G_tilde, so the trace against it is invariant to
                    rotations inside any degenerate eigenspace.
                    """
                    return (U * phi_vals) @ U.T

                def pair_indices(n_non_per: int, n_per: int) -> Array:
                    """Start index of every staggered (re, im) pair."""
                    return n_non_per + 2 * jnp.arange(n_per)

                def periodic_argmask(n_non_per: int, n_per: int) -> Array:
                    """Flat indices of all periodic components, for pi_argmask (FIX(11))."""
                    idx = pair_indices(n_non_per, n_per)
                    return jnp.concatenate([idx, idx + 1])

                def offdiag_mask(n: int, n_non_per: int, n_per: int) -> Array:
                    """
                    1 off the diagonal, 0 on it AND 0 on the within-pair (re_c, im_c) entries.

                    The within-pair entry of E[G^2] is ~0.5 for a perfect circle (verified), so
                    it must not enter the gradient-overlap penalty.  Cross-pair entries are 0
                    for distinct circles and ~1 for harmonics -- that is the signal we want.
                    """
                    m = 1.0 - jnp.eye(n)
                    if n_per > 0:
                        i = pair_indices(n_non_per, n_per)
                        m = m.at[i, i + 1].set(0.0).at[i + 1, i].set(0.0)
                    return m

                def pair_blocks(A: Array, n_non_per: int, n_per: int):
                    """Extract (A_xx, A_xy, A_yx, A_yy) for every periodic pair from a matrix."""
                    i = pair_indices(n_non_per, n_per)
                    return A[i, i], A[i, i + 1], A[i + 1, i], A[i + 1, i + 1]

                def ring_terms(C00: Array, C00_sq: Array, n_non_per: int, n_per: int, r0_sq: float = 2.0):
                    """
                    Radial + improperness losses from the SECOND-MOMENT matrices alone.

                    r0_sq = 2 because KL(C00 || I) drives each real component to unit variance,
                    so a proper circular pair has E[x^2] = E[y^2] = 1 and r^2 = 2.  Targeting
                    r^2 = 1 would fight loss_ortho: different fixed points, training stalls at a
                    compromise radius.

                    Requires C00 to be UNCENTRED on the periodic indices -- see FIX(11).
                    """
                    A, B, C, D = (jnp.real(t) for t in pair_blocks(C00, n_non_per, n_per))
                    A2, B2, C2, D2 = (jnp.real(t) for t in pair_blocks(C00_sq, n_non_per, n_per))

                    E_r2 = A + D  # E[|z|^2]
                    E_x2y2 = 0.5 * (B2 + C2)  # E[x^2 y^2]
                    E_r4 = A2 + D2 + 2.0 * E_x2y2  # E[|z|^4]

                    # E[(r^2 - r0^2)^2] = Var(r^2) + (E[r^2] - r0^2)^2
                    radial = (E_r4 - 2.0 * r0_sq * E_r2 + r0_sq**2) / r0_sq**2

                    # pseudo-covariance E[z^2] = (E[x^2] - E[y^2]) + 2i E[xy]  -> 0
                    E_xy = 0.5 * (B + C)
                    improper = ((A - D) ** 2 + 4.0 * E_xy**2) / r0_sq**2

                    return jnp.mean(radial), jnp.mean(improper)

                def conformality(C_gen: Array, n_non_per: int, n_per: int, eps: float = 1e-12):
                    """
                    |P^G_cc|^2 / (tr G_cc)^2  with  P^G_cc = (G_xx - G_yy) + 2i G_xy.

                    Zero iff the 2x2 Dirichlet block is isotropic, i.e. the channel traverses
                    its circle at constant speed.  Scale-free, so it cannot be cheated by
                    shrinking the gradients.
                    """
                    Gxx, Gxy, Gyx, Gyy = (jnp.real(t) for t in pair_blocks(C_gen, n_non_per, n_per))
                    Gxy = 0.5 * (Gxy + Gyx)
                    num = (Gxx - Gyy) ** 2 + 4.0 * Gxy**2
                    den = (Gxx + Gyy) ** 2 + eps
                    return jnp.mean(num / den)

                def gradient_overlap(rho_gen_sq: Array, mask: Array, eps: float = 1e-12):
                    """
                    Harmonic / redundancy detector.  Cauchy-Schwarz normalised so it lives in
                    [0, 1]: E[G_ij^2] <= sqrt(E[G_ii^2] E[G_jj^2]).

                    Measured: 0.000 for two distinct circles, 1.010 for k=1 and k=2 of the same
                    circle.  Nothing else in the objective separates these.
                    """
                    R = jnp.real(rho_gen_sq)
                    d2 = jnp.diag(R)
                    norm = jnp.sqrt(jnp.outer(d2, d2)) + eps
                    return jnp.sum((R / norm) * mask) / jnp.sum(mask)

                # --------------------------------------------------------------------------- #
                # sector escape                                                                #
                # --------------------------------------------------------------------------- #
                def mean_term(pi_0: Array, n_non_per: int, n_per: int, r0_sq: float = 2.0):
                    """
                    |E[z_c]|^2 / r0^2  -- the term whose null space does NOT contain an arc.

                    For a uniform arc of half-width a, E[exp(i k theta)] = sin(ka)/(ka).  The
                    k=2 (improperness) term vanishes at a = pi/2, so a HALF RING is an exact
                    stationary point of it: dL_imp/da = 0.0000 there.  The k=1 term reads
                    0.392 with slope -0.516, which is the restoring force that escapes the
                    sector.  Equivalently, at fixed E|z|^2,
                        E|z_a - z_b|^2 = 2 E|z|^2 - 2 |E[z]|^2
                    so minimising |E[z]|^2 IS maximising pairwise angular spread.

                    Requires pi to be TRACKED on the periodic indices (see get_pi patch);
                    pi_argmask as originally written discards it.
                    """
                    i = pair_indices(n_non_per, n_per)
                    mu = jnp.real(pi_0)
                    return jnp.mean((mu[i] ** 2 + mu[i + 1] ** 2)) / r0_sq

                def circular_moments(z: Array, k_max: int = 4, weights: Array | None = None):
                    """
                    sum_k |E[u^k]|^2 / k^2  with u = z/|z|, over k = 1..k_max.

                    Strictly decreasing in arc coverage on (0, pi] with a unique zero at full
                    coverage -- no spurious stationary points anywhere.  Needs the phases, so
                    it cannot be reconstructed from S and S^(4) alone (those reach k=2 only);
                    add it as its own accumulator if the k=1 term proves insufficient.
                    """
                    u = z / (jnp.abs(z) + 1e-8)
                    w = jnp.ones(z.shape[0]) / z.shape[0] if weights is None else weights
                    out = 0.0
                    uk = jnp.ones_like(u)
                    for k in range(1, k_max + 1):
                        uk = uk * u
                        out = out + jnp.abs(jnp.einsum("nc,n->c", uk, w)) ** 2 / k**2
                    return jnp.mean(out)

                def winding_number(z: Array):
                    """
                    Per-channel winding along the sample ordering (needs a trajectory).
                    |W| = 1 is the target; W = 0 is the trapped sector.  Not differentiable --
                    use as a diagnostic / restart trigger, not a loss.
                    """
                    th = jnp.arctan2(z.imag, z.real)
                    d = jnp.diff(th, axis=0)
                    d = (d + jnp.pi) % (2 * jnp.pi) - jnp.pi
                    return jnp.sum(d, axis=0) / (2 * jnp.pi)

                # --------------------------------------------------------------------------- #
                # objective                                                                    #
                # --------------------------------------------------------------------------- #
                @partial(jit_decorator, static_argnames=["entropy_reg", "chunk_size", "hessian"])  # noqa: F821
                def slow_smooth_objective(
                    params,
                    eta,
                    sigma_0,
                    x_0,
                    x_t,
                    w,
                    w_t,
                    wdyn,
                    nl,
                    hp: LossHP,  # noqa: F821
                    entropy_reg=None,
                    chunk_size=None,
                    epoch: int = 0,
                    key: Array | None = None,
                    hessian: bool = True,
                ):
                    """Single scalar loss: -(slow-and-smooth reward) + structural constraints.

                    loss_main       slowness of the generator, via the resolvent -1/(1+lambda).
                    loss_ortho      KL(C00 || I): unit variance, decorrelated outputs.
                    loss_grad       masked, CS-normalised gradient overlap.  Separates distinct
                                    circles from harmonics of one circle.  FIX(8)
                    loss_radial     E[(r^2-2)^2] from C00 / C_00_sq.  FIX(9)
                    loss_improper   |E[z^2]|^2: uniform angular density.
                    loss_conformal  |P^G|^2: constant angular speed.  FIX(10)
                    """
                    _trans = trans.apply_learnable_params(params)  # noqa: F821

                    scale_deriv = eta ** (-1 / 2)

                    bp = BendParams(  # noqa: F821
                        R0=hp.R0 * jnp.sqrt(eta),
                        ridge=1e-5 / eta,
                        mode="projected",
                        reduce="hinge",
                        n_probe=16,
                        hinge_p=2,
                        W=W_baro,  # noqa: F821
                    )

                    # FIX(11): never subtract the mean on periodic components -- the ring must be
                    # measured about the same fixed origin that atan2 uses downstream, and the
                    # radial/properness algebra below assumes uncentred second moments there.
                    if periodicities is not None:  # noqa: F821
                        pi_argmask = periodic_argmask(n_non_per, n_per)  # noqa: F821
                    else:
                        pi_argmask = None

                    cov = Covariances.create(  # noqa: F821
                        cv_0=[x_0],
                        cv_1=[x_t] if time_series else None,  # noqa: F821
                        nl=nl,
                        nl_t=nl,
                        w=[w],
                        w_t=[w_t] if time_series else None,  # noqa: F821
                        trans_f=_trans,
                        trans_g=_trans if time_series else None,  # noqa: F821
                        shrink=False,
                        calc_pi=calc_pi,  # noqa: F821
                        get_diff=False,
                        symmetric=symmetric,  # noqa: F821
                        generator=True,
                        get_hess=hessian,
                        scale_deriv=scale_deriv,
                        key=key,
                        bp=bp,
                        macro_chunk=batch_chunk_size,  # noqa: F821
                        gen_squared=True,
                        cov_squared=True,
                        bessel_correct=False,
                        pi_argmask=pi_argmask,
                        reduction_fun=CvTrans.from_cv_function(KoopmanModel._periodic_trans_pushback),
                    )

                    N = cov.C00.shape[0]

                    # jax.debug.print(" reduction array {}", cov.reduction_array)

                    # ---- slow reward ------------------------------------------------------- #
                    lambdas = jnp.real(jnp.diag(cov.rho_gen))

                    if periodicities is not None:  # noqa: F821
                        lambdas_per = jnp.concatenate(
                            [lambdas[:n_non_per], 0.5 * (lambdas[n_non_per::2] + lambdas[n_non_per + 1 :: 2])]  # noqa: F821
                        )
                    else:
                        lambdas_per = lambdas

                    n_eff = lambdas_per.shape[0]
                    loss_main = jnp.sum(-1.0 / (1.0 + lambdas_per)) / n_eff

                    # ---- hermitian KL(C00 || I) -------------------------------------------- #
                    def loss_ortho_fn(x, jitter=1e-6):
                        x = 0.5 * (x + jnp.conj(x).T)
                        D = x.shape[0]
                        x = x + jitter * jnp.eye(D, dtype=x.dtype)
                        trace_val = jnp.real(jnp.trace(x))
                        _, logdet_val = jnp.linalg.slogdet(x)
                        return trace_val - jnp.real(logdet_val) - D

                    loss_ortho = loss_ortho_fn(cov.C00) / N

                    # ---- FIX(8): masked, CS-normalised gradient overlap -------------------- #
                    if periodicities is not None:  # noqa: F821
                        _mask = offdiag_mask(N, n_non_per, n_per)  # noqa: F821
                    else:
                        _mask = 1.0 - jnp.eye(N)
                    loss_grad = gradient_overlap(cov.rho_gen_sq, _mask)

                    # ---- FIX(9)/(10): periodic structure, no second covariance pass -------- #
                    loss_radial = jnp.asarray(0.0)
                    loss_improper = jnp.asarray(0.0)
                    loss_conformal = jnp.asarray(0.0)

                    if periodicities is not None:  # noqa: F821z
                        # _, loss_improper = ring_terms(
                        #     cov.C00,
                        #     cov.C_00_sq,
                        #     n_non_per,
                        #     n_per,
                        #     r0_sq=2.0,  # noqa: F821
                        # )
                        loss_conformal = conformality(cov.C_gen, n_non_per, n_per)  # noqa: F821

                        radial_cv, angular_cv = cov.reduction_cv.split()

                        loss_radial = jnp.sum(jnp.linalg.norm(cov.reduction_cv.cv))

                        # arr_ang = cov.reduction_cv[n_per : 2 * n_per]

                        loss_improper = jnp.sum(jnp.linalg.norm(angular_cv.cv, axis=0) ** 2) / n_per

                        jax.debug.print(
                            "loss_radial={:.3E} | loss_improper={:.3E} | loss_conformal={:.3E}",
                            loss_radial,
                            loss_improper,
                            loss_conformal,
                        )

                    loss_periodic = (
                        hp.w_radial * loss_radial + hp.w_improper * loss_improper + hp.w_conformal * loss_conformal
                    )

                    # ---- FIX(12): separate, per-mode-normalised constraint weights --------- #
                    loss = loss_main + hp.w_ortho * loss_ortho + hp.w_grad * loss_grad + hp.w_periodic * loss_periodic

                    # ---- telemetry --------------------------------------------------------- #
                    lam_sg = lambdas_per * eta
                    # FIX(13): was hp.eta * min(lam_sg), re-mixing the two etas that FIX(6)
                    # unified.  eta is the single source of truth in this function.
                    eta_next = jnp.min(lam_sg)

                    a = 0.05
                    ema_lambda_min = a * eta_next + (1 - a) * eta

                    timescales = 1.0 / (
                        lam_sg
                        / (beta_timecon if beta_timecon is not None else 1 / (300 * boltzmann))  # noqa: F821
                        * (tau_thermo if tau_thermo is not None else 100 * femtosecond)  # noqa: F821
                    )

                    aux = dict(
                        timescales=timescales,
                        resolvent_weights=1.0 / (1.0 + lambdas_per),
                        sat=0.0,
                        loss_main=loss_main,
                        loss_ortho=loss_ortho,
                        eta_next=ema_lambda_min,
                        sigma_next=None,
                        loss_ortho_gradient=loss_grad,
                        loss_center=loss_periodic,
                        loss_radial=loss_radial,
                        loss_improper=loss_improper,
                        loss_conformal=loss_conformal,
                        lambdas=lambdas_per,
                    )
                    return loss, aux

                # --------------------------------------------------------------------------- #
                # Optimiser step -- one value_and_grad, no per-component vjp / gradnorm.        #
                # --------------------------------------------------------------------------- #

                @partial(jit_decorator, static_argnames=["hessian"])
                def _body_fun(carry, hessian=True):
                    (opt_state, params, eta, sigma, (cv, cv_t, w, w_t, w_dyn, nl), _, iter_num) = carry

                    print(f"inside body {sigma=}")

                    w = w / jnp.sum(w)

                    if time_series:
                        w_t = w_t / jnp.sum(w_t)

                    N_eff = (jnp.sum(w) ** 2) / jnp.sum(w**2)

                    hp = LossHP(
                        eta=100.0,
                        R0=curvature_radius,
                        rate_floor=1e-6,
                        rel_jitter=1e-4,
                        ortho_weight=0.1,
                        n_skip=0,
                        curvature_mode="per_mode",
                        reward_mode="hinge",
                        additive_mu=2.0,
                        hinge_beta=10.0,
                        grad_floor=1e-6,
                    )

                    def loss_fn(p):
                        return slow_smooth_objective(
                            p,
                            eta,
                            sigma,
                            cv,
                            cv_t,
                            w,
                            w_t,
                            w_dyn,
                            nl,
                            hp=hp,
                            entropy_reg=entropy_reg,
                            chunk_size=chunk_size,
                            epoch=iter_num,
                            key=jax.random.fold_in(jax.random.PRNGKey(0), iter_num),
                            hessian=hessian,
                        )

                    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

                    aux["N_eff"] = N_eff

                    # Cheap insurance: a single bad Cholesky / eigh step won't poison state.
                    grads = jax.tree_util.tree_map(jnp.nan_to_num, grads)

                    gnorm = optax.global_norm(grads)

                    updates, opt_state = optimizer.update(grads, opt_state, params)

                    params = optax.apply_updates(params, updates)

                    output = dict(
                        iter_num=iter_num,
                        loss=loss,
                        eta=eta,
                        sigma=sigma,
                        gnorm=gnorm,
                        **aux,
                    )

                    # with jnp.printoptions(formatter={"float": "{:0.3e}".format}):
                    #     jax.debug.print(
                    #         "epoch={e} | timescales (ns)={ts} | sat={s:.3f} | "
                    #         "loss_main={lm:.3E} | ortho={o:.3E} | eta_next={eta_next:.3E} | gnorm={gnorm:.3E}",
                    #         e=iter_num,
                    #         ts=aux["timescales"][::-1] / nanosecond,
                    #         s=aux["sat"],
                    #         lm=aux["loss_main"],
                    #         o=aux["loss_ortho"],
                    #         eta_next=aux["eta_next"],
                    #         gnorm=gnorm,
                    #     )

                    b = iter_num < epochs
                    return (opt_state, params, output, aux["eta_next"], aux["sigma_next"], b, iter_num + 1, loss)

            cov: Covariances | None = None

            print(f"{init_steps=}, {epochs=}, {entropy_reg=}")

            t0 = time.time()

            print(f"{nl=}")

            eta = 100
            sigma = None

            for epoch in range(epochs):
                if (epoch % 10 == 0) and epoch > 0:
                    t = time.time()

                    end = t0 + (t - t0) / 10 * (epochs - epoch)

                    print(
                        f"Epoch {epoch} time {time.strftime('%H:%M:%S', time.localtime(t))} estimated end {time.strftime('%H:%M:%S', time.localtime(end))}"
                    )

                    t0 = t

                if batch_size is not None:
                    key, (x0_chunk, xt_chunk, w_chunk, wt_chunk, wdyn_chunk) = get_batch(key)

                (
                    opt_state,
                    learable_params,
                    output,
                    eta,
                    sigma,
                    # cov,
                    b,
                    iters,
                    log_beta_gamma_new,
                ) = _body_fun(
                    (
                        opt_state,
                        learable_params,
                        eta,
                        sigma,
                        (x0_chunk, xt_chunk, w_chunk, wt_chunk, wdyn_chunk, nl),
                        # cov,
                        True,
                        epoch,
                    ),
                    hessian=hessian,
                )

                if (epoch % print_nonlin_every) == 0:
                    # print(f"{output['loss_center']}")

                    with jnp.printoptions(formatter={"float": "{:0.3e}".format}):
                        print(
                            f"epoch={epoch} | timescales (ns)={output['timescales'][::-1] / nanosecond} | "
                            f"sat={output['sat']:.3f} | loss_main={output['loss_main']:.3E} | "
                            f"ortho={output['loss_ortho']:.3E} | eta_next={output['eta_next']:.3E} | "
                            f"ortho gradient={output['loss_ortho_gradient']:.3E} | "
                            f"center loss={output['loss_center']:.3E} | "
                            f"gnorm={output['gnorm']:.3E} |"
                            f"sigma_next={output['sigma_next']} ... | "
                            f"N_eff={output['N_eff']:.1f} |"
                            f"lr={lr_schedule(epoch):.3E} "
                        )

                if not b:
                    print(f"Convergence criteria met at epoch {epoch}, stopping optimization")
                    break

            # if hessian:
            #     print("running for 200 epochs with hessian to get smoothness")
            #     for epoch in range(int(epochs * hessian_steps)):
            #         if (epoch % 10 == 0) and epoch > 0:
            #             t = time.time()

            #             end = t0 + (t - t0) / 10 * (200 - epoch)

            #             print(
            #                 f"Epoch {epoch} time {time.strftime('%H:%M:%S', time.localtime(t))} estimated end {time.strftime('%H:%M:%S', time.localtime(end))}"
            #             )

            #             t0 = t

            #         if batch_size is not None:
            #             key, (x0_chunk, xt_chunk, w_chunk, wt_chunk, wdyn_chunk) = get_batch(key)

            #         (
            #             opt_state,
            #             learable_params,
            #             output,
            #             eta,
            #             sigma,
            #             # cov,
            #             b,
            #             iters,
            #             log_beta_gamma_new,
            #         ) = _body_fun(
            #             (
            #                 opt_state,
            #                 learable_params,
            #                 eta,
            #                 sigma,
            #                 (x0_chunk, xt_chunk, w_chunk, wt_chunk, wdyn_chunk, nl),
            #                 # cov,
            #                 True,
            #                 epoch,
            #             ),
            #             hessian=True,
            #         )

            #         if (epoch % print_nonlin_every) == 0:
            #             with jnp.printoptions(formatter={"float": "{:0.3e}".format}):
            #                 print(
            #                     f"epoch={epoch} | timescales (ns)={output['timescales'][::-1] / nanosecond} | "
            #                     f"sat={output['sat']:.3f} | loss_main={output['loss_main']:.3E} | "
            #                     f"ortho={output['loss_ortho']:.3E} | eta_next={output['eta_next']:.3E} | "
            #                     f"gnorm={output['gnorm']:.3E} |"
            #                     f"sigma_next={output['sigma_next']} ... | "
            #                     f"N_eff={output['N_eff']:.1f} |"
            #                     f"lr={lr_schedule(epoch):.3E} "
            #                 )

            if batch_size is not None:
                del pools
                del x0_chunk, xt_chunk, w_chunk, wt_chunk, wdyn_chunk

            trans = trans.apply_learnable_params(learable_params)
            print(f"optimized trans done ")

        else:
            eta = None
        print(f"{nl=}")

        cov = Covariances.create(
            cv_0=cv_0,  # type: ignore
            cv_1=cv_t,  # type: ignore
            nl=nl,
            nl_t=nl_t,
            w=w_tot if use_w else None,
            w_t=w_tot if use_w else None,
            dynamic_weights=dynamic_weights,
            calc_pi=calc_pi,
            only_diag=only_diag,
            symmetric=symmetric,
            # chunk_size=batch_chunk_size,
            # macro_chunk=macro_chunk,
            trans_f=trans,
            trans_g=trans,
            verbose=verbose,
            shrink=False,
            shrinkage_method=shrinkage_method,
            pi_argmask=jnp.array([-1]) if add_1 else None,
            eps_shrink=eps_shrink,
            generator=True,
            scale_deriv=eta ** (-1 / 2) if variational else None,
            macro_chunk=batch_chunk_size,
            # halve_even_odd=angle,
        )

        # if halve_even_odd:
        #     cov_full = cov
        #     cov = cov.halve_even_odd()

        argmask = cov.mask(eps_pre, max_features_pre, auto_cov_threshold)

        if not generator:
            # first mask before shrinking
            if shrink:
                cov = cov.shrink()

        idx, s, W = cov.decompose_diag(
            out_dim,
            non_per=n_non_per,
            per=n_per,
        )

        s = (
            s
            / (beta_timecon if beta_timecon is not None else 1 / (300 * boltzmann))
            * (tau_thermo if tau_thermo is not None else 100 * picosecond)
        )

        # x = cov.decompose(
        #     out_dim,
        #     eps=eps,
        #     out_eps=out_eps,
        #     glasso_whiten=glasso,
        #     verbose=verbose,
        #     scale_gen=eta
        #     / (beta_timecon if beta_timecon is not None else 1 / (300 * boltzmann))
        #     * (tau_thermo if tau_thermo is not None else 100 * femtosecond)
        #     # * amu
        #     if variational
        #     else None,
        #     angle=angle,
        # )

        # if halve_even_odd:
        #     """
        #     Per-mode 2x2 analysis of a shared-Q complex slow-mode problem, with COMPLEX Q.

        #     What changed vs the real-Q version
        #     ----------------------------------
        #     With q = a + i b complex, the mode is Z_n = (e + i o) q_n, so

        #         Re Z_n = e.a - o.b        Im Z_n = e.b + o.a

        #     i.e. the pair is no longer (even, odd) -- both components mix parities.  The
        #     2D plane is therefore defined by the (2n x 2) real coefficient matrix

        #         P_n = [[ a,  b],          (rows indexed by the stacked feature vector
        #             [-b,  a]]           f = [e; o]; columns are Re Z_n, Im Z_n)

        #     and the restricted forms are simply

        #         C2 = P' C P ,   G2 = P' G P            (real symmetric 2x2)

        #     which automatically picks up the antisymmetric parts of the cross-blocks that
        #     a real q cannot see.  Everything downstream (2x2 pencil, rotation, T) is
        #     unchanged and real.

        #     The n-dimensional problem is now the Hermitian pencil

        #         H_G q = lam H_C q ,   H_M = (M_ee + M_oo) + i (M_eo - M_oe)

        #     which has real eigenvalues (symmetric-definite Hermitian pencil) and complex
        #     eigenvectors.  If Im H_C = Im H_G = 0 (inversion-symmetric sampling) this
        #     reduces exactly to the previous real problem.

        #     Gauge
        #     -----
        #     q -> exp(i alpha) q rotates (Re Z, Im Z) into each other, so:

        #         INVARIANT     mu_lo, mu_hi, ratio, aniso, lam_e, lam_o, rho, kappa,
        #                     and the extracted coordinates u, v as functions
        #         GAUGE-DEP.    angle, R, T (angle shifts by -alpha)

        #     Use `uv` (coefficients of u, v on the stacked features) rather than `angle`
        #     when q is complex -- `angle` is only meaningful relative to a fixed phase, and
        #     eigh returns eigenvectors with an arbitrary phase.

        #     The parity diagnostics are defined as traces over the plane,

        #         V_e = a' C_ee a + b' C_ee b,   D_e = a' G_ee a + b' G_ee b,   (same for o)
        #         lam_e = D_e / V_e,  rho = V_e / (V_e + V_o),  kappa = cross / sqrt(V_e V_o)

        #     which are invariant under the phase and reduce to the old definitions at b = 0.
        #     """

        #     # --------------------------------------------------------------------------
        #     # helpers
        #     # --------------------------------------------------------------------------

        #     def _sym(M):
        #         return 0.5 * (M + jnp.swapaxes(M, -1, -2))

        #     def _herm(M):
        #         return 0.5 * (M + jnp.conj(jnp.swapaxes(M, -1, -2)))

        #     def _ct(M):
        #         """Conjugate transpose (plain transpose for real input)."""
        #         return jnp.conj(jnp.swapaxes(M, -1, -2))

        #     def psd_inv_sqrt(M, rcond=1e-10):
        #         """Symmetric/Hermitian inverse square root, pseudo-inverted below rcond."""
        #         M = _herm(M) if jnp.iscomplexobj(M) else _sym(M)
        #         w, V = jnp.linalg.eigh(M)
        #         wmax = jnp.max(w, axis=-1, keepdims=True)
        #         keep = w > rcond * wmax
        #         safe = jnp.where(keep, w, 1.0)  # double-where: no NaN grad
        #         inv = jnp.where(keep, 1.0 / jnp.sqrt(safe), 0.0)
        #         return (V * inv[..., None, :].astype(V.dtype)) @ _ct(V)

        #     def _eigh2(S):
        #         """Analytic eigendecomposition of a real 2x2 symmetric matrix.

        #         Returns (mu_lo, mu_hi, phi); columns of R(phi) are the (low, high)
        #         eigenvectors.
        #         """
        #         a, b, d = S[..., 0, 0], S[..., 0, 1], S[..., 1, 1]
        #         tr, diff = a + d, a - d
        #         r = jnp.hypot(diff, 2.0 * b)
        #         mu_lo, mu_hi = 0.5 * (tr - r), 0.5 * (tr + r)
        #         theta = 0.5 * jnp.arctan2(2.0 * b, diff)  # direction of high eigvec
        #         phi = theta + 0.5 * jnp.pi  # direction of low  eigvec
        #         return mu_lo, mu_hi, phi

        #     def _rot(phi):
        #         c, s = jnp.cos(phi), jnp.sin(phi)
        #         return jnp.stack([jnp.stack([c, -s], -1), jnp.stack([s, c], -1)], -2)

        #     def block_index(two_n, layout="block"):
        #         n = two_n // 2
        #         if layout == "block":
        #             return jnp.arange(n), jnp.arange(n, two_n)
        #         if layout == "interleaved":
        #             return jnp.arange(0, two_n, 2), jnp.arange(1, two_n, 2)
        #         raise ValueError(layout)

        #     def split_blocks(M, layout="block"):
        #         """(2n,2n) -> (ee, eo, oe, oo), each (n,n)."""
        #         ie, io = block_index(M.shape[-1], layout)
        #         return (M[jnp.ix_(ie, ie)], M[jnp.ix_(ie, io)], M[jnp.ix_(io, ie)], M[jnp.ix_(io, io)])

        #     def hermitian_halve(M, layout="block"):
        #         """<zbar_j z_k> = (M_ee + M_oo) + i (M_eo - M_oe).  Hermitian for symmetric M."""
        #         ee, eo, oe, oo = split_blocks(M, layout)
        #         return (ee + oo) + 1j * (eo - oe)

        #     def pair_coeffs(q, two_n, layout="block"):
        #         """(2n, 2) real coefficient matrix P for (Re Z, Im Z) on f = [e; o]."""
        #         ie, io = block_index(two_n, layout)
        #         a, b = jnp.real(q), jnp.imag(q)
        #         P = jnp.zeros((two_n, 2), dtype=a.dtype)
        #         P = P.at[ie, 0].set(a).at[io, 0].set(-b)
        #         P = P.at[ie, 1].set(b).at[io, 1].set(a)
        #         return P

        #     # --------------------------------------------------------------------------
        #     # main
        #     # --------------------------------------------------------------------------

        #     @partial(jax.jit, static_argnames=("layout", "rcond"))
        #     def analyse(C, G, W0=None, layout="block", rcond=1e-10):
        #         """Solve the (Hermitian) shared-Q pencil and the per-mode 2x2 pencils.

        #         Parameters
        #         ----------
        #         C, G : (2n, 2n) real covariance / Dirichlet matrices over [e; o].
        #         W0   : optional (n, n) whitening of H_C, real or complex, with
        #             W0^dagger H_C W0 = I.  Check `white_err` in the output.

        #         All returned eigenvalues are real.  Per-mode quantities are ordered by
        #         increasing lam and indexed along axis 0.
        #         """
        #         two_n = C.shape[-1]
        #         C, G = _sym(C), _sym(G)
        #         Cee, Ceo, Coe, Coo = split_blocks(C, layout)
        #         Gee, Geo, Goe, Goo = split_blocks(G, layout)

        #         H_C, H_G = _herm(hermitian_halve(C, layout)), _herm(hermitian_halve(G, layout))

        #         if W0 is None:
        #             W0 = psd_inv_sqrt(H_C, rcond)
        #         W0 = W0.astype(H_C.dtype)

        #         # --- Hermitian generalised eigenproblem: real eigenvalues -------------
        #         A = _herm(_ct(W0) @ H_G @ W0)
        #         lam, U = jnp.linalg.eigh(A)  # lam real, ascending; U unitary
        #         lam = jnp.real(lam)
        #         Q = W0 @ U  # columns q_n, H_C-orthonormal

        #         # eigh returns each eigenvector up to an arbitrary complex phase.  Fix it
        #         # so the largest-|.| component is real positive: reproducible `angle`, and
        #         # Q comes out real whenever the underlying problem is real.
        #         piv = jnp.take_along_axis(Q, jnp.argmax(jnp.abs(Q), axis=0)[None, :], axis=0)[0]
        #         Q = Q * (jnp.conj(piv) / jnp.maximum(jnp.abs(piv), jnp.finfo(lam.dtype).tiny))

        #         white_err = jnp.linalg.norm(_ct(W0) @ H_C @ W0 - jnp.eye(W0.shape[0]))

        #         # --- per-mode 2D plane -------------------------------------------------
        #         P = jax.vmap(lambda q: pair_coeffs(q, two_n, layout), out_axes=0)(Q.T)  # (n,2n,2)
        #         C2 = jnp.einsum("nia,ij,njb->nab", P, C, P)
        #         G2 = jnp.einsum("nia,ij,njb->nab", P, G, P)
        #         C2, G2 = _sym(C2), _sym(G2)

        #         # --- per-mode 2x2 pencil (real) ---------------------------------------
        #         Wc = jax.vmap(lambda M: psd_inv_sqrt(M, rcond))(C2)
        #         mu_lo, mu_hi, phi = _eigh2(_sym(Wc @ G2 @ Wc))
        #         R = _rot(phi)
        #         T = jnp.swapaxes(R, -1, -2) @ Wc  # (Re Z, Im Z) -> (u, v)
        #         uv = P @ jnp.swapaxes(T, -1, -2)  # (n, 2n, 2): u, v on [e; o]
        #         # deterministic sign: largest-|.| coefficient positive
        #         idx = jnp.argmax(jnp.abs(uv), axis=1)  # (n,2)
        #         sgn = jnp.sign(jnp.take_along_axis(uv, idx[:, None, :], axis=1))
        #         uv = uv * jnp.where(sgn == 0, 1.0, sgn)

        #         # --- parity diagnostics (phase-invariant traces over the plane) --------
        #         a, b = jnp.real(Q).T, jnp.imag(Q).T  # (n, n_half)
        #         tr = lambda M: jnp.einsum("nj,jk,nk->n", a, M, a) + jnp.einsum("nj,jk,nk->n", b, M, b)
        #         V_e, V_o, D_e, D_o = tr(Cee), tr(Coo), tr(Gee), tr(Goo)
        #         cross = tr(_sym(Ceo))

        #         eps = jnp.finfo(lam.dtype).tiny
        #         safe = lambda x: jnp.where(x > 0, x, 1.0)
        #         lam_e = jnp.where(V_e > 0, D_e / safe(V_e), 0.0)
        #         lam_o = jnp.where(V_o > 0, D_o / safe(V_o), 0.0)
        #         rho = V_e / jnp.maximum(V_e + V_o, eps)
        #         kappa = cross / jnp.sqrt(jnp.maximum(V_e * V_o, eps))

        #         # --- how much complex structure is actually present --------------------
        #         nrm = lambda M: jnp.linalg.norm(M)
        #         asym_C = nrm(jnp.imag(H_C)) / jnp.maximum(nrm(jnp.real(H_C)), eps)
        #         asym_G = nrm(jnp.imag(H_G)) / jnp.maximum(nrm(jnp.real(H_G)), eps)

        #         return dict(
        #             lam=lam,
        #             Q=Q,
        #             mu=jnp.stack([mu_lo, mu_hi], -1),
        #             ratio=mu_hi / jnp.maximum(mu_lo, eps),
        #             aniso=(mu_hi - mu_lo) / jnp.maximum(mu_hi + mu_lo, eps),
        #             R=R,
        #             T=T,
        #             uv=uv,
        #             angle=jnp.mod(phi, jnp.pi),  # gauge-dependent if Q is complex
        #             lam_e=lam_e,
        #             lam_o=lam_o,
        #             ratio_eo=lam_e / jnp.maximum(lam_o, eps),
        #             rho=rho,
        #             kappa=kappa,
        #             imag_frac=jnp.abs(jnp.imag(Q)).sum() / jnp.maximum(jnp.abs(Q).sum(), eps),
        #             C2=C2,
        #             G2=G2,
        #             white_err=white_err,
        #             asym_C=asym_C,
        #             asym_G=asym_G,
        #         )

        #     def classify(res, ratio_circle=1.3, ratio_line=1.8):
        #         out = []
        #         for r in res["ratio"]:
        #             r = float(r)
        #             out.append("loop" if r < ratio_circle else ("line" if r > ratio_line else "ambiguous"))
        #         return out

        #     def report(res, k=None):
        #         n = res["lam"].shape[0]
        #         k = n if k is None else k
        #         print(
        #             f"|Im H_C|/|Re H_C| = {float(res['asym_C']):.3e}   "
        #             f"|Im H_G|/|Re H_G| = {float(res['asym_G']):.3e}   "
        #             f"whitening err = {float(res['white_err']):.2e}   "
        #             f"Im(Q) frac = {float(res['imag_frac']):.3f}"
        #         )
        #         hdr = (
        #             "mode      lam    mu_lo    mu_hi    ratio   angle    lam_e    lam_o   e/o     rho    kappa   class"
        #         )
        #         print(hdr)
        #         print("-" * len(hdr))
        #         lab = classify(res)
        #         for i in range(k):
        #             print(
        #                 f"{i:>4d} {res['lam'][i]:8.4f} {res['mu'][i, 0]:8.4f} "
        #                 f"{res['mu'][i, 1]:8.4f} {res['ratio'][i]:8.3f} "
        #                 f"{jnp.rad2deg(res['angle'][i]):7.1f} "
        #                 f"{res['lam_e'][i]:8.4f} {res['lam_o'][i]:8.4f} "
        #                 f"{res['ratio_eo'][i]:6.3f} {res['rho'][i]:7.3f} "
        #                 f"{res['kappa'][i]:8.3f}   {lab[i]}"
        #             )

        #     W0, _, _ = x
        #     Q = jnp.diag(cov.sigma_0_inv) @ W0.T
        #     res = analyse(cov_full.C00, cov_full.C_gen, W0=Q, layout="block", rcond=1e-10)

        #     report(res, k=min(10, res["lam"].shape[0]))

        #     print(f"{res["R"].shape=}")

        #     pair_info = (res["ratio"], res["T"])
        # else:
        #     pair_info = None

        W0 = W
        W1 = None
        argmask = idx

        # if only_diag:
        #     argmask_s, s = x
        #     W0 = None
        #     W1 = None

        #     argmask = argmask
        # else:
        #     W0, W1, s, idx = x

        if out_dim is not None:
            if s.shape[0] < out_dim:
                print(f"found only {s.shape[0]} singular values")

        if verbose:
            if generator:
                _ts = 1 / s / nanosecond
            else:
                _ts = -tau / jnp.log(s) / nanosecond

            print(f"timescales {_ts[0:min(10, s.shape[0])]=}  (ns)")

        print(f"{add_1=} new weights")  # if self.periodicities is not None:
        #     print("applying periodicities to f")

        #     _real = CvTrans.from_cv_function(lambda cv, nl, shmap, shmap_kwargs: cv.replace(cv=jnp.abs(cv.cv)))

        #     tr *= _real

        km = KoopmanModel(
            cov=cov,
            W0=W0,
            W1=W1,
            s=s,
            cv_0=cv_0,
            cv_t=cv_t,
            nl=nl,
            nl_t=nl_t,
            w=w,
            w_t=w_t,
            rho=rho,
            rho_t=rho_t,
            dynamic_weights=dynamic_weights,
            add_1=add_1,
            max_features=max_features,
            max_features_pre=max_features_pre,
            out_dim=out_dim,
            # method=method,
            calc_pi=calc_pi,
            tau=tau,
            trans=trans,
            argmask=argmask,
            only_diag=only_diag,
            eps=eps,
            eps_pre=eps_pre,
            shape=cov.rho_00.shape[0],
            scaled_tau=scaled_tau,
            correlation_whiten=correlation_whiten,
            constant_threshold=constant_threshold,
            verbose=verbose,
            shrink=shrink,
            shrinkage_method=shrinkage_method,
            eps_shrink=eps_shrink,
            # use_w=use_w,
            generator=generator,
            # periodicities=periodicities[idx],
            periodicities=jnp.array([False] * n_non_per + [True] * n_per),
            entropy_reg=entropy_reg,
            target_smoothness=target_smoothness,
            alpha_smooth=alpha_smooth,
            halve_even_odd=halve_even_odd,
            # pair_info=pair_info,
            argmask_s=None,
        )

        return km

    @staticmethod
    def _add_1(
        cv,
        nl,
        shmap,
        shmap_kwargs,
    ):
        return cv.replace(cv=jnp.hstack([cv.cv, jnp.array([1])]))

    # def Tk(self, out_dim=None):
    #     # Optimal Data-Driven Estimation of Generalized Markov State Models for Non-Equilibrium Dynamics eq. 30
    #     # T_k = C11^{-1/2} K^T C00^{1/2}
    #     #     = w1.T K.T w0 C00

    #     if out_dim is None:
    #         out_dim = self.s.shape[0]

    #     Tk = self.[:out_dim, :].T @ jnp.diag(self.s[:out_dim])

    #     return Tk

    @staticmethod
    def _periodic_trans_pre(
        cv: CV,
        nl,
        shmap,
        shmap_kwargs,
        w_nonper: jax.Array,
        w_per: jax.Array,
        mu_per: jax.Array,
        eps=1e-8,
        idx_per: jax.Array | None = None,
        idx_nonper: jax.Array | None = None,
    ):
        cv_even, cv_odd = CV.split(cv)

        x_even_per = cv_even.cv[idx_per]
        x_odd_per = cv_odd.cv[idx_per]

        x_even_nonper = cv_even.cv[idx_nonper]
        x_odd_nonper = cv_odd.cv[idx_nonper]

        assert x_even_per.shape[0] == x_odd_per.shape[0]
        assert x_even_nonper.shape[0] == x_odd_nonper.shape[0]

        assert w_nonper.shape == (x_even_nonper.shape[0], 2)
        assert w_per.shape == (x_even_per.shape[0], 2, 2)
        assert mu_per.shape == (x_even_per.shape[0], 2)

        @jax.vmap
        def _per(u, v, pi, w):
            out = (jnp.hstack([u, v]) - pi) @ w

            return out[0], out[1]

        @jax.vmap
        def _nonper(u, v, w):
            return jnp.hstack([u, v]) @ w

        x_per, y_per = _per(x_even_per, x_odd_per, mu_per, w_per)
        x_nonper = _nonper(x_even_nonper, x_odd_nonper, w_nonper)

        new_cv_nonper = CV(cv=x_nonper, _combine_dims=None)
        new_cv_per = CV(cv=jnp.array(list(zip(x_per, y_per))).reshape((-1,)), _combine_dims=None)

        print(f"{new_cv_nonper.cv.shape=}, {new_cv_per.cv.shape=}")

        return CV.combine(new_cv_nonper, new_cv_per)

    @staticmethod
    def _periodic_trans_pushback(
        cv: CV,
        nl,
        shmap,
        shmap_kwargs,
        K_max=5,
        # pi: jax.Array,
        # sigma: jax.Array,
    ):
        # cv = cv.replace(cv=(cv.cv - pi) / sigma)

        _, cv_ang = CV.split(cv)

        ang_x = cv_ang.cv[::2]
        ang_y = cv_ang.cv[1::2]

        r = jnp.sqrt(ang_x**2 + ang_y**2 + 1e-8)

        angle = (ang_x + 1j * ang_y) / (r)

        constraints = []

        # fix radius at 1, push away from origin
        constraints.append((r + 2 / r - 2 * jnp.sqrt(2)))

        k_array = jnp.arange(1, K_max + 1)
        constraints.append((angle[:, None] ** k_array[None, :] / k_array[None, :]).reshape((-1,)))

        cvs = []

        for c in constraints:
            cvs.append(CV(cv=c, _combine_dims=None))

        print(f"{cvs=}")

        cv_out = CV.combine(*cvs)

        print(f"{cv_out=}")

        return cv_out

        # loss = -jnp.log(ang_x**2 + ang_y**2 + 1e-8)

        # return CV(cv=loss, _combine_dims=None)

    @staticmethod
    def _periodic_trans_post(
        cv: CV,
        nl,
        shmap,
        shmap_kwargs,
        pi: jax.Array,
        sigma: jax.Array,
        w_per: jax.Array,
    ):
        print(f"{cv=}")

        cv = cv.replace(cv=(cv.cv - pi) / sigma)

        cv_lin, cv_ang = CV.split(cv)

        ang_x = cv_ang.cv[::2]
        ang_y = cv_ang.cv[1::2]

        @jax.vmap
        def _per(x, y, w):
            out = jnp.hstack([x, y]) @ w.T

            return jnp.arctan2(out[1], out[0])

        cv_ang_x = _per(ang_x, ang_y, w_per)

        return CV.combine(
            cv_lin.replace(cv=cv_lin.cv),
            cv_ang.replace(cv=cv_ang_x),
        )

        # @jax.vmap
        # def pair(x, y, b):
        #     return jnp.where(b, jnp.atan2(y, x), x)

        # out_x = pair(cv_even.cv, cv_odd.cv, bools)

        # return CV(cv=out_x, _combine_dims=None)

    def _idx_select(
        cv: CV,
        nl,
        shmap,
        shmap_kwargs,
        idx: jax.Array,
    ):
        return cv.replace(cv=cv.cv[idx], _combine_dims=None)

    @staticmethod
    def _unpair(
        cv,
        nl,
        shmap,
        shmap_kwargs,
        pair_info,
    ):
        ratios, uv = pair_info

        cv_even, cv_odd = CV.split(cv)

        def _inner(x, uv, ratios):
            out = x @ uv

            return jnp.where(ratios < 1.3, jnp.atan2(out[1], out[0]), out[0])

        cvs = jax.vmap(_inner, in_axes=(0, 0, 0))(jnp.stack([cv_even.cv, cv_odd.cv], axis=-1), uv, ratios)
        return CV(cv=cvs, _combine_dims=None)

    # def radius_norm(eps=1e-8, idx=None) -> CvTrans:
    #     return CvTrans.from_cv_function(
    #         _radius_norm,
    #         eps=eps,
    #         idx=idx,
    #     )

    @property
    def tot_trans(self):
        tr = self.trans

        # if self.add_1:
        #     if tr is not None:
        #         tr *= CvTrans.from_cv_function(KoopmanModel._add_1)

        #     else:
        #         tr = CvTrans.from_cv_function(KoopmanModel._add_1)

        return tr

    @staticmethod
    def _transform(
        cv: CV,
        nl,
        shmap,
        shmap_kwargs,
        argmask: jax.Array | None = None,
        pi: jax.Array | None = None,
        add_1: bool = False,
        add_1_pre: bool = False,
        q: jax.Array | None = None,
        l: jax.Array | None = None,
        # halve_even_odd: bool = False,
    ) -> jax.Array:
        print(f"{cv=}")

        x = cv.cv
        # def _inner(x: jax.Array, q: jax.Array | None, pi: jax.Array | None = None) -> jax.Array:
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

        return CV(cv=x, _combine_dims=None)

        # if not halve_even_odd:
        #     x = _inner(cv.cv, q, pi)
        #     return cv.replace(cv=x, _combine_dims=None)

        # pi_even = jnp.real(pi) if pi is not None else None
        # pi_odd = jnp.imag(pi) if pi is not None else None

        # q_real = jnp.real(q)
        # q_imag = jnp.imag(q)

        # cv_even, cv_odd = cv.split()

        # x_even = _inner(cv_even.cv, q_real, pi_even) - _inner(cv_odd.cv, q_imag, pi_odd)
        # x_odd = _inner(cv_odd.cv, q_real, pi_odd) + _inner(cv_even.cv, q_imag, pi_even)

        # return CV.combine(cv.replace(cv=x_even, _combine_dims=None), cv.replace(cv=x_odd, _combine_dims=None))

    @staticmethod
    def _radius_norm(cv, nl, shmap, shmap_kwargs, eps=1e-8):
        cv_even, cv_odd = CV.split(cv)

        r_sq = cv_even.cv**2 + cv_odd.cv**2 + eps
        r_inv = 1 / jnp.sqrt(r_sq)

        return CV.combine(
            cv_even.replace(cv=cv_even.cv * r_inv, _combine_dims=None),
            cv_odd.replace(cv=cv_odd.cv * r_inv, _combine_dims=None),
        )

    def f_stiefel(
        self,
        out_dim=None,
        n_skip=None,
        remove_constant=True,
        mask: jax.Array | None = None,
        optimizer_kwargs: dict = {},
    ):
        W0, _ = self.cov.decompose_pymanopt(out_dim=out_dim, optimizer_kwargs=optimizer_kwargs)

        # sigma_1_inv = self.cov.sigma_1_inv
        # assert sigma_1_inv is not None
        o = W0.T  # jnp.diag(sigma_1_inv) @ self.W1.T
        # s = self.s

        # n_skip = self.get_n_skip(
        #     n_skip=n_skip,
        #     remove_constant=remove_constant,
        # )

        # # s = s[n_skip:]
        # o = o[:, n_skip:]

        if mask is not None:
            # s = s[mask]
            o = o[:, mask]

        o = o[:, :out_dim]

        tr = CvTrans.from_cv_function(
            KoopmanModel._transform,
            static_argnames=["add_1", "add_1_pre"],
            add_1=False,
            add_1_pre=False,
            q=o,
            pi=self.cov.pi_1,
            argmask=self.argmask,
        )

        if self.tot_trans is not None:
            tr = self.tot_trans * tr

        return tr

    def f(
        self,
        out_dim=None,
        remove_constant=True,
        n_skip: int | None = None,
        mask: jax.Array | None = None,
        inv_exp=True,
    ):
        tr = self.trans

        _real = CvTrans.from_cv_function(
            KoopmanModel._periodic_trans_post,
            pi=self.cov.pi_0,
            sigma=self.cov.sigma_0,
            w_per=self.W0,
        )

        tr *= _real

        tr *= CvTrans.from_cv_function(
            KoopmanModel._idx_select,
            idx=self.argmask,
        )

        per = self.periodicities[self.argmask] if self.periodicities is not None else None

        # if self.only_diag:
        #     argmask_s = self.argmask_s

        # else:
        #     # o = self.W0.T
        #     o = jnp.diag(self.cov.sigma_0_inv) @ self.W0.T

        # s = self.s

        # n_skip = self.get_n_skip(
        #     n_skip=n_skip,
        #     remove_constant=remove_constant,
        # )

        # s = s[n_skip:]

        # # pair_info = self.pair_info if self.halve_even_odd else None

        # if self.only_diag:
        #     argmask_s = argmask_s[n_skip:]
        # else:
        #     o = o[:, n_skip:]

        #     # if pair_info is not None:
        #     #     pair_info = (pair_info[0][n_skip:], pair_info[1][n_skip:])

        # if mask is not None:
        #     s = s[mask]
        #     if self.only_diag:
        #         argmask_s = argmask_s[mask]
        #     else:
        #         o = o[:, mask]

        #     # if pair_info is not None:
        #     #     pair_info = (pair_info[0][mask], pair_info[1][mask])

        # if self.only_diag:
        #     argmask_s = argmask_s[:out_dim]
        # else:
        #     o = o[:, :out_dim]

        #     # if self.halve_even_odd:
        #     #     pair_info = (pair_info[0][:out_dim], pair_info[1][:out_dim])

        # tr = CvTrans.from_cv_function(
        #     KoopmanModel._transform,
        #     static_argnames=["add_1", "add_1_pre"],
        #     add_1=False,
        #     add_1_pre=False,
        #     q=o if not self.only_diag else None,
        #     pi=self.cov.pi_0[argmask_s] if (self.only_diag and self.cov.pi_0 is not None) else self.cov.pi_0,
        #     argmask=self.argmask if not self.only_diag else self.argmask[argmask_s],
        #     # halve_even_odd=self.halve_even_odd,
        # )

        # if self.trans is not None:
        #     tr = self.trans * tr

        # per = None

        # if self.periodicities is not None:
        #     print("applying periodicities to f")

        # if self.halve_even_odd:
        #     # tr = tr * CvTrans.from_cv_function(
        #     #     KoopmanModel._angle_transform,
        #     # )

        #     tr = tr * CvTrans.from_cv_function(
        #         KoopmanModel._unpair,
        #         pair_info=pair_info,
        #     )

        #     per = pair_info[0] < 1.3

        # if self.cov.C00.dtype == jnp.complex_ is not None and inv_exp:
        #     print("applying periodicities to f")

        #     _real = CvTrans.from_cv_function(PeriodicKoopmanModel._inv_exp_periodic)

        #     tr *= _real

        return tr, per

    def g(
        self,
        out_dim=None,
        n_skip=None,
        remove_constant=True,
        mask: jax.Array | None = None,
    ):
        sigma_1_inv = self.cov.sigma_1_inv
        assert sigma_1_inv is not None
        o = self.W1.T  # jnp.diag(sigma_1_inv) @ self.W1.T
        s = self.s

        n_skip = self.get_n_skip(
            n_skip=n_skip,
            remove_constant=remove_constant,
        )

        s = s[n_skip:]
        o = o[:, n_skip:]

        if mask is not None:
            s = s[mask]
            o = o[:, mask]

        o = o[:, :out_dim]

        tr = CvTrans.from_cv_function(
            KoopmanModel._transform,
            static_argnames=["add_1", "add_1_pre"],
            add_1=False,
            add_1_pre=False,
            q=o,
            pi=self.cov.pi_1,
            argmask=self.argmask,
        )

        if self.tot_trans is not None:
            tr = self.tot_trans * tr

        return tr

    def koopman_weight(
        self,
        verbose=False,
        chunk_size=None,
        macro_chunk=1000,
        retarget=True,
        epsilon=1e-6,
        max_entropy=True,
        out_dim=None,
    ) -> tuple[list[Array], list[Array], list[Array] | None, list[Array] | None, bool]:
        # Optimal Data-Driven Estimation of Generalized Markov State Models, page 18-19
        # create T_k in the trans basis
        # C00^{-1} C11 T_k
        # W0 C00 W0.T = I
        # W1 C11 W1.T = I
        # W0 C01 W1.T = sigma
        # _______________
        # W1 C11 W0.T sigma x = x   <- eigensolver
        # mu_ref = W1.T@x

        # assert not self.shrink, "cannot compute equilibrium density from shrunken covariances"

        if out_dim is None:
            out_dim = int(jnp.sum(jnp.abs(1 - self.s) < epsilon))

        if out_dim == 0:
            # print("no eigenvalues found close to 1")
            # return self.w, None, False
            out_dim = 1

            i = int(jnp.min(jnp.array([5, self.s.shape[0]])))

            print(f"using closest eigenvalue to 1: {self.s[:i]=}")

        print("reweighing,  A")

        # A = self.W1 @ self.cov.rho_11 @ jnp.diag(self.cov.sigma_1 * self.cov.sigma_0_inv) @ self.W0.T @ jnp.diag(self.s)
        A = self.W1 @ self.cov.C11 @ self.W0.T @ jnp.diag(self.s)

        lv, v = jnp.linalg.eig(A)

        print(f"eigenvalues {lv=}")

        # remove complex eigenvalues, as they cannot be tshe ground state
        real = jnp.abs(jnp.imag(lv)) <= 1e-3
        out_idx = jnp.argwhere(real).reshape((-1))  # out_idx[real]
        out_idx = out_idx[jnp.argsort(jnp.abs(lv[out_idx] - 1))]
        # sort

        n = jnp.min(jnp.array([out_idx.shape[0], 10]))
        print(f"{lv[out_idx[:n]]=} ")

        lv, v = lv[out_idx], v[:, out_idx]

        lv, v = jnp.real(lv), jnp.real(v)

        lv, v = lv[(0,)], v[:, (0,)]

        sigma_1_inv = self.cov.sigma_1_inv

        assert sigma_1_inv is not None

        mu_ref = self.W1.T @ v  # jnp.diag(sigma_1_inv) @ self.W1.T @ v

        f_trans_2 = CvTrans.from_cv_function(
            KoopmanModel._transform,
            q=None,
            l=None,
            pi=self.cov.pi_0 if self.calc_pi else None,
            argmask=self.argmask,
            add_1_pre=False,
            static_argnames=["add_1_pre"],
        )

        @partial(CvTrans.from_cv_function, v=mu_ref)
        def _get_w(cv: CV, _nl, shmap, shmap_kwargs, v: Array):
            x = cv.cv

            return cv.replace(cv=jnp.einsum("i,ij->j", x, v), _combine_dims=None)

        tr = f_trans_2 * _get_w

        if self.trans is not None:
            tr = self.trans * tr

        print(f"{self.cv_0[0].shape}")

        from IMLCV.base.dataset import DataLoaderOutput

        w_out_cv, w_out_cv_t = DataLoaderOutput.apply_cv(
            f=tr,
            x=self.cv_0,  # type:ignore
            x_t=self.cv_t,  # type:ignore
            nl=self.nl,
            # nl_t=self.nl_t,
            macro_chunk=macro_chunk,
            chunk_size=chunk_size,
            verbose=verbose,
        )
        assert w_out_cv_t is not None

        w_corr = CV.stack(*w_out_cv).cv
        w_corr_t = CV.stack(*w_out_cv_t).cv

        def _norm(w):
            w = jnp.where(jnp.sum(w > 0) - jnp.sum(w < 0) > 0, w, -w)  # majaority determines sign
            n_neg = jnp.sum(w < 0)

            w_pos: jax.Array = jnp.where(w >= 0, w, 0)  # type: ignore
            w_neg: jax.Array = jnp.where(w < 0, -w, 0)  # type: ignore

            n = jnp.sum(w_pos)

            w_pos /= n
            w_neg /= n

            return w_pos, jnp.sum(w_neg), n_neg / w.shape[0]

        w_pos, wf_neg, f_neg = vmap_decorator(_norm, in_axes=1)(w_corr)
        w_pos_t, wf_neg_t, f_neg_t = vmap_decorator(_norm, in_axes=1)(w_corr_t)

        x = jnp.logical_and(wf_neg == 0, f_neg == 0)

        nm = jnp.sum(x)

        if nm == 0:
            print(f"didn't find modes with positive weights, aborting {wf_neg=} {f_neg=}")
            assert self.w is not None
            assert self.w_t is not None

            return self.w, self.w_t, None, None, False

        if nm > 1:
            print(f"found multiple modes with positive weights, merging {wf_neg=} {f_neg=}")
        else:
            print(f"succes")

        def norm_w_corr(w: jax.Array):
            log_w = jnp.log(w)
            log_w -= jnp.max(log_w)
            log_w_norm = jnp.log(jnp.sum(jnp.exp(log_w)))

            w_norm = jnp.exp(log_w - log_w_norm) * w.shape[0]
            # w_norm = DataLoaderOutput._unstack_weights([cvi.shape[0] for cvi in self.cv_0], w_norm)

            return w_norm

        print(f"{w_pos.shape=}")

        w_corr = norm_w_corr(jnp.sum(w_pos[x, :], axis=0))
        w_corr_t = norm_w_corr(jnp.sum(w_pos_t[x, :], axis=0))

        print(f"  {jnp.std(w_corr)=}  ")

        def get_new_w(w_orig: list[Array], w_corr: Array):
            log_w_orig = jnp.log(jnp.hstack(w_orig))

            w_new_log = jnp.log(w_corr) + log_w_orig

            mm = jnp.max(w_new_log)

            w_new_log -= mm

            w_new_log_norm = jnp.log(jnp.sum(jnp.exp(w_new_log)))

            w_new = jnp.exp(w_new_log - w_new_log_norm)

            w_new = jnp.real(w_new)

            w_new = DataLoaderOutput._unstack_weights([cvi.shape[0] for cvi in self.cv_0], w_new)

            return w_new

        assert self.w is not None
        assert self.w_t is not None

        w_new = get_new_w(self.w, w_corr)
        w_new_t = get_new_w(self.w_t, w_corr_t)

        # print(f"{  jnp.isnan(jnp.hstack(w_new)).any()=}")

        w_corr = DataLoaderOutput._unstack_weights([cvi.shape[0] for cvi in self.cv_0], w_corr)
        w_corr_t = DataLoaderOutput._unstack_weights([cvi.shape[0] for cvi in self.cv_t], w_corr_t)

        return w_new, w_new_t, w_corr, w_corr_t, True

    def weighted_model(
        self,
        chunk_size=None,
        macro_chunk=1000,
        # verbose=False,
        out_dim=None,
        **kwargs,
    ) -> KoopmanModel:
        if self.only_diag:
            print("cannot reweight model with only diagonal covariances")
            return self

        new_w, new_w_t, _, _, b = self.koopman_weight(
            verbose=self.verbose,
            chunk_size=chunk_size,
            macro_chunk=macro_chunk,
            out_dim=out_dim,
        )

        # if not b:
        #     return self

        kw = dict(
            w=new_w,
            rho=self.rho,
            w_t=new_w_t,
            rho_t=self.rho_t,
            dynamic_weights=self.dynamic_weights,
            cv_0=self.cv_0,
            cv_t=self.cv_t,
            nl=self.nl,
            nl_t=self.nl_t,
            add_1=self.add_1,
            eps=self.eps,
            eps_pre=self.eps_pre,
            # method=self.method,
            symmetric=False,
            out_dim=self.out_dim,
            tau=self.tau,
            macro_chunk=macro_chunk,
            chunk_size=chunk_size,
            trans=self.trans,
            # T_scale=self.T_scale,
            verbose=self.verbose,
            calc_pi=self.calc_pi,
            max_features=self.max_features,
            max_features_pre=self.max_features_pre,
            only_diag=self.only_diag,
            correlation_whiten=self.correlation_whiten,
            constant_threshold=self.constant_threshold,
            shrink=self.shrink,
            shrinkage_method=self.shrinkage_method,
            eps_shrink=self.eps_shrink,
            use_w=self.use_w,
            generator=self.generator,
            entropy_reg=self.entropy_reg,
            target_smoothness=self.target_smoothness,
            alpha_smooth=self.alpha_smooth,
            # periodicities=self.periodicities,
            # argmask_s=self.argmask_s,
        )

        kw.update(**kwargs)

        return KoopmanModel.create(**kw)  # type:ignore

    def get_n_skip(
        self,
        n_skip: int | None = None,
        remove_constant=True,
    ):
        s = self.s

        if n_skip is None:
            if self.add_1 or not self.calc_pi:
                n_skip = 1
            else:
                n_skip = 0

        s = s[n_skip:]

        if remove_constant:
            nc = jnp.abs(1 - s) < self.constant_threshold

            if jnp.sum(nc) > 0:
                print(f"found {jnp.sum(nc)} constant eigenvalues,removing from timescales")

            s = s[jnp.logical_not(nc)]

            n_skip += int(jnp.sum(nc))

        print(f"got {n_skip=}")

        return n_skip

    def timescales(
        self,
        remove_constant=True,
        n_skip=None,
        mask: jax.Array | None = None,
    ):
        s = self.s

        n_skip = self.get_n_skip(
            n_skip=n_skip,
            remove_constant=remove_constant,
        )

        s = s[n_skip:]

        if mask is not None:
            s = s[mask]

        tau = self.tau
        if tau is None:
            tau = 1
            print("tau not set, assuming 1")

        if self.cov.rho_gen is not None:
            print("using generator timescales")
            return 1 / s

        return -tau / jnp.log(s)
