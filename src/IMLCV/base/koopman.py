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
        periodicities: jax.Array | None = None,
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
    ):
        #  see Optimal Data-Driven Estimation of Generalized Markov State Models
        if verbose:
            print(f"getting covariancesm {generator=}")

        epochs = int(epochs)

        print(f"{out_dim=} {eps=} {eps_pre=}")
        print(f"{calc_pi=} {add_1=}   ")

        print(f"{use_w=}")

        if w_t is not None:
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

        # if not generator:
        assert cv_t is not None
        assert w_t is not None
        assert rho_t is not None

        w_tot_t = tot_w(w_t, rho_t)

        if periodicities is not None:
            _tr_per = CvTrans.from_cv_function(PeriodicKoopmanModel._exp_periodic, periodicities=periodicities)

            if trans is not None:
                trans *= _tr_per
            else:
                trans = _tr_per

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

            key = jax.random.PRNGKey(42)

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
                init_value=1e-4,
                peak_value=1e-3,
                warmup_steps=200 if epochs > 200 else epochs // 5,
                decay_steps=epochs,
                end_value=1e-4,
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

                    wt_chunk /= wtot_chunk
                    w_chunk /= wtot_chunk

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

                # --------------------------------------------------------------------------- #
                # Linear-algebra kernels (verified against explicit inverse / per-mode sum)    #
                # --------------------------------------------------------------------------- #
                def _sym(A):
                    return 0.5 * (A + A.T)

                def _cholesky_jittered(C00, rel_jitter):
                    C = _sym(C00)
                    N = C.shape[0]
                    jit = rel_jitter * jnp.trace(C) / N
                    return jnp.linalg.cholesky(C + jit * jnp.eye(N))

                def _whiten(L, A):
                    """Return L^{-1} A L^{-T} via two triangular solves (symmetrised)."""
                    Y = solve_triangular(L, A, lower=True)  # L^{-1} A
                    M = solve_triangular(L, Y.T, lower=True).T  # L^{-1} A L^{-T}
                    return _sym(M)

                def _filter_matrix(U, phi_vals):
                    """phi(G_tilde) restricted to the columns of U:  U diag(phi_vals) U^T.

                    Smooth function of G_tilde, so the trace against it is invariant to
                    rotations inside any degenerate eigenspace.
                    """
                    return (U * phi_vals) @ U.T

                # --------------------------------------------------------------------------- #
                # The objective                                                                #
                # --------------------------------------------------------------------------- #
                @partial(jit_decorator, static_argnames=["entropy_reg", "chunk_size", "hessian"])
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
                    hp: LossHP,
                    entropy_reg=None,
                    chunk_size=None,
                    epoch: int = 0,
                    key: Array | None = None,
                    hessian: bool = True,
                ):
                    """Single scalar loss: -(slow-and-smooth reward) + ortho constraint."""
                    _trans = trans.apply_learnable_params(params)

                    scale_deriv = eta ** (-1 / 2)

                    bp = BendParams(
                        # R0=hp.R0
                        # * scale_deriv
                        # * jnp.sqrt(
                        #     hp.eta
                        # ),  # compare directly against the slowest mode length. This is a choice, not a physical prinicple
                        R0=hp.R0 * jnp.sqrt(hp.eta),
                        ridge=1e-5 / hp.eta,
                        mode="projected",
                        reduce="hinge",
                        n_probe=16,
                        hinge_p=2,
                        W=W_baro,
                    )

                    print(f"{key=}")

                    cov = Covariances.create(
                        cv_0=[x_0],
                        cv_1=[x_t],
                        nl=nl,
                        nl_t=nl,
                        w=[w],
                        w_t=[w],
                        trans_f=_trans,
                        trans_g=_trans,
                        shrink=False,
                        calc_pi=calc_pi,
                        get_diff=False,
                        symmetric=symmetric,
                        generator=True,
                        get_hess=hessian,
                        scale_deriv=scale_deriv,
                        curvature=False,
                        sigma_0=sigma_0,
                        key=key,
                        bp=bp,
                        # chunk_size=batch_chunk_size,
                        macro_chunk=batch_chunk_size,
                    )

                    # jax.debug.print("covariances computed {cov}", cov=cov)

                    N = cov.C00.shape[0]

                    ##################
                    # resolvent loss #
                    ##################
                    C = _sym(cov.rho_00)
                    M = _sym(cov.rho_gen) + C  # G/eta + C  (> 0)
                    L = _cholesky_jittered(M, hp.rel_jitter)  # L L^T = M
                    rho_res = _whiten(L, C)  # L^-1 C L^-T ; eig = 1/(1+lambda_i/eta) = mu_i
                    reward = jnp.trace(rho_res)  # sum_i mu_i   (eigh-free)

                    loss_main = -reward

                    ##################
                    # bending loss   #
                    ##################

                    if hessian:
                        bending_loss = jnp.trace(_whiten(L, cov.rho_bend))

                        loss_main += hp.additive_mu * bending_loss
                    else:
                        bending_loss = 0.0

                    ##################
                    # telemetry only #
                    ##################

                    mu_sg = jnp.linalg.eigvalsh(rho_res)  # eigenvalues only; R_sg already detached
                    lam_eta_inv_sg = jnp.where(mu_sg < hp.rate_floor, jnp.inf, 1.0 / mu_sg - 1.0)
                    lam_sg = lam_eta_inv_sg * eta  # implied timescales (the output you need)
                    sat = bending_loss

                    ##################
                    # orhto loss     #
                    ##################

                    v = jnp.diag(cov.C00)  # σ_a² = variances

                    x, _ = jnp.linalg.eigh(cov.C00)  # decorrelate + rescale to σ_a = 1
                    loss_ortho = jnp.sum(jnp.log(x) ** 2) + jnp.sum((cov.rho_00 - jnp.eye(N)) ** 2) / (
                        N**2
                    )  # pushes σ_a → 1 and ρ₀₀ → I

                    # pushes eigenvalues of C00 → 1 (decorrelation + scale)
                    # loss_scale = jnp.sum(jnp.log(v) ** 2)  # pushes σ_a → 1  (scale gauge)
                    # loss_decorr = jnp.sum((cov.rho_00 - jnp.eye(N))**2)    # pushes ρ₀₀ → I  (decorrelation)

                    # loss_ortho = jnp.sum((cov.C00 - jnp.eye(N)) ** 2)

                    loss = loss_main + hp.ortho_weight * loss_ortho

                    eta_next = hp.eta * jnp.min(lam_sg)

                    a = 0.05
                    ema_lambda_min = a * eta_next + (1 - a) * eta
                    sigma_next = a * v**0.5 + (1 - a) * sigma_0 if sigma_0 is not None else v**0.5

                    timescales = 1.0 / (
                        lam_sg
                        / (beta_timecon if beta_timecon is not None else 1 / (300 * boltzmann))
                        * (tau_thermo if tau_thermo is not None else 100 * femtosecond)
                        # / amu
                    )

                    aux = dict(
                        timescales=timescales,
                        resolvent_weights=mu_sg,
                        sat=sat,
                        loss_main=loss_main,
                        loss_ortho=loss_ortho,
                        eta_next=ema_lambda_min,
                        sigma_next=sigma_next,
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
                    with jnp.printoptions(formatter={"float": "{:0.3e}".format}):
                        print(
                            f"epoch={epoch} | timescales (ns)={output['timescales'][::-1] / nanosecond} | "
                            f"sat={output['sat']:.3f} | loss_main={output['loss_main']:.3E} | "
                            f"ortho={output['loss_ortho']:.3E} | eta_next={output['eta_next']:.3E} | "
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
        )

        argmask = cov.mask(eps_pre, max_features_pre, auto_cov_threshold)

        if not generator:
            # first mask before shrinking
            if shrink:
                cov = cov.shrink()

        x = cov.decompose(
            out_dim,
            eps=eps,
            out_eps=out_eps,
            glasso_whiten=glasso,
            verbose=verbose,
            scale_gen=eta
            / (beta_timecon if beta_timecon is not None else 1 / (300 * boltzmann))
            * (tau_thermo if tau_thermo is not None else 100 * femtosecond)
            # * amu
            if variational
            else None,
        )

        #    / (beta_timecon if beta_timecon is not None else 1 / (300 * boltzmann))
        #                 * (tau_thermo if tau_thermo is not None else 100 * picosecond)

        if only_diag:
            argmask_s, s = x
            W0 = None
            W1 = None

            argmask = argmask
        else:
            W0, W1, s = x

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
            # periodicities=periodicities,
            argmask_s=argmask_s if only_diag else None,
            entropy_reg=entropy_reg,
            target_smoothness=target_smoothness,
            alpha_smooth=alpha_smooth,
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
        if self.only_diag:
            argmask_s = self.argmask_s

        else:
            # o = self.W0.T
            o = jnp.diag(self.cov.sigma_0_inv) @ self.W0.T

        s = self.s

        n_skip = self.get_n_skip(
            n_skip=n_skip,
            remove_constant=remove_constant,
        )

        s = s[n_skip:]

        if self.only_diag:
            argmask_s = argmask_s[n_skip:]
        else:
            o = o[:, n_skip:]

        if mask is not None:
            s = s[mask]
            if self.only_diag:
                argmask_s = argmask_s[mask]
            else:
                o = o[:, mask]

        if self.only_diag:
            argmask_s = argmask_s[:out_dim]
        else:
            o = o[:, :out_dim]

        tr = CvTrans.from_cv_function(
            KoopmanModel._transform,
            static_argnames=["add_1", "add_1_pre"],
            add_1=False,
            add_1_pre=False,
            q=o if not self.only_diag else None,
            pi=self.cov.pi_0[argmask_s] if (self.only_diag and self.cov.pi_0 is not None) else self.cov.pi_0,
            argmask=self.argmask if not self.only_diag else self.argmask[argmask_s],
        )

        if self.trans is not None:
            tr = self.trans * tr

        if self.cov.C00.dtype == jnp.complex_ is not None and inv_exp:
            print("applying periodicities to f")

            _real = CvTrans.from_cv_function(PeriodicKoopmanModel._inv_exp_periodic)

            tr *= _real

        return tr, None

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


class PeriodicKoopmanModel(MyPyTreeNode):
    km_periodic: KoopmanModel
    km_nonperiodic: KoopmanModel
    periodicities: jax.Array
    trans_periodic: CvTrans
    trans_nonperiodic: CvTrans
    trans: CvTrans | None = None

    @staticmethod
    def _reorder(cv: CV, _nl, shmap, shmap_kwargs, idx):
        return cv.replace(cv=cv.cv.at[idx].set(cv.cv))

    @staticmethod
    def _center_periodic(cv: CV, _nl, shmap, shmap_kwargs, periodicities, angles):
        return cv.replace(
            cv=jnp.mod(cv.cv - angles + jnp.pi, 2 * jnp.pi) - jnp.pi,  # center around angles
        )

    @staticmethod
    def _exp_periodic(cv: CV, _nl, shmap, shmap_kwargs, periodicities=None):
        if periodicities is None:
            return cv.replace(cv=jnp.exp(1.0j * cv.cv))

        return cv.replace(
            cv=jnp.where(
                periodicities,
                jnp.exp(1.0j * cv.cv),
                cv.cv,
            )
        )

    @staticmethod
    def _inv_exp_periodic(cv: CV, _nl, shmap, shmap_kwargs, periodicities=None):
        return cv.replace(cv=jnp.real(cv.cv))

        # if periodicities is None:
        #     return cv.replace(cv=jnp.angle(cv.cv))

        # return cv.replace(
        #     cv=jnp.where(
        #         periodicities,
        #         jnp.real(cv.cv),
        #         cv.cv,
        #     )
        # )

    def _get_s_ext(
        self,
        remove_constant=True,
        n_skip: int | None = None,
    ):
        s_ext_per = jnp.vstack([self.km_periodic.s, jnp.full_like(self.km_periodic.s, 0)])

        skip_per = self.km_periodic.get_n_skip(
            n_skip=n_skip,
            remove_constant=remove_constant,
        )

        s_ext_nonper = jnp.vstack([self.km_nonperiodic.s, jnp.full_like(self.km_nonperiodic.s, 1)])

        n_skip_nonper = self.km_nonperiodic.get_n_skip(
            n_skip=n_skip,
            remove_constant=remove_constant,
        )

        s_ext = jnp.hstack([s_ext_per[:, skip_per:], s_ext_nonper[:, n_skip_nonper:]])

        s_ext = jnp.concatenate(  # type: ignore
            [
                s_ext,
                jnp.arange(s_ext.shape[1]).reshape((1, -1)),
            ],
            axis=0,
        )

        # print(f"{s_ext=}")

        s_ext = s_ext[:, jnp.argsort(s_ext[0, :], descending=True)]

        return s_ext

    def f(
        self,
        out_dim=None,
        remove_constant=True,
        n_skip: int | None = None,
        mask: jax.Array | None = None,
        inv_exp=True,
    ):
        s_ext = self._get_s_ext(
            remove_constant=remove_constant,
            n_skip=n_skip,
        )

        # print(f"{s_ext=}")

        o = s_ext[:, :out_dim]

        n_per = jnp.sum(o[1, :] == 0)
        n_nonper = jnp.sum(o[1, :] == 1)

        # print(f"{n_per=} {n_nonper=}")

        tr_per, _ = self.km_periodic.f(
            out_dim=n_per,
            remove_constant=remove_constant,
            n_skip=n_skip,
            mask=mask,
            inv_exp=inv_exp,
        )

        tr_nonper, _ = self.km_nonperiodic.f(
            out_dim=n_nonper,
            remove_constant=remove_constant,
            n_skip=n_skip,
            mask=mask,
            inv_exp=inv_exp,
        )

        idx = jnp.hstack(
            [
                jnp.argwhere(o[1, :] == 0, size=int(n_per), fill_value=-1).reshape(-1),
                jnp.argwhere(o[1, :] == 1, size=int(n_nonper), fill_value=-1).reshape(-1),
            ]
        )

        print(f"{idx=} {n_per=} {n_nonper=} ")

        out_tr = (self.trans_periodic * tr_per + self.trans_nonperiodic * tr_nonper) * CvTrans.from_cv_function(
            PeriodicKoopmanModel._reorder,
            idx=idx,
        )

        if self.trans is not None:
            out_tr = self.trans * out_tr

        return out_tr, o[1, :] == 0

    def timescales(
        self,
        remove_constant=True,
        n_skip: int | None = None,
    ):
        s_ext = self._get_s_ext(
            remove_constant=remove_constant,
            n_skip=n_skip,
        )

        return -self.km_periodic.tau / jnp.log(s_ext[0, :])

    def weighted_model(
        self,
        chunk_size=None,
        macro_chunk=1000,
        out_dim=None,
        **kwargs,
    ) -> PeriodicKoopmanModel:
        km_per = self.km_periodic.weighted_model(
            chunk_size=chunk_size,
            macro_chunk=macro_chunk,
            out_dim=out_dim,
            **kwargs,
        )

        # km_nonper = self.km_nonperiodic.weighted_model(
        #     chunk_size=chunk_size,
        #     macro_chunk=macro_chunk,
        #     out_dim=out_dim,
        #     **kwargs,
        # )

        return PeriodicKoopmanModel(
            km_periodic=km_per,
            km_nonperiodic=self.km_nonperiodic,
            periodicities=self.periodicities,
            trans_periodic=self.trans_periodic,
            trans_nonperiodic=self.trans_nonperiodic,
            trans=self.trans,
        )
