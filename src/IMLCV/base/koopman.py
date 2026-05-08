from __future__ import annotations

import time
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from IMLCV.base.covariances import Covariances
from IMLCV.base.CV import CvTrans
from IMLCV.base.dataobjects import CV, NeighbourList, SystemParams, chunk_selector
from IMLCV.base.datastructures import MyPyTreeNode, Partial_decorator, jit_decorator, vmap_decorator
from IMLCV.base.UnitsConstants import boltzmann, kjmol, nanosecond


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
    nl: list[NeighbourList] | NeighbourList | None
    nl_t: list[NeighbourList] | NeighbourList | None

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

    @staticmethod
    def create(
        w: list[jax.Array] | None,
        rho: list[jax.Array] | None,
        w_t: list[jax.Array] | None,
        rho_t: list[jax.Array] | None,
        cv_0: list[CV] | list[SystemParams],
        cv_t: list[CV] | list[SystemParams] | None,
        dynamic_weights: list[jax.Array] | None = None,
        nl: list[NeighbourList] | NeighbourList | None = None,
        nl_t: list[NeighbourList] | NeighbourList | None = None,
        add_1=True,
        eps: float = 1e-14,
        eps_pre: float = 0,
        # method="tcca",
        symmetric=False,
        out_dim=-1,  # maximum dimension for koopman model
        max_features=5000,
        max_features_pre=5000,
        tau=None,
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
        print_nonlin_every=1,
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
    ):
        #  see Optimal Data-Driven Estimation of Generalized Markov State Models
        if verbose:
            print(f"getting covariancesm {generator=}")

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

            assert not generator

            tau = tau if tau is not None else 1.0

            init_steps = 100

            cv_out_shape, _ = jax.eval_shape(trans.compute_cv, cv_0[0], nl)
            dim = cv_out_shape.cv.shape[1]

            N = cv_out_shape.cv.shape[0]
            sigma_silverman = (4 / (N * (dim + 2))) ** (1 / (dim + 4))

            print(f"{N=} {dim=} {sigma_silverman=}")

            print(f"{cv_out_shape.cv.shape=} {entropy_reg=} {target_smoothness=} ")

            # if batch_size is None:
            grid_size = int(2000 ** (1 / dim))
            # else:
            #     grid_size = int(min(batch_size, 1000) ** (1 / dim))
            print(f"entropy grid size per dim: {grid_size}")

            print("Using AdamW optimizer for mini-batch optimization")

            # 1. Define a Schedule: Warmup + Cosine Decay
            # A CV needs to settle in before you hit it with high rates
            # total_steps = 1000  # Adjust based on your total epochs/batches
            # warmup_steps = max(epochs // 5, 100)  # Warmup for the first 20% of training or at least 100 steps

            lr_schedule = optax.warmup_cosine_decay_schedule(
                init_value=1e-2,
                peak_value=1e-1,
                warmup_steps=20,
                decay_steps=epochs,
                end_value=1e-2,
            )

            # 2. Add Gradient Clipping
            # This is the "shield" against those massive spikes in your plot
            optimizer = optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adamw(learning_rate=lr_schedule, weight_decay=1e-4),
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

                if nl is not None:
                    if isinstance(x0_chunk, SystemParams):
                        nl_chunk = nl.slow_update_nl(x0_chunk) if nl is not None else None

                        nl = nl.replace(update=nl_chunk.update)

                    else:
                        nl_chunk = nl
                else:
                    nl_chunk = None

                lbfgs = True

            else:

                def _get_batch(key):
                    subsample = True
                    key, subkey = jax.random.split(key)

                    w_subsample = w_tot_dyn if use_girsanov else w_tot

                    if subsample:
                        x0_chunk, xt_chunk, w_chunk, wt_chunk, wdyn_chunk, wtot_chunk = chunk_selector(
                            [
                                cv_0,
                                cv_t,
                                w_tot if use_w else None,
                                w_tot_t if use_w else None,
                                dynamic_weights if dynamic_weights is not None else None,
                                w_subsample,
                            ],
                            batch_size,
                            subkey,
                            weight_list=w_tot_dyn if use_girsanov else w_tot,
                        )

                        wt_chunk /= wtot_chunk
                        w_chunk /= wtot_chunk
                    else:
                        x0_chunk, xt_chunk, w_chunk, wt_chunk, wdyn_chunk = chunk_selector(
                            [
                                cv_0,
                                cv_t,
                                w_tot if use_w else None,
                                w_tot_t if use_w else None,
                                dynamic_weights if dynamic_weights is not None else None,
                            ],
                            batch_size,
                            subkey,
                        )

                        # print(f"batch weights sum {jnp.sum(w_chunk)=} {jnp.sum(wt_chunk)=}")

                        w_chunk /= jnp.sum(w_chunk)
                        wt_chunk /= jnp.sum(wt_chunk)

                    # if nl is not None:
                    #     if isinstance(x0_chunk, SystemParams):
                    #         nl_chunk = nl2.slow_update_nl(x0_chunk)
                    #         nl2 = nl2.replace(update=nl_chunk.update)
                    #     else:
                    #         nl_chunk = nl2

                    batch = dict(
                        x0_i=x0_chunk,
                        xt_i=xt_chunk,
                        w_i=w_chunk,
                        wt_i=wt_chunk,
                        wdyn_i=wdyn_chunk if dynamic_weights is not None else None,
                        # nl_i=nl_chunk,
                    )

                    return batch, key

                if prefetch:
                    from queue import Queue
                    from threading import Thread

                    def prefetch_worker(
                        queue,
                        key,
                    ):
                        # nl2 = nl

                        # if nl2 is not None:
                        #     nl2.info.r_skin = 0.0

                        # w_alpha = [wi**alpha_smooth for wi in w_tot] if use_w else None

                        while True:
                            batch, key = _get_batch(key)

                            print("-", end="", flush=True)

                            queue.put(batch)

                    lbfgs = False

                    # Setup queue and thread
                    batch_queue = Queue(maxsize=4)  # Prefetch 4 batches ahead
                    thread = Thread(
                        target=prefetch_worker,
                        args=(
                            batch_queue,
                            key,
                        ),
                    )
                    thread.daemon = True
                    thread.start()

                    def get_batch():
                        batch = batch_queue.get()
                        return key, (batch["x0_i"], batch["xt_i"], batch["w_i"], batch["wt_i"], batch["wdyn_i"])

                else:
                    lbfgs = False

                    def get_batch(key):
                        batch, key = _get_batch(key)
                        return key, (batch["x0_i"], batch["xt_i"], batch["w_i"], batch["wt_i"], batch["wdyn_i"])

            # compute effective diffusion

            @partial(jit_decorator, static_argnames=["entropy_reg", "chunk_size"])
            def objective_fn(
                params,
                log_beta_gamma,
                x_0: CV | SystemParams,
                x_t: CV | SystemParams,
                w: jax.Array | None,
                w_t: jax.Array | None,
                wdyn: jax.Array | None,
                nl: NeighbourList | None,
                # cov_tot: Covariances | None,
                entropy_reg,
                chunk_size=None,
                epoch: int = 0,
            ):
                learnable_params = params

                _trans = trans.apply_learnable_params(learnable_params)

                cov = Covariances.create(
                    cv_0=x_0,
                    cv_1=x_t,
                    nl=nl,
                    nl_t=nl,
                    w=[w],
                    w_t=[w_t],
                    dynamic_weights=[wdyn] if wdyn is not None else None,
                    trans_f=_trans,
                    trans_g=_trans,
                    shrink=False,
                    calc_pi=calc_pi,
                    get_diff=False,
                    pi_argmask=jnp.array([-1]) if add_1 else None,
                    symmetric=symmetric,
                    generator=True,
                    chunk_size=None,
                    macro_chunk=None,
                )

                # jax.lax.cond(epoch == 0, lambda: None, lambda: jax.debug.print("cov {cov}", cov=cov))

                cov_new = cov

                W0, lambda_i = cov_new.whiten_C(
                    "rho_00",
                    apply_mask=False,
                    epsilon=1e-12,
                    cholesky=False,
                    tikhonov=1e-2,
                    return_eigh=True,
                )

                loss_ortho = jnp.sum(lambda_i) - jnp.sum(jnp.log(lambda_i)) - lambda_i.shape[0]

                loss = 1e-1 * loss_ortho

                if symmetric:
                    W1 = W0

                else:
                    print(f"whitening C11 separately")

                    W1 = cov_new.whiten_C(
                        "rho_11",
                        apply_mask=False,
                        epsilon=1e-12,
                        cholesky=False,
                        tikhonov=1e-2,
                    )

                T = W1 @ cov_new.C01.T @ W0.T

                # if not symmetric:
                #     T = W0 @ cov_new.C11 @ W1.T @ T

                sigmas = jnp.linalg.svd(T, compute_uv=False)
                loss_vamp = -jnp.sum(sigmas**vamp_r) + dim

                timescales_vamp = -tau / jnp.log(sigmas)

                # https://proceedings.neurips.cc/paper_files/paper/2024/file/89edef87915d31de3437b6b2ac5f79e7-Paper-Conference.pdf

                dK = W0 @ cov_new.C_gen @ W0.T

                if beta_timecon is not None:
                    dK /= beta_timecon

                timescales_gen = 1 / (jnp.linalg.eigvalsh(dK))

                if calc_pi:
                    log_beta_gamma_new = jnp.log(timescales_vamp[0] / timescales_gen[0])
                else:
                    log_beta_gamma_new = jnp.log(timescales_vamp[1] / timescales_gen[1])

                loss_gen = -jnp.sum(jnp.exp(-vamp_r * tau / timescales_gen)) + dim

                # https://chemrxiv.org/doi/pdf/10.26434/chemrxiv-2024-8752d-v2?download=true&redirectToLatest=false
                # scaling constant to make independed of thermostat
                # loss_gen = jnp.diag(dK) / beta_timecon
                # loss_gen = jnp.sum(dK**2)

                mix = 0.1

                # softmax
                loss += loss_vamp

                return loss, (
                    (
                        loss_vamp,
                        loss_gen,
                        # loss_betagamma,
                        # norm_indep,
                        # loss_border,
                        loss_ortho,
                        timescales_vamp,
                        timescales_gen,
                    ),
                    # cov_new,
                    log_beta_gamma_new,
                )

            # @jax.jit(static_argnames=["chunk_size"])
            @partial(jit_decorator)
            def _body_fun_2(x):
                (
                    opt_state,
                    params,
                    log_beta_gamma,
                    (cv, cv_t, w, w_t, w_dyn, nl),
                    # cov,
                    _,
                    iter_num,
                ) = x

                w /= jnp.sum(w)

                if w_dyn is not None:
                    _wd = w * w_dyn
                else:
                    _wd = w
                ess = jnp.sum(_wd) ** 2 / jnp.sum(_wd**2)

                w_t /= jnp.sum(w_t)

                _objective_fn = Partial_decorator(
                    objective_fn,
                    log_beta_gamma=log_beta_gamma,
                    x_0=cv,
                    x_t=cv_t,
                    w=w,
                    w_t=w_t,
                    wdyn=w_dyn,
                    nl=nl,
                    entropy_reg=entropy_reg,
                    chunk_size=batch_chunk_size,
                    # cov_tot=cov,
                    epoch=iter_num,
                )

                (
                    (
                        loss_val,
                        (
                            losses,
                            # cov_new,
                            log_beta_gamma_new,
                        ),
                    ),
                    grads,
                ) = jax.value_and_grad(_objective_fn, has_aux=True)(params)

                # print(f"computed gradients with {grads=}")

                def without_aux(params):
                    loss_val, _ = _objective_fn(params)
                    return loss_val

                gnorm = optax.tree.norm(grads, ord=2)

                if lbfgs:
                    updates, opt_state = optimizer.update(
                        grads,
                        opt_state,
                        params,
                        value=loss_val,
                        grad=grads,
                        value_fn=without_aux,
                    )

                    zoomstate: ScaleByZoomLinesearchState = opt_state[-1]
                    steps = zoomstate.info.num_linesearch_steps

                    crit = jnp.logical_and(
                        gnorm > 1e-3,
                        steps < max_steps,
                    )

                else:
                    updates, opt_state = optimizer.update(
                        grads,
                        opt_state,
                        params,
                    )

                    # Adam state is usually the first element in the Optax state tuple
                    # adam_s: ScaleByAdamState = opt_state[0]

                    # # 1. Step Count
                    # steps = adam_s.count

                    # # 2. Update Magnitude (Average of the second moment)
                    # # This tells you how much "energy" is left in the updates
                    # nu_leaves = jax.tree_util.tree_leaves(adam_s.nu)
                    # avg_nu = jnp.mean(jnp.array([jnp.mean(l) for l in nu_leaves]))

                    # crit = avg_nu > 1e-7

                    crit = True

                    # Inside your loop:
                    # health = get_optimizer_health(opt_state)

                    steps = 0

                # eps = optimizer.hyperparams.eps if hasattr(optimizer, "hyperparams") else 1e-8

                params = optax.apply_updates(params, updates)

                b = jnp.logical_and(
                    jnp.logical_or(
                        crit,
                        iter_num < init_steps,  # warmup
                    ),
                    iter_num < epochs,
                )

                # loss_vamp, total_entropy, loss_border, loss_ortho = losses
                (
                    loss_vamp,
                    loss_gen,
                    loss_ortho,
                    timescales_vamp,
                    timescales_gen,
                ) = losses

                if not calc_pi:
                    timescales_vamp = timescales_vamp[1:]
                    timescales_gen = timescales_gen[1:]

                jax.debug.print(
                    "epoch={e} loss train {tl:.3E} loss vamp {v:.3E} loss_gen {loss_gen:.3E} timescales_vamp {timescales_vamp} ns timescales_gen {timescales_gen} ns ortho {o:.3E} gnorm {g:.3E} scaling vamp gen {scaling:.3E} ess {ess:.3E}",
                    e=iter_num,
                    tl=loss_val,
                    v=loss_vamp,
                    loss_gen=loss_gen,
                    g=gnorm,
                    o=loss_ortho,
                    timescales_vamp=timescales_vamp / nanosecond,
                    timescales_gen=timescales_gen / nanosecond,
                    scaling=jnp.exp(log_beta_gamma),
                    ess=ess,
                )

                return (
                    opt_state,
                    params,
                    log_beta_gamma,
                    # cov_new,
                    b,
                    iter_num + 1,
                    log_beta_gamma_new,
                )

            @jax.jit
            def _cond_fun(x):
                opt_state, params, cov, b, iter_num = x

                return b

            cov: Covariances | None = None

            print(f"{init_steps=}, {epochs=}, {entropy_reg=}")

            t0 = time.time()

            print(f"{nl=}")

            log_beta_gamma = 0

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
                    log_beta_gamma,
                    # cov,
                    b,
                    iters,
                    log_beta_gamma_new,
                ) = _body_fun_2(
                    (
                        opt_state,
                        learable_params,
                        log_beta_gamma,
                        (x0_chunk, xt_chunk, w_chunk, wt_chunk, wdyn_chunk, nl),
                        # cov,
                        True,
                        epoch,
                    ),
                )

                log_beta_gamma = log_beta_gamma_new * 0.1 + log_beta_gamma * 0.9

                if not b:
                    print(f"Convergence criteria met at epoch {epoch}, stopping optimization")
                    break

            if batch_size is not None:
                if prefetch:
                    thread.join(timeout=0)

            trans = trans.apply_learnable_params(learable_params)
            print(f"optimized trans done ")

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
            macro_chunk=macro_chunk,
            trans_f=trans,
            trans_g=trans,
            verbose=verbose,
            shrink=False,
            shrinkage_method=shrinkage_method,
            pi_argmask=jnp.array([-1]) if add_1 else None,
            eps_shrink=eps_shrink,
            generator=generator,
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
            # generator=generator,
        )

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
            print(f"{s[0:min(10, s.shape[0]) ]=}")

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
            DataLoaderOutput._transform,
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
            o = self.W0.T  # jnp.diag(self.cov.sigma_0_inv) @ self.W0.T

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
            DataLoaderOutput._transform,
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
            DataLoaderOutput._transform,
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
            DataLoaderOutput._transform,
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
