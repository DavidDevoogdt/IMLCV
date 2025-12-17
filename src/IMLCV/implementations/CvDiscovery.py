from functools import partial

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
from jax import Array, jit, random

from IMLCV.base.CV import CV, CvTrans, NeighbourList, SystemParams, CvMetric, CvTrans
from IMLCV.base.CVDiscovery import Transformer
from IMLCV.base.datastructures import jit_decorator, vmap_decorator
from IMLCV.base.rounds import Covariances, DataLoaderOutput
from IMLCV.base.UnitsConstants import nanosecond
from IMLCV.implementations.CV import trunc_svd, un_atomize


class Encoder(nn.Module):
    latents: int
    layers: int
    nunits: int
    dim: int

    @nn.compact
    def __call__(self, x):
        for i in range(self.layers):
            x = nn.Dense(
                self.nunits,
                name=f"encoder_{i}",
                # bias_init=jax.nn.initializers.normal(),
                kernel_init=jax.nn.initializers.xavier_normal(),
            )(x)
            x = nn.tanh(x)
        mean_x = nn.Dense(
            self.latents,
            name="fc2_mean",
            # bias_init=jax.nn.initializers.normal(),
            # kernel_init=jax.nn.initializers.normal(),
        )(x)
        logvar_x = nn.Dense(
            self.latents,
            name="fc2_logvar",
            # bias_init=jax.nn.initializers.normal(),
            # kernel_init=jax.nn.initializers.normal(),
        )(x)
        return mean_x, logvar_x


class Decoder(nn.Module):
    latents: int
    layers: int
    nunits: int
    dim: int

    @nn.compact
    def __call__(self, z):
        for i in range(self.layers):
            z = nn.Dense(
                self.nunits,
                name=f"decoder_{i}",
                # bias_init=jax.nn.initializers.normal(),
                kernel_init=jax.nn.initializers.xavier_normal(),
            )(z)
            z = nn.tanh(z)
        z = nn.Dense(
            self.dim,
            name="fc2",
            # bias_init=jax.nn.initializers.normal(),
            kernel_init=jax.nn.initializers.normal(),
        )(z)
        return z


class VAE(nn.Module):
    latents: int
    layers: int
    nunits: int
    dim: int

    def setup(self):
        self.encoder = Encoder(self.latents, self.layers, self.nunits, self.dim)
        self.decoder = Decoder(self.latents, self.layers, self.nunits, self.dim)

    def __call__(self, x, z_rng):
        mean, logvar = self.encoder(x)
        z = VAE.reparameterize(z_rng, mean, logvar)
        recon_x = self.decoder(z)
        return recon_x, mean, logvar

    # def generate(self, z):
    #     return nn.sigmoid(self.decoder(z))

    def encode(self, x):
        mean, _ = self.encoder(x)
        return mean

    @classmethod
    def reparameterize(cls, rng, mean, logvar):
        std = jnp.exp(0.5 * logvar)
        eps = random.normal(rng, logvar.shape)
        return mean + eps * std


class TranformerAutoEncoder(Transformer):
    nunits: int = 250
    nlayers: int = 3
    lr: float = 1e-4
    num_epochs: int = 100
    batch_size: int = 32

    def _fit(
        self,
        cv: list[CV],
        cv_t: list[CV],
        w: list[jax.Array],
        dlo: DataLoaderOutput,
        chunk_size=None,
        verbose=True,
        macro_chunk=1000,
    ):
        # import wandb

        # wandb.init(
        #     project="TranformerAutoEncode",
        #     config={
        #         "cv": cv,
        #         "nunits": nunits,
        #         "nlayers": nlayers,
        #         "lr": lr,
        #         "num_epochs": num_epochs,
        #         "batch_size": batch_size,
        #         "outdim": self.outdim,
        #         **self.fit_kwargs,
        #         **kwargs,
        #     },
        # )

        _cv = CV.stack(*cv)
        _cv = un_atomize.compute_cv(_cv, None)[0]

        _cv_t = CV.stack(*cv_t)
        _cv_t = un_atomize.compute_cv(_cv_t, None)[0]

        rng = random.PRNGKey(0)

        dim = _cv.shape[1]

        vae_args = {
            "latents": self.outdim,
            "layers": self.nlayers,
            "nunits": self.nunits,
            "dim": dim,
        }

        @vmap_decorator
        def kl_divergence(mean, logvar):
            return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))

        @vmap_decorator
        def mean_Squared_error(x1, x2):
            return 0.5 * jnp.linalg.norm(x1 - x2) ** 2

        def compute_metrics(recon_x, x, mean, logvar):
            bce_loss = mean_Squared_error(recon_x, x).mean()
            kld_loss = 0.01 * kl_divergence(mean, logvar).mean()
            return {"bce": bce_loss, "kld": kld_loss, "loss": bce_loss + kld_loss}

        # @vmap_decorator
        # def binary_cross_entropy_with_logits(logits, labels):
        #     logits = nn.log_sigmoid(logits)
        #     return -jnp.sum(
        #         labels * logits + (1.0 - labels) * jnp.log(-jnp.expm1(logits))
        #     )

        import optax

        @jit_decorator
        def train_step(state: train_state.TrainState, batch, z_rng) -> train_state.TrainState:
            def loss_fn(params):
                recon_x, mean, logvar = VAE(**vae_args).apply(  # type:ignore
                    {"params": params},
                    batch,
                    z_rng,
                )

                bce_loss = mean_Squared_error(recon_x, batch).mean()
                # bce_loss = mean_Squared_error(recon_x, batch)
                kld_loss = kl_divergence(mean, logvar).mean()
                loss = bce_loss + kld_loss
                return loss

            grads = jax.grad(loss_fn)(state.params)  # type:ignore

            return state.apply_gradients(grads=grads)  # type:ignore

        @jit_decorator
        def eval(params, x, z_rng):
            def eval_model(vae):
                recon_x, mean, logvar = vae(x, z_rng)
                comparison = jnp.stack([x, recon_x])

                metrics = compute_metrics(recon_x, x, mean, logvar)
                return metrics, comparison

            return nn.apply(eval_model, VAE(**vae_args))({"params": params})  # type:ignore

        # Test encoder implementation
        # Random key for initialization
        key, rng = jax.random.split(rng, 2)

        init_data = jax.random.normal(key, (self.batch_size, dim), jnp.float32)

        key, rng = jax.random.split(rng, 2)

        state = train_state.TrainState.create(
            apply_fn=VAE(**vae_args).apply,  # type:ignore
            params=VAE(**vae_args).init(key, init_data, rng)["params"],  # type:ignore
            tx=optax.adam(self.lr),
        )

        rng, key, eval_rng = random.split(rng, 3)

        x = _cv.cv
        x = random.permutation(key, x)

        split = x.shape[0] // 10

        x_train = x[0:-split, :]
        x_test = x[-split:, :]

        steps_per_epoch = x_train.shape[0] // self.batch_size

        for epoch in range(self.num_epochs):
            rng, key = random.split(rng)
            indices = jax.random.choice(
                key=key,
                a=int(x.shape[0]),
                shape=(steps_per_epoch, self.batch_size),
                replace=False,
            )

            for n, index in enumerate(indices):
                rng, key = random.split(rng)
                state = train_step(state, x_train[index, :], key)

                if n % 5000:
                    metrics, comparison = eval(state.params, x_test, eval_rng)
                    # wandb.log(
                    #     {
                    #         "loss": metrics["loss"],
                    #         "kld": metrics["kld"],
                    #         "distance": metrics["bce"],
                    #     }
                    # )

                    print(
                        "eval epoch: {}, loss: {:.4f}, BCE: {:.4f}, KLD: {:.4f}".format(
                            epoch + 1,
                            metrics["loss"],
                            metrics["bce"],
                            metrics["kld"],
                        ),
                    )

            # @partial( jit_decorator, static_argnums=(0,))

        def forward(x: CV, nl, y: list[CV] | None = None, _=None, smap=False):
            assert y is None
            encoded: Array = VAE(**vae_args).apply(  # type:ignore
                {"params": state.params},
                x.cv,
                method=VAE.encode,
            )
            return x.replace(cv=encoded)

        f_enc = CvTrans.from_cv_function(f=forward)

        cv = f_enc.compute_cv(_cv)[0].unstack()
        if cv_t is not None:
            cv_t = f_enc.compute_cv(_cv_t)[0].unstack()

        return cv, cv_t, un_atomize * f_enc, w, None, None


def _LDA_trans(cv: CV, nl: NeighbourList | None, shmap, shmap_kwargs, alpha, outdim, solver):
    if solver == "eigen":

        def f1(cv, scalings):
            return cv @ scalings

        _f = partial(f1, scalings=jnp.array(alpha.scalings_)[:, :outdim])

    elif solver == "svd":

        def f2(cv, scalings, xbar):
            return (cv - xbar) @ scalings

        _f = partial(
            f2,
            scalings=jnp.array(alpha.scalings_),
            xbar=jnp.array(alpha.xbar_),
        )

    else:
        raise NotImplementedError

    return CV(
        cv=_f(cv.cv),
        _stack_dims=cv._stack_dims,
        _combine_dims=cv._combine_dims,
        atomic=cv.atomic,
        mapped=cv.mapped,
    )


def _LDA_rescale(cv: CV, nl: NeighbourList | None, shmap, shmap_kwargs, mean):
    return CV(
        cv=(cv.cv - mean[0]) / (mean[1] - mean[0]),
        _stack_dims=cv._stack_dims,
        _combine_dims=cv._combine_dims,
        atomic=cv.atomic,
        mapped=cv.mapped,
    )


def _scale_trans(cv: CV, nl: NeighbourList | None, shmap, shmap_kwargs, alpha: jax.Array, scale_factor: jax.Array):
    return CV(
        cv=(alpha.T @ cv.cv - scale_factor[0, :]) / (scale_factor[1, :] - scale_factor[0, :]),
        _stack_dims=cv._stack_dims,
        _combine_dims=cv._combine_dims,
        atomic=cv.atomic,
        mapped=cv.mapped,
    )


class TransoformerLDA(Transformer):
    kernel = False
    optimizer = None
    solver: str = "eigen"
    method: str = "pymanopt"
    harmonic = True
    min_gradient_norm: float = 1e-3
    min_step_size: float = 1e-3
    max_iterations: int = 25

    def _fit(
        self,
        cv_list: list[CV],
        cv_t_list: list[CV],
        w: list[jax.Array],
        dlo: DataLoaderOutput,
        chunk_size=None,
        verbose=True,
        macro_chunk=1000,
    ):
        # nl_list = dlo.nl

        if self.kernel:
            raise NotImplementedError("kernel not implemented for lda")

        cv = CV.stack(*cv_list)
        cv, _ = un_atomize.compute_cv(cv)

        cv_t = CV.stack(*cv_t_list)
        cv_t, _ = un_atomize.compute_cv(cv_t)

        if self.method == "sklearn":
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

            labels = []
            for i, cvi in enumerate(cv_list):
                labels.append(jnp.full(cvi.shape[0], i))

            labels = jnp.hstack(labels)

            alpha = LDA(n_components=self.outdim, solver=self.solver, shrinkage="auto").fit(  # type:ignore
                cv.cv.__array__(),
                labels,
            )

            lda_cv = CvTrans.from_cv_function(_LDA_trans, alpha=alpha, outdim=self.outdim, solver=self.solver)
            cv, _ = lda_cv.compute_cv(cv)

            cvs = CV.unstack(cv)

            mean = []
            for i, cvs_i in enumerate(cvs):
                mean.append(jnp.mean(cvs_i.cv, axis=0))
            mean = jnp.array(mean)

            assert self.outdim == 1

            lda_rescale = CvTrans.from_cv_function(_LDA_rescale, mean=mean)
            cv, __ = lda_rescale.compute_cv(cv)

            full_trans = un_atomize * lda_cv * lda_rescale

        elif self.method == "pymanopt":
            import pymanopt  # type:ignore

            assert isinstance(cv_list, list)
            if self.optimizer is None:
                optimizer = pymanopt.optimizers.TrustRegions(
                    max_iterations=self.max_iterations,
                    min_gradient_norm=self.min_gradient_norm,
                    min_step_size=self.min_step_size,
                )

            cv, _f = trunc_svd(cv)
            normed_cv_u = CV.unstack(cv)

            mu_i = [jnp.mean(x.cv, axis=0) for x in normed_cv_u]
            mu = jnp.einsum(
                "i,ij->j",
                jnp.array(cv.stack_dims) / jnp.sum(jnp.array(cv.stack_dims)),
                jnp.array(mu_i),
            )

            obersevations_within = CV.stack(*[cv_i - mu_i for mu_i, cv_i in zip(mu_i, normed_cv_u)])
            observations_between = CV.stack(*[cv_i - mu for cv_i in normed_cv_u])

            from sklearn.covariance import LedoitWolf

            cov_w = LedoitWolf(assume_centered=True).fit(obersevations_within.cv.__array__()).covariance_
            cov_b = LedoitWolf(assume_centered=True).fit(observations_between.cv.__array__()).covariance_

            manifold = pymanopt.manifolds.stiefel.Stiefel(n=mu.shape[0], p=self.outdim)

            @pymanopt.function.jax(manifold)
            @jit
            def cost(x):
                a = jnp.trace(x.T @ cov_w @ x)
                b = jnp.trace(x.T @ cov_b @ x)

                if self.harmonic:
                    out = a / b
                else:
                    out = -(b / a)

                return out

            problem = pymanopt.Problem(manifold, cost)
            result = optimizer.run(problem)

            alpha = result.point

            scale_factor = vmap_decorator(lambda x: alpha.T @ x)(jnp.array(mu_i))

            _g = CvTrans.from_cv_function(_scale_trans, alpha=alpha, scale_factor=scale_factor)

            cv, _ = _g.compute_cv(cv)
            cv_t, _ = _g.compute_cv(cv_t)

            full_trans = un_atomize * _f * _g

        return cv.unstack(), cv_t.unstack(), full_trans, w, None, None


class TransformerMAF(Transformer):
    # Maximum Autocorrelation Factors
    outdim: int
    eps: float = 1e-5
    eps_pre: float = 1e-5
    max_features: int = 500
    max_features_pre: int = 500

    sym: bool = True
    use_w: bool = True

    min_t_frac: float = 0.1
    max_t_cutoff = 1 * nanosecond
    periodicities: jax.Array | None = None

    add_1: bool = True

    trans: CvTrans | None = None
    T_scale: float = 1.0

    disciminating_CVs: CV | None = None

    generator: bool = False

    shrink: bool = True

    @staticmethod
    def _transform(
        cv,
        nl,
        shmap,
        shmap_kwargs,
        argmask: jax.Array | None = None,
        pi: jax.Array | None = None,
        W: jax.Array | None = None,
    ) -> jax.Array:
        x = cv.cv

        # print(f"inside {x.shape=} {q=} {argmask=} ")

        if argmask is not None:
            x = x[argmask]

        if pi is not None:
            x = x - pi

        if W is not None:
            x = x @ W

        return cv.replace(cv=x, _combine_dims=None)

    def _fit(
        self,
        x: list[CV] | list[SystemParams],
        x_t: list[CV] | list[SystemParams] | None,
        w: list[jax.Array],
        # w_t: list[jax.Array],
        dlo: DataLoaderOutput,
        macro_chunk=1000,
        chunk_size=None,
        verbose=True,
        trans: CvTrans | None = None,
    ):
        print("getting koopman")

        outdim = self.outdim

        # def tot_w(w, rho):
        #     w_log = [(jnp.log(wi) + jnp.log(rhoi)) / self.T_scale for wi, rhoi in zip(w, rho)]

        #     z = jnp.hstack(w_log)
        #     z_max = jnp.max(z)
        #     norm = jnp.log(jnp.sum(jnp.exp(z - z_max))) + z_max

        #     w_tot = [jnp.exp(w_log_i - norm) for w_log_i in w_log]

        #     s = 0
        #     for wi in w_tot:
        #         s += jnp.sum(wi)

        #     print(f"{s=}")

        #     return w_tot

        # w_tot = tot_w(w, dlo._rho)
        # # w_tot_t = tot_w(w_t, rho_t)

        # # print(f" {calc_pi=}")

        # cov = Covariances.create(
        #     cv_0=x,  # type: ignore
        #     cv_1=x,  # type: ignore
        #     nl=dlo.nl,
        #     nl_t=dlo.nl_t,
        #     w=w_tot,
        #     calc_pi=True,
        #     # only_diag=only_diag,
        #     symmetric=False,
        #     chunk_size=chunk_size,
        #     macro_chunk=macro_chunk,
        #     trans_f=self.trans,
        #     trans_g=self.trans,
        #     verbose=verbose,
        #     shrink=False,
        #     # shrinkage_method=shrinkage_method,
        # )

        # argmask = cov.mask(
        #     eps_pre=self.eps_pre,
        #     max_features=self.max_features,
        #     auto_cov_threshold=0.1,
        # )

        # W, _ = cov.decompose_pymanopt(out_dim=outdim)

        # trans_km = CvTrans.from_cv_function(
        #     TransformerMAF._transform,
        #     W=W,
        #     pi=cov.pi_0,
        #     argmask=argmask,
        # )

        trans_tot = trans
        if self.trans is not None:
            if trans_tot is None:
                trans_tot = self.trans
            else:
                trans_tot *= self.trans

        if self.generator:
            km = dlo.koopman_model(
                cv_0=x,
                cv_t=x_t,
                nl=dlo.nl,
                nl_t=dlo.nl_t,
                w=w if self.use_w else [jnp.ones_like(x) for x in w],
                rho=dlo._rho if self.use_w else [jnp.ones_like(x) for x in w],
                chunk_size=chunk_size,
                macro_chunk=macro_chunk,
                calc_pi=True,
                add_1=True,
                eps_pre=self.eps_pre,
                eps=self.eps,
                symmetric=True,
                trans=trans_tot,
                verbose=True,
                # auto_cov_threshold=0.1,
                max_features=self.max_features,
                max_features_pre=self.max_features_pre,
                T_scale=self.T_scale,
                shrink=True,
                generator=True,
                periodicities=self.periodicities,
                # shrinkage_method="BC",
            )

        else:
            km = dlo.koopman_model(
                cv_0=x,
                cv_t=x_t,
                nl=dlo.nl,
                nl_t=dlo.nl_t,
                w=w if self.use_w else [jnp.ones_like(x) for x in w],
                rho=dlo._rho if self.use_w else [jnp.ones_like(x) for x in w],
                chunk_size=chunk_size,
                macro_chunk=macro_chunk,
                calc_pi=True,
                add_1=self.add_1,
                eps_pre=self.eps_pre,
                eps=self.eps,
                symmetric=False,
                trans=trans_tot,
                verbose=True,
                # auto_cov_threshold=0.1,
                max_features=self.max_features,
                max_features_pre=self.max_features_pre,
                T_scale=self.T_scale,
                shrink=False,
                periodicities=self.periodicities,
                # shrinkage_method="BC",
            )

            if self.sym:
                km = km.weighted_model(
                    symmetric=True,
                    shrink=True,
                    add_1=self.add_1,
                    # T_scale=self.T_scale,
                )

        # assert km.w is not None
        # w = km.w

        #########

        # trans_km = km.f_stiefel(
        #     out_dim=outdim,
        #     # n_skip=None,
        #     # remove_constant=True,
        #     # mask=b,
        # )

        # if self.disciminating_CVs is not None:
        #     cvs_d, _ = km.f(out_dim=None).compute_cv(self.disciminating_CVs)

        #     @partial(vmap_decorator, in_axes=(0, None))
        #     @partial(vmap_decorator, in_axes=(None, 0))
        #     def close(x1, x2):
        #         return jnp.abs(x1 - x2)

        #     dist = jax.vmap(lambda x: x[jnp.triu_indices_from(x, k=1)], in_axes=(2))(close(cvs_d.cv, cvs_d.cv))

        #     b = (dist > 0.1).any(axis=1)

        #     print(f"close results {dist} {b=}  {cvs_d} ")
        # else:
        #     b = None

        ts = (
            km.timescales(
                n_skip=None,
                remove_constant=True,
                # mask=b,
            )
            / nanosecond
        )

        print(f"timescales: {ts[: jnp.min(jnp.array([10, ts.shape[0]]))]} ns")

        for i in range(self.outdim):
            if (ts[i] / ts[0] < self.min_t_frac) and (ts[i] < self.max_t_cutoff / nanosecond):
                print(
                    f"cv {i} is too small compared to ref (fraction= {ts[i] / ts[0]}, {self.min_t_frac=}), cutting off "
                )
                outdim = i
                break

        exta_info = [f"{ts[i]:.4f}" for i in range(self.outdim)]

        trans_km, per = km.f(
            out_dim=outdim,
            n_skip=None,
            remove_constant=True,
            # mask=b,
        )

        print("applying transformation")

        x, x_t = dlo.apply_cv(
            trans_km,
            x=x,
            x_t=x_t,
            nl=dlo.nl,
            nl_t=dlo.nl_t,
            macro_chunk=macro_chunk,
            chunk_size=chunk_size,
            verbose=True,
        )

        return (x, x_t, trans_km, w, exta_info, per)


# def _smoothstep(u: jnp.ndarray) -> jnp.ndarray:
#     u = jnp.clip(u, 0.0, 1.0)
#     return 3.0 * u**2 - 2.0 * u**3


# def _soft_triangle(
#     x_in: jax.Array,
#     center: jax.Array,
#     width: jax.Array,
#     smooth: bool = True,
#     periodic: bool = False,
#     order: int = 1,
# ) -> jax.Array:
#     d = jnp.where(
#         periodic,
#         jnp.mod(x_in - center + jnp.pi, 2 * jnp.pi) - jnp.pi,
#         x_in - center,
#     )

#     t = jnp.clip(1.0 - jnp.abs(d) / (width + 1e-12), 0.0, 1.0)
#     return t ** jnp.arange(order) * (_smoothstep(t) if smooth else t)


def _forward_apply_rule(
    kwargs,
    static_kwargs,
):
    print(f"apply rule linear layer")

    weights = kwargs["weights"]
    min_val = kwargs.get("min_val", 0.1)
    max_val = kwargs.get("max_val", 1.0)

    weights = (jnp.tanh(weights) + 1) / 2 * (max_val - min_val) + min_val

    # jax.debug.print(
    #     "w in boudns {}",
    #     jnp.all(
    #         jnp.logical_and(
    #             weights >= min_val,
    #             weights <= max_val,
    #         )
    #     ),
    # )

    kwargs["weights"] = weights

    return kwargs


def _forward(
    cv: CV,
    nl: NeighbourList | None,
    shmap,
    shmap_kwargs,
    bounds: jax.Array,
    weights: jax.Array,
    M: jax.Array,
    periodicities: jax.Array | None = None,
    min_val: float = 0.1,
    max_val: float = 1.0,
    n: int = 8,
) -> CV:
    xvals = cv.cv

    # def smoothstep(x):
    #     return jnp.where(x <= 0, 0, jnp.where(x >= 1, 1, x**2 * (3 - 2 * x)))

    def smoothstep(x):
        # return jnp.where(x <= 0, 0, jnp.where(x >= 1, 1, x**2 * (3 - 2 * x)))
        return (jnp.tanh(x) + 1) / 2

    def smooth(x, center, width, periodic):
        d = jnp.where(periodic, (jnp.mod(x - center + jnp.pi, 2 * jnp.pi) - jnp.pi), (x - center))

        d = jnp.where(
            periodic,
            jnp.select(
                [jnp.abs(d) <= jnp.pi / 2, jnp.abs(d) > jnp.pi / 2],
                [d, jnp.sign(d) * (jnp.pi - jnp.abs(d))],
            ),
            d,
        )

        return smoothstep(d / width)

    @partial(jax.vmap, in_axes=[0, 0, 0, 0])
    def _get(bounds, cv, weights, periodicities):
        grid = jnp.linspace(bounds[0], bounds[1], num=n, endpoint=True)
        centers = 0.5 * (grid[1:] + grid[:-1])
        half_width = centers[1] - centers[0]

        return jnp.einsum(
            "w,w...->...",
            weights,
            jax.vmap(
                smooth,
                in_axes=(
                    None,
                    0,
                    None,
                    None,
                ),
            )(
                cv,
                centers,
                half_width,
                periodicities,
            ),
        )

    print(f"{bounds.shape=} {xvals.shape=} {weights.shape=} {periodicities=}")

    out = _get(bounds, xvals, weights, periodicities)

    print(f"{out.shape=} {M.shape=}")

    out = out @ M

    return CV(
        cv=out.reshape(-1),
        _stack_dims=cv._stack_dims,
    )


class IndicatorSplitterTransformer(Transformer):
    """Split a 1D collective variable into n smooth triangular indicator functions.

    The transformer assumes the input CV values are scalar (or uses the first
    dimension if multi-dimensional). It creates a CvTrans that maps each input
    sample to an n-dimensional vector of indicators using a smooth triangle
    (hat) function.

    Parameters:
      n: number of indicator functions (output dimension)
      lower: lower bound for centers (if None, determined from data)
      upper: upper bound for centers (if None, determined from data)
      smooth: whether to apply cubic smoothstep to the linear triangle
    """

    n: int = 8
    lower: float | None = None
    upper: float | None = None
    smooth: bool = True
    periodicities: jax.Array | None = None

    min_val: float = 0.1
    max_val: float = 1.0
    outdim: int = 2

    def _fit(
        self,
        x: list[CV] | list[SystemParams],
        x_t: list[CV] | list[SystemParams] | None,
        w: list[jax.Array],
        dlo: DataLoaderOutput,
        chunk_size: int | None = None,
        verbose=True,
        macro_chunk=1000,
    ) -> tuple[list[CV], list[CV] | None, CvTrans, list[jax.Array] | None, jax.Array | None]:
        # Expect list of CV objects
        assert isinstance(x, list) and len(x) > 0, "x must be a non-empty list of CV"
        assert isinstance(x[0], CV), "x must be a list of CV objects"

        bounds, _, _ = CvMetric.bounds_from_cv(x)

        if self.periodicities is None:
            per = jnp.full((len(bounds),), False)
        else:
            per = self.periodicities

        bounds = jax.vmap(
            lambda x, y: jnp.where(
                x,
                jnp.array([-jnp.pi, jnp.pi]),
                y,
            )
        )(per, jnp.array(bounds))

        print(f"{bounds=}")

        print(f"IndicatorSplitterTransformer: using bounds {bounds}")

        # create CvTrans from forward
        f_trans = CvTrans.from_cv_function(
            f=_forward,
            static_argnames=["n"],
            bounds=jnp.array(bounds),
            n=self.n,
            periodicities=per,
            learnable_argnames=["weights", "M"],
            apply_rule=_forward_apply_rule,
            min_val=self.min_val,
            max_val=self.max_val,
            weights=jnp.zeros(
                (
                    bounds.shape[0],
                    self.n - 1,
                )
            ),
            M=jnp.zeros((bounds.shape[0], self.outdim)),
        )

        print(f"{f_trans.get_learnable_params()=}")

        x, x_t = dlo.apply_cv(
            x=x,
            x_t=x_t,
            f=f_trans,
            verbose=verbose,
            chunk_size=chunk_size,
            macro_chunk=macro_chunk,
            shmap=False,
        )

        print(f"{x[0]=}")

        return x, x_t, f_trans, w, None, None
