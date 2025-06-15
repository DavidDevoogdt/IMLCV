from functools import partial

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
from jax import Array, jit, random

from IMLCV.base.CV import CV, CvTrans, NeighbourList, SystemParams
from IMLCV.base.CVDiscovery import Transformer
from IMLCV.base.datastructures import jit_decorator, vmap_decorator
from IMLCV.base.rounds import DataLoaderOutput
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
    def _fit(
        self,
        cv: list[CV],
        cv_t: list[CV],
        w: list[jax.Array],
        dlo: DataLoaderOutput,
        nunits=250,
        nlayers=3,
        lr=1e-4,
        num_epochs=100,
        batch_size=32,
        chunk_size=None,
        verbose=True,
        macro_chunk=1000,
        **kwargs,
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
            "layers": nlayers,
            "nunits": nunits,
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

            return nn.apply(eval_model, VAE(**vae_args))({"params": params})

        # Test encoder implementation
        # Random key for initialization
        key, rng = jax.random.split(rng, 2)

        init_data = jax.random.normal(key, (batch_size, dim), jnp.float32)

        key, rng = jax.random.split(rng, 2)

        state = train_state.TrainState.create(
            apply_fn=VAE(**vae_args).apply,
            params=VAE(**vae_args).init(key, init_data, rng)["params"],
            tx=optax.adam(lr),
        )

        rng, key, eval_rng = random.split(rng, 3)

        x = _cv.cv
        x = random.permutation(key, x)

        split = x.shape[0] // 10

        x_train = x[0:-split, :]
        x_test = x[-split:, :]

        steps_per_epoch = x_train.shape[0] // batch_size

        for epoch in range(num_epochs):
            rng, key = random.split(rng)
            indices = jax.random.choice(
                key=key,
                a=int(x.shape[0]),
                shape=(steps_per_epoch, batch_size),
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
            encoded: Array = VAE(**vae_args).apply(
                {"params": state.params},
                x.cv,
                method=VAE.encode,
            )  # type:ignore
            return x.replace(cv=encoded)

        f_enc = CvTrans.from_cv_function(f=forward)

        cv = f_enc.compute_cv(_cv)[0].unstack()
        if cv_t is not None:
            cv_t = f_enc.compute_cv(_cv_t)[0].unstack()

        return cv, cv_t, un_atomize * f_enc, w


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
    def __init__(
        self,
        outdim: int,
        kernel=False,
        optimizer=None,
        solver="eigen",
        method="pymanopt",
        harmonic=True,
        min_gradient_norm: float = 1e-3,
        min_step_size: float = 1e-3,
        max_iterations=25,
        **kwargs,
    ):
        super().__init__(
            outdim=outdim,
            kernel=kernel,
            optimizer=optimizer,
            solver=solver,
            method=method,
            harmonic=harmonic,
            min_gradient_norm=min_gradient_norm,
            min_step_size=min_step_size,
            max_iterations=max_iterations,
            **kwargs,
        )

    def _fit(
        self,
        cv_list: list[CV],
        cv_t_list: list[CV],
        w: list[jax.Array],
        dlo: DataLoaderOutput,
        kernel=False,
        optimizer=None,
        chunk_size=None,
        solver="eigen",
        method="pymanopt",
        harmonic=True,
        min_gradient_norm: float = 1e-3,
        min_step_size: float = 1e-3,
        max_iterations=25,
        verbose=True,
        macro_chunk=1000,
        **kwargs,
    ):
        # nl_list = dlo.nl

        if kernel:
            raise NotImplementedError("kernel not implemented for lda")

        cv = CV.stack(*cv_list)
        cv, _ = un_atomize.compute_cv(cv)

        cv_t = CV.stack(*cv_t_list)
        cv_t, _ = un_atomize.compute_cv(cv_t)

        if method == "sklearn":
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

            labels = []
            for i, cvi in enumerate(cv_list):
                labels.append(jnp.full(cvi.shape[0], i))

            labels = jnp.hstack(labels)

            alpha = LDA(n_components=self.outdim, solver=solver, shrinkage="auto").fit(  # type:ignore
                cv.cv.__array__(),
                labels,
            )

            lda_cv = CvTrans.from_cv_function(_LDA_trans, alpha=alpha, outdim=self.outdim, solver=solver)
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

        elif method == "pymanopt":
            import pymanopt  # type:ignore

            assert isinstance(cv_list, list)
            if optimizer is None:
                optimizer = pymanopt.optimizers.TrustRegions(
                    max_iterations=max_iterations,
                    min_gradient_norm=min_gradient_norm,
                    min_step_size=min_step_size,
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

                if harmonic:
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

        return cv.unstack(), cv_t.unstack(), full_trans, w


class TransformerMAF(Transformer):
    # Maximum Autocorrelation Factors

    def _fit(
        self,
        x: list[CV] | list[SystemParams],
        x_t: list[CV] | list[SystemParams] | None,
        w: list[jax.Array],
        w_t: list[jax.Array],
        dlo: DataLoaderOutput,
        max_features=500,
        max_features_pre=5000,
        macro_chunk=1000,
        chunk_size=None,
        trans=None,
        eps=1e-6,
        eps_pre=1e-6,
        outdim=None,
        correlation=True,
        use_w=True,
        **fit_kwargs,
    ):
        print("getting koopman")

        if outdim is None:
            outdim = self.outdim

        print(f"{outdim=}")

        # print(f"looking for constant mode with {num_regions=}")

        print(f"{dlo.nl=}, {dlo.nl_t=}")

        # if dlo.labels is not None:
        #     n_skip = int(jnp.sum(jnp.unique(jnp.hstack(dlo.labels))))
        # else:
        #     n_skip = 1

        km = dlo.koopman_model(
            cv_0=x,
            cv_t=x_t,
            nl=dlo.nl,
            nl_t=dlo.nl_t,
            method="tcca",
            max_features=max_features,
            max_features_pre=max_features_pre,
            w=w if use_w else [jnp.ones_like(x) for x in w],
            w_t=w_t if use_w else [jnp.ones_like(x) for x in w],
            calc_pi=True,
            add_1=False,
            trans=trans,
            chunk_size=chunk_size,
            macro_chunk=macro_chunk,
            verbose=True,
            out_dim=-1,
            eps=eps,
            eps_pre=eps_pre,
            symmetric=True,
            correlation=correlation,
        )

        # km = km.weighted_model(
        #     symmetric=True,
        #     verbose=True,
        # )

        ##########

        ts = (
            km.timescales(
                n_skip=None,
                remove_constant=True,
            )
            / nanosecond
        )

        print(f"timescales {ts} ns")

        for i in range(self.outdim):
            if ts[i] / ts[0] < 1 / 10:
                (print(f"cv {i} is too small compared to ref (fraction= {ts[i] / ts[0]}), cutting off "),)
                outdim = i
                break

        trans_km = km.f(
            out_dim=outdim,
            n_skip=None,
            remove_constant=True,
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

        return x, x_t, trans_km, w
