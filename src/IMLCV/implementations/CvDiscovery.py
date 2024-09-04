from functools import partial

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
from jax import Array, jit, random, vmap
from molmod.units import nanosecond

from IMLCV.base.CV import CV, CvFun, CvTrans, NeighbourList
from IMLCV.base.CVDiscovery import Transformer
from IMLCV.base.rounds import data_loader_output
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
        cv_t: list[CV] | None,
        w: list[jax.Array],
        dlo: data_loader_output,
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

        cv = CV.stack(*cv)
        cv = un_atomize.compute_cv_trans(cv, None)[0]

        rng = random.PRNGKey(0)

        dim = cv.shape[1]

        vae_args = {
            "latents": self.outdim,
            "layers": nlayers,
            "nunits": nunits,
            "dim": dim,
        }

        @jax.vmap
        def kl_divergence(mean, logvar):
            return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))

        @jax.vmap
        def mean_Squared_error(x1, x2):
            return 0.5 * jnp.linalg.norm(x1 - x2) ** 2

        def compute_metrics(recon_x, x, mean, logvar):
            bce_loss = mean_Squared_error(recon_x, x).mean()
            kld_loss = 0.01 * kl_divergence(mean, logvar).mean()
            return {"bce": bce_loss, "kld": kld_loss, "loss": bce_loss + kld_loss}

        # @jax.vmap
        # def binary_cross_entropy_with_logits(logits, labels):
        #     logits = nn.log_sigmoid(logits)
        #     return -jnp.sum(
        #         labels * logits + (1.0 - labels) * jnp.log(-jnp.expm1(logits))
        #     )

        import optax

        @jax.jit
        def train_step(state: optax.TraceState, batch, z_rng):
            def loss_fn(params):
                recon_x, mean, logvar = VAE(**vae_args).apply(
                    {"params": params},
                    batch,
                    z_rng,
                )

                bce_loss = mean_Squared_error(recon_x, batch).mean()
                # bce_loss = mean_Squared_error(recon_x, batch)
                kld_loss = kl_divergence(mean, logvar).mean()
                loss = bce_loss + kld_loss
                return loss

            grads = jax.grad(loss_fn)(state.params)

            return state.apply_gradients(grads=grads)

        @jax.jit
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

        x = cv.cv
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

            # @partial(jit, static_argnums=(0,))

        def forward(x: CV, nl, y: list[CV] | None = None, _=None, smap=False):
            assert y is None
            encoded: Array = VAE(**vae_args).apply(
                {"params": state.params},
                x.cv,
                method=VAE.encode,
            )
            return x.replace(cv=encoded)

        f_enc = CvTrans(trans=(CvFun(forward=forward),))

        cv = f_enc.compute_cv_trans(cv)[0].unstack()
        if cv_t is not None:
            cv_t = f_enc.compute_cv_trans(cv_t)[0].unstack()

        return cv, cv_t, un_atomize * f_enc, w


def _LDA_trans(cv: CV, nl: NeighbourList | None, _, shmap, alpha, outdim, solver):
    if solver == "eigen":

        def f(cv, scalings):
            return cv @ scalings

        f = partial(f, scalings=jnp.array(alpha.scalings_)[:, :outdim])

    elif solver == "svd":

        def f(cv, scalings, xbar):
            return (cv - xbar) @ scalings

        f = partial(
            f,
            scalings=jnp.array(alpha.scalings_),
            xbar=jnp.array(alpha.xbar_),
        )

    else:
        raise NotImplementedError

    return CV(
        cv=f(cv.cv),
        _stack_dims=cv._stack_dims,
        _combine_dims=cv._combine_dims,
        atomic=cv.atomic,
        mapped=cv.mapped,
    )


def _LDA_rescale(cv: CV, nl: NeighbourList | None, _, shmap, mean):
    return CV(
        cv=(cv.cv - mean[0]) / (mean[1] - mean[0]),
        _stack_dims=cv._stack_dims,
        _combine_dims=cv._combine_dims,
        atomic=cv.atomic,
        mapped=cv.mapped,
    )


def _scale_trans(cv: CV, nl: NeighbourList | None, _, shmap, alpha, scale_factor):
    return CV(
        (alpha.T @ cv.cv - scale_factor[0, :]) / (scale_factor[1, :] - scale_factor[0, :]),
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
        cv_t: list[CV] | None,
        w: list[jax.Array],
        dlo: data_loader_output,
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
        # nl = NeighbourList.stack(*nl_list)
        cv, _, _ = un_atomize.compute_cv_trans(cv)

        if method == "sklearn":
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

            labels = []
            for i, cvi in enumerate(cv_list):
                labels.append(jnp.full(cvi.shape[0], i))

            labels = jnp.hstack(labels)

            alpha = LDA(n_components=self.outdim, solver=solver, shrinkage="auto").fit(
                cv.cv,
                labels,
            )

            lda_cv = CvTrans.from_cv_function(_LDA_trans, alpha=alpha, outdim=self.outdim, solver=solver)
            cv, _, _ = lda_cv.compute_cv_trans(cv)

            cvs = CV.unstack(cv)

            mean = []
            for i, cvs_i in enumerate(cvs):
                mean.append(jnp.mean(cvs_i.cv, axis=0))
            mean = jnp.array(mean)

            assert self.outdim == 1

            lda_rescale = CvTrans.from_cv_function(_LDA_rescale, mean=mean)
            cv, _, _ = lda_rescale.compute_cv_trans(cv)

            full_trans = un_atomize * lda_cv * lda_rescale

        elif method == "pymanopt":
            import pymanopt

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

            cov_w = LedoitWolf(assume_centered=True).fit(obersevations_within.cv).covariance_
            cov_b = LedoitWolf(assume_centered=True).fit(observations_between.cv).covariance_

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

            scale_factor = vmap(lambda x: alpha.T @ x)(jnp.array(mu_i))

            _g = CvTrans.from_cv_function(_scale_trans, alpha=alpha, scale_factor=scale_factor)

            cv = _g.compute_cv_trans(cv)[0]
            if cv_t is not None:
                cv_t = _g.compute_cv_trans(cv_t)[0]

            full_trans = un_atomize * _f * _g

        return cv.unstack(), cv_t.unstack(), full_trans, w


class TransformerMAF(Transformer):
    # Maximum Autocorrelation Factors

    def _fit(
        self,
        x: list[CV],
        x_t: list[CV] | None,
        w: list[jax.Array],
        dlo: data_loader_output,
        correct_bias=False,
        pre_selction_epsilon=1e-10,
        max_features=2000,
        max_functions=2500,
        koopman_weighting=False,
        method="tcca",
        macro_chunk=1000,
        chunk_size=None,
        T_scale=10,
        **fit_kwargs,
    ) -> tuple[CV, CvTrans]:
        assert dlo.time_series
        assert x_t is not None

        print("unatomizing")

        trans = un_atomize
        x, x_t = dlo.apply_cv_trans(
            un_atomize,
            x=x,
            x_t=x_t,
            macro_chunk=macro_chunk,
            chunk_size=chunk_size,
        )

        print("getting koopman")

        km = dlo.koopman_model(
            cv_0=x,
            cv_tau=x_t,
            # eps=1e-10,
            method=method,
            max_features=max_features,
            w=w,
            koopman_weight=True,
            add_1=True,
            chunk_size=chunk_size,
            macro_chunk=macro_chunk,
            verbose=True,
            T_scale=T_scale,
        )

        ts = km.timescales() / nanosecond

        print(f"timescales {  ts[1: min(self.outdim+5,len(ts-1))  ]   } ns")

        trans_km = km.f(out_dim=self.outdim)

        del km

        trans *= trans_km

        print("applying transformation")

        x, x_t = dlo.apply_cv_trans(
            trans_km,
            x=x,
            x_t=x_t,
            macro_chunk=macro_chunk,
            chunk_size=chunk_size,
        )

        return x, x_t, trans, w
