from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
from IMLCV.base.CV import CV
from IMLCV.base.CV import CvFun
from IMLCV.base.CV import CvTrans
from IMLCV.base.CV import NeighbourList
from IMLCV.base.CVDiscovery import Transformer
from IMLCV.implementations.CV import get_sinkhorn_divergence
from IMLCV.implementations.CV import stack_reduce
from IMLCV.implementations.CV import trunc_svd
from IMLCV.implementations.CV import un_atomize
from jax import Array
from jax import jit
from jax import random
from jax import vmap


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
        nl: list[NeighbourList] | None,
        nunits=250,
        nlayers=3,
        lr=1e-4,
        num_epochs=100,
        batch_size=32,
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

        def forward(x: CV, nl, y: list[CV] | None = None):
            assert y is None
            encoded: Array = VAE(**vae_args).apply(
                {"params": state.params},
                x.cv,
                method=VAE.encode,
            )
            return CV(cv=encoded)

        f_enc = CvTrans(trans=[CvFun(forward=forward)])

        return f_enc.compute_cv_trans(cv)[0], un_atomize * f_enc


class TransoformerLDA(Transformer):
    def _fit(
        self,
        cv_list: list[CV],
        nl_list: list[NeighbourList] | None,
        kernel=False,
        harmonic=True,
        sort="rematch",
        shrinkage=0,
        alpha_rematch=1e-1,
        max_iterations=50,
        **kwargs,
    ):
        try:
            import pymanopt
        except ImportError:
            raise ImportError(
                "pymanopt not installed, please install it to use LDA collective variable",
            )

        assert isinstance(cv_list, list)

        from IMLCV.base.CV import NeighbourList

        # @vmap

        if kernel:
            raise NotImplementedError("kernel not implemented for lda")

        if sort == "l2" or sort == "average":
            norm_data = [get_sinkhorn_divergence(None, None, sort=sort, alpha_rematch=alpha_rematch)]
        elif sort == "rematch":
            norm_data = []
            assert nl_list is not None, "Neigbourlist required for rematch"
            for cvi, nli in zip(cv_list, nl_list):
                norm_data.append(
                    get_sinkhorn_divergence(
                        nli=nli[0],
                        pi=jnp.average(cvi.cv, axis=0),
                        sort=sort,
                        alpha_rematch=alpha_rematch,
                    ),
                )
        else:
            raise NotImplementedError

        cv = CV.stack(*cv_list)
        nl = NeighbourList.stack(*nl_list) if nl_list is not None else None

        hs = []

        cv_out = []

        for nd in norm_data:
            cv_i, _ = jax.jit(nd.compute_cv_trans)(cv, nl)
            cv_i, _f = trunc_svd(cv_i)
            normed_cv_u = CV.unstack(cv_i)

            mu_i = [jnp.mean(x.cv, axis=0) for x in normed_cv_u]
            mu = jnp.einsum(
                "i,ij->j",
                jnp.array(cv.stack_dims) / jnp.sum(jnp.array(cv.stack_dims)),
                jnp.array(mu_i),
            )

            correlation_w = CV.stack(*[cv_i - mu_i for mu_i, cv_i in zip(mu_i, normed_cv_u)])
            correlation_b = CV.stack(*[cv_i - mu for cv_i in normed_cv_u])

            manifold = pymanopt.manifolds.stiefel.Stiefel(n=mu.shape[0], p=self.outdim)

            @pymanopt.function.jax(manifold)
            @jit
            def cost(x):
                a = jnp.einsum("ab, ia,ic, cb ", x, correlation_w.cv, correlation_w.cv, x)
                b = jnp.einsum("ab, ja, jc, cb ", x, correlation_b.cv, correlation_b.cv, x)

                if harmonic:
                    out = ((1 - shrinkage) * a + shrinkage) / ((1 - shrinkage) * b + shrinkage)
                else:
                    out = -((1 - shrinkage) * b + shrinkage) / ((1 - shrinkage) * a + shrinkage)

                return out

            optimizer = pymanopt.optimizers.TrustRegions(max_iterations=max_iterations)

            problem = pymanopt.Problem(manifold, cost)
            result = optimizer.run(problem)

            alpha = result.point

            scale_factor = vmap(lambda x: alpha.T @ x)(jnp.array(mu_i))

            def scale_trans(cv: CV, nl: NeighbourList | None, _, alpha=alpha, scale_factor=scale_factor):
                return CV(
                    (alpha.T @ cv.cv - scale_factor[0, :]) / (scale_factor[1, :] - scale_factor[0, :]),
                    _stack_dims=cv._stack_dims,
                    _combine_dims=cv._combine_dims,
                    atomic=cv.atomic,
                    mapped=cv.mapped,
                )

            _g = CvTrans.from_cv_function(partial(scale_trans, alpha=alpha, scale_factor=scale_factor))

            cv_i = _g.compute_cv_trans(cv_i, nl)[0]

            cv_out.append(cv_i)
            hs.append(nd * _f * _g)

        lda_cv = CvTrans.stack(*hs) * stack_reduce()
        cv_out = stack_reduce().compute_cv_trans(CV.combine(*cv_out), nl)[0]

        # assert jnp.allclose(cv_out.cv, jax.jit(lda_cv.compute_cv_trans)(cv, nl)[0].cv)

        return cv_out, lda_cv
