import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from flax import linen as nn
from flax.training import train_state
from IMLCV.base.CV import CV
from IMLCV.base.CV import CvFun
from IMLCV.base.CV import CvTrans
from IMLCV.base.CVDiscovery import Transformer
from jax import Array
from jax import random


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
        cv,
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

        def forward(x: CV, *y):
            assert y == (None,)
            encoded: Array = VAE(**vae_args).apply(
                {"params": state.params},
                x.cv,
                method=VAE.encode,
            )
            return CV(cv=encoded)

        f_enc = CvTrans(trans=[CvFun(forward=forward)])

        return f_enc.compute_cv_trans(cv)[0], f_enc
