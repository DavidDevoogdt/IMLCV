import jax
import jax.numpy as jnp
import keras
import matplotlib.pyplot as plt
import umap
from flax import linen as nn
from jax import random
from tensorflow import keras

from IMLCV.implementations.CV import KerasTrans, PeriodicLayer

plt.rcParams["text.usetex"] = True

from IMLCV.base.CVDiscovery import Transformer


class TranformerUMAP(Transformer):
    def _fit(
        self,
        x,
        # indices,
        decoder=False,
        nunits=256,
        nlayers=3,
        parametric=True,
        metric=None,
        **kwargs,
    ):
        dims = x.shape[1:]

        kwargs["n_components"] = self.outdim

        if metric is None:
            pl = PeriodicLayer(bbox=self.bounding_box, periodicity=self.periodicity)

            kwargs["output_metric"] = pl.metric
        else:
            kwargs["output_metric"] = metric

        if parametric:
            act = keras.activations.tanh
            layers = [
                keras.layers.InputLayer(input_shape=dims),
                *[
                    keras.layers.Dense(
                        units=nunits,
                        activation=act,
                    )
                    for _ in range(nlayers)
                ],
                keras.layers.Dense(units=self.outdim),
            ]
            if metric is None:
                layers.append(pl)

            encoder = keras.Sequential(layers)
            kwargs["encoder"] = encoder

            if decoder:
                decoder = keras.Sequential(
                    [
                        keras.layers.InputLayer(input_shape=(self.outdim)),
                        *[
                            keras.layers.Dense(units=nunits, activation=act)
                            for _ in range(nlayers)
                        ],
                        keras.layers.Dense(units=jnp.prod(jnp.array(x.shape[1:]))),
                    ]
                )
                kwargs["decoder"] = decoder

            reducer = umap.parametric_umap.ParametricUMAP(**kwargs)
        else:
            reducer = umap.UMAP(**kwargs)

        reducer.fit(x)

        assert parametric

        f = KerasTrans(reducer.encoder)
        return f.compute_cv_trans(x), f


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
