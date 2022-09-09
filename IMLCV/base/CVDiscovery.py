import itertools
import os
from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import keras
import matplotlib.pyplot as plt
import numpy as np
import optax
import umap
from flax import linen as nn
from flax.training import train_state
from jax import jacrev, jit, random, vmap
from jax.random import PRNGKey, choice
from matplotlib import gridspec
from matplotlib.colors import hsv_to_rgb
from tensorflow import keras

from IMLCV.base.bias import Bias
from IMLCV.base.CV import (
    CV,
    CvFlow,
    CvTrans,
    KerasTrans,
    PeriodicLayer,
    SystemParams,
    coulomb_descriptor_cv_flow,
    scale_cv_trans,
)
from IMLCV.base.metric import Metric
from IMLCV.base.rounds import RoundsMd
from IMLCV.external.parsl_conf.bash_app_python import bash_app_python
from molmod.constants import boltzmann
from parsl.data_provider.files import File

# keras: KerasAPI = import_module("tensorflow.keras")


plt.rcParams["text.usetex"] = True


class Transformer:
    def __init__(self, outdim, periodicity=None, bounding_box=None) -> None:
        self.outdim = outdim

        if periodicity is None:
            periodicity = [False for _ in range(self.outdim)]
        if bounding_box is None:
            bounding_box = np.array([[0.0, 10.0] for _ in periodicity])

        self.periodicity = periodicity
        self.bounding_box = bounding_box

    def pre_fit(
        self, z: SystemParams, svd=True, scale=True
    ) -> Tuple[jnp.ndarray, CvFlow]:
        # x, f = SystemParamss.flatten_f(z, scale=prescale)
        x, f = coulomb_descriptor_cv_flow(z)

        if scale:
            x, g = scale_cv_trans(x)
            f = f * g

        return x, f

    def fit(
        self, sp: SystemParams, indices, prescale=True, postscale=True, **kwargs
    ) -> CV:
        x, f = self.pre_fit(sp, scale=prescale)
        y, g = self._fit(x, indices, **kwargs)
        z, h = self.post_fit(y, scale=postscale)

        cv = CV(
            f=f * g * h,
            metric=Metric(periodicities=self.periodicity),
            jac=jacrev,
        )

        return cv

    def _fit(self, x, indices, **kwargs) -> Tuple[jnp.ndarray, CvFlow]:
        raise NotImplementedError

    def post_fit(self, y: jnp.ndarray, scale) -> Tuple[jnp.ndarray, CvTrans]:
        if not scale:
            return y, CvTrans(lambda x: x)
        return scale_cv_trans(y)


class TranformerUMAP(Transformer):
    def _fit(
        self,
        x,
        indices,
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

        reducer.fit(x[indices, :])

        assert parametric

        f = KerasTrans(reducer.encoder)
        return f.compute(x), f


class Encoder(nn.Module):
    latents: int
    layers: int
    nunits: int
    dim: int

    @nn.compact
    def __call__(self, x):
        for i in range(self.layers):
            x = nn.Dense(self.nunits, name=f"encoder_{i}")(x)
            x = nn.relu(x)
        mean_x = nn.Dense(self.latents, name="fc2_mean")(x)
        logvar_x = nn.Dense(self.latents, name="fc2_logvar")(x)
        return mean_x, logvar_x


class Decoder(nn.Module):
    latents: int
    layers: int
    nunits: int
    dim: int

    @nn.compact
    def __call__(self, z):
        for i in range(self.layers):
            z = nn.Dense(self.nunits, name=f"decoder_{i}")(z)
            z = nn.relu(z)
        z = nn.Dense(self.dim, name="fc2")(z)
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

    def generate(self, z):
        return nn.sigmoid(self.decoder(z))

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
        x,
        indices,
        decoder=False,
        nunits=256,
        nlayers=3,
        parametric=True,
        metric=None,
        **kwargs,
    ):

        # import tensorflow_datasets as tfds

        # x = x[indices, :]

        # EPS = 1e-8
        key = random.PRNGKey(0)

        dim = x.shape[1]

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
            return ((x1 - x2) ** 2).mean()

        def compute_metrics(recon_x, x, mean, logvar):
            bce_loss = mean_Squared_error(recon_x, x).mean()
            kld_loss = kl_divergence(mean, logvar).mean()
            return {"bce": bce_loss, "kld": kld_loss, "loss": bce_loss + kld_loss}

        # @jax.vmap
        # def binary_cross_entropy_with_logits(logits, labels):
        #     logits = nn.log_sigmoid(logits)
        #     return -jnp.sum(labels * logits + (1. - labels) * jnp.log(-jnp.expm1(logits)))

        @jax.jit
        def train_step(state, batch, z_rng):
            def loss_fn(params):
                recon_x, mean, logvar = VAE(**vae_args).apply(
                    {"params": params}, batch, z_rng
                )

                bce_loss = mean_Squared_error(recon_x, batch).mean()
                kld_loss = kl_divergence(mean, logvar).mean()
                loss = bce_loss + kld_loss
                return loss

            grads = jax.grad(loss_fn)(state.params)
            return state.apply_gradients(grads=grads)

        @jax.jit
        def eval(params, x, z, z_rng):
            def eval_model(vae):
                recon_x, mean, logvar = vae(x, z_rng)
                comparison = jnp.stack([x, recon_x])

                metrics = compute_metrics(recon_x, x, mean, logvar)
                return metrics, comparison

            return nn.apply(eval_model, VAE(**vae_args))({"params": params})

        # Test encoder implementation
        # Random key for initialization
        rng = jax.random.PRNGKey(0)

        lr = 1e-4
        num_epochs = 10
        batch_size = 10

        init_data = jnp.ones((batch_size, dim), jnp.float32)

        state = train_state.TrainState.create(
            apply_fn=VAE(**vae_args).apply,
            params=VAE(**vae_args).init(key, init_data, rng)["params"],
            tx=optax.adam(lr),
        )

        rng, z_key, eval_rng = random.split(rng, 3)
        z = random.normal(z_key, (64, self.outdim))

        rng, key = random.split(rng, 2)

        x = random.permutation(key, x)
        x_train = x[0:-1000, :]
        x_test = x[-1000:, :]

        steps_per_epoch = x_train.shape[0] // batch_size

        for epoch in range(num_epochs):

            rng, key = random.split(rng)
            indices = jax.random.choice(
                key=key,
                a=int(x.shape[0]),
                shape=(steps_per_epoch, batch_size),
                replace=False,
            )

            for index in indices:

                rng, key = random.split(rng)

                state = train_step(state, x_train[index, :], key)

                metrics, comparison = eval(state.params, x_test, z, eval_rng)

                print(
                    "eval epoch: {}, loss: {:.4f}, BCE: {:.4f}, KLD: {:.4f}".format(
                        epoch + 1, metrics["loss"], metrics["bce"], metrics["kld"]
                    )
                )

        class NNtrans(CvTrans):
            def __init__(self, params, vae_args) -> None:
                self.params = params
                self.vae_args = vae_args

            @partial(jit, static_argnums=(0,))
            def compute(self, x):
                encoded: jnp.ndarray = VAE(**self.vae_args).apply(
                    {"params": self.params}, x, method=VAE.encode
                )[0]
                return encoded

        f_enc = NNtrans(state.params, vae_args)

        # a: jnp.ndarray =

        return f_enc.compute(x), f_enc


class CVDiscovery:
    """convert set of coordinates to good collective variables."""

    def __init__(self, transformer: Transformer) -> None:
        # self.rounds = rounds
        self.transformer = transformer

    def _get_data(self, rounds: RoundsMd, num=4, out=1e4):

        sp_arr = []

        cvs_mapped_arr = []
        weights = []

        cv = None

        for dictionary in rounds.iter(num=num):

            # map cvs
            bias = Bias.load(dictionary["attr"]["name_bias"])

            if cv is None:
                cv = bias.cvs

                def map_cv(x):
                    return cv.compute(x)[0]

                map_metric = jit(vmap(cv.metric.map))

            sp = SystemParams(
                coordinates=jnp.array(dictionary["positions"]),
                cell=dictionary.get("cell", None),
                masses=jnp.array(
                    [1, 6, 1, 1, 6, 8, 7, 1, 6, 1, 6, 1, 1, 1, 6, 8, 7, 1, 6, 1, 1, 1]
                ),
            )

            # execute all the mappings
            cvs = map_cv(sp)
            cvs_mapped = map_metric(cvs)
            biases = bias.compute(cvs=cvs_mapped, map=False)[0]

            beta = 1 / (dictionary["round"]["T"] * boltzmann)
            weight = jnp.exp(beta * biases)

            sp_arr.append(sp)
            cvs_mapped_arr.append(cvs_mapped)

            weights.append(weight)

        # todo modify probability
        probs = jnp.hstack(weights)
        probs = probs / jnp.sum(probs)

        key = PRNGKey(0)

        indices = choice(
            key=key,
            a=probs.shape[0],
            shape=(int(out),),
            p=probs,
            replace=False,
        )

        system_params = SystemParams.stack(sp_arr)
        cvs = jnp.vstack(cvs_mapped_arr)

        return [system_params, cvs, indices, cv]

    def compute(
        self, rounds: RoundsMd, samples=3e3, plot=True, name=None, **kwargs
    ) -> CV:

        sps, _, indices, cv_old = self._get_data(num=1, out=samples, rounds=rounds)

        new_cv = self.transformer.fit(
            sps,
            indices,
            **kwargs,
        )

        if plot:
            base_name = f"{rounds.folder}/round_{rounds.round}/cvdicovery"

            plot_app(
                name=base_name,
                old_cv=cv_old,
                new_cv=new_cv,
                sps=sps[
                    np.random.choice(
                        a=sps.shape[0], size=min(3000, sps.shape[0]), replace=False
                    )
                ],
                stdout=f"{base_name}.stdout",
                stderr=f"{base_name}.stderr",
            )

        return new_cv


@bash_app_python()
def plot_app(sps, old_cv: CV, new_cv: CV, name, outputs=[]):
    def color(c, per):
        c2 = (c - c.min()) / (c.max() - c.min())
        if not per:
            c2 *= 330.0 / 360.0

        col = np.ones((len(c), 3))
        col[:, 0] = c2

        return hsv_to_rgb(col)

    cv_data = []
    cv_data_mapped = []

    cvs = [old_cv, new_cv]
    for cv in cvs:

        cvd, _ = cv.compute(sps)
        cvdm = vmap(cv.metric.map)(cvd)

        cv_data.append(np.array(cvd))
        cv_data_mapped.append(np.array(cvdm))

    # for z, data in enumerate([cv_data, cv_data_mapped]):
    for z, data in enumerate([cv_data]):

        # plot setting
        kwargs = {"s": 0.2}

        labels = [[r"$\Phi$", r"$\Psi$"], ["umap 1", "umap 2", "umap 3"]]
        for [i, j] in [[0, 1], [1, 0]]:  # order

            indim = cvs[i].n
            outdim = cvs[j].n

            if outdim == 2:
                proj = None
                wr = 1
            elif outdim == 3:
                proj = "3d"
                wr = 1
            else:
                raise NotImplementedError

            indim_pairs = list(itertools.combinations(range(indim), r=2))
            print(indim_pairs)

            fig = plt.figure()

            if outdim == 2:
                spec = gridspec.GridSpec(
                    nrows=len(indim_pairs) * 2,
                    ncols=2,
                    width_ratios=[1, wr],
                    wspace=0.5,
                )
            elif outdim == 3:
                spec = gridspec.GridSpec(nrows=len(indim_pairs) * 2, ncols=3)

            for id, inpair in enumerate(indim_pairs):

                for cc in range(2):
                    print(f"cc={cc}")

                    col = color(
                        data[i][:, inpair[cc]], cvs[i].metric.periodicities[inpair[cc]]
                    )

                    if outdim == 2:
                        l = fig.add_subplot(spec[id * 2 + cc, 0])
                        r = fig.add_subplot(spec[id * 2 + cc, 1], projection=proj)
                    elif outdim == 3:
                        l = fig.add_subplot(spec[id * 2 + cc, 0])
                        r = [
                            fig.add_subplot(spec[id * 3 + cc, 1], projection=proj),
                            fig.add_subplot(spec[id * 3 + cc, 2], projection=proj),
                        ]

                    print(f"scatter={cc}")
                    l.scatter(*[data[i][:, l] for l in inpair], c=col, **kwargs)
                    l.set_xlabel(labels[i][inpair[0]])
                    l.set_ylabel(labels[i][inpair[1]])

                    if outdim == 2:
                        print("plot r 2d")
                        r.scatter(*[data[j][:, l] for l in range(2)], c=col, **kwargs)
                        r.set_xlabel(labels[j][0])
                        r.set_ylabel(labels[j][1])

                    elif outdim == 3:
                        print("plot r 3d")

                        def plot3d(data, ax, colors=None, labels=labels[j], mode=0):

                            ax.set_xlabel(labels[0])
                            ax.set_ylabel(labels[1])
                            ax.set_zlabel(labels[2])

                            if mode == 0:

                                ax.scatter(
                                    data[:, 0],
                                    data[:, 1],
                                    data[:, 2],
                                    **kwargs,
                                    c=colors,
                                    zorder=1,
                                )

                                for a, b, z in [[0, 1, "z"], [0, 2, "y"], [1, 2, "x"]]:

                                    Z, X, Y = np.histogram2d(data[:, a], data[:, b])

                                    X = (X[1:] + X[:-1]) / 2
                                    Y = (Y[1:] + Y[:-1]) / 2

                                    X, Y = np.meshgrid(X, Y)

                                    Z = (Z - Z.min()) / (Z.max() - Z.min())

                                    kw = {
                                        "facecolors": plt.cm.Greys(Z),
                                        "shade": True,
                                        "alpha": 1.0,
                                        "zorder": 0,
                                    }

                                    # im = NonUniformImage(ax, interpolation='bilinear')

                                    zz = np.zeros(X.shape) - 0.1
                                    # zz = - Z

                                    if z == "z":
                                        ax.plot_surface(X, Y, zz, **kw)
                                    elif z == "y":
                                        ax.plot_surface(X, zz, Y, **kw)
                                    else:
                                        ax.plot_surface(zz, X, Y, **kw)

                            else:

                                zz = np.zeros(data[:, 0].shape)
                                for z in ["x", "y", "z"]:

                                    if z == "z":
                                        ax.scatter(
                                            data[:, 0],
                                            data[:, 1],
                                            zz,
                                            **kwargs,
                                            zorder=1,
                                            c=colors,
                                        )
                                    elif z == "y":
                                        ax.scatter(
                                            data[:, 0],
                                            zz,
                                            data[:, 2],
                                            **kwargs,
                                            zorder=1,
                                            c=colors,
                                        )
                                    else:
                                        ax.scatter(
                                            zz,
                                            data[:, 1],
                                            data[:, 2],
                                            **kwargs,
                                            zorder=1,
                                            c=colors,
                                        )

                            ax.view_init(elev=20, azim=45)

                        plot3d(data=data[j], colors=col, ax=r[0], mode=0)
                        plot3d(data=data[j], colors=col, ax=r[1], mode=1)

            # fig.set_size_inches([10, 16])

            n = f"{name}_{ 'mapped' if z==1 else ''}_{'old_new' if i == 0 else 'new_old'}.pdf"
            os.makedirs(os.path.dirname(n), exist_ok=True)
            fig.savefig(n)

            outputs.append(File(n))


if __name__ == "__main__":
    from IMLCV.test.test_CV_disc import test_cv_discovery

    test_cv_discovery()
