import functools
import os
from typing import Iterable

import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import umap
from IMLCV.base.bias import Bias
from IMLCV.base.CV import CV, CvFlow, SystemParams, cv
from IMLCV.base.metric import Metric, MetricUMAP
from IMLCV.base.rounds import Rounds, RoundsMd
from jax import custom_jvp, grad, jacrev, jit, make_jaxpr, value_and_grad, vmap
from jax.experimental import jax2tf
from jax.experimental.host_callback import call
from jax.interpreters import batching
from jax.random import PRNGKey, choice
from keras import backend
from molmod.constants import boltzmann


plt.rcParams['text.usetex'] = True


class Transformer:
    def __init__(self) -> None:
        pass

    def fit(self, x, metric: Metric, **kwargs) -> CV:
        pass


class TranformerUMAP(Transformer):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, z: Iterable[SystemParams], output_metric: MetricUMAP, **kwargs):

        x = SystemParams.flatten(z)

        ncomps = 2

        import tensorflow as tf
        dims = x.shape[1]

        n_components = ncomps
        nlayers = 5

        encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=dims),
            * [tf.keras.layers.Dense(units=256, activation="relu")
               for _ in range(nlayers)],
            tf.keras.layers.Dense(units=n_components),
        ])

        decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(n_components)),
            * [tf.keras.layers.Dense(units=256, activation="relu")
               for _ in range(nlayers)],
            tf.keras.layers.Dense(units=dims),

        ])

        reducer = umap.parametric_umap.ParametricUMAP(
            n_neighbors=20,
            min_dist=0.5,
            n_components=ncomps,
            output_metric=output_metric,
            encoder=encoder,
            # decoder=decoder,
            batch_size=100,
            # n_epochs=200,
            n_training_epochs=1,
            # parametric_reconstruction=True,
            # autoencoder_loss=True,
            # run_eagerly=True,
        )

        reducer.fit(x)

        class KerasFlow(CvFlow):
            def __init__(self, encoder) -> None:
                self.encoder = encoder

            def __call__(self, x: SystemParams):
                cc = SystemParams.flatten(x).reshape((1, -1))
                out = jax2tf.call_tf(self.encoder.call)(cc)
                return jnp.reshape(out, out.shape[1:])

        cv = CV(f=KerasFlow(reducer.encoder),
                metric=output_metric, jac=jacrev)

        return cv


class CVDiscovery:
    """convert set of coordinates to good collective variables."""

    def __init__(self, transformer: Transformer = TranformerUMAP()) -> None:
        # self.rounds = rounds
        self.transformer = transformer

    def _get_data(self, rounds: RoundsMd, num=4, out=1e4):

        sp_arr = []

        cvs_mapped_arr = []
        # energies = []
        # times = []
        # taus = []
        weights = []

        cv = None

        for dictionary in rounds.iter(num=num):

            # map cvs
            bias = Bias.load(dictionary['attr']["name_bias"])
            map_bias = jit(vmap(lambda x:  bias.compute(cvs=x, map=False)[0]))

            if cv is None:
                cv = bias.cvs
                map_cv = cv.map_cv
                map_metric = jit(vmap(cv.metric.map))

            pos = dictionary["positions"]
            cell = dictionary.get("cell", None)
            sp = SystemParams.map_params(pos, cell)

            # execute all the mappings
            cvs = map_cv(sp)
            cvs_mapped = map_metric(cvs)
            biases = map_bias(cvs_mapped)

            beta = 1/(dictionary['round']['T']*boltzmann)
            weight = jnp.exp(beta * biases)

            # time = dictionary["t"]

            # def integr(fx, x):
            #     y = np.array(
            #         [0, *np.cumsum((fx[1:] + fx[:-1]) * (x[1:] - x[:-1]) / 2)])
            #     return y

            # beta = 1/(dictionary['round']['T']*boltzmann)
            # tau = integr(np.exp(beta * biases), time)

            sp_arr.append(sp)
            cvs_mapped_arr.append(cvs_mapped)

            weights.append(weight)

        # start_tuas = [0, * np.cumsum([tau[-1] for tau in taus[:-1]])]
        # taus = [tau+st for tau, st in zip(taus, start_tuas)]

        # taus = np.hstack(taus)
        # new_taus = np.linspace(
        #     start=taus.min(), stop=taus.max(), num=int(out))

        # def _interp(x_new, x, y):
        #     if y is not None:
        #         return jnp.apply_along_axis(
        #             lambda yy: jnp.interp(x_new, x, yy), arr=y, axis=0)
        #     return None

        # pos = np.vstack(pos_arr)
        # new_pos = _interp(new_taus, taus, pos)

        # if cell is not None:
        #     cell = np.vstack(cell_arr)
        #     new_cell = _interp(new_taus, taus, cell)
        # else:
        #     new_cell = None

        # #
        # cvs = get_cvs(new_pos, new_cell, cv)

        probs = jnp.hstack(weights)
        probs = probs/jnp.sum(probs)

        key = PRNGKey(0)

        indices = choice(key=key, a=probs.shape[0], shape=(
            int(out),), p=probs, replace=False)

        system_params = [b for a in sp_arr for b in a]
        sps = [system_params[i] for i in indices]
        cvs = jnp.vstack(cvs_mapped_arr)[indices]

        return [sps, cvs]

    def compute(self, rounds: RoundsMd,  plot=True, name=None, **kwargs) -> CV:

        sps, cv = self._get_data(num=6, out=5e3, rounds=rounds)

        fit = True

        newCV = self.transformer.fit(
            sps,
            output_metric=MetricUMAP(
                periodicities=[True, True],
                bounding_box=[[-10, 10], [-10, 10]],
            ),
            ** kwargs,
        )

        if plot:
            if name is None:
                name = f"{rounds.folder}/round_{rounds.round}/new_cv.pdf"

            def color(c):
                c2 = (c[:]-c.min(axis=0))/(c.max(axis=0) - c.min(axis=0))
                l = c2.shape[0]
                w = c2.shape[1]

                cv1 = np.ones((l, 3))
                cv1[:, 0:w] = c2[:]

                cv_hsv = matplotlib.colors.hsv_to_rgb(cv1)
                cv_hsv2 = matplotlib.colors.hsv_to_rgb(cv1[:,  [1, 0, 2]])

                return cv_hsv, cv_hsv2

            tf = jit(vmap(newCV.metric.map))(newCV.map_cv(sps=sps))

            fig, ax_arr = plt.subplots(4, 2)

            # plot setting
            kwargs = {'s': 1}

            data = [cv, tf]
            labels = [[r'$\Phi$', r'$\Psi$'], ['umap1', 'umap2']]
            cmaps = [None, ]

            for [i, j] in [[0, 1], [1, 0]]:  # order

                colors = color(data[i])

                for k, col in enumerate(colors):

                    l, r = ax_arr[2*i+k]

                    l.scatter(data[i][:, 0], data[i]
                              [:, 1], c=col, **kwargs)

                    l.set_xlabel(labels[i][0])
                    l.set_ylabel(labels[i][1])

                    r.scatter(data[j][:, 0], data[j]
                              [:, 1], c=col, **kwargs)

                    r.set_xlabel(labels[j][0])
                    r.set_ylabel(labels[j][1])

            # plt.show()

            os.makedirs(os.path.dirname(name), exist_ok=True)

            fig.set_size_inches([12, 8])

            fig.savefig(name)

        return newCV


if __name__ == "__main__":
    from IMLCV.test.test_CV_disc import test_cv_discovery
    test_cv_discovery()
