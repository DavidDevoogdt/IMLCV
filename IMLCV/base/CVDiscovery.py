import functools
import os
import tempfile
from calendar import different_locale
from importlib import import_module
from typing import Iterable

import dill
import jax
import jax.numpy as jnp
import keras
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import umap
from IMLCV.base.bias import Bias
from IMLCV.base.CV import CV, CvFlow, KerasFlow, SystemParams, cv
from IMLCV.base.metric import Metric, MetricUMAP
from IMLCV.base.rounds import Rounds, RoundsMd
from IMLCV.launch.parsl_conf.bash_app_python import bash_app_python
from jax import custom_jvp, grad, jacrev, jit, make_jaxpr, value_and_grad, vmap
from jax.experimental import jax2tf
from jax.experimental.host_callback import call
from jax.interpreters import batching
from jax.random import PRNGKey, choice
from keras.api._v2 import keras as KerasAPI
from molmod.constants import boltzmann
from parsl.data_provider.files import File

# using the import module import the tensorflow.keras module
# and typehint that the type is KerasAPI module
keras: KerasAPI = import_module("tensorflow.keras")


# from tensorflow import keras


plt.rcParams['text.usetex'] = True


class Transformer:
    def __init__(self) -> None:
        pass

    def fit(self, x, metric: Metric, **kwargs) -> CV:
        pass


class TranformerUMAP(Transformer):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, z: Iterable[SystemParams], indices, decoder, parametric=True,  **kwargs):

        x, f = SystemParams.flatten_f(z)

        ncomps = 2

        import tensorflow as tf
        dims = x.shape[1]

        n_components = ncomps
        nlayers = 3

        if parametric:

            bbox = np.array([
                [0.0, 20.0],
                [0.0, 20.0]
            ])

            @tf.function
            def output_metric(r):
                a = tf.convert_to_tensor(bbox[:, 1]-bbox[:, 0], dtype=r.dtype)

                r = tf.math.mod(r, a)
                r = tf.where(r > a/2, r-a, r)
                return tf.norm(r, axis=1)

            encoder = keras.Sequential([
                keras.layers.InputLayer(input_shape=dims),
                * [keras.layers.Dense(units=256, activation="relu")
                   for _ in range(nlayers)],
                keras.layers.Dense(units=n_components),
            ])

            if decoder:

                decoder = keras.Sequential([
                    keras.layers.InputLayer(input_shape=(n_components)),
                    * [keras.layers.Dense(units=256, activation="relu")
                       for _ in range(nlayers)],
                    keras.layers.Dense(units=dims),

                ])
            else:
                decoder = None

            reducer = umap.parametric_umap.ParametricUMAP(
                n_neighbors=20,
                min_dist=0.7,
                n_components=ncomps,
                encoder=encoder,
                decoder=decoder,
                output_metric=output_metric,
                n_training_epochs=1,
                **kwargs
            )

        else:

            reducer = umap.UMAP(
                n_neighbors=50,
                min_dist=0.9,
                n_components=ncomps,
                output_metric=MetricUMAP(periodicities=[True, True]).umap_f

            )

        # tf.data.experimental.enable_debug_mode()
        # tf.config.run_functions_eagerly(True)

        reducer.fit(x[indices, :])

        if parametric:
            total_embedding = reducer.transform(x)
        else:
            total_embedding = reducer.embedding_

        # bounding_box = np.array([
        #     total_embedding.min(axis=0),
        #     total_embedding.max(axis=0)
        # ]).T
        # margin = 1.0
        # delta = (bounding_box[:, 1]-bounding_box[:, 0])*margin/2.0
        # bounding_box[0, :] -= delta
        # bounding_box[1, :] += delta

        if parametric:
            cv = CV(
                f=KerasFlow(reducer.encoder, f=f),
                metric=Metric(
                    periodicities=[True for _ in range(ncomps)],
                    bounding_box=bbox,
                ),
                jac=jacrev,
            )
        else:
            raise NotImplementedError

            def func(sp: SystemParams):
                with jax.disable_jit():
                    out = reducer.transform(f([sp]))
                    return np.reshape(out, out.shape[1:])

            cv = CV(
                f=CvFlow(func=func),
                metric=Metric(
                    periodicities=[False for _ in range(ncomps)],
                    bounding_box=bounding_box,
                ),
                jac=jacrev,
            )

            cv.compute(z[0])

        return cv


class CVDiscovery:
    """convert set of coordinates to good collective variables."""

    def __init__(self, transformer: Transformer = TranformerUMAP()) -> None:
        # self.rounds = rounds
        self.transformer = transformer

    def _get_data(self, rounds: RoundsMd, num=4, out=1e4):

        sp_arr = []

        cvs_mapped_arr = []
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

            sp_arr.append(sp)
            cvs_mapped_arr.append(cvs_mapped)

            weights.append(weight)

        # todo modify probability
        probs = jnp.hstack(weights)
        probs = probs/jnp.sum(probs)

        key = PRNGKey(0)

        indices = choice(
            key=key,
            a=probs.shape[0],
            shape=(int(out),),
            p=probs,
            replace=False,
        )

        system_params = [b for a in sp_arr for b in a]
        cvs = jnp.vstack(cvs_mapped_arr)

        return [system_params, cvs, indices, cv]

    def compute(self, rounds: RoundsMd, samples=3e3,  plot=True, name=None, **kwargs) -> CV:

        sps, _, indices, cv_old = self._get_data(
            num=6, out=samples, rounds=rounds)

        new_cv = self.transformer.fit(
            sps, indices,
            ** kwargs,
        )

        @bash_app_python()
        def plot_app(sps, old_cv: CV, new_cv: CV, outputs=[]):
            for o in outputs:
                os.makedirs(os.path.dirname(o), exist_ok=True)

            def color(c):
                c2 = (c[:]-c.min(axis=0))/(c.max(axis=0) - c.min(axis=0))
                l = c2.shape[0]
                w = c2.shape[1]

                cv1 = np.ones((l, 3))
                cv1[:, 0:w] = c2[:]

                cv_hsv = matplotlib.colors.hsv_to_rgb(cv1)
                cv_hsv2 = matplotlib.colors.hsv_to_rgb(cv1[:,  [1, 0, 2]])

                return cv_hsv, cv_hsv2

            cv_data = []
            cv_data_mapped = []

            for cv in [old_cv, new_cv]:

                cvd = cv.map_cv(sps)
                cvdm = jit(vmap(cv.metric.map))(cvd)

                cv_data.append(cvd)
                cv_data_mapped.append(cvdm)

            for z, data in enumerate([cv_data, cv_data_mapped]):

                # if True:
                fig, ax_arr = plt.subplots(4, 2)

                # plot setting
                kwargs = {'s': 1}

                labels = [[r'$\Phi$', r'$\Psi$'], ['umap1', 'umap2']]
                for [i, j] in [[0, 1], [1, 0]]:  # order

                    colors = color(data[i])
                    # if i == 0:
                    #     colors = color(data[i])
                    # else:
                    #     colors = [data[i][:, 0],  data[i][:, 1]]

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

                fig.set_size_inches([12, 8])
                fig.savefig(outputs[z])

        if plot:
            base_name = f"{rounds.folder}/round_{rounds.round}/new_cv"

            plot_app(
                outputs=[
                    File(f"{base_name}.pdf"),
                    File(f"{base_name}_mapped.pdf")
                ],
                old_cv=cv_old,
                new_cv=new_cv,
                sps=[sps[i] for i in indices],
                stdout=f'{base_name}.stdout',
                stderr=f'{base_name}.stderr',
            )

        return new_cv


if __name__ == "__main__":
    from IMLCV.test.test_CV_disc import test_cv_discovery
    test_cv_discovery()
