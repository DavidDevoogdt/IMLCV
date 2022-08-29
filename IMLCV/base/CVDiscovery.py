
import itertools
import os
from importlib import import_module
from typing import Iterable

import jax
import jax.numpy as jnp
import keras
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import umap
from IMLCV.base.bias import Bias
from IMLCV.base.CV import (CV, CvFlow, KerasFlow, PeriodicLayer, SystemParams,
                           cv)
from IMLCV.base.metric import Metric, MetricUMAP
from IMLCV.base.rounds import Rounds, RoundsMd
from IMLCV.launch.parsl_conf.bash_app_python import bash_app_python
from jax import custom_jvp, grad, jacrev, jit, make_jaxpr, value_and_grad, vmap
from jax.random import PRNGKey, choice
# using the import module import the tensorflow.keras module
# and typehint that the type is KerasAPI module
from keras.api._v2 import keras as KerasAPI
from matplotlib import gridspec
from matplotlib.colors import hsv_to_rgb
from matplotlib.transforms import Bbox
from molmod.constants import boltzmann
from parsl.data_provider.files import File

keras: KerasAPI = import_module("tensorflow.keras")


plt.rcParams['text.usetex'] = True


class Transformer:
    def __init__(self, outdim,  periodicity=None, bounding_box=None) -> None:
        self.outdim = outdim

        if periodicity is None:
            periodicity = [False for _ in range(self.outdim)]
        if bounding_box is None:
            bounding_box = np.array([
                [0.0, 10.0] for _ in periodicity])

        self.periodicity = periodicity
        self.bounding_box = bounding_box

    def fit(self, x, metric: Metric, **kwargs) -> CV:
        pass


class TranformerUMAP(Transformer):

    def fit(self, z: Iterable[SystemParams], indices, decoder=False, prescale=True, nunits=256, nlayers=3, parametric=True, metric=None, **kwargs):

        # x, f = SystemParams.flatten_f(z, scale=prescale)
        f = SystemParams.get_descriptor_coulomb(z_array=np.array(
            [1, 6, 1, 1, 6, 8, 7, 1, 6, 1, 6, 1, 1, 1, 6, 8, 7, 1, 6, 1, 1, 1]), permutation='none')
        x = jnp.stack([f(zi) for zi in z])

        dims = x.shape[1:]

        if parametric:

            act = keras.activations.tanh

            layers = [
                keras.layers.InputLayer(input_shape=dims),
                * [keras.layers.Dense(units=nunits,
                                      activation=act,
                                      )
                   for _ in range(nlayers)],
                keras.layers.Dense(units=self.outdim),
            ]

            periodicity = self.periodicity
            if metric is None:

                pl = PeriodicLayer(bbox=self.bounding_box,
                                   periodicity=periodicity)

                layers.append(pl)

            encoder = keras.Sequential(layers)

            if decoder:
                decoder = keras.Sequential([
                    keras.layers.InputLayer(input_shape=(self.outdim)),
                    * [keras.layers.Dense(units=nunits, activation=keras.activations.tanh,)
                       for _ in range(nlayers)],
                    keras.layers.Dense(units=dims),
                ])
            else:
                decoder = None

            reducer = umap.parametric_umap.ParametricUMAP(

                n_components=self.outdim,
                encoder=encoder,
                decoder=decoder,
                output_metric=pl.metric if metric is None else metric,
                n_training_epochs=1,
                **kwargs
            )

        else:

            reducer = umap.UMAP(
                n_neighbors=50,
                min_dist=0.9,
                n_components=self.outdim,
                output_metric=MetricUMAP(periodicities=[True, True]).umap_f

            )
        # import tensorflow as tf
        # tf.data.experimental.enable_debug_mode()
        # tf.config.run_functions_eagerly(True)

        reducer.fit(x[indices, :])

        if parametric:

            # calculate accurate bbox for non periodic dims
            if metric is None:
                bbox = np.array(pl.bbox)
            else:
                bbox = self.bounding_box
            out = reducer.transform(x)
            mask = np.logical_not(np.array(self.periodicity))

            bbox[mask, 1] = out.max(axis=0)[mask]
            bbox[mask, 0] = out.min(axis=0)[mask]

            if metric is not None:

                # extend bbox a little
                extension_factor = 0.1
                delta = (bbox[:, 1] - bbox[:, 0]) * extension_factor/2
                bbox[:, 0] -= delta
                bbox[:, 1] += delta

            cv = CV(
                f=KerasFlow(reducer.encoder, f=f),
                metric=Metric(
                    periodicities=self.periodicity,
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
                    bounding_box=bbox,
                ),
                jac=jacrev,
            )

            cv.compute(z[0])

        return cv


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
            # p=probs,
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
        def plot_app(sps, old_cv: CV, new_cv: CV, name, outputs=[]):

            def color(c, per):
                c2 = (c-c.min())/(c.max() - c.min())
                if not per:
                    c2 *= 330.0/360.0

                col = np.ones((len(c), 3))
                col[:, 0] = c2

                return hsv_to_rgb(col)

            cv_data = []
            cv_data_mapped = []

            cvs = [old_cv, new_cv]
            for cv in cvs:

                cvd = cv.map_cv(sps)
                cvdm = jit(vmap(cv.metric.map))(cvd)

                cv_data.append(np.array(cvd))
                cv_data_mapped.append(np.array(cvdm))

            for z, data in enumerate([cv_data, cv_data_mapped]):

                # plot setting
                kwargs = {'s': 0.2}

                labels = [[r'$\Phi$', r'$\Psi$'], [
                    'umap 1', 'umap 2', 'umap 3']]
                for [i, j] in [[0, 1], [1, 0]]:  # order

                    indim = cvs[i].n
                    outdim = cvs[j].n

                    if outdim == 2:
                        proj = None
                        wr = 1
                    elif outdim == 3:
                        proj = '3d'
                        wr = 1
                    else:
                        raise NotImplementedError

                    indim_pairs = list(
                        itertools.combinations(range(indim), r=2))
                    print(indim_pairs)

                    fig = plt.figure()

                    spec = gridspec.GridSpec(nrows=len(indim_pairs)*2, ncols=2,
                                             width_ratios=[1, wr], wspace=0.5)

                    for id, inpair in enumerate(indim_pairs):

                        for cc in range(2):
                            print(f"cc={cc}")

                            col = color(
                                data[i][:, inpair[cc]],  cvs[i].metric.periodicities[inpair[cc]])

                            l = fig.add_subplot(spec[id*2+cc, 0])
                            r = fig.add_subplot(
                                spec[id*2+cc, 1], projection=proj)

                            print(f"scatter={cc}")
                            l.scatter(
                                *[data[i][:, l] for l in inpair],  c=col, **kwargs)
                            l.set_xlabel(labels[i][inpair[0]])
                            l.set_ylabel(labels[i][inpair[1]])

                            if outdim == 2:
                                print("plot r 2d")
                                r.scatter(
                                    *[data[j][:, l] for l in range(2)],  c=col, **kwargs)
                                r.set_xlabel(labels[j][0])
                                r.set_ylabel(labels[j][1])

                            elif outdim == 3:
                                print("plot r 3d")

                                def plot3d(data,  ax, colors=None, labels=labels[j]):

                                    ax.set_xlabel(labels[0])
                                    ax.set_ylabel(labels[1])
                                    ax.set_zlabel(labels[2])

                                    mm = data.min(axis=0)
                                    MM = data.max(axis=0)
                                    offset_extra = (MM-mm)*0

                                    ax.scatter(data[:, 0], data[:, 1],  mm[2]-offset_extra[2], **kwargs,
                                               c=colors, zdir='z')
                                    ax.scatter(data[:, 0], data[:, 2], MM[1]+offset_extra[1], **kwargs,
                                               c=colors, zdir='y')
                                    ax.scatter(data[:, 1], data[:, 2], mm[0]-offset_extra[0], **kwargs,
                                               c=colors, zdir='x', )

                                plot3d(data=data[j], colors=col, ax=r)

                    # fig.set_size_inches([10, 16])

                    n = f"{name}_{ 'mapped' if z==1 else ''}_{'old_new' if i == 0 else 'new_old'}.pdf"
                    os.makedirs(os.path.dirname(n), exist_ok=True)
                    fig.savefig(n)

                    outputs.append(File(n))

        if plot:
            base_name = f"{rounds.folder}/round_{rounds.round}/cvdicovery"

            plot_app(
                name=base_name,
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
