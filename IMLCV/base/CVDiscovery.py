import functools
import os

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

    def fit(self, x, output_metric: MetricUMAP, **kwargs):

        shape = x.shape

        x = np.reshape(x, (shape[0], -1))

        ncomps = 2

        reducer = umap.ParametricUMAP(
            n_neighbors=20,
            min_dist=0.99,
            n_components=ncomps,
            output_metric=output_metric,
            n_epochs=10,
            # autoencoder_loss=True,

        )

        import IMLCV.test.tf2jax
        from IMLCV.test.tf2jax import call_tf_p, loop_batcher
        from jax.interpreters import batching
        batching.primitive_batchers[call_tf_p] = functools.partial(
            loop_batcher, call_tf_p)
        # batching.primitive_batchers[call_tf_p] = functools.partial(
        #     par_batcher, call_tf_p)

        transf = reducer.fit(x)

        # transform to function of 1 variable with correct shape

        @jit
        def func(y: SystemParams):
            assert y.cell is None, "implement cell parameter passing"

            cc = y.coordinates.reshape((-1, x.shape[1]))
            out = jax2tf.call_tf(transf.encoder.call)(cc)
            return jnp.reshape(out, out.shape[1:])

        # fast
        # a = func(SystemParams(coordinates=x[0:100, :], cell=None))
        # b = vmap(lambda y: func(SystemParams(y, None)))(x[0:10000, :])

        cv = CV(f=CvFlow(func=func),
                metric=output_metric, jac=jacrev)

        return cv


class CVDiscovery:
    """convert set of coordinates to good collective variables."""

    def __init__(self, transformer: Transformer = TranformerUMAP()) -> None:
        # self.rounds = rounds
        self.transformer = transformer

    def _get_data(self, rounds: RoundsMd, num=4, out=1e4):

        pos_arr = []
        cell_arr = []
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
                #  jit(vmap(lambda x, y: cv.compute(
                #     SystemParams(coordinates=x, cell=y))[0]))
                map_metric = jit(vmap(cv.metric.map))

            pos = dictionary["positions"]
            cell = dictionary.get("cell", None)

            # execute all the mappings
            cvs = map_cv(pos, cell)
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

            # times.append(time)
            # taus.append(tau)
            pos_arr.append(pos)
            cvs_mapped_arr.append(cvs_mapped)
            # energies.append(biases)
            cell_arr.append(cell)
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

        pos = jnp.vstack(pos_arr)[indices]
        if cell is not None:
            cell = jnp.vstack(cell_arr)[indices]
        else:
            cell = None
        cvs = jnp.vstack(cvs_mapped_arr)[indices]

        return [pos, cell, cvs]

    def compute(self, rounds: RoundsMd,  plot=True, **kwargs) -> CV:

        x, _, cv = self._get_data(num=2, out=1e3, rounds=rounds)

        fit = True

        if fit:
            newCV = self.transformer.fit(x, output_metric=MetricUMAP(
                periodicities=[False, False]), **kwargs)

        if plot:

            if fit:

                def color(c):
                    c2 = (c[:]-c.min(axis=0))/(c.max(axis=0) - c.min(axis=0))
                    l = c2.shape[0]
                    w = c2.shape[1]

                    cv1 = np.ones((l, 3))
                    cv1[:, 0:w] = c2[:]

                    cv_hsv = matplotlib.colors.hsv_to_rgb(cv1)

                    return cv_hsv

                cv_func = jit(vmap(lambda x: newCV.compute(
                    sp=SystemParams(coordinates=x, cell=None))))
                tf = cv_func(x)[0]

                # tf = newCV(jnp.array(x))
                # tf = t(jnp.array(x))

                fig, ax_arr = plt.subplots(2, 2)

                # plot setting
                kwargs = {'s': 1}

                data = [cv, tf]
                labels = [[r'$\Phi$', r'$\Psi$'], ['umap1', 'umap2']]
                cmaps = [None, ]

                for [i, j] in [[0, 1], [1, 0]]:  # order

                    cols = color(data[i])
                    l, r = ax_arr[i]

                    # l.hist2d(data[i][:, 0], data[i][:, 1],
                    #          bins=(50, 50), cmap=plt.cm.gray, alpha=0.5)

                    l.scatter(data[i][:, 0], data[i]
                              [:, 1], c=cols, **kwargs)

                    l.set_xlabel(labels[i][0])
                    l.set_ylabel(labels[i][1])

                    r.scatter(data[j][:, 0], data[j]
                              [:, 1], c=cols, **kwargs)

                    # r.hist2d(data[j][:, 0], data[j][:, 1],
                    #          bins=(50, 50), cmap=plt.cm.gray, alpha=0.5)

                    r.set_xlabel(labels[j][0])
                    r.set_ylabel(labels[j][1])

                plt.show()

            else:
                fig, ax = plt.subplots(2, 2)
                ax.scatter(cv[:, 0], cv[:, 1])

        raise NotImplementedError
