from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import umap
from IMLCV.base.CV import CV
from IMLCV.base.CV import CvTrans
from IMLCV.base.CV import NeighbourList
from IMLCV.base.CVDiscovery import Transformer
from IMLCV.implementations.CV import get_sinkhorn_divergence
from IMLCV.implementations.CV import un_atomize
from IMLCV.implementations.tensorflow.CV import KerasFunBase
from IMLCV.implementations.tensorflow.CV import PeriodicLayer
from jax import jit
from jax import vmap
from pynndescent import NNDescent
from umap.umap_ import nearest_neighbors


def get_nn(x, nl, n_neigh, chunk_size=None):
    @partial(jit, static_argnames=("n_neigh", "chunk_size"))
    def _get_tree(x, nl, n_neigh, chunk_size=None):
        dist, _ = NeighbourList.sinkhorn_divergence(
            x.cv,
            x.cv,
            nl,
            nl,
            matching="average",
            alpha=1e-2,
            chunk_size=chunk_size,
        )

        dist_i, i = jax.lax.approx_min_k(dist, n_neigh)
        return i, dist_i

    i, di = _get_tree(x=x, nl=nl, n_neigh=n_neigh, chunk_size=chunk_size)
    return np.array(i), np.array(di), object.__new__(NNDescent)


class TranformerUMAP(Transformer):
    def _fit(
        self,
        x: list[CV],
        nl: list[NeighbourList] | None = None,
        decoder=False,
        nunits=256,
        nlayers=3,
        parametric=True,
        n_neighbors=20,
        # metric=None,
        chunk_size=None,
        **kwargs,
    ):
        x = CV.stack(*x)
        nl = sum(nl[1:], nl[0])

        tree = get_nn(x, nl, n_neighbors, chunk_size=chunk_size)

        x = un_atomize.compute_cv_trans(x, None)[0]

        dims = x.shape[1:]

        kwargs["n_components"] = self.outdim

        # kwargs["metric"] = "precomputed"

        # if metric is None:
        #     pl = PeriodicLayer(bbox=self.bounding_box, periodicity=self.periodicity)

        #     kwargs["output_metric"] = pl.metric
        # else:
        #     kwargs["output_metric"] = metric

        if parametric:
            from tensorflow import keras

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
            # if metric is None:
            #     layers.append(pl)

            encoder = keras.Sequential(layers)

            kwargs["encoder"] = encoder

            if decoder:
                decoder = keras.Sequential(
                    [
                        keras.layers.InputLayer(input_shape=(self.outdim)),
                        *[keras.layers.Dense(units=nunits, activation=act) for _ in range(nlayers)],
                        keras.layers.Dense(units=jnp.prod(jnp.array(x.shape[1:]))),
                    ],
                )
                kwargs["decoder"] = decoder

            reducer = umap.parametric_umap.ParametricUMAP(
                **kwargs,
                precomputed_knn=tree,
                force_approximation_algorithm=True,
            )
        else:
            reducer = umap.UMAP(**kwargs, precomputed_knn=tree, force_approximation_algorithm=True)

        reducer.fit_transform(X=x.cv)

        assert parametric
        f = CvTrans(trans=(KerasFunBase(reducer.encoder, reducer.decoder),))

        return f.compute_cv_trans(x)[0], un_atomize * f
