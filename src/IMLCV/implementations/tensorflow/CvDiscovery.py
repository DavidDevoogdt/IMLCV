import jax.numpy as jnp
import matplotlib.pyplot as plt
import umap
from IMLCV.base.CV import CvTrans
from IMLCV.base.CVDiscovery import Transformer
from IMLCV.implementations.tensorflow.CV import KerasFunBase
from IMLCV.implementations.tensorflow.CV import PeriodicLayer


class TranformerUMAP(Transformer):
    def _fit(
        self,
        x,
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
            if metric is None:
                layers.append(pl)

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

            reducer = umap.parametric_umap.ParametricUMAP(**kwargs)
        else:
            reducer = umap.UMAP(**kwargs)

        reducer.fit(x)

        assert parametric

        f = CvTrans(trans=[KerasFunBase(reducer.encoder)])
        return f.compute_cv_trans(x)[0], f
