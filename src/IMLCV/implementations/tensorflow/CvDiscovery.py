from dataclasses import KW_ONLY

import haiku as hk
import jax.numpy as jnp
import umap
from equinox import Partial
from flax.struct import field
from flax.struct import PyTreeNode
from IMLCV.base.CV import CV
from IMLCV.base.CV import CvFunBase
from IMLCV.base.CV import CvTrans
from IMLCV.base.CV import NeighbourList
from IMLCV.base.CVDiscovery import Transformer
from IMLCV.implementations.CV import un_atomize
import numpy
from IMLCV.base.rounds import data_loader_output


def umap_function(x: CV, nl: NeighbourList, c, enc):
    assert not x.batched

    print(f"{x.cv.shape=}")

    cv = enc(x.batch().cv)
    cv = cv.reshape(cv.shape[1:])

    return x.replace(cv=cv)


def umap_encoder(x, nlayers, nunits, outdim):
    for i in range(nlayers):
        x = hk.Linear(nunits)(x)
        x = jnp.tanh(x)

    return hk.Linear(outdim)(x)


class hkFunBase(CvFunBase, PyTreeNode):
    _: KW_ONLY
    fwd_params: dict
    fwd_kwargs: dict = field(pytree_node=False)

    bwd_params: dict | None = field(pytree_node=True, default=None)
    bwd_kwargs: dict | None = field(pytree_node=False, default=None)

    def _calc(
        self,
        x: CV,
        nl: NeighbourList,
        reverse=False,
        conditioners: list[CV] | None = None,
        shmap=True,
    ) -> CV:
        assert conditioners is None
        assert not reverse

        batched = x.batched
        if not batched:
            y = x.cv.reshape((1, -1))
        else:
            y = x.cv

        if reverse:
            assert self.bwd is not None, "No backward model defined"
            raise NotImplementedError

        else:
            fwd = hk.transform(Partial(umap_encoder, **self.fwd_kwargs))
            out = fwd.apply(self.fwd_params, None, y)

        if not batched:
            out = out.reshape((-1,))

        return x.replace(cv=out)


class TranformerUMAP(Transformer):
    def __init__(
        self,
        outdim=2,
        decoder=False,
        nunits=256,
        nlayers=3,
        parametric=True,
        densmap=False,
        n_neighbors=20,
        **kwargs,
    ):
        super().__init__(
            outdim=outdim,
            decoder=decoder,
            nunits=nunits,
            nlayers=nlayers,
            parametric=parametric,
            n_neighbors=n_neighbors,
            densmap=densmap,
            **kwargs,
        )

    def _fit(
        self,
        x: list[CV],
        x_t: list[CV] | None,
        dlo: data_loader_output,
        decoder=False,
        nunits=256,
        nlayers=3,
        parametric=True,
        densmap=False,
        n_neighbors=20,
        # metric=None,
        chunk_size=None,
        verbose=True,
        macro_chunk=1000,
        **kwargs,
    ):
        x = CV.stack(*x)
        x = un_atomize.compute_cv_trans(x, None)[0]
        if x_t is not None:
            x_t = CV.stack(*x_t)
            x_t = un_atomize.compute_cv_trans(x_t, None)[0]

        x_train = x

        print(f"{x_train.cv.shape=}")

        dims = (x_train.shape[1],)

        kwargs["n_components"] = self.outdim
        kwargs["n_neighbors"] = n_neighbors
        kwargs["densmap"] = densmap

        if parametric:
            # import tensorflow as tf
            import tf_keras as keras

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

            kwargs["encoder"] = keras.Sequential(layers)

            if decoder:
                raise NotImplementedError
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
                dims=dims,
            )
        else:
            reducer = umap.UMAP(**kwargs)

        trans_1 = reducer.fit_transform(X=numpy.array(x_train.cv))

        fwd_kwargs = dict(
            nlayers=nlayers,
            nunits=nunits,
            outdim=self.outdim,
        )

        hk_encoder = hk.transform(Partial(umap_encoder, **fwd_kwargs))

        params = {}

        for i, layer in enumerate(reducer.encoder.layers):
            w, b = layer.get_weights()
            name = f"linear_{i}" if i != 0 else "linear"
            params[name] = {"w": jnp.array(w), "b": jnp.array(b)}

        trans_2 = hk_encoder.apply(params, None, x_train.cv)

        assert jnp.allclose(trans_1, trans_2, atol=1e-4, rtol=1e-4)

        assert parametric
        f = CvTrans.from_cv_fun(hkFunBase(fwd_params=params, fwd_kwargs=fwd_kwargs))

        cv_0 = f.compute_cv_trans(x, chunk_size=chunk_size)[0].unstack()
        cv_tau = f.compute_cv_trans(x_t, chunk_size=chunk_size)[0].unstack() if x_t is not None else None

        return cv_0, cv_tau, un_atomize * f, None
