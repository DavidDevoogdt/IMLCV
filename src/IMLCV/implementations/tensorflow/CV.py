from __future__ import annotations

import tempfile
from importlib import import_module
from typing import TYPE_CHECKING

import numpy as np
import tensorflow as tfl
from IMLCV.base.CV import CV
from IMLCV.base.CV import CvFunBase
from jax.experimental.jax2tf import call_tf

# from IMLCV.base.CV import CV, CvTrans
# import numpy as np

if TYPE_CHECKING:
    pass

from keras.api._v2 import keras as KerasAPI

keras: KerasAPI = import_module("tensorflow.keras")


class PeriodicLayer(keras.layers.Layer):
    import tensorflow as tfl

    def __init__(self, bbox, periodicity, **kwargs):
        super().__init__(**kwargs)

        self.bbox = tfl.Variable(np.array(bbox))
        self.periodicity = np.array(periodicity)

    def call(self, inputs):
        # maps to periodic box
        bbox = self.bbox

        inputs_mod = tfl.math.mod(inputs - bbox[:, 0], bbox[:, 1] - bbox[:, 0]) + bbox[:, 0]
        return tfl.where(self.periodicity, inputs_mod, inputs)

    def metric(self, r):
        # maps difference
        a = self.bbox[:, 1] - self.bbox[:, 0]

        r = tfl.math.mod(r, a)
        r = tfl.where(r > a / 2, r - a, r)
        return tfl.norm(r, axis=1)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "bbox": np.array(self.bbox),
                "periodicity": self.periodicity,
            },
        )
        return config


class KerasFunBase(CvFunBase):
    def __init__(self, encoder) -> None:
        self.encoder = encoder

    def _calc(self, x: CV, *conditioners: CV, reverse=False) -> CV:
        assert len(conditioners) == 0
        assert not reverse

        return call_tf(self.encoder.call)(x.cv)

    def __getstate__(self):
        # https://stackoverflow.com/questions/48295661/how-to-pickle-keras-model

        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("tensorflow not installed, cannot pickle keras model")

        model_str = ""
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as fd:
            tf.keras.models.save_model(self.encoder, fd.name, overwrite=True)
            model_str = fd.read()
        d = {"model_str": model_str}
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as fd:
            fd.write(state["model_str"])
            fd.flush()

            custom_objects = {"PeriodicLayer": PeriodicLayer}
            with keras.utils.custom_object_scope(custom_objects):
                model = keras.models.load_model(fd.name)

        self.encoder = model