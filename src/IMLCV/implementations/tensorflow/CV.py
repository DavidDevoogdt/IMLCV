import tempfile
from importlib import import_module

import numpy as np
import tensorflow as tfl
from IMLCV.base.CV import CV
from IMLCV.base.CV import CvFunBase
from IMLCV.base.CV import NeighbourList
from jax.experimental.jax2tf import call_tf
from keras.api._v2 import keras as KerasAPI

keras: KerasAPI = import_module("tensorflow.keras")


class PeriodicLayer(keras.layers.Layer):
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
    def __init__(self, forward: tfl.Module, backward: tfl.Module | None) -> None:
        self.fwd = forward
        self.bwd = backward

    def _calc(
        self,
        x: CV,
        nl: NeighbourList,
        reverse=False,
        conditioners: list[CV] | None = None,
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
            out = call_tf(self.bwd.call, has_side_effects=False)(y)
        else:
            assert self.fwd is not None
            out = call_tf(self.fwd.call, has_side_effects=False)(y)

        if not batched:
            out = out.reshape((-1,))

        return x.replace(cv=out)

    def __getstate__(self):
        # https://stackoverflow.com/questions/48295661/how-to-pickle-keras-model

        fwd_str = ""
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as fd:
            tfl.keras.models.save_model(self.fwd, fd.name, overwrite=True)
            fwd_str = fd.read()
        d = {"fwd_str": fwd_str}

        if self.bwd is not None:
            bwd_str = ""
            with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as fd:
                tfl.keras.models.save_model(self.bwd, fd.name, overwrite=True)
                bwd_str = fd.read()
            d["bwd_str"] = bwd_str

        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as fd:
            fd.write(state["fwd_str"])
            fd.flush()

            custom_objects = {"PeriodicLayer": PeriodicLayer}
            with keras.utils.custom_object_scope(custom_objects):
                fwd = keras.models.load_model(fd.name)
        self.fwd = fwd

        if (bwd_str := state.get("bwd_str", None)) is not None:
            with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as fd:
                fd.write(bwd_str)
                fd.flush()

                custom_objects = {"PeriodicLayer": PeriodicLayer}
                with keras.utils.custom_object_scope(custom_objects):
                    bwd = keras.models.load_model(fd.name)

            self.bwd = bwd
