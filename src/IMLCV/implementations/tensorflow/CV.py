import tempfile
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
from flax.struct import dataclass, field
from jax.custom_batching import custom_vmap
from jax.experimental.jax2tf import call_tf

from IMLCV.base.CV import CV, CvFunBase, NeighbourList

if TYPE_CHECKING:
    import tensorflow as tfl


@partial(dataclass, frozen=False, eq=False)
class tfl_module:
    mod: tfl.Module = field(pytree_node=False, default=None)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "bbox": np.array(self.bbox),
                "periodicity": self.periodicity,
            },
        )
        return config

    def __getstate__(self):
        # https://stackoverflow.com/questions/48295661/how-to-pickle-keras-model
        import tensorflow as tfl

        with tempfile.NamedTemporaryFile(suffix=".keras", delete=True) as fd:
            tfl.keras.models.save_model(self.mod, fd.name, overwrite=True)
            mod_str = fd.read()
        return {"mod_str": mod_str}

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix=".keras", delete=True) as fd:
            fd.write(state["mod_str"])
            fd.flush()

            import tensorflow as tfl
            import tf_keras as keras

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

            custom_objects = {"PeriodicLayer": PeriodicLayer}
            with keras.utils.custom_object_scope(custom_objects):
                mod = keras.models.load_model(fd.name)

        self.__init__(mod=mod)


class KerasFunBase(CvFunBase):
    fwd: tfl_module = field(pytree_node=False)
    bwd: tfl_module | None = field(pytree_node=False, default=None)

    def create(fwd: tfl.Module, bwd: tfl.Module = None):
        return KerasFunBase(fwd=tfl_module(fwd), bwd=tfl_module(bwd) if bwd is not None else None)

    def _calc(
        self,
        x: CV,
        nl: NeighbourList,
        reverse=False,
        conditioners: list[CV] | None = None,
        shmap=False,
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

            out = call_tf(self.bwd.mod.call, has_side_effects=False)(y)
        else:
            assert self.fwd is not None

            @custom_vmap
            def forward(y):
                return call_tf(self.fwd.mod.call, has_side_effects=False)(y)

            @forward.def_vmap
            def _forward_vmap(axis_size, in_batched, y):
                (x_batched,) = in_batched

                assert x_batched

                return forward(y.reshape((axis_size * y.shape[1], *y.shape[2:]))), True

            out = forward(y)

        if not batched:
            out = out.reshape((-1,))

        return x.replace(cv=out)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__init__(**state)
