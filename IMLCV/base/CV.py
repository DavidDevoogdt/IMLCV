from __future__ import annotations

import tempfile
import typing
from abc import abstractmethod
from ast import Raise
from dataclasses import dataclass
from functools import partial
from importlib import import_module
from types import MethodType
from typing import (Callable, Collection, Iterable, Iterator, List, Optional,
                    Tuple, Union)

import dill
import jax
# import numpy as np
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import tensorflow
import tensorflow as tfl
from IMLCV.base.metric import Metric
from jax import grad, jacfwd, jit, vmap
from jax.experimental.jax2tf import call_tf
# using the import module import the tensorflow.keras module
# and typehint that the type is KerasAPI module
from keras.api._v2 import keras as KerasAPI

keras: KerasAPI = import_module("tensorflow.keras")


@jdc.pytree_dataclass
class SystemParams:
    coordinates: jnp.ndarray
    cell: Optional[jnp.ndarray]

    @staticmethod
    def flatten_f(sps: Union[SystemParams, Iterable[SystemParams]], scale=True) -> jnp.ndarray:

        def f(sps):
            out = [jnp.stack([sp.coordinates for sp in sps], axis=0)]

            if sps[0].cell is not None:
                if sps[0].cell.shape[0] != 0:
                    out.append(jnp.stack([sp.cell for sp in sps], axis=0))
            return out

        out = f(sps)

        if scale:
            bounds = []

            for a in out:
                a = jnp.reshape(a, (-1, 3))
                bounds.append(jnp.array([a.min(axis=0), a.max(axis=0)]))

            def g(x):
                return [(y-b[0, :])/(b[1, :] - b[0, :]) for y, b in zip(x, bounds)]

            out = g(out)

            def h(x): return g(f(x))
        else:
            h = f

        def i(x): return jnp.hstack(
            [jnp.reshape(y, (y.shape[0], -1)) for y in x])

        return i(out), lambda x: i(h(x))

    @ staticmethod
    def map_params(coordinates, cells):
        if cells is None:
            return [SystemParams(coordinates=a, cell=None) for a in coordinates]
        else:
            return [SystemParams(coordinates=a, cell=b) for a, b in zip(coordinates, cells)]


sf = Callable[[SystemParams], jnp.ndarray]
tf = Callable[[jnp.ndarray], jnp.ndarray]


class CvTrans:
    def __init__(self, f: tf) -> None:
        self.f = f

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.f(x)


class CvFlow:
    def __init__(self, func: sf) -> None:
        self.f0 = func

    def __call__(self, x: SystemParams):
        return self.f0(x)

    def __add__(self, other):
        assert isinstance(other, CvFlow)

        def f0(x):
            return jnp.array([self(x), other(x)])

        return CvFlow(func=f0)

    def __mul__(self, other):
        assert isinstance(
            other, CvTrans), 'can only multiply by CvTrans object'

        def f0(x):
            return other(self(x))

        return CvFlow(func=f0)


class PeriodicLayer(keras.layers.Layer):
    def __init__(self, bbox, periodicity, **kwargs):
        super().__init__(**kwargs)

        self.bbox = np.array(bbox, dtype=np.float32)
        self.periodicity = np.array(periodicity)

    def call(self, inputs):
        # maps to periodic box
        bbox = self.bbox

        inputs_mod = tfl.math.mod(
            inputs - bbox[:, 0], bbox[:, 1] - bbox[:, 0])+bbox[:, 0]
        return tfl.where(self.periodicity,  inputs_mod, inputs)

    def metric(self, r):
        # maps difference
        a = self.bbox[:, 1] - self.bbox[:, 0]

        r = tfl.math.mod(r, a)
        r = tfl.where(r > a/2, r-a, r)
        return tfl.norm(r, axis=1)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "bbox": self.bbox,
            "periodicity": self.periodicity,
        })
        return config


class KerasFlow(CvFlow):
    def __init__(self, encoder, f) -> None:
        self.encoder = encoder
        self.f = f

    def __call__(self, x: SystemParams):
        cc = self.f([x])
        out = call_tf(self.encoder.call)(cc)
        return jnp.reshape(out, out.shape[1:])

    def __getstate__(self):
        # https://stackoverflow.com/questions/48295661/how-to-pickle-keras-model
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            tensorflow.keras.models.save_model(
                self.encoder, fd.name, overwrite=True)
            model_str = fd.read()
        d = {'model_str': model_str,
             'f': self.f}
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()

            custom_objects = {"PeriodicLayer": PeriodicLayer}
            with keras.utils.custom_object_scope(custom_objects):
                model = keras.models.load_model(fd.name)

        self.encoder = model
        self.f = state['f']

# decorators definition for functions


def cv(func: sf):
    """decorator to make a CV"""
    ff = CvFlow(func=func)
    return ff


def cvtrans(f: tf):
    """decorator to make a CV tranformation func"""
    ff = CvTrans(f=f)
    return ff


class CV:

    def __init__(self, f: CvFlow, metric: Metric, jac=jacfwd) -> None:
        "jac: kind of jacobian. Default is jacfwd (more efficient for tall matrices), but functions with custom jvp's only support jacrev"

        # self.f = f
        # self.jac = jac
        self.metric = metric
        self.cv = jit(lambda sp: (jnp.ravel(f(sp))))
        self.jac_p = jit(jac(self.cv))

    def compute(self, sp: SystemParams, jac_p=False, jac_c=False):

        val = self.cv(sp)
        jac = self.jac_p(sp) if jac_p or jac_c else None

        return [val, jac]

    def __eq__(self, other):
        if not isinstance(other, CV):
            return NotImplemented
        return dill.dumps(self.cv) == dill.dumps(other.cv)

    def map_cv(self, sps: Iterable[SystemParams]):
        return jnp.array([self.compute(x)[0] for x in sps])

    @ property
    def n(self):
        return self.metric.ndim


def dihedral(numbers):
    """from https://stackoverflow.com/questions/20305272/dihedral-torsion-
    angle-from-four-points-in-cartesian- coordinates-in-python.

    args:
        numbers: list with index of 4 atoms that form dihedral
    """

    @ cv
    def f(sp: SystemParams):
        p0 = sp.coordinates[numbers[0]]
        p1 = sp.coordinates[numbers[1]]
        p2 = sp.coordinates[numbers[2]]
        p3 = sp.coordinates[numbers[3]]

        b0 = -1.0 * (p1 - p0)
        b1 = p2 - p1
        b2 = p3 - p2

        b1 /= jnp.linalg.norm(b1)

        v = b0 - jnp.dot(b0, b1) * b1
        w = b2 - jnp.dot(b2, b1) * b1

        x = jnp.dot(v, w)
        y = jnp.dot(jnp.cross(b1, v), w)
        return jnp.arctan2(y, x)

    return f


@ cv
def Volume(sp: SystemParams):
    return jnp.abs(jnp.dot(sp.cell[0], jnp.cross(sp.cell[1], sp.cell[2])))


def rotate_2d(alpha):
    @ cvtrans
    def f(cv):
        return jnp.array([
            [jnp.cos(alpha), jnp.sin(alpha)],
            [-jnp.sin(alpha), jnp.cos(alpha)]
        ]) @ cv
    return f
