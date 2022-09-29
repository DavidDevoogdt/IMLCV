from __future__ import annotations

import itertools
import tempfile
from collections.abc import Iterable
from functools import partial
from importlib import import_module
from typing import Callable

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import tensorflow
import tensorflow as tfl
from jax import jacfwd, jit, vmap
from jax.experimental.jax2tf import call_tf
from keras.api._v2 import keras as KerasAPI

from IMLCV.base.metric import Metric

keras: KerasAPI = import_module("tensorflow.keras")


@jdc.pytree_dataclass
class SystemParams:
    coordinates: jnp.ndarray
    cell: jnp.ndarray | None = None

    def __post_init__(self):
        if isinstance(self.coordinates, np.ndarray):
            self.__dict__["coordinates"] = jnp.array(self.coordinates)
        if isinstance(self.cell, np.ndarray):
            self.__dict__["cell"] = jnp.array(self.cell)
        if isinstance(self.cell, jnp.ndarray):
            if jnp.size(self.cell) == 0:
                self.__dict__["cell"] = None

    def __getitem__(self, slices):
        return SystemParams(
            coordinates=self.coordinates[slices],
            cell=(self.cell[slices] if self.cell is not None else None),
        )

    def __iter__(self):
        if not self.batched:
            yield self
            return
        for i in range(self.coordinates.shape[0]):
            yield SystemParams(
                coordinates=self.coordinates[i, :, :],
                cell=self.cell[i, :, :] if self.cell is not None else None,
            )
        return

    @property
    def batched(self):
        return len(self.coordinates.shape) == 3

    @property
    def shape(self):
        return self.coordinates.shape

    @staticmethod
    def stack(arr: list[SystemParams]):
        coordinates = []
        cell = []
        has_cell = arr[0].cell is not None

        for x in arr:
            coordinates.append(x.coordinates)

            if has_cell:
                assert x.cell is not None
                cell.append(x.cell)
            else:
                assert x.cell is None

        ncoordinates = jnp.vstack(coordinates)
        if has_cell:
            ncell = jnp.vstack(cell)
        else:
            ncell = None

        return SystemParams(coordinates=ncoordinates, cell=ncell)


sf = Callable[[SystemParams], jnp.ndarray]
tf = Callable[[jnp.ndarray], jnp.ndarray]


class CvTrans:
    def __init__(self, f: tf, batched=False) -> None:
        self.f = f
        self.batched = batched

    @partial(jit, static_argnums=(0,))
    def compute(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.batched:
            return self.f(x)
        else:
            return vmap(self.f)(x)


class CvFlow:
    def __init__(
        self,
        func: sf,
        trans: CvTrans | Iterable[CvTrans] | None = None,
        batched=False,
    ) -> None:
        self.f0 = func
        if trans is None:
            self.f1: Iterable[CvTrans] = []
        else:
            if isinstance(trans, Iterable):
                self.f1 = trans
            else:
                self.f1 = [trans]

        self.batched = batched

    @partial(jit, static_argnums=(0,))
    def compute(self, x: SystemParams):

        # make output batched
        if x.batched:
            if self.batched:  # fallback for unbatched f0's
                out = self.f0(x)
            else:
                out = jax.lax.map(self.f0, x)

            if len(out.shape) == 1:  # single cv
                out = out.reshape((*out.shape, 1))
        else:
            if self.batched:
                return self.f0(x)
            else:
                out = jnp.array([self.f0(x)])

        # prepare for batchehd cftrans
        if len(out.shape) == 1:
            out = out.reshape((1, *out.shape))

        for other in self.f1:
            out = other.compute(out)

        # undo batching for unbatched systemparams
        if not x.batched:
            if len(out.shape) == 2:
                assert out.shape[0] == 1
                out = out[0, :]

        return out

    def __add__(self, other):
        assert isinstance(other, CvFlow)

        def f0(x):
            return jnp.hstack([self.compute(x), other.compute(x)])

        return CvFlow(func=f0, batched=True)

    def __mul__(self, other):
        assert isinstance(other, CvTrans), "can only multiply by CvTrans object"

        return CvFlow(func=self.f0, trans=[*self.f1, other], batched=self.batched)


class PeriodicLayer(keras.layers.Layer):
    def __init__(self, bbox, periodicity, **kwargs):
        super().__init__(**kwargs)

        self.bbox = tfl.Variable(np.array(bbox, dtype=np.float32))
        self.periodicity = np.array(periodicity)

    def call(self, inputs):
        # maps to periodic box
        bbox = self.bbox

        inputs_mod = (
            tfl.math.mod(inputs - bbox[:, 0], bbox[:, 1] - bbox[:, 0]) + bbox[:, 0]
        )
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
            }
        )
        return config


class KerasTrans(CvTrans):
    def __init__(self, encoder) -> None:
        self.encoder = encoder

    @partial(jit, static_argnums=(0,))
    def compute(self, cc: jnp.ndarray):
        out = call_tf(self.encoder.call)(cc)
        return out

    def __getstate__(self):
        # https://stackoverflow.com/questions/48295661/how-to-pickle-keras-model
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as fd:
            tensorflow.keras.models.save_model(self.encoder, fd.name, overwrite=True)
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


# decorators definition for functions


def cvflow(func: sf):
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

        self.metric = metric
        self.f = f
        self.jac = jac

    @partial(jit, static_argnums=(0, 2, 3))
    def compute(self, sp: SystemParams, jacobian=False, map=False):

        if map:

            def cvf(x):
                return vmap(self.metric.map)(self.f.compute(x))

        else:
            cvf = self.f.compute

        if sp.batched:
            jac = vmap(self.jac(cvf))
        else:
            jac = self.jac(cvf)

        val = cvf(sp)
        jac = jac(sp) if jacobian else None

        return [val, jac]

    @property
    def n(self):
        return self.metric.ndim


def dihedral(numbers):
    """from https://stackoverflow.com/questions/20305272/dihedral-torsion-
    angle-from-four-points-in-cartesian- coordinates-in-python.

    args:
        numbers: list with index of 4 atoms that form dihedral
    """

    @cvflow
    def f(sp: SystemParams):

        # @partial(vmap, in_axes=(0), out_axes=(0))
        coor = sp.coordinates
        p0 = coor[numbers[0]]
        p1 = coor[numbers[1]]
        p2 = coor[numbers[2]]
        p3 = coor[numbers[3]]

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


@cvflow
def Volume(sp: SystemParams):
    assert sp.cell is not None, "can only calculate volume if there is a unit cell"
    return jnp.abs(jnp.dot(sp.cell[0], jnp.cross(sp.cell[1], sp.cell[2])))


def rotate_2d(alpha):
    @cvtrans
    def f(cv):
        return (
            jnp.array(
                [[jnp.cos(alpha), jnp.sin(alpha)], [-jnp.sin(alpha), jnp.cos(alpha)]]
            )
            @ cv
        )

    return f


def scale_cv_trans(array=jnp.ndarray):
    "axis 0 is batch axis"
    maxi = jnp.max(array, axis=0)
    mini = jnp.min(array, axis=0)
    diff = maxi - mini
    mask = jnp.abs(diff) > 1e-6

    def f0(x):
        return (x[mask] - mini[mask]) / diff[mask]

    f = CvTrans(f=f0)

    return f.compute(array), f


def coulomb_descriptor_cv_flow(sps: SystemParams, permutation="l2"):
    @cvflow
    def h(x: SystemParams):
        raise NotImplementedError
        assert x.masses is not None, "Z array in systemparams for coulomb descriptor"

        coor = x.coordinates

        n = coor.shape[0]
        out = jnp.zeros((n, n))

        for i in range(n):
            d = 0.5 * x.masses[i] ** 2.4
            out = out.at[i, i].set(d)

        for i, j in itertools.combinations(range(n), 2):
            d = jnp.linalg.norm(coor[i, :] - coor[j, :], 2)
            d = x.masses[i] * x.masses[j] / d
            out = out.at[i, j].set(d)
            out = out.at[j, i].set(d)

        if permutation == "l2":

            ind = jnp.argsort(jnp.linalg.norm(out, 2, axis=(0)))
            out = out[ind, :]
            out = out[:, ind]
        elif permutation == "none":
            pass
        else:
            raise NotImplementedError

        return out
        # flatten relevant coordinates
        # return out[jnp.triu_indices(n)]
        # return g(x.coordinates)

    return h.compute(sps), h
