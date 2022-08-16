from __future__ import annotations

import typing
from abc import abstractmethod
from ast import Raise
from dataclasses import dataclass
from functools import partial
from types import MethodType
from typing import (Callable, Collection, Iterable, Iterator, List, Optional,
                    Tuple, Union)

import dill
import jax
# import numpy as np
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
from IMLCV.base.metric import Metric
from jax import grad, jacfwd, jit, vmap


@jdc.pytree_dataclass
class SystemParams:
    coordinates: jnp.ndarray
    cell: Optional[jnp.ndarray]

    @staticmethod
    def flatten(sps: Union[SystemParams, Iterable[SystemParams]]) -> jnp.ndarray:

        def fl(sp):
            if sp.cell is not None:
                if sp.cell.shape[0] != 0:
                    return jnp.vstack([jnp.ravel(sp.coordinates), jnp.ravel(sp.cell)])

            return jnp.ravel(sp.coordinates)

        if isinstance(sps, Iterable):
            return jnp.vstack([fl(sp) for sp in sps])

        return fl(sps)

    @staticmethod
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


def cv(func: sf):
    """decorator to make a CV"""
    ff = CvFlow(func=func)
    return ff


def cvtrans(f: tf):
    """decorator to make a CV tranformation func"""
    ff = CvTrans(f=f)
    return ff


# converts system params to cv array


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
