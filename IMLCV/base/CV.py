from __future__ import annotations

import typing
from abc import abstractmethod
from ast import Raise
from dataclasses import dataclass
from functools import partial
from typing import Callable, Collection, Iterable, List, Optional, Tuple, Union

import dill
import jax
# import numpy as np
import jax.numpy as jnp
import jax_dataclasses as jdc
from IMLCV.base.metric import Metric
from jax import jit


@jdc.pytree_dataclass
class SystemParams:
    coordinates: jnp.ndarray
    cell: Optional[jnp.ndarray]


sf = Callable[[SystemParams], jnp.ndarray]
tf = Callable[[jnp.ndarray], jnp.ndarray]  # flow from cv to cv


class cvflow:
    def __init__(self, cvs: Union[sf, Iterable[sf]], tranf: Optional[Union[tf, Iterable[tf]]] = None) -> None:

        def f0(x):
            # compose
            if isinstance(cvs, Iterable):
                y = jnp.array([h(x) for h in cvs])
            else:
                y = cvs(x)

            # serial
            if tranf is not None:
                if isinstance(tranf, Iterable):
                    for h in tranf:
                        y = h(y)
                else:
                    y = tranf(y)
            return y

        self.f0 = f0

    def __call__(self, x: SystemParams):
        return self.f0(x)


# converts system params to cv array


class CV:
    """base class for CVs.

    args:
        f: list of
        **kwargs: arguments of custom function f
    """

    def __init__(self, f: cvflow, metric: Metric) -> None:

        self.f = f
        self._update_params()

        self.metric = metric

    def compute(self, sp: SystemParams, jac_p=False, jac_c=False):
        """
        args:
            coodinates: cartesian coordinates, as numpy array of form (number of atoms,3)
            cell: cartesian coordinates of cell vectors, as numpy array of form (3,3)
            grad: if not None, is set to the to gradient of CV wrt coordinates
            vir: if not None, is set to the to gradient of CV wrt cell params
        """
        val = self.cv(sp)
        jac = self.jac_p(sp) if jac_p or jac_c else None

        return [val, jac]

    def _update_params(self):
        """update the CV functions."""

        self.cv = jit(lambda sp: (jnp.ravel(self.f(sp))))
        self.jac_p = jit(jax.jacfwd(self.cv))

    def __eq__(self, other):
        if not isinstance(other, CV):
            return NotImplemented
        return dill.dumps(self.cv) == dill.dumps(other.cv)

    @ property
    def n(self):
        return self.metric.ndim


class CVUtils:
    """collection of predifined CVs. Intended to be used as argument to CV
    class.

    args:
        coordinates (np.array(n_atoms,3)): cartesian coordinates
        cell (np.array((3,3)): cartesian coordinates of cell vectors
    """

    @ staticmethod
    def dihedral(numbers):
        """from https://stackoverflow.com/questions/20305272/dihedral-torsion-
        angle-from-four-points-in-cartesian- coordinates-in-python.

        args:
            numbers: list with index of 4 atoms that form dihedral
        """

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

    @ staticmethod
    def Volume():
        def f(sp: SystemParams):
            return jnp.abs(jnp.dot(sp.cell[0], jnp.cross(sp.cell[1], sp.cell[2])))
        return f

    @ staticmethod
    def rotate(alpha):
        def f(cv):
            return jnp.array([[jnp.cos(alpha), jnp.sin(alpha)], [-jnp.sin(alpha), jnp.cos(alpha)]]) @ cv
        return f
