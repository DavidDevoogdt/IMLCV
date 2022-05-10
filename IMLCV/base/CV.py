from __future__ import annotations

from functools import partial
from typing import Iterable

import dill
import jax
# import numpy as np
import jax.numpy as jnp
from IMLCV.base.metric import Metric
from jax import jit


class CV:
    """base class for CVs.

    args:
        f: function f(coordinates, cell, **kwargs) that returns a single CV
        **kwargs: arguments of custom function f
    """

    def __init__(self, f, metric: Metric, n=1, **kwargs) -> None:

        self.f = f
        self.kwargs = kwargs
        self._update_params(**kwargs)
        self.n = n

        self.metric = metric

    def compute(self, coordinates, cell, jac_p=False, jac_c=False):
        """
        args:
            coodinates: cartesian coordinates, as numpy array of form (number of atoms,3)
            cell: cartesian coordinates of cell vectors, as numpy array of form (3,3)
            grad: if not None, is set to the to gradient of CV wrt coordinates
            vir: if not None, is set to the to gradient of CV wrt cell params
        """
        val = self.cv(coordinates, cell)
        jac_p_val = self.jac_p(coordinates, cell) if jac_p else None
        jac_c_val = self.jac_c(coordinates, cell) if jac_c else None

        return [val, jac_p_val, jac_c_val]

    def _update_params(self, **kwargs):
        """update the CV functions."""

        self.cv = jit(lambda x, y: (jnp.ravel(partial(self.f, **kwargs)
                                              (x, y))))
        self.jac_p = jit(jax.jacfwd(self.cv, argnums=(0)))
        self.jac_c = jit(jax.jacfwd(self.cv, argnums=(1)))

    def __eq__(self, other):
        if not isinstance(other, CV):
            return NotImplemented
        return dill.dumps(self.cv) == dill.dumps(other.cv)


class CombineCV(CV):
    """combine multiple CVs into one CV."""

    def __init__(self, cvs: Iterable[CV]) -> None:
        self.n = 0
        self.cvs = cvs
        metric = None

        for cv in cvs:
            self.n += cv.n
            if metric is None:
                metric = cv.metric
            else:
                metric += cv.metric

        self.metric = metric
        self._update_params()

    def _update_params(self):
        """function selects on ouput according to index."""

        def f(x, y, cvs):
            return jnp.array([cv.cv(x, y) for cv in cvs])

        self.f = partial(f, cvs=self.cvs)

        super()._update_params()


class CVUtils:
    """collection of predifined CVs. Intended to be used as argument to CV
    class.

    args:
        coordinates (np.array(n_atoms,3)): cartesian coordinates
        cell (np.array((3,3)): cartesian coordinates of cell vectors
    """

    @ staticmethod
    def dihedral(coordinates, _, numbers):
        """from https://stackoverflow.com/questions/20305272/dihedral-torsion-
        angle-from-four-points-in-cartesian- coordinates-in-python.

        args:
            numbers: list with index of 4 atoms that form dihedral
        """
        p0 = coordinates[numbers[0]]
        p1 = coordinates[numbers[1]]
        p2 = coordinates[numbers[2]]
        p3 = coordinates[numbers[3]]

        b0 = -1.0 * (p1 - p0)
        b1 = p2 - p1
        b2 = p3 - p2

        b1 /= jnp.linalg.norm(b1)

        v = b0 - jnp.dot(b0, b1) * b1
        w = b2 - jnp.dot(b2, b1) * b1

        x = jnp.dot(v, w)
        y = jnp.dot(jnp.cross(b1, v), w)
        return jnp.arctan2(y, x)

    @ staticmethod
    def Volume(_, cell):
        return jnp.abs(jnp.dot(cell[0], jnp.cross(cell[1], cell[2])))

    @ staticmethod
    def linear_combination(cv1, cv2, a=1, b=1):
        return lambda x, y: a * cv1(x, y) + b * cv2(x, y)

    @ staticmethod
    def rotate(alpha, cv1: CV, cv2: CV):

        def f(x, y):
            a = cv1(x, y)
            b = cv2(x, y)
            return jnp.array([jnp.cos(alpha)*a + jnp.sin(alpha)*b, -jnp.sin(alpha)*a + jnp.cos(alpha)*b])
        return f
