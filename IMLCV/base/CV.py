from functools import partial
import jax.numpy as np
from jax import jit, grad


class CV:
    """base class for CVs.

    args:
        f: function f(coordinates, cell, **kwargs) that returns the CV
        **kwargs: arguments of custom function f
    """

    def __init__(self, f, **kwargs) -> None:

        self.f = f
        self.kwargs = kwargs
        self.update_params(**kwargs)

    def compute(self, coordinates, cell, grad=None, vir=None):
        """
        args:
            coodinates: cartesian coordinates, as numpy array of form (number of atoms,3)
            cell: cartesian coordinates of cell vectors, as numpy array of form (3,3)
            grad: if not None, is set to the to gradient of CV wrt coordinates
            vir: if not None, is set to the to gradient of CV wrt cell params
        """

        if (grad is not None):
            grad = self.grad(coordinates, cell)
        if (vir is not None):
            vir = self.vir(coordinates, cell)

        return self.cv(coordinates, cell)

    def update_params(self, **kwargs):
        """update the CV functions."""
        self.cv = jit(partial(self.f, **kwargs))
        self.grad = jit(grad(self.cv, argnums=(0)))
        self.vir = jit(grad(self.cv, argnums=(1)))


class CVUtils:
    """collection of predifined CVs. Intended to be used as argument to CV
    class.

    args:
        coordinates (np.array(n_atoms,3)): cartesian coordinates
        cell (np.array((3,3)): cartesian coordinates of cell vectors
    """

    @staticmethod
    def dihedral(coordinates, cell, numbers):
        """from https://stackoverflow.com/questions/20305272/dihedral-torsion-
        angle-from-four-points-in-cartesian-coordinates-in-python.

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

        b1 /= np.linalg.norm(b1)

        v = b0 - np.dot(b0, b1) * b1
        w = b2 - np.dot(b2, b1) * b1

        x = np.dot(v, w)
        y = np.dot(np.cross(b1, v), w)
        return np.arctan2(y, x)


def Volume(coordinates, cell):
    return np.abs(np.dot(cell[0], np.cross(cell[1], cell[2])))
