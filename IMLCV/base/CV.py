from functools import partial
import jax
import numpy as np
import jax.numpy as jnp
from jax import jit, grad


class CV:
    """base class for CVs.

    args:
        f: function f(coordinates, cell, **kwargs) that returns a single CV
        **kwargs: arguments of custom function f
    """

    def __init__(self, f, n=1, periodicity=None, **kwargs) -> None:

        self.f = f
        self.kwargs = kwargs
        self._update_params(**kwargs)
        self.n = n

        #figure out bounds for CV
        if periodicity is not None:
            if isinstance(periodicity, float):
                assert n == 1
                periodicity = [[0.0, periodicity]]
            if isinstance(periodicity, list):
                periodicity = np.array(periodicity)
            assert periodicity.shape[0] == n
            if periodicity.shape[1] == 1:
                periodicity = np.concatenate((0.0 * periodicity, periodicity), axis=1)

            assert periodicity.shape == (n, 2)
        else:
            periodicity = np.array([[np.NaN, np.NaN] * n])

        self.periodicity = periodicity

    def compute(self, coordinates, deriv=True):
        """used as a thermolib collective variable."""

        if deriv == True:
            gpos = np.zeros(coordinates.shape)
            ener = self.compute(coordinates, cell=None, gpos=gpos)
            return ener, gpos
        else:
            return self.compute(coordinates, cell=None)

    def compute(self, coordinates, cell, gpos=None, vir=None):
        """
        args:
            coodinates: cartesian coordinates, as numpy array of form (number of atoms,3)
            cell: cartesian coordinates of cell vectors, as numpy array of form (3,3)
            grad: if not None, is set to the to gradient of CV wrt coordinates
            vir: if not None, is set to the to gradient of CV wrt cell params
        """

        if (gpos is not None):
            self.cv_grad(coordinates, cell, gpos)
        if (vir is not None):
            self.vir(coordinates, cell, vir)

        return self.cv(coordinates, cell)

    def _update_params(self, **kwargs):
        """update the CV functions."""

        if kwargs is not None:
            self.cv = jit(partial(self.f, **kwargs))
        else:
            self.cv = jit(self.f)

        @jit
        def cv_grad(coordinates, cell, gpos):
            """calculates viral as tau*dcv(coor,cell)/dtau."""
            gpos = gpos.at[:].set(grad(self.cv, argnums=(0))(coordinates, cell))

        self.cv_grad = cv_grad

        @jit
        def vir(coord, cell, vir):
            """calculates viral as tau*dcv(coor,cell)/dtau."""
            if cell is not None:
                if cell.shape == (3, 3):
                    vir = vir.at[:].set(jnp.matmul(jnp.transpose(cell), grad(self.cv, argnums=(1))(coord, cell)))
                    return
            vir = vir.at[:].set(0.0)

        self.vir = vir

    def split_cv(self):
        """Split the given CV in list of n=1 CVs."""
        if self.n == 1:
            return [self]

        class splitCV(CV):

            def __init__(self2, cvs, index) -> None:

                def f2(x, y):
                    return self.cv(x, y)[index]

                self2.f = f2
                self2.n = 1
                self2._update_params()

                self2.periodicity = self.periodicity[index, :]

        return [splitCV(self.cv, i) for i in range(self.n)]


class CombineCV(CV):
    """combine multiple CVs into one CV."""

    def __init__(self, cvs) -> None:
        self.n = 0
        self.cvs = cvs
        periodicity = np.empty((0, 2))
        for cv in cvs:
            self.n += cv.n
            periodicity = np.concatenate([periodicity, cv.periodicity], axis=0)
        self.periodicity = periodicity
        self._update_params()

    def _update_params(self):
        """function selects on ouput according to index."""

        def f(x, y, cvs):
            return jnp.array([cv.cv(x, y) for cv in cvs])

        self.f = partial(f, cvs=self.cvs)

        super()._update_params()


class CVUtils:
    """collection of predifined CVs. Intended to be used as argument to CV class.

    args:
        coordinates (np.array(n_atoms,3)): cartesian coordinates
        cell (np.array((3,3)): cartesian coordinates of cell vectors
    """

    @staticmethod
    def dihedral(coordinates, cell, numbers):
        """from https://stackoverflow.com/questions/20305272/dihedral-torsion-angle-from-four-points-in-cartesian-
        coordinates-in-python.

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

    @staticmethod
    def Volume(coordinates, cell):
        return jnp.abs(jnp.dot(cell[0], jnp.cross(cell[1], cell[2])))
