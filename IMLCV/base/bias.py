from functools import partial
from typing import Iterable

import molmod.constants

import numpy as np
import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit, grad, jacfwd

from IMLCV.base.CV import CV
# from IMLCV.base.MdEngine import YaffEngine

import molmod
from abc import ABC, abstractmethod

from yaff.pes import ForceField

import dill


class Energy(ABC):
    """base class for biased Energy of MD simulation."""

    def __init__(self) -> None:
        """"args:
                cvs: collective variables
                start: number of md steps before update is called
                step: steps between update is called"""
        pass

    def compute_coor(self, coordinates, cell, gpos=None, vir=None):
        """Computes the bias, the gradient of the bias wrt the coordinates and the virial."""
        raise NotImplementedError

    def save(self, filename):
        with open(filename, 'wb') as f:
            dill.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            self = dill.load(f)
        return self


class Bias(Energy, ABC):
    """base class for biased MD runs."""

    def __init__(self, cvs: CV, start=None, step=None) -> None:
        """"args:
                cvs: collective variables
                start: number of md steps before update is called
                step: steps between update is called"""

        self.cvs = cvs
        self.start = start
        self.step = step

    def update_bias(self, coordinates, cell):
        """update the bias.

        Can only change the properties from _get_args
        """
        pass

    class _HashableArrayWrapper:
        """#see https://github.com/google/jax/issues/4572"""

        def __init__(self, val):
            self.val = val

        def __hash__(self):
            return hash(self.val.tobytes())

        def __eq__(self, other):
            eq = isinstance(other, BiasMTD._HashableArrayWrapper) and (self.__hash__() == other.__hash__())
            return eq

    @staticmethod
    def _gnool_jit(fun, static_array_argnums=(), static_argnums=()):
        """#see https://github.com/google/jax/issues/4572"""

        @partial(jit, static_argnums=static_array_argnums + static_argnums)
        def callee(*args):
            args = list(args)
            for i in static_array_argnums:
                args[i] = args[i].val
            return fun(*args)

        def caller(*args):
            args = list(args)
            for i in static_array_argnums:
                args[i] = BiasMTD._HashableArrayWrapper(args[i])
            return callee(*args)

        return caller

    @partial(jit, static_argnums=(0,))
    def compute_coor(self, coordinates, cell, gpos=None, vir=None):
        """Computes the bias, the gradient of the bias wrt the coordinates and the virial."""

        if not (cell.shape == (0, 3) or cell.shape == (3, 3)):
            raise NotImplementedError("other cell shapes not yet supported")

        bv = (vir is not None)
        bg = (gpos is not None)

        [cvs, jac_p_val, jac_c_val] = self.cvs.compute(coordinates, cell, jac_p=bg, jac_c=bv)
        [ener, de] = self.compute(cvs, diff=(bv or bg))

        if bg:
            gpos = gpos.at[:].set(jnp.einsum('j,jkl->kl', de, jac_p_val))

        if bv:
            vir = vir.at[:].set(jnp.einsum('ji,k,kjl->il', cell, de, jac_c_val))

        return ener

    @partial(jit, static_argnums=(0, 2))
    def compute(self, cvs, diff=False):
        args = self._get_args()
        static_array_argnums = tuple(i + 1 for i in range(len(args)))

        self.e = Bias._gnool_jit(partial(self._compute), static_array_argnums=static_array_argnums)
        self.de = Bias._gnool_jit(jacfwd(self.e, argnums=(0,)), static_array_argnums=static_array_argnums)

        E = self.e(cvs, *self._get_args())
        if diff:
            diffE = self.de(cvs, *self._get_args())[0]
        else:
            diffE = None

        return E, diffE

    @abstractmethod
    def _compute(self, cvs, *args):
        """function that calculates the bias potential."""
        raise NotImplementedError

    @abstractmethod
    def _get_args(self):
        """function that return dictionary with kwargs of _compute."""
        return []

    def finalize_bias(self):
        """Should be called at end of metadynamics simulation.

        Optimises compute
        """

        def update_bias(coordinates, cell):
            """update the bias.

            Used in metadyanmics to deposit hills.
            """
            pass

        self.update_bias = update_bias

    @partial(jit, static_argnums=(0, 2))
    def _periodic_wrap(self, cvs, min=False):
        """Translate cvs such over periodic vector

        Args:
            cvs: array of cvs
            min (bool): if False, translate to cv range. I true, minimises norm of vector
        """

        x = 0.5 if min else 0.0

        per = self.cvs.periodicity

        coor = (cvs - per[:, 0]) / (per[:, 1] - per[:, 0]) - x
        coor = jnp.modf(coor)[0]  # fractional part
        coor = (coor + x) * (per[:, 1] - per[:, 0]) + per[:, 0]

        return jnp.where(jnp.isnan(per).any(axis=1), cvs, coor)


class CompositeBias(Bias):
    """Class that combines several biases in one single bias
    """

    def __init__(self, biases: Iterable[Bias]) -> None:

        f_biases = []
        cvlist = []
        start_list = []
        step_list = []

        #flatten composite biases
        for b in biases:
            if isinstance(b, CompositeBias):
                f_biases = [*f_biases, *b.biases]
            else:
                f_biases.append(b)

        self.biases = f_biases

        for b in self.biases:
            cvlist.append(b.cvs)
            start_list.append(b.start)
            step_list.append(b.step)

        cvs = cvlist[0]
        for cvsi in cvlist[1:]:
            assert cvsi is cvs, "CV should be same instance"

        self.start_list = np.array(start_list)
        self.step_list = np.array(step_list)
        self.args_shape = np.array([0, *np.cumsum([len(b._get_args()) for b in self.biases])])

        super().__init__(cvs=cvs, start=0, step=1)

    def _compute(self, cvs, *args):
        e = 0.0

        for i in range(len(self.biases)):
            e += self.biases[i]._compute(cvs, *args[self.args_shape[i]:self.args_shape[i + 1]])

        return e

    def update_bias(self, coordinates, cell):

        mask = self.start_list == 0

        self.start_list[mask] += self.step_list[mask]
        self.start_list[~mask] -= 1

        for i in np.argwhere(mask):
            self.biases[int(i)].update_bias(coordinates=coordinates, cell=cell)

    def _get_args(self):
        return [a for b in self.biases for a in b._get_args()]


class BiasF(Bias):
    """Bias according to CV."""

    def __init__(self, cvs: CV, f=None):

        self.f = f if (f is not None) else lambda _: 0.0
        super().__init__(cvs, start=None, step=None)

    def _compute(self, cvs):
        return self.f(cvs)[0]

    def _get_args(self):
        return []


class BiasMTD(Bias):
    """A sum of Gaussian hills, for instance used in metadynamics: Adapted from Yaff.

    V = \sum_{\\alpha} K_{\\alpha}} \exp{-\sum_{i} \\frac{(q_i-q_{i,\\alpha}^0)^2}{2\sigma^2}}

    where \\alpha loops over deposited hills and i loops over collective
    variables.
    """

    def __init__(self, cvs: CV, K, sigmas, tempering=0.0, start=None, step=None):
        """_summary_

        Args:
            cvs: _description_
            K: _description_
            sigmas: _description_
            start: _description_. Defaults to None.
            step: _description_. Defaults to None.
            tempering: _description_. Defaults to 0.0.
        """

        if isinstance(sigmas, float):
            sigmas = jnp.array([sigmas])
        if isinstance(sigmas, np.ndarray):
            sigmas = jnp.array(sigmas)

        self.ncv = cvs.n
        assert sigmas.ndim == 1
        assert sigmas.shape[0] == self.ncv
        assert jnp.all(sigmas > 0)
        self.sigmas = sigmas
        self.sigmas_isq = 1.0 / (2.0 * sigmas**2.0)

        self.Ks = jnp.zeros((0,))
        self.q0s = jnp.zeros((0, self.ncv))

        self.tempering = tempering
        self.K = K

        super().__init__(cvs, start, step)

    def _add_hill_cv(self, q0, K):
        """Deposit a single hill.

        Args:
            q0: A NumPy array [Ncv] specifying the Gaussian center for each collective variable, or a single float if there is only one collective variable
            K: the force constant of this hill
        """
        if isinstance(q0, float):
            assert self.ncv == 1
            q0 = jnp.array([q0])
        assert q0.ndim == 1
        assert q0.shape[0] == self.ncv
        self.q0s = jnp.append(self.q0s, jnp.array([q0]), axis=0)
        self.Ks = jnp.append(self.Ks, jnp.array([K]), axis=0)

    def _add_hills_cv(self, q0s, Ks):
        """Deposit multiple hills.

        Args:
        q0s: A NumPy array [Nhills,Ncv]. Each row represents a hill, specifying the Gaussian center for each collective variable
        K: A NumPy array [Nhills] providing the force constant of each hill.
        """
        assert q0s.ndim == 2
        assert q0s.shape[1] == self.ncv
        self.q0s = jnp.concatenate((self.q0s, q0s), axis=0)
        assert Ks.ndim == 1
        assert Ks.shape[0] == q0s.shape[0]
        self.Ks = jnp.concatenate((self.Ks, Ks), axis=0)

    def update_bias(self, coordinates, cell):
        """_summary_

        Args:
            coordinates: _description_
            cell: _description_

        Raises:
            NotImplementedError: _description_
        """
        # Compute current CV values
        q0s, _, _ = self.cvs.compute(coordinates, cell)
        # Compute force constant
        K = self.K
        if self.tempering != 0.0:
            raise NotImplementedError("untested")
            K *= jnp.exp(-self.compute() / molmod.constants.boltzmann / self.tempering)
        # Add a hill
        self._add_hill_cv(q0s, K)

    def _compute(self, cvs, q0s, Ks):
        """Computes sum of hills"""

        deltas = cvs - q0s
        deltas = jnp.apply_along_axis(self._periodic_wrap, axis=1, arr=deltas, min=True)

        exparg = jnp.einsum('ji,ji,i -> j', deltas, deltas, self.sigmas_isq)
        energy = jnp.sum(Ks * jnp.exp(-exparg))

        return energy

    def _get_args(self):
        return [self.q0s, self.Ks]


class GridBias(Bias):
    """Bias interpolated from lookup table on uniform grid."""

    def __init__(self, cvs: CV, vals, start=None, step=None) -> None:
        super().__init__(cvs, start, step)

        self.per = np.array(self.cvs.periodicity)
        assert ~jnp.isnan(self.cvs.periodicity).any()
        self.vals = vals

    def _compute(self, cvs):
        #inspiration taken from https://github.com/adam-coogan/jaxinterp2d
        coords = jnp.array((cvs + self.per[:, 0]) / (self.per[:, 1] - self.per[:, 0]) * (np.array(self.vals.shape) - 1))
        return jsp.ndimage.map_coordinates(self.vals, coords, mode='wrap', order=1)

    def _get_args(self):
        return []


class HarmonicBias(BiasF):
    """Harmonic bias potential centered arround q0 with force constant k"""

    def __init__(self, cvs: CV, q0, k):
        """generate harmonic potentia;

        Args:
            cvs: CV
            q0: rest pos spring
            k: force constant spring
        """
        super().__init__(cvs, lambda q: k * (q - q0)**2 / 2.0)


class BiasPlumed(Bias):
    pass
