from functools import partial

import molmod.constants

import numpy as np
import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit, grad, jacfwd
import jaxinterp2d

# from .CV import CV, YaffCv
from IMLCV.base.CV import CV
import molmod
from abc import ABC, abstractmethod

import dill


class Bias(ABC):
    """base class for biased MD runs."""

    def __init__(self, cvs: CV, start=None, step=None) -> None:
        """"args:
                cvs: collective variables
                start: number of md steps before update is called
                step: steps between update is called"""

        self.cvs = cvs
        self.start = start
        self.step = step

        args = self._get_args()
        static_array_argnums = tuple(i + 1 for i in range(len(args)))

        self.e = BiasMTD._gnool_jit(partial(self._compute), static_array_argnums=static_array_argnums)
        self.de = BiasMTD._gnool_jit(jacfwd(self.e, argnums=(0,)), static_array_argnums=static_array_argnums)

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
            # return self.val.shape[0]

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

    def save_bias(self, filename):
        with open(filename, 'wb') as f:
            dill.dump(self, f)

    @staticmethod
    def load_bias(filename):
        with open(filename, 'rb') as f:
            self = dill.load(f)
        return self


class CompositeBias(Bias):
    pass


class BiasF(Bias):
    """Bias according to CV."""

    def __init__(self, cvs: CV, f=None):

        self.f = f if (f is not None) else lambda _: 0.0
        super().__init__(cvs, start=None, step=None)

    def _compute(self, cvs):
        return self.f(cvs)[0]


class BiasMTD(Bias):
    """A sum of Gaussian hills, for instance used in metadynamics: Adapted from Yaff.

    V = \sum_{\\alpha} K_{\\alpha}} \exp{-\sum_{i} \\frac{(q_i-q_{i,\\alpha}^0)^2}{2\sigma^2}}

    where \\alpha loops over deposited hills and i loops over collective
    variables.
    """

    def __init__(self, cvs: CV, K, sigmas, start=None, step=None, tempering=0.0, f=None):
        """_summary_

        Args:
            cvs: _description_
            K: _description_
            sigmas: _description_
            start: _description_. Defaults to None.
            step: _description_. Defaults to None.
            tempering: _description_. Defaults to 0.0.
            f: _description_. Defaults to None.
        """

        if isinstance(sigmas, float):
            sigmas = jnp.array([sigmas])
        if isinstance(sigmas, np.ndarray):
            sigmas = jnp.array(sigmas)

        periodicities = cvs.periodicity

        self.ncv = cvs.n
        assert sigmas.ndim == 1
        assert sigmas.shape[0] == self.ncv
        assert jnp.all(sigmas > 0)
        self.sigmas = sigmas
        self.sigmas_isq = 1.0 / (2.0 * sigmas**2.0)

        self.Ks = jnp.zeros((0,))
        self.q0s = jnp.zeros((0, self.ncv))

        #reorganise periodicity
        self.q_filt = ~jnp.isnan(periodicities).any(axis=1)
        self.d_q = periodicities[self.q_filt, 1] - periodicities[self.q_filt, 0]
        self.q_min = periodicities[self.q_filt, 0]

        self.tempering = tempering
        self.K = K

        self.f = f

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
        """_summary_

        Args:
            cvs: _description_
            q0s: _description_
            Ks: _description_

        Returns:
            _description_
        """

        deltas = cvs - q0s

        #find smalles diff by shifting over cell vector
        diff = jnp.floor(0.5 + deltas[:, self.q_filt] / self.d_q) * self.d_q
        deltas = deltas.at[:, self.q_filt].set(deltas[:, self.q_filt] - diff)

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


class BiasPlumed(Bias):
    pass


class BiasThermolib(Bias):
    """use FES from thermolib as bias potential."""
    pass