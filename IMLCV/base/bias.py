from functools import partial

import molmod.constants

import numpy as np
import jax.numpy as jnp
from jax import jit, grad, jacfwd

# from .CV import CV, YaffCv
from IMLCV.base.CV import CV
import molmod
from abc import ABC

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

    def compute_coor(self, coordinates, cell, gpos=None, vir=None):
        """Computes the bias for arbitrary coordinates+cell."""

        if not (cell.shape == (0, 3) or cell.shape == (3, 3)):
            raise NotImplementedError("other cell shapes not yet supported")

        bv = (vir is not None)
        bg = (gpos is not None)

        [cvs, jac_p_val, jac_c_val] = self.cvs.compute(coordinates, cell, jac_p=bg, jac_c=bv)
        [ener, de] = self.compute(cvs, diff=(bv or bg))

        if bg:
            gpos = np.einsum('ij,jkl->kl', de, jac_p_val)
            #gpos = gpos.at[:].set(jnp.einsum('ij,jkl->kl', de, jac_p_val))

        if bv:
            vir[:] = np.einsum('ji,kz,kjl->il', cell, de, jac_c_val)
            #vir = vir.at[:].set(jnp.einsum('ji,kz,kjl->il', cell, de, jac_c_val))

        return ener

    def compute(self, cvs, diff=False):
        """function that calculates the bias potential."""
        raise NotImplementedError

    def update_bias(self, coordinates, cell):
        """update the bias.

        Used in metadyanmics to deposit hills.
        """
        raise NotImplementedError

    def finalize_bias(self):
        """Should be called at end of metadynamics simulation.

        Optimises compute
        """
        self.compute = jit(self.compute, static_argnames="diff")

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

        self.fcomp = f if (f is not None) else lambda x: 0.0
        self.dfcomp = jacfwd(self.fcomp)

        def compute(cvs, diff=False):
            de = self.dfcomp(cvs) if diff else None
            return self.fcomp(cvs), de

        self.compute = jit(compute, static_argnums=(1,))

        super().__init__(cvs, start=None, step=None)


class BiasMTD(Bias):
    """A sum of Gaussian hills, for instance used in metadynamics: Adapted from Yaff.

    V = \sum_{\\alpha} K_{\\alpha}} \exp{-\sum_{i} \\frac{(q_i-q_{i,\\alpha}^0)^2}{2\sigma^2}}

    where \\alpha loops over deposited hills and i loops over collective
    variables.
    """

    def __init__(self, cvs: CV, K, sigmas, start=None, step=None, tempering=0.0, f=None):
        '''
           args:
                sigmas:  The width of the Gaussian or a NumPy array [Ncv] specifying thewidths of the Gaussians
                periodicity: todo
        '''

        if isinstance(sigmas, float):
            sigmas = np.array([sigmas])

        periodicities = cvs.periodicity

        self.ncv = cvs.n
        assert sigmas.ndim == 1
        assert sigmas.shape[0] == self.ncv
        assert np.all(sigmas > 0)
        self.sigmas = sigmas
        self.sigmas_isq = 1.0 / (2.0 * sigmas**2.0)

        self.Ks = jnp.zeros((0,))
        self.q0s = jnp.zeros((0, self.ncv))
        if periodicities is not None:
            assert periodicities.shape[0] == self.ncv

        self.q_filt = ~np.isnan(periodicities).any(axis=1)
        self.d_q = periodicities[self.q_filt, 1] - periodicities[self.q_filt, 0]
        self.q_min = periodicities[self.q_filt, 0]

        self.tempering = tempering
        self.K = K

        self.f = f

        self.fcomp = BiasMTD._gnool_jit(partial(BiasMTD._compute,
                                                q_filt=self.q_filt,
                                                d_q=self.d_q,
                                                sigmas_isq=self.sigmas_isq),
                                        static_array_argnums=(1, 2))
        self.dfcomp = BiasMTD._gnool_jit(jacfwd(self.fcomp, argnums=(0,)), static_array_argnums=(1, 2))

        super().__init__(cvs, start, step)

    class _HashableArrayWrapper:
        """#see https://github.com/google/jax/issues/4572"""

        def __init__(self, val):
            self.val = val

        def __hash__(self):
            t = self.val.shape
            return t[0]  # number of hills

        def __eq__(self, other):
            return isinstance(other, BiasMTD._HashableArrayWrapper) and (self.val.shape == other.val.shape)

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

    def _add_hill_cv(self, q0, K):
        """Deposit a single hill.

        args:
            q0:
            A NumPy array [Ncv] specifying the Gaussian center for each
            collective variable, or a single float if there is only one
            collective variable

            K:
            The force constant of this hill
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

        **Arguments:**

        q0s
            A NumPy array [Nhills,Ncv]. Each row represents a hill,
            specifying the Gaussian center for each collective variable

        K
            A NumPy array [Nhills] providing the force constant of each
            hill.
        """
        assert q0s.ndim == 2
        assert q0s.shape[1] == self.ncv
        self.q0s = jnp.concatenate((self.q0s, q0s), axis=0)
        assert Ks.ndim == 1
        assert Ks.shape[0] == q0s.shape[0]
        self.Ks = jnp.concatenate((self.Ks, Ks), axis=0)

    def update_bias(self, coordinates, cell):
        """hook to update the bias.

        Used in metadyanmics to deposit hills
        """
        # Compute current CV values
        q0s, _, _ = self.cvs.compute(coordinates, cell)
        # Compute force constant
        K = self.K
        if self.tempering != 0.0:
            K *= np.exp(-self.compute() / molmod.constants.boltzmann / self.tempering)
        # Add a hill
        self._add_hill_cv(q0s, K)

    def compute(self, cvs, diff=False):
        de = self.dfcomp(cvs, self.q0s, self.Ks) if diff else None
        return self.fcomp(cvs, self.q0s, self.Ks), de

    def _compute(cvs, q0s, Ks, q_filt, d_q, sigmas_isq):
        deltas = cvs - q0s
        diff = jnp.floor(0.5 + deltas[:, q_filt] / d_q) * d_q
        deltas = deltas.at[:, q_filt].set(deltas[:, q_filt] - diff)

        exparg = deltas * deltas
        exparg = jnp.multiply(exparg, sigmas_isq)
        exparg = jnp.sum(exparg, axis=1)
        # Compute the bias energy
        exponents = jnp.exp(-exparg)
        energy = jnp.sum(Ks * exponents)

        return energy


class BiasPlumed(Bias):
    pass


class BiasThermolib(Bias):
    """use FES from thermolib as bias potential."""
    pass