from collections.abc import Iterable

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import scipy
import yaff
from flax.struct import field
from flax.struct import PyTreeNode
from IMLCV.base.bias import Bias
from IMLCV.base.bias import CompositeBias
from IMLCV.base.CV import CollectiveVariable
from IMLCV.base.CV import CV
from IMLCV.base.MdEngine import MDEngine
from IMLCV.tools._rbf_interp import RBFInterpolator
from jax import Array
from molmod.units import kjmol
from molmod.units import nanometer
from molmod.units import picosecond
from typing_extensions import Self


class MinBias(CompositeBias):
    @classmethod
    def create(clz, biases: Iterable[Bias]) -> Self:
        b: clz = CompositeBias.create(biases=biases, fun=jnp.min)
        return b


class HarmonicBias(Bias):
    """Harmonic bias potential centered arround q0 with force constant k."""

    q0: CV
    k: Array
    k_max: Array | None = field(pytree_node=False, default=None)
    y0: Array | None = field(pytree_node=False, default=None)
    r0: Array | None = field(pytree_node=False, default=None)

    @classmethod
    def create(
        clz,
        cvs: CollectiveVariable,
        q0: CV,
        k,
        k_max: Array | float | None = None,
        start=None,
        step=None,
        finalized=None,
    ) -> Self:
        """generate harmonic potentia;

        Args:
            cvs: CV
            q0: rest pos spring
            k: force constant spring
        """

        if isinstance(k, float):
            k = jnp.zeros_like(q0.cv) + k
        else:
            assert k.shape == q0.cv.shape

        if k_max is not None:
            if isinstance(k_max, float):
                k_max = jnp.zeros_like(q0.cv) + k_max
            else:
                assert k_max.shape == q0.cv.shape

        assert np.all(k > 0)
        k = jnp.array(k)

        if k_max is not None:
            assert np.all(k_max > 0)
            r0 = k_max / k
            y0 = jnp.einsum("i,i,i->", k, r0, r0) / 2
        else:
            r0 = None
            y0 = None

        return clz(
            collective_variable=cvs,
            q0=q0,
            k=k,
            k_max=k_max,
            r0=r0,
            y0=y0,
            start=start,
            step=step,
            finalized=finalized,
        )

    def _compute(self, cvs: CV):
        # assert isinstance(cvs, CV)
        r = self.collective_variable.metric.difference(cvs, self.q0)

        def parabola(r):
            return jnp.einsum("i,i,i->", self.k, r, r) / 2

        if self.k_max is None:
            return parabola(r)

        return jnp.where(
            jnp.linalg.norm(r / self.r0) < 1,
            parabola(r),
            jnp.sqrt(
                jnp.einsum(
                    "i,i,i,i->",
                    self.k_max,
                    self.k_max,
                    jnp.abs(r) - self.r0,
                    jnp.abs(r) - self.r0,
                ),
            )
            + self.y0,
        )


class BiasMTD(Bias):
    r"""A sum of Gaussian hills, for instance used in metadynamics:
    Adapted from Yaff.

    V = \sum_{\\alpha} K_{\\alpha}} \exp{-\sum_{i}
    \\frac{(q_i-q_{i,\\alpha}^0)^2}{2\sigma^2}}

    where \\alpha loops over deposited hills and i loops over collective
    variables.
    """

    q0s: jax.Array
    sigmas: jax.Array = field(pytree_node=False)
    K: jax.Array
    Ks: jax.Array
    tempering: bool = field(pytree_node=False)

    @classmethod
    def create(
        self,
        cvs: CollectiveVariable,
        K,
        sigmas,
        tempering=0.0,
        start=None,
        step=None,
        finalized=False,
    ) -> Self:
        """_summary_

        Args:
            cvs: _description_
            K: _description_
            sigmas: _description_
            start: _description_. Defaults to None.
            step: _description_. Defaults to None.
            tempering: _description_. Defaults to 0.0.
        """

        # raise NotImplementedError

        if isinstance(sigmas, float):
            sigmas = jnp.array([sigmas])
        if isinstance(sigmas, Array):
            sigmas = jnp.array(sigmas)

        ncv = cvs.n
        assert sigmas.ndim == 1
        assert sigmas.shape[0] == ncv
        assert jnp.all(sigmas > 0)

        Ks = jnp.zeros((0,))
        q0s = jnp.zeros((0, ncv))
        tempering = tempering
        K = K

        return self(
            collective_variable=cvs,
            start=start,
            step=step,
            q0s=q0s,
            sigmas=sigmas,
            K=K,
            Ks=Ks,
            tempering=tempering,
            finalized=finalized,
        )

    def update_bias(
        self,
        md: MDEngine,
    ):
        b, self = self._update_bias()

        if not b:
            return self

        assert md.last_cv is not None

        q0s = md.last_cv.cv
        K = self.K

        if self.tempering != 0.0:
            # update K
            raise NotImplementedError("untested")

        q0s = jnp.vstack([self.q0s, q0s])
        Ks = jnp.array([*self.Ks, K])

        return self.replace(q0s=q0s, Ks=Ks, K=K)

    def _compute(self, cvs):
        """Computes sum of hills."""

        def f(x):
            return self.collective_variable.metric.difference(x1=CV(cv=x), x2=cvs)

        deltas = jnp.apply_along_axis(f, axis=1, arr=self.q0s)

        exparg = jnp.einsum("ji,ji,i -> j", deltas, deltas, 1.0 / (2.0 * self.sigmas**2.0))
        energy = jnp.sum(jnp.exp(-exparg) * self.Ks)

        return energy


class RbfBias(Bias):
    """Bias interpolated from lookup table on uniform grid.

    values are caluclated in bin centers
    """

    rbf: RBFInterpolator = field(pytree_node=False)

    @classmethod
    def create(
        clz,
        cvs: CollectiveVariable,
        vals: Array,
        cv: CV,
        start=None,
        step=None,
        kernel="linear",
        epsilon=None,
        smoothing=0.0,
        degree=None,
        finalized=False,
    ) -> Self:
        assert cv.batched
        assert cv.shape[1] == cvs.n
        assert len(vals.shape) == 1
        assert cv.shape[0] == vals.shape[0]

        rbf = RBFInterpolator.create(
            y=cv,
            kernel=kernel,
            d=vals,
            metric=cvs.metric,
            smoothing=smoothing,
            epsilon=epsilon,
            degree=degree,
        )

        return clz(collective_variable=cvs, start=start, step=step, rbf=rbf, finalized=finalized)

    def _compute(self, cvs: CV):
        out = self.rbf(cvs)
        if cvs.batched:
            return out
        return out[0]


class GridBias(Bias):
    """Bias interpolated from lookup table on uniform grid.

    values are caluclated in bin centers
    """

    n: jax.Array
    bounds: jax.Array
    vals: jax.Array

    @classmethod
    def create(
        self,
        cvs: CollectiveVariable,
        vals,
        bounds,
        start=None,
        step=None,
        centers=True,
        finalized=False,
    ) -> Self:
        if not centers:
            raise NotImplementedError

        # extend periodically
        n = np.array(vals.shape)

        bias = vals
        for i, p in enumerate(self.collective_variable.metric.periodicities):
            # extend array and fill boundary values if periodic
            def sl(a, b):
                out = [slice(None) for _ in range(self.collective_variable.n)]
                out[a] = b

                return tuple(out)

            def get_ext(i):
                a = np.array(bias.shape)
                a[i] = 1
                part = bias[sl(i, 0)].reshape(a)

                return part * jnp.nan

            bias = np.concatenate((get_ext(i), bias, get_ext(i)), axis=i)

            if p:
                bias[sl(i, 0)] = bias[sl(i, -2)]
                bias[sl(i, -1)] = bias[sl(i, 1)]

        # do general interpolation
        inds_pairs = np.array(np.indices(bias.shape))
        mask = np.isnan(bias)
        inds_pairs[:, mask]

        # TODO: change wiht own rbf interpollator
        rbf = scipy.interpolate.RBFInterpolator(
            np.array([i[~mask] for i in inds_pairs]).T,
            bias[~mask],
        )

        bias[mask] = rbf(np.array([i[mask] for i in inds_pairs]).T)

        vals = bias
        bounds = jnp.array(bounds)

        return self(
            collective_variable=cvs,
            start=start,
            step=step,
            n=n,
            vals=vals,
            bounds=bounds,
            finalized=finalized,
        )

    def _compute(self, cvs: CV):
        # overview of grid points. stars are addded to allow out of bounds extension.
        # the bounds of the x square are per.
        #  ___ ___ ___ ___
        # |   |   |   |   |
        # | * | * | * | * |
        # |___|___|___|___|
        # |   |   |   |   |
        # | * | x | x | * |
        # |___|___|___|___|
        # |   |   |   |   |
        # | * | x | x | * |
        # |___|___|___|___|
        # |   |   |   |   |
        # | * | * | * | * |
        # |___|___|___|___|
        # gridpoints are in the middle of

        # map between vals 0 and 1
        # if self.bounds is not None:
        coords = (cvs.cv - self.bounds[:, 0]) / (self.bounds[:, 1] - self.bounds[:, 0])
        # else:
        #     coords = self.collective_variable.metric.__map(cvs).cv

        # map between vals matrix edges
        coords = (coords * self.n - 0.5) / (self.n - 1)
        # scale to array size and offset extra row
        coords = coords * (self.n - 1) + 1

        # type: ignore
        return jsp.ndimage.map_coordinates(self.vals, coords, mode="nearest", order=1)


# class PlumedBias(Bias):
#     def __init__(
#         self,
#         collective_variable: CollectiveVariable,
#         timestep,
#         kernel=None,
#         fn="plumed.dat",
#         fn_log="plumed.log",
#     ) -> None:
#         super().__init__(
#             collective_variable,
#             start=0,
#             step=1,
#         )

#         self.fn = fn
#         self.kernel = kernel
#         self.fn_log = fn_log
#         self.plumedstep = 0
#         self.hooked = False

#         self.setup_plumed(timestep, 0)

#         raise NotImplementedError("this is untested")

#     def setup_plumed(self, timestep, restart):
#         r"""Send commands to PLUMED to make it computation-ready.

#         **Arguments:**

#         timestep
#             The timestep (in au) of the integrator

#         restart
#             Set to an integer value different from 0 to let PLUMED know that
#             this is a restarted run
#         """
#         # Try to load the plumed Python wrapper, quit if not possible
#         try:
#             from plumed import Plumed
#         except ImportError as e:
#             raise e

#         self.plumed = Plumed(kernel=self.kernel)
#         # Conversion between PLUMED internal units and YAFF internal units
#         # Note that PLUMED output will follow the PLUMED conventions
#         # concerning units
#         self.plumed.cmd("setMDEnergyUnits", 1.0 / kjmol)
#         self.plumed.cmd("setMDLengthUnits", 1.0 / nanometer)
#         self.plumed.cmd("setMDTimeUnits", 1.0 / picosecond)
#         # Initialize the system in PLUMED
#         self.plumed.cmd("setPlumedDat", self.fn)
#         self.plumed.cmd("setNatoms", self.system.natom)
#         self.plumed.cmd("setMDEngine", "IMLCV")
#         self.plumed.cmd("setLogFile", self.fn_log)
#         self.plumed.cmd("setTimestep", timestep)
#         self.plumed.cmd("setRestart", restart)
#         self.plumed.cmd("init")

#     def update_bias(self, md: MDEngine):
#         r"""When this point is reached, a complete time integration step was
#         finished and PLUMED should be notified about this.
#         """
#         if not self.hooked:
#             self.setup_plumed(timestep=self.plumedstep, restart=int(md.step > 0))
#             self.hooked = True

#         # PLUMED provides a setEnergy command, which should pass the
#         # current potential energy. It seems that this is never used, so we
#         # don't pass anything for the moment.
#         #        current_energy = sum([part.energy for part in iterative.ff.parts[:-1] if not isinstance(part, ForcePartPlumed)])
#         #        self.plumed.cmd("setEnergy", current_energy)
#         # Ensure the plumedstep is an integer and not a numpy data type
#         self.plumedstep += 1
#         self._internal_compute(None, None)
#         self.plumed.cmd("update")

#     def _internal_compute(self, gpos, vtens):
#         self.plumed.cmd("setStep", self.plumedstep)
#         self.plumed.cmd("setPositions", self.system.pos)
#         self.plumed.cmd("setMasses", self.system.masses)
#         if self.system.charges is not None:
#             self.plumed.cmd("setCharges", self.system.charges)
#         if self.system.cell.nvec > 0:
#             rvecs = self.system.cell.rvecs.copy()
#             self.plumed.cmd("setBox", rvecs)
#         # PLUMED always needs arrays to write forces and virial to, so
#         # provide dummy arrays if Yaff does not provide them
#         # Note that gpos and forces differ by a minus sign, which has to be
#         # corrected for when interacting with PLUMED
#         if gpos is None:
#             my_gpos = np.zeros(self.system.pos.shape)
#         else:
#             gpos[:] *= -1.0
#             my_gpos = gpos
#         self.plumed.cmd("setForces", my_gpos)
#         if vtens is None:
#             my_vtens = np.zeros((3, 3))
#         else:
#             my_vtens = vtens
#         self.plumed.cmd("setVirial", my_vtens)
#         # Do the actual calculation, without an update; this should
#         # only be done at the end of a time step
#         self.plumed.cmd("prepareCalc")
#         self.plumed.cmd("performCalcNoUpdate")
#         if gpos is not None:
#             gpos[:] *= -1.0
#         # Retrieve biasing energy
#         energy = np.zeros((1,))
#         self.plumed.cmd("getBias", energy)
#         return energy[0]
