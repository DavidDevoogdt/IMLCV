from collections.abc import Iterable

import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import scipy
import yaff
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

yaff.log.set_level(yaff.log.silent)


class MinBias(CompositeBias):
    def __init__(self, biases: Iterable[Bias]) -> None:
        super().__init__(biases, fun=jnp.min)


class HarmonicBias(Bias):
    """Harmonic bias potential centered arround q0 with force constant k."""

    def __init__(
        self,
        cvs: CollectiveVariable,
        q0: CV,
        k,
        k_max: Array | float | None = None,
    ):
        """generate harmonic potentia;

        Args:
            cvs: CV
            q0: rest pos spring
            k: force constant spring
        """
        super().__init__(cvs)

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
        self.k = jnp.array(k)
        self.q0 = q0

        self.k_max = k_max
        if k_max is not None:
            assert np.all(k_max > 0)
            self.r0 = k_max / k
            self.y0 = jnp.einsum("i,i,i->", k, self.r0, self.r0) / 2

    def _compute(self, cvs: CV, *args):
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

    def get_args(self):
        return []


class BiasMTD(Bias):
    r"""A sum of Gaussian hills, for instance used in metadynamics:
    Adapted from Yaff.

    V = \sum_{\\alpha} K_{\\alpha}} \exp{-\sum_{i}
    \\frac{(q_i-q_{i,\\alpha}^0)^2}{2\sigma^2}}

    where \\alpha loops over deposited hills and i loops over collective
    variables.
    """

    def __init__(
        self,
        cvs: CollectiveVariable,
        K,
        sigmas,
        tempering=0.0,
        start=None,
        step=None,
    ):
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

    def update_bias(
        self,
        md: MDEngine,
    ):
        if not self._update_bias():
            return

        assert md.trajectory_info is not None
        sp = md.sp

        if self.finalized:
            return
        # Compute current CV values
        nl = sp.get_neighbour_list(
            md.static_trajectory_info.r_cut,
            z_array=md.static_trajectory_info.atomic_numbers,
        )
        q0s = self.collective_variable.compute_cv(sp=sp, nl=nl)[0].cv

        K = self.K
        if self.tempering != 0.0:
            raise NotImplementedError("untested")

        self.q0s = jnp.vstack([self.q0s, q0s])
        self.Ks = jnp.array([*self.Ks, K])

    def _compute(self, cvs, q0s, Ks):
        """Computes sum of hills."""

        def f(x):
            return self.collective_variable.metric.difference(x1=CV(cv=x), x2=cvs)

        deltas = jnp.apply_along_axis(f, axis=1, arr=q0s.val)

        exparg = jnp.einsum("ji,ji,i -> j", deltas, deltas, self.sigmas_isq)
        energy = jnp.sum(jnp.exp(-exparg) * Ks.val)

        return energy

    def get_args(self):
        return [self.q0s, self.Ks]


class RbfBias(Bias):
    """Bias interpolated from lookup table on uniform grid.

    values are caluclated in bin centers
    """

    def __init__(
        self,
        cvs: CollectiveVariable,
        vals: Array,
        cv: CV,
        start=None,
        step=None,
        kernel="linear",
        epsilon=None,
        smoothing=0.0,
        degree=None,
    ) -> None:
        super().__init__(cvs, start, step)

        assert cv.batched
        assert cv.shape[1] == cvs.n
        assert len(vals.shape) == 1
        assert cv.shape[0] == vals.shape[0]

        self.rbf = RBFInterpolator(
            y=cv,
            kernel=kernel,
            d=vals,
            metric=cvs.metric,
            smoothing=smoothing,
            epsilon=epsilon,
            degree=degree,
        )

        # assert jnp.allclose(vals, self.rbf(cv), atol=1e-7)

    def _compute(self, cvs: CV, *args):
        out = self.rbf(cvs)
        if cvs.batched:
            return out
        return out[0]

    def get_args(self):
        return []


class GridBias(Bias):
    """Bias interpolated from lookup table on uniform grid.

    values are caluclated in bin centers
    """

    def __init__(
        self,
        cvs: CollectiveVariable,
        vals,
        bounds,
        start=None,
        step=None,
        centers=True,
    ) -> None:
        super().__init__(cvs, start, step)

        if not centers:
            raise NotImplementedError

        # extend periodically
        self.n = np.array(vals.shape)

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

        rbf = scipy.interpolate.RBFInterpolator(
            np.array([i[~mask] for i in inds_pairs]).T,
            bias[~mask],
        )

        bias[mask] = rbf(np.array([i[mask] for i in inds_pairs]).T)

        self.vals = bias
        self.bounds = jnp.array(bounds)

    def _compute(self, cvs: CV, *args):
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

    def get_args(self):
        return []


class GridBiasNd(Bias):
    # inspiration fromhttps://github.com/stanbiryukov/Nyx/tree/main/nyx/jax

    def __init__(
        self,
        collective_variable: CollectiveVariable,
        vals,
        bounds=None,
        start=None,
        step=None,
    ) -> None:
        super().__init__(collective_variable, start, step)

    def _compute(self, cvs, *args):
        return super()._compute(cvs, *args)


class PlumedBias(Bias):
    def __init__(
        self,
        collective_variable: CollectiveVariable,
        timestep,
        kernel=None,
        fn="plumed.dat",
        fn_log="plumed.log",
    ) -> None:
        super().__init__(
            collective_variable,
            start=0,
            step=1,
        )

        self.fn = fn
        self.kernel = kernel
        self.fn_log = fn_log
        self.plumedstep = 0
        self.hooked = False

        self.setup_plumed(timestep, 0)

        raise NotImplementedError("this is untested")

    def setup_plumed(self, timestep, restart):
        r"""Send commands to PLUMED to make it computation-ready.

        **Arguments:**

        timestep
            The timestep (in au) of the integrator

        restart
            Set to an integer value different from 0 to let PLUMED know that
            this is a restarted run
        """
        # Try to load the plumed Python wrapper, quit if not possible
        try:
            from plumed import Plumed
        except ImportError as e:
            raise e

        self.plumed = Plumed(kernel=self.kernel)
        # Conversion between PLUMED internal units and YAFF internal units
        # Note that PLUMED output will follow the PLUMED conventions
        # concerning units
        self.plumed.cmd("setMDEnergyUnits", 1.0 / kjmol)
        self.plumed.cmd("setMDLengthUnits", 1.0 / nanometer)
        self.plumed.cmd("setMDTimeUnits", 1.0 / picosecond)
        # Initialize the system in PLUMED
        self.plumed.cmd("setPlumedDat", self.fn)
        self.plumed.cmd("setNatoms", self.system.natom)
        self.plumed.cmd("setMDEngine", "IMLCV")
        self.plumed.cmd("setLogFile", self.fn_log)
        self.plumed.cmd("setTimestep", timestep)
        self.plumed.cmd("setRestart", restart)
        self.plumed.cmd("init")

    def update_bias(self, md: MDEngine):
        r"""When this point is reached, a complete time integration step was
        finished and PLUMED should be notified about this.
        """
        if not self.hooked:
            self.setup_plumed(timestep=self.plumedstep, restart=int(md.step > 0))
            self.hooked = True

        # PLUMED provides a setEnergy command, which should pass the
        # current potential energy. It seems that this is never used, so we
        # don't pass anything for the moment.
        #        current_energy = sum([part.energy for part in iterative.ff.parts[:-1] if not isinstance(part, ForcePartPlumed)])
        #        self.plumed.cmd("setEnergy", current_energy)
        # Ensure the plumedstep is an integer and not a numpy data type
        self.plumedstep += 1
        self._internal_compute(None, None)
        self.plumed.cmd("update")

    def _internal_compute(self, gpos, vtens):
        self.plumed.cmd("setStep", self.plumedstep)
        self.plumed.cmd("setPositions", self.system.pos)
        self.plumed.cmd("setMasses", self.system.masses)
        if self.system.charges is not None:
            self.plumed.cmd("setCharges", self.system.charges)
        if self.system.cell.nvec > 0:
            rvecs = self.system.cell.rvecs.copy()
            self.plumed.cmd("setBox", rvecs)
        # PLUMED always needs arrays to write forces and virial to, so
        # provide dummy arrays if Yaff does not provide them
        # Note that gpos and forces differ by a minus sign, which has to be
        # corrected for when interacting with PLUMED
        if gpos is None:
            my_gpos = np.zeros(self.system.pos.shape)
        else:
            gpos[:] *= -1.0
            my_gpos = gpos
        self.plumed.cmd("setForces", my_gpos)
        if vtens is None:
            my_vtens = np.zeros((3, 3))
        else:
            my_vtens = vtens
        self.plumed.cmd("setVirial", my_vtens)
        # Do the actual calculation, without an update; this should
        # only be done at the end of a time step
        self.plumed.cmd("prepareCalc")
        self.plumed.cmd("performCalcNoUpdate")
        if gpos is not None:
            gpos[:] *= -1.0
        # Retrieve biasing energy
        energy = np.zeros((1,))
        self.plumed.cmd("getBias", energy)
        return energy[0]
