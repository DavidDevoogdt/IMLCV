from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from typing_extensions import Self

from IMLCV.base.bias import Bias, CompositeBias
from IMLCV.base.CV import CV, CollectiveVariable, CvMetric
from IMLCV.base.datastructures import field
from IMLCV.base.MdEngine import MDEngine
from IMLCV.tools._rbf_interp import RBFInterpolator

######################################
# helper functions that are pickable #
######################################


def _clip(x, a_min, a_max):
    return jnp.clip(x, a_min, a_max)


######################################


class MinBias(CompositeBias):
    @classmethod
    def create(cls, biases: list[Bias]) -> CompositeBias:
        b = CompositeBias.create(biases=biases, fun=jnp.min)
        return b


class DTBias(Bias):
    # scales the bias with dt

    T: float
    dT: float
    bias: Bias

    @classmethod
    def create(cls, bias: Bias, T: float, dT: float) -> Bias:
        if dT == 0.0:
            return bias

        colvar = bias.collective_variable
        bias.collective_variable = None

        return DTBias(
            collective_variable=colvar,
            collective_variable_path=bias.collective_variable_path,
            start=bias.start,
            step=bias.step,
            finalized=bias.finalized,
            slice_exponent=bias.slice_exponent,
            slice_mean=bias.slice_mean,
            dT=dT,
            bias=bias,
            T=T,
        )

    def _compute(self, cvs: CV) -> Array:
        b = self.bias._compute(cvs)

        return b * (self.T / (self.T + self.dT))


class HarmonicBias(Bias):
    """Harmonic bias potential centered arround q0 with force constant k."""

    # __: KW_ONLY

    q0: CV
    k: float
    k_max: float | None = None
    size: Array | None = None
    y0: float | None = None
    r0: float | None = None
    metric: CvMetric

    @staticmethod
    def create(
        cvs: CollectiveVariable,
        q0: CV,
        k: float | Array,
        k_max: float | None = None,
        start=None,
        step=None,
        finalized=True,
    ) -> HarmonicBias:
        """generate harmonic potentia;

        Args:
            cvs: CV
            q0: rest pos spring
            k: force constant spring
        """

        _k: Array

        if isinstance(k, float):
            _k = jnp.zeros_like(q0.cv) + k
        else:
            _k = k  # type:ignore

        assert _k.shape == q0.cv.shape

        k_float = float(jnp.mean(_k))
        size = jnp.sqrt(_k / k_float)

        if k_max is not None:
            r0 = k_max / k_float
            y0 = k_float * r0**2 / 2
        else:
            r0 = None
            y0 = None

        return HarmonicBias(
            collective_variable=cvs,
            q0=q0,
            k=k_float,
            size=size,
            k_max=k_max,
            r0=r0,
            y0=y0,
            start=start,
            step=step,
            finalized=finalized,
            metric=cvs.metric,
        )

    def _compute(self, cvs: CV):
        # assert isinstance(cvs, CV)
        r = jnp.linalg.norm(self.metric.difference(cvs, self.q0) * self.size)

        para = self.k * r**2 / 2

        if self.k_max is None:
            return para

        o = jnp.where(
            r < self.r0,
            para,
            (r - self.r0) * self.k_max + self.y0,
        )

        return o


class BiasMTD(Bias):
    r"""A sum of Gaussian hills, for instance used in metadynamics:
    Adapted from Yaff.

    V = \sum_{\\alpha} K_{\\alpha}} \exp{-\sum_{i}
    \\frac{(q_i-q_{i,\\alpha}^0)^2}{2\sigma^2}}

    where \\alpha loops over deposited hills and i loops over collective
    variables.
    """

    # __: KW_ONLY

    q0s: jax.Array
    sigmas: jax.Array
    K: jax.Array
    Ks: jax.Array
    tempering: float = field(pytree_node=False, default=0.0)

    @classmethod
    def create(
        cls,
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
        # tempering = tempering
        K = K

        return cls(
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
        raise

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

    # __: KW_ONLY

    rbf: RBFInterpolator
    offset: float | jax.Array = 0.0

    @classmethod
    def create(
        cls,
        cvs: CollectiveVariable,
        vals: Array,
        cv: CV,
        start=None,
        step=None,
        kernel="thin_plate_spline",
        epsilon=None,
        smoothing: int | Array = 0.0,
        degree=None,
        finalized=True,
        slice_exponent=1,
        log_exp_slice=True,
        slice_mean=False,
    ) -> RbfBias:
        assert cv.batched
        assert cv.shape[1] == cvs.n, f"{cv.shape}[1] != {cvs.n}"
        assert len(vals.shape) == 1
        assert cv.shape[0] == vals.shape[0]

        # lift
        offset = jnp.min(vals)

        rbf = RBFInterpolator.create(
            y=cv,
            kernel=kernel,
            d=vals - offset,
            metric=cvs.metric,
            smoothing=smoothing,
            epsilon=epsilon,
            degree=degree,
        )

        return RbfBias(
            collective_variable=cvs,
            start=start,
            step=step,
            rbf=rbf,
            finalized=finalized,
            slice_exponent=slice_exponent,
            log_exp_slice=log_exp_slice,
            slice_mean=slice_mean,
            offset=offset,
        )

    def _compute(self, cvs: CV):
        out = self.rbf(cvs) + self.offset
        if cvs.batched:
            return out
        return out[0]


class GridBias(Bias):
    """Bias interpolated from lookup table on uniform grid.

    values are caluclated in bin centers
    """

    n: int
    bounds: jax.Array
    vals: jax.Array
    order: int = field(pytree_node=False, default=1)

    @classmethod
    def create(
        cls,
        cvs: CollectiveVariable,
        bias: Bias,
        n=30,
        bounds: Array | None = None,
        margin=0.1,
        order=1,
    ) -> GridBias:
        grid, _, cv, _, bounds = cvs.metric.grid(
            n=n,
            bounds=bounds,
            margin=margin,
        )

        vals, _ = bias.compute_from_cv(cv)

        vals = vals.reshape((n,) * cvs.n)

        return GridBias(
            collective_variable=cvs,
            n=n,
            vals=vals,
            bounds=bounds,
            order=order,
        )

    def _compute(self, cvs: CV):
        # map between vals 0 and 1
        # if self.bounds is not None:
        coords = (cvs.cv - self.bounds[:, 0]) / (self.bounds[:, 1] - self.bounds[:, 0])

        import jax.scipy as jsp

        return jsp.ndimage.map_coordinates(
            self.vals,
            coords * (self.n - 1),  # type:ignore
            mode="constant",
            cval=jnp.nan,
            order=self.order,
        )
