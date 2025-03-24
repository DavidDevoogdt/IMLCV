from collections.abc import Iterable
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax.struct import dataclass, field
from jax import Array
from typing_extensions import Self

from IMLCV.base.bias import Bias, CompositeBias
from IMLCV.base.CV import CV, CollectiveVariable
from IMLCV.base.MdEngine import MDEngine
from IMLCV.tools._rbf_interp import RBFInterpolator

######################################
# helper functions that are pickable #
######################################


def _clip(x, a_min, a_max):
    return jnp.clip(x, a_min, a_max)


######################################


@partial(dataclass, frozen=False, eq=False)
class MinBias(CompositeBias):
    @classmethod
    def create(clz, biases: Iterable[Bias]) -> Self:
        b: clz = CompositeBias.create(biases=biases, fun=jnp.min)
        return b


# @partial(dataclass, frozen=False, eq=False)
# class DiffBias(Bias):
#     biases: Iterable[Bias]
#     beta: float

#     def _compute(self, cvs: CV):
#         return -jnp.exp(-self.beta * self.biases[0]._compute(cvs))* jnp.abs(self.biases[0]._compute(cvs) - self.biases[1]._compute(cvs))
#         # return jnp.exp(-self.beta * self.biases[0]._compute(cvs)) * jnp.abs(
#         #     self.biases[0]._compute(cvs) - self.biases[1]._compute(cvs)
#         # )

#     @classmethod
#     def create(clz, biases: Iterable[Bias], T=300 * kelvin) -> Self:
#         assert len(biases) == 2

#         return DiffBias(
#             biases=biases,
#             beta=1 / (T * boltzmann),
#             collective_variable=biases[0].collective_variable,
#         )


@partial(dataclass, frozen=False, eq=False)
class HarmonicBias(Bias):
    """Harmonic bias potential centered arround q0 with force constant k."""

    q0: CV
    k: Array
    k_max: Array | None = field(default=None)
    y0: Array | None = field(default=None)
    r0: Array | None = field(default=None)

    @classmethod
    def create(
        clz,
        cvs: CollectiveVariable,
        q0: CV,
        k,
        k_max: Array | float | None = None,
        start=None,
        step=None,
        finalized=True,
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

        # assert np.all(k > 0)
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


@partial(dataclass, frozen=False, eq=False)
class BiasMTD(Bias):
    r"""A sum of Gaussian hills, for instance used in metadynamics:
    Adapted from Yaff.

    V = \sum_{\\alpha} K_{\\alpha}} \exp{-\sum_{i}
    \\frac{(q_i-q_{i,\\alpha}^0)^2}{2\sigma^2}}

    where \\alpha loops over deposited hills and i loops over collective
    variables.
    """

    q0s: jax.Array
    sigmas: jax.Array
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


@partial(dataclass, frozen=False, eq=False)
class RbfBias(Bias):
    """Bias interpolated from lookup table on uniform grid.

    values are caluclated in bin centers
    """

    rbf: RBFInterpolator

    @classmethod
    def create(
        clz,
        cvs: CollectiveVariable,
        vals: Array,
        cv: CV,
        start=None,
        step=None,
        kernel="multiquadric",
        epsilon=None,
        smoothing=0.0,
        degree=None,
        finalized=True,
        slice_exponent=1,
        log_exp_slice=True,
        slice_mean=False,
    ) -> Self:
        assert cv.batched
        assert cv.shape[1] == cvs.n, f"{cv.shape}[1] != {cvs.n}"
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

        return clz(
            collective_variable=cvs,
            start=start,
            step=step,
            rbf=rbf,
            finalized=finalized,
            slice_exponent=slice_exponent,
            log_exp_slice=log_exp_slice,
            slice_mean=slice_mean,
        )

    def _compute(self, cvs: CV):
        out = self.rbf(cvs)
        if cvs.batched:
            return out
        return out[0]


@partial(dataclass, frozen=False, eq=False)
class GridBias(Bias):
    """Bias interpolated from lookup table on uniform grid.

    values are caluclated in bin centers
    """

    n: jax.Array
    bounds: jax.Array
    vals: jax.Array
    order: int = field(pytree_node=False, default=1)

    @classmethod
    def create(
        self,
        cvs: CollectiveVariable,
        bias: Bias,
        n=30,
        bounds: Array | None = None,
        margin=0.1,
        order=1,
    ) -> RbfBias:
        grid, cv, _, bounds = cvs.metric.grid(
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
            coords * (self.n - 1),
            mode="nearest",
            order=1,
        )
