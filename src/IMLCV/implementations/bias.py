from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from typing_extensions import Self

from IMLCV.base.bias import Bias, CompositeBias
from IMLCV.base.CV import CV, CollectiveVariable
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


# #@partial(dataclass, frozen=False, eq=False)
# @dataclass
# class DiffBias(Bias):
#     biases: Iterable[Bias]
#     beta: float

#     def _compute(self, cvs: CV):
#         return -jnp.exp(-self.beta * self.biases[0]._compute(cvs))* jnp.abs(self.biases[0]._compute(cvs) - self.biases[1]._compute(cvs))
#         # return jnp.exp(-self.beta * self.biases[0]._compute(cvs)) * jnp.abs(
#         #     self.biases[0]._compute(cvs) - self.biases[1]._compute(cvs)
#         # )

#     @classmethod
#     def create(cls, biases: Iterable[Bias], T=300 * kelvin) -> Self:
#         assert len(biases) == 2

#         return DiffBias(
#             biases=biases,
#             beta=1 / (T * boltzmann),
#             collective_variable=biases[0].collective_variable,
#         )


class HarmonicBias(Bias):
    """Harmonic bias potential centered arround q0 with force constant k."""

    # __: KW_ONLY

    q0: CV
    k: Array
    k_max: Array | None = None
    y0: Array | None = None
    r0: Array | None = None

    @staticmethod
    def create(
        cvs: CollectiveVariable,
        q0: CV,
        k: float | Array,
        k_max: Array | float | None = None,
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

        if isinstance(k, float):
            _k = jnp.zeros_like(q0.cv) + k
        else:
            _k = k

        assert _k.shape == q0.cv.shape

        if k_max is not None:
            k_max = jnp.array(k_max)

            assert k_max.shape == q0.cv.shape  # type:ignore

        if k_max is not None:
            assert np.all(k_max > 0)
            r0 = k_max / _k
            y0 = jnp.einsum("i,i,i->", _k, r0, r0) / 2
        else:
            r0 = None
            y0 = None

        return HarmonicBias(
            collective_variable=cvs,
            q0=q0,
            k=_k,
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
        kernel="gaussian",
        epsilon=None,
        smoothing=0.0,
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
            coords * (self.n - 1),  # type:ignore
            mode="nearest",
            order=1,
        )
