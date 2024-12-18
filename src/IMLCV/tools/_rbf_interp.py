"""Module for RBF interpolation."""

import warnings
from functools import partial
from itertools import combinations_with_replacement

import jax
import jax.numpy as jnp
from flax.struct import dataclass, field
from scipy.special import comb

from IMLCV.base.CV import CV, CvMetric
from IMLCV.tools._rbfinterp_pythran import _eval_system, cv_vals, evaluate_system

__all__ = ["RBFInterpolator", "cv_vals"]


# These RBFs are implemented.
_AVAILABLE = {
    "linear",
    "thin_plate_spline",
    "cubic",
    "quintic",
    "multiquadric",
    "inverse_multiquadric",
    "inverse_quadratic",
    "gaussian",
}


# The shape parameter does not need to be specified when using these RBFs.
_SCALE_INVARIANT = {"linear", "thin_plate_spline", "cubic", "quintic"}


# For RBFs that are conditionally positive definite of order m, the interpolant
# should include polynomial terms with degree >= m - 1. Define the minimum
# degrees here. These values are from Chapter 8 of Fasshauer's "Meshfree
# Approximation Methods with MATLAB". The RBFs that are not in this dictionary
# are positive definite and do not need polynomial terms.
_NAME_TO_MIN_DEGREE = {
    "multiquadric": 0,
    "linear": 0,
    "thin_plate_spline": 1,
    "cubic": 1,
    "quintic": 2,
}


def _monomial_powers(ndim, degree):
    """Return the powers for each monomial in a polynomial.

    Parameters
    ----------
    ndim : int
        Number of variables in the polynomial.
    degree : int
        Degree of the polynomial.

    Returns
    -------
    (nmonos, ndim) int ndarray
        Array where each row contains the powers for each variable in a
        monomial.

    """
    nmonos = comb(degree + ndim, ndim, exact=True)

    out = jnp.zeros((nmonos, ndim), dtype=int)
    count = 0
    for deg in range(degree + 1):
        for mono in combinations_with_replacement(range(ndim), deg):
            # `mono` is a tuple of variables in the current monomial with
            # multiplicity indicating power (e.g., (0, 1, 1) represents x*y**2)
            for var in mono:
                out = out.at[count, var].set(out[count, var] + 1)

            count += 1

    return jnp.array(out)


def _build_and_solve_system(
    y: CV,
    metric: CvMetric,
    d,
    smoothing,
    kernel,
    epsilon,
    powers,
):
    """Build and solve the RBF interpolation system of equations.

    Parameters
    ----------
    y : (P, N) float ndarray
        Data point coordinates.
    d : (P, S) float ndarray
        Data values at `y`.
    smoothing : (P,) float ndarray
        Smoothing parameter for each data point.
    kernel : str
        Name of the RBF.
    epsilon : float
        Shape parameter.
    powers : (R, N) int ndarray
        The exponents for each monomial in the polynomial.

    Returns
    -------
    coeffs : (P + R, S) float ndarray
        Coefficients for each RBF and monomial.
    shift : (N,) float ndarray
        Domain shift used to create the polynomial matrix.
    scale : (N,) float ndarray
        Domain scaling used to create the polynomial matrix.

    """

    from jax.tree_util import Partial

    s = d.shape[1]
    r = powers.shape[0]

    f = Partial(
        _eval_system,
        y=y,
        metric=metric,
        d=d,
        smoothing=smoothing,
        kernel=kernel,
        epsilon=epsilon,
        powers=powers,
    )

    coeffs, _ = jax.scipy.sparse.linalg.cg(
        A=f,
        b=jnp.vstack([d, jnp.zeros((r, s))]),
        tol=1e-12,
    )

    return coeffs


@partial(dataclass, frozen=False, eq=False)
class RBFInterpolator:
    """Radial basis function (RBF) interpolation in N dimensions. adapted from scipy"""

    _coeffs: jax.Array
    y: CV
    d: jax.Array
    smoothing: jax.Array
    epsilon: jax.Array
    powers: jax.Array
    metric: CvMetric
    d_shape: tuple[int] = field(pytree_node=False)
    kernel: str = field(pytree_node=False)
    d_dtype: jnp.dtype | None = field(pytree_node=False, default=None)

    @classmethod
    def create(
        self,
        y: CV,
        metric: CvMetric,
        d: jax.Array,
        smoothing=0.0,
        kernel="thin_plate_spline",
        epsilon=None,
        degree=None,
    ):
        ny, ndim = y.shape

        if jnp.iscomplexobj(d):
            raise NotImplementedError("Complex-valued data is not supported. ")

        # d_dtype = jnp.complex64 if jnp.iscomplexobj(d) else jnp.float32
        # d_dtype = d.dtype

        # d = jnp.asarray(d, dtype=d_dtype)
        if d.shape[0] != ny:
            raise ValueError(f"Expected the first axis of `d` to have length {ny}.")

        d_shape = d.shape[1:]
        d = d.reshape((ny, -1))
        # If `d` is complex, convert it to a float array with twice as many
        # columns. Otherwise, the LHS matrix would need to be converted to
        # complex and take up 2x more memory than necessary.
        # d = d.view(jnp.float32)

        if jnp.isscalar(smoothing):
            smoothing = jnp.full(ny, smoothing, dtype=float)
        else:
            smoothing = jnp.asarray(smoothing, dtype=float)
            if smoothing.shape != (ny,):
                raise ValueError(
                    "Expected `smoothing` to be a scalar or have shape " f"({ny},).",
                )

        kernel = kernel.lower()
        if kernel not in _AVAILABLE:
            raise ValueError(f"`kernel` must be one of {_AVAILABLE}.")

        if metric.periodicities.any():
            if kernel != "multiquadric" and kernel != "linear":
                print(f" The chosen kernel {kernel} is not suitable for periodic data. Switching to linear kernel")
                kernel = "linear"

        if epsilon is None:
            if kernel in _SCALE_INVARIANT:
                epsilon = 1.0
            else:
                raise ValueError(
                    "`epsilon` must be specified if `kernel` is not one of " f"{_SCALE_INVARIANT}.",
                )
        else:
            epsilon = jnp.array(epsilon)

        min_degree = _NAME_TO_MIN_DEGREE.get(kernel, -1)
        if degree is None:
            degree = max(min_degree, 0)
        else:
            degree = int(degree)
            if degree < -1:
                raise ValueError("`degree` must be at least -1.")
            elif degree < min_degree:
                warnings.warn(
                    f"`degree` should not be below {min_degree} when `kernel` "
                    f"is '{kernel}'. The interpolant may not be uniquely "
                    "solvable, and the smoothing parameter may have an "
                    "unintuitive effect.",
                    UserWarning,
                )

        nobs = ny

        powers = _monomial_powers(ndim, degree)

        # The polynomial matrix must have full column rank in order for the
        # interpolant to be well-posed, which is not possible if there are
        # fewer observations than monomials.
        if powers.shape[0] > nobs:
            raise ValueError(
                f"At least {powers.shape[0]} data points are required when "
                f"`degree` is {degree} and the number of dimensions is {ndim}.",
            )

        coeffs = _build_and_solve_system(
            y,
            metric,
            d,
            smoothing,
            kernel,
            epsilon,
            powers,
        )

        return RBFInterpolator(
            _coeffs=coeffs,
            y=y.replace(_stack_dims=None),
            d=d,
            d_shape=d_shape,
            smoothing=smoothing,
            kernel=kernel,
            epsilon=epsilon,
            powers=powers,
            metric=metric,
        )

    def __call__(self, x: CV):
        """Evaluate the interpolant at `x`.

        Parameters
        ----------
        x : (Q, N) array_like
            Evaluation point coordinates.

        Returns
        -------
        (Q, ...) ndarray
            Values of the interpolant at `x`.

        """

        isbatched = x.batched

        if not isbatched:
            x = x.batch()

        nx, ndim = x.shape

        if ndim != self.y.shape[1]:
            raise ValueError(
                "Expected the second axis of `x` to have length " f"{self.y.shape[1]}.",
            )

        out = evaluate_system(
            self._coeffs,
            x,
            self.y,
            self.metric,
            self.kernel,
            self.epsilon,
            self.powers,
        )

        if isbatched:
            return out.reshape((nx, -1))
        else:
            return out.reshape((-1,))

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state: dict):
        if "d_dtype" in state:
            del state["d_dtype"]

        if state["y"]._stack_dims is not None:
            state["y"] = state["y"].replace(_stack_dims=None)

        self.__init__(**state)
