"""Module for RBF interpolation."""

import warnings
from datetime import datetime
from itertools import combinations_with_replacement

import jax
import jax.numpy as jnp
from scipy.special import comb

from IMLCV.base.CV import CV, CvMetric
from IMLCV.base.datastructures import MyPyTreeNode, field
from IMLCV.tools._rbfinterp_pythran import (
    NAME_TO_FUNC,
    cv_vals,
    eval_kernel_matrix,
    eval_polynomial_matrix,
    evaluate_system,
)

# __all__ = ["RBFInterpolator", "cv_vals"]


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
_SCALE_INVARIANT = {
    "linear",
    "thin_plate_spline",
    "cubic",
    "quintic",
}


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


def _monomial_powers(ndim, degree, periodicities):
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

    ndim += jnp.sum(periodicities)  # every periodic dim is split in sin and cos component

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
    smoothing: jax.Array | None,
    kernel,
    epsilon,
    powers,
    periodicities: tuple[bool],
    check=True,
    return_lcov=False,
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
    p = y.shape[0]
    s = d.shape[1]
    r = powers.shape[0]

    kernel_func = NAME_TO_FUNC[kernel]

    K = eval_kernel_matrix(
        y,
        y,
        metric,
        epsilon,
        kernel_func,
        periodicities=periodicities,
    )

    if smoothing is not None:
        K += smoothing**2 * jnp.eye(p)
    # dK = eval_kernel_matrix(y, y, metric, epsilon, kernel_func, norm_jacobian=True)
    P = eval_polynomial_matrix(
        y,
        metric=metric,
        powers=powers,
        periodicities=periodicities,
    )

    # print(f"{P=}")

    # print(f"{dK=} {dK.shape}")

    A = jnp.block(
        [
            [K, P],
            [P.T, jnp.zeros((r, r))],
        ]
    )

    l, U = jnp.linalg.eigh(A)

    # Check condition number and lightly regularize tiny/zero eigenvalues to avoid numerical blow-up.
    l_abs = jnp.abs(l)
    lmax = jnp.max(l_abs)
    lmin = jnp.min(l_abs)

    # estimated condition number (use abs of eigenvalues)
    cond = jnp.where(lmin == 0, jnp.inf, lmax / lmin)

    # # warn for ill-conditioning or singularity
    # if jnp.isfinite(cond) and cond > 1e12:
    #     warnings.warn(f"Ill-conditioned system: estimated condition number {float(cond):.3e}", UserWarning)
    # elif not jnp.isfinite(cond):
    #     warnings.warn("Singular system detected (zero eigenvalue). Regularizing tiny eigenvalues.", UserWarning)

    # regularize eigenvalues smaller than machine-tolerance-scaled threshold
    tol = lmax * max(A.shape) * jnp.finfo(l.dtype).eps * 10
    # print(f"{tol=}")
    small = jnp.abs(l) < tol
    if jnp.any(small):
        num_small = int(jnp.sum(small))
        warnings.warn(f"Regularizing {num_small} eigenvalue(s) below tol={float(tol):.3e}", UserWarning)
        l = jnp.where(small, tol, l)

    b = jnp.vstack([d, jnp.zeros((r, s))])

    # print(f"{l=}")

    A_inv = U @ jnp.diag(l ** (-1)) @ U.T

    # coeffs = jax.scipy.linalg.solve(A, b, assume_a="sym")

    coeffs = A_inv @ b

    if return_lcov:
        soln = A_inv[:, :p] @ d
        errors = soln[:p] / jnp.diag(A_inv[:p, :p])[:, None]
        loocv = jnp.linalg.norm(errors)

        # print(f"{loocv=}")

        return coeffs, loocv

    return coeffs


class RBFInterpolator(MyPyTreeNode):
    """Radial basis function (RBF) interpolation in N dimensions. adapted from scipy"""

    _coeffs: jax.Array
    y: CV
    d: jax.Array
    smoothing: jax.Array | None
    epsilon: jax.Array
    powers: jax.Array
    metric: CvMetric
    d_shape: tuple[int, ...] = field(pytree_node=False)
    kernel: str = field(pytree_node=False)
    d_dtype: jnp.dtype | None = field(pytree_node=False, default=None)
    periodicities: tuple[bool] = field(pytree_node=False, default=None)

    @classmethod
    def create(
        cls,
        y: CV,
        metric: CvMetric,
        d: jax.Array,
        smoothing=-1,
        kernel="thin_plate_spline",
        epsilon=None,
        degree=None,
    ):
        ny, ndim = y.shape

        if jnp.iscomplexobj(d):
            raise NotImplementedError("Complex-valued data is not supported. ")

        if d.shape[0] != ny:
            raise ValueError(f"Expected the first axis of `d` to have length {ny}.")

        d_shape = d.shape[1:]
        d = d.reshape((ny, -1))

        if smoothing is not None:
            if jnp.isscalar(smoothing):
                smoothing = jnp.full(ny, smoothing, dtype=float)
            else:
                smoothing = jnp.asarray(smoothing, dtype=float)
                if smoothing.shape != (ny,):
                    raise ValueError(
                        f"Expected `smoothing` to be a scalar or have shape ({ny},).",
                    )

        kernel = kernel.lower()
        if kernel not in _AVAILABLE:
            raise ValueError(f"`kernel` must be one of {_AVAILABLE}.")

        # if metric.periodicities.any():
        #     if kernel in ["linear", "thin_plate_spline", "cubic", "quintic", "thin_plate_spline"]:
        #         print(f"for periodic CV, choose decaying kernel. Switching from {kernel=} to inverse_quadratic ")

        #         kernel = "inverse_quadratic"

        if epsilon is None:
            if kernel in _SCALE_INVARIANT:
                epsilon = 1.0
            else:
                raise ValueError(
                    f"`epsilon` must be specified if `kernel` is not one of {_SCALE_INVARIANT}.",
                )

        epsilon = jnp.array(epsilon)

        min_degree = _NAME_TO_MIN_DEGREE.get(kernel, -1)
        if degree is None:
            degree = min_degree
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

        # if metric.periodicities.any():
        #     assert degree <= 0, "degree>0 is untested!"

        nobs = ny

        powers = _monomial_powers(ndim, degree, periodicities=metric.periodicities)

        # The polynomial matrix must have full column rank in order for the
        # interpolant to be well-posed, which is not possible if there are
        # fewer observations than monomials.
        if powers.shape[0] > nobs:
            raise ValueError(
                f"At least {powers.shape[0]} data points are required when "
                f"`degree` is {degree} and the number of dimensions is {ndim}.",
            )

        periodicities: tuple[bool] = tuple(bool(a) for a in metric.periodicities)

        loocv = False

        if smoothing is not None:
            if jnp.any(smoothing < 0):
                loocv = True

        def score_fn(s_min, s_max, step):
            s_vals = 10.0 ** (jnp.arange(s_min, s_max, step))
            scores = []

            print(f"Using LOOCV to determine smoothing parameter, {s_vals=}")

            for s in s_vals:
                coeffs, score = _build_and_solve_system(
                    y,
                    metric,
                    d,
                    s,
                    kernel,
                    epsilon,
                    powers,
                    periodicities=periodicities,
                    return_lcov=loocv,
                )
                if jnp.isnan(coeffs).any() or jnp.isinf(coeffs).any():
                    continue

                scores.append(score)

            scores = jnp.array(scores)
            best = jnp.nanargmin(scores)
            smoothing = s_vals[best]
            print(f"Choosing smoothing={smoothing} with l_cov={scores[best]} {scores=}")
            return smoothing

        if loocv:
            smoothing = score_fn(-5.0, 5.0, 1.0)
            print(f"first order s {smoothing=}")
            smoothing = score_fn(jnp.log10(smoothing) - 0.5, jnp.log10(smoothing) + 0.5, 0.1)
            print(f"second order s {smoothing=}")

        coeffs = _build_and_solve_system(
            y,
            metric,
            d,
            smoothing,
            kernel,
            epsilon,
            powers,
            periodicities=periodicities,
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
            periodicities=periodicities,  # type:ignore
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
                f"Expected the second axis of `x` to have length {self.y.shape[1]}.",
            )

        out = evaluate_system(
            self._coeffs,
            x,
            self.y,
            self.metric,
            self.kernel,
            self.epsilon,
            self.powers,
            self.periodicities,
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

        if "periodicities" not in state:
            state["periodicities"] = tuple(bool(a) for a in state["metric"].periodicities)

        self.__init__(**state)
