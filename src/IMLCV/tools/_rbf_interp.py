"""Module for RBF interpolation."""

import warnings
from functools import partial
from itertools import combinations_with_replacement

import jax
import jax.numpy as jnp
from flax.struct import dataclass, field
from scipy.special import comb

from IMLCV.base.CV import CV, CvMetric
from IMLCV.tools._rbfinterp_pythran import (
    NAME_TO_FUNC,
    cv_vals,
    eval_kernel_matrix,
    eval_polynomial_matrix,
    evaluate_system,
)

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
    check=True,
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

    K = eval_kernel_matrix(y, y, metric, epsilon, kernel_func) + jnp.diag(smoothing)
    P = eval_polynomial_matrix(y, metric=metric, powers=powers)

    A = jnp.block(
        [
            [K, P],
            [P.T, jnp.zeros((r, r))],
        ]
    )

    b = jnp.vstack([d, jnp.zeros((r, s))])

    # A can be decomposed wiht shur complement
    # K is not psd and can be (pivoted) chol decomposed

    # L_K = jnp.linalg.cholesky(K)
    # L_K_inv = jax.scipy.linalg.solve_triangular(L_K, jnp.eye(L_K.shape[0]))
    # mS = P.T @ L_K_inv.T @ L_K_inv @ P  # minus shur complement
    # L_S = jnp.linalg.cholesky(mS)

    # print(f"{L_K=} {K=} {L_S=}")

    # print(f"{jnp.linalg.eigh(K)=}")

    # y1 = jax.scipy.linalg.solve_triangular(
    #     jnp.block(
    #         [
    #             [L_K, jnp.zeros((p, r))],
    #             [P.T @ L_K_inv, -L_S],
    #         ]
    #     ),
    #     d,
    #     lower=True,
    # )

    # coeffs = jax.scipy.linalg.solve_triangular(
    #     jnp.block(
    #         [
    #             [L_K.T, L_K_inv.T @ P],
    #             [jnp.zeros((r, p)), L_S.T],
    #         ]
    #     ),
    #     y1,
    # )

    # K = W W.T  #chol, W triu
    # S = 0- P K^(-1) P.T

    # import scipy

    # # this is pivoted cholesky
    # cho = scipy.linalg.lapack.dpstrf

    # X, P, r, info = cho(A, tol=1e-12, lower=True)
    # X = jnp.array(X)
    # pi = jnp.eye(P.shape[0])[:, P - 1][:, :r]
    # X = X.at[jnp.triu_indices(X.shape[0], 1)].set(0)
    # X = X[:, :r][:r, :]

    # print(f"rbf {r=} {A.shape=} {A=}")

    # y1 = jax.scipy.linalg.solve_triangular(X, pi.T @ b)
    # y2 = jax.scipy.linalg.solve_triangular(X.T, y1)

    # coeffs = pi @ y2

    # # print(f"{P=}")

    # # might be unstable

    # l, U = jnp.linalg.eigh(A)

    # # print(f"{l}  {U[:,0]} ")

    # l_max = jnp.max(jnp.abs(l))
    # l_min = jnp.min(jnp.abs(l))

    # print(f"Condition number of A: {l_max/l_min} {l_max=} {l_min=}")

    # coeffs = U @ jnp.diag(jnp.where(jnp.abs(l) / jnp.abs(l_max) > 1e-10, 1 / l, 0)) @ U.T @ b

    # print(f"{b.shape=} {coeffs.shape=} {coeffs=}")

    # if check:
    coeffs = jax.scipy.linalg.solve(A, b, assume_a="sym")

    # print(f"{coeffs-coeffs2=} {jnp.linalg.norm(coeffs-coeffs2)=}")

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
        kernel="multiquadric",
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
                    f"Expected `smoothing` to be a scalar or have shape ({ny},).",
                )

        kernel = kernel.lower()
        if kernel not in _AVAILABLE:
            raise ValueError(f"`kernel` must be one of {_AVAILABLE}.")

        if metric.periodicities.any():
            min_degree = _NAME_TO_MIN_DEGREE.get(kernel, -1)

            if min_degree > 0:
                print(
                    f" The chosen kernel {kernel} is not suitable for periodic data because it requires degree {min_degree=}>0.  Switching to linear kernel"
                )

                kernel = "linear"

        if epsilon is None:
            if kernel in _SCALE_INVARIANT:
                epsilon = 1.0
            else:
                raise ValueError(
                    f"`epsilon` must be specified if `kernel` is not one of {_SCALE_INVARIANT}.",
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
