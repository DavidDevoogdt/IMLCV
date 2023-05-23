import jax
import jax.numpy as jnp
from IMLCV.base.CV import CV
from IMLCV.base.CV import CvMetric


def linear(r):
    return -r


def thin_plate_spline(r):
    return jax.lax.cond(r == 0, lambda r: r, lambda r: r**2 * jnp.log(r), r)


def cubic(r):
    return r**3


def quintic(r):
    return -(r**5)


def multiquadric(r):
    return -jnp.sqrt(r**2 + 1)


def inverse_multiquadric(r):
    return 1 / jnp.sqrt(r**2 + 1)


def inverse_quadratic(r):
    return 1 / (r**2 + 1)


def gaussian(r):
    return jnp.exp(-(r**2))


NAME_TO_FUNC = {
    "linear": linear,
    "thin_plate_spline": thin_plate_spline,
    "cubic": cubic,
    "quintic": quintic,
    "multiquadric": multiquadric,
    "inverse_multiquadric": inverse_multiquadric,
    "inverse_quadratic": inverse_quadratic,
    "gaussian": gaussian,
}


def scale(val, metric):
    return val / (metric.bounding_box[:, 1] - metric.bounding_box[:, 0]) * 2


def cv_norm(x: CV, y: CV, metric: CvMetric, eps):
    return jnp.linalg.norm(
        scale(metric.min_cv(x.cv) - metric.min_cv(y.cv), metric=metric) * eps,
    )


def cv_vals(x: CV, powers, metric: CvMetric):
    return scale(metric.min_cv(x.cv), metric=metric) ** powers


def kernel_vector(x: CV, y: CV, metric: CvMetric, epsilon, kernel_func):
    """Evaluate RBFs, with centers at `y`, at the point `x`."""

    def f0(y):
        return kernel_func(cv_norm(x, y, metric, epsilon))

    f1 = jax.vmap(f0)

    out0 = f1(y)

    return out0


def polynomial_vector(x: CV, powers, metric: CvMetric):
    """Evaluate monomials, with exponents from `powers`, at the point `x`."""

    def g(x, powers):
        return cv_vals(x, powers, metric=metric)

    def f0(powers):
        return jnp.prod(g(x, powers))

    f1 = jax.vmap(f0)
    out0 = f1(powers)

    return out0


def kernel_matrix(x: CV, metric: CvMetric, eps, kernel_func):
    """Evaluate RBFs, with centers at `x`, at `x`."""

    def f00(x, y):
        return cv_norm(x, y, metric, eps)

    f10 = jax.vmap(f00, in_axes=(0, None), out_axes=0)
    f11 = jax.vmap(f10, in_axes=(None, 0), out_axes=1)

    out_norm = f11(x, x)
    out_kernel = jax.vmap(jax.vmap(kernel_func))(out_norm)

    return out_kernel


def polynomial_matrix(x: CV, metric: CvMetric, powers):
    """Evaluate monomials, with exponents from `powers`, at `x`."""

    def g(x, powers):
        return cv_vals(x, powers, metric=metric)

    def f00(x, powers):
        return jnp.prod(g(x, powers))

    f10 = jax.vmap(f00, in_axes=(0, None), out_axes=0)
    f11 = jax.vmap(f10, in_axes=(None, 0), out_axes=1)

    return f11(x, powers)


def _polynomial_matrix(x: CV, powers, metric):
    """Return monomials, with exponents from `powers`, evaluated at `x`."""
    assert isinstance(x, CV)

    out = polynomial_matrix(x=x, metric=metric, powers=powers)
    return out


def _build_system(y: CV, metric: CvMetric, d, smoothing, kernel, epsilon, powers):
    """Build the system used to solve for the RBF interpolant coefficients.

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
    lhs : (P + R, P + R) float ndarray
        Left-hand side matrix.
    rhs : (P + R, S) float ndarray
        Right-hand side matrix.
    shift : (N,) float ndarray
        Domain shift used to create the polynomial matrix.
    scale : (N,) float ndarray
        Domain scaling used to create the polynomial matrix.

    """
    # p = d.shape[0]
    s = d.shape[1]
    r = powers.shape[0]
    kernel_func = NAME_TO_FUNC[kernel]

    # Transpose to make the array fortran contiguous. This is required for
    # dgesv to not make a copy of lhs.
    K = kernel_matrix(y, metric, epsilon, kernel_func) + jnp.diag(smoothing)
    P = polynomial_matrix(y, metric=metric, powers=powers)
    lhs = jnp.block([[K, P], [P.T, jnp.zeros((P.shape[1], P.shape[1]))]])

    # Transpose to make the array fortran contiguous.
    rhs = jnp.vstack([d, jnp.zeros((r, s))])

    return lhs, rhs


def _build_evaluation_coefficients(
    x: CV,
    y: CV,
    metric: CvMetric,
    kernel,
    epsilon,
    powers,
):
    """Construct the coefficients needed to evaluate
    the RBF.

    Parameters
    ----------
    x : (Q, N) float ndarray
        Evaluation point coordinates.
    y : (P, N) float ndarray
        Data point coordinates.
    kernel : str
        Name of the RBF.
    epsilon : float
        Shape parameter.
    powers : (R, N) int ndarray
        The exponents for each monomial in the polynomial.
    shift : (N,) float ndarray
        Shifts the polynomial domain for numerical stability.
    scale : (N,) float ndarray
        Scales the polynomial domain for numerical stability.

    Returns
    -------
    (Q, P + R) float ndarray

    """
    kernel_func = NAME_TO_FUNC[kernel]

    def kv(x):
        return kernel_vector(x, y, metric, epsilon, kernel_func)

    kv0 = jax.vmap(kv)(x)

    def pv(x):
        return polynomial_vector(x, powers, metric=metric)

    pv0 = jax.vmap(pv)(x)

    vec0 = jnp.hstack([kv0, pv0])

    return vec0
