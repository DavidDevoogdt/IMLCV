from functools import partial

import jax
import jax.numpy as jnp

from IMLCV.base.CV import CV, CvMetric


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


def scale(val, metric: CvMetric):
    return (val - metric.bounding_box[:, 0]) / (metric.bounding_box[:, 1] - metric.bounding_box[:, 0])


def cv_norm(x: CV, y: CV, metric: CvMetric, eps):
    # periodic dimensions are mapped to circle first, and then the distance is calculated

    x_min = metric.min_cv(x.cv)
    y_min = metric.min_cv(y.cv)

    x_min = scale(x_min, metric=metric)
    y_min = scale(y_min, metric=metric)

    return jnp.sqrt(
        jnp.sum(
            jnp.where(
                metric.periodicities,
                (jnp.sin(2 * x_min * jnp.pi) - jnp.sin(2 * y_min * jnp.pi)) ** 2
                + (jnp.cos(2 * x_min * jnp.pi) - jnp.cos(2 * y_min * jnp.pi)) ** 2,
                (x_min - y_min) ** 2,
            )
            * eps**2
        )
    )


def cv_vals(x: CV, powers, metric: CvMetric):
    return scale(metric.min_cv(x.cv), metric=metric) ** powers


@partial(jax.jit, static_argnums=(4))
def eval_kernel_matrix(x: CV, y: CV, metric: CvMetric, eps, kernel_func):
    """Evaluate RBFs, with centers at `x`, at `x`."""

    @partial(jax.vmap, in_axes=(None, 0), out_axes=1)
    @partial(jax.vmap, in_axes=(0, None), out_axes=0)
    def f00(x, y):
        return kernel_func(cv_norm(x, y, metric, eps))

    return f00(x, y)


@jax.jit
def eval_polynomial_matrix(x: CV, metric: CvMetric, powers):
    """Evaluate monomials, with exponents from `powers`, at `x`."""

    def g(x, powers):
        return cv_vals(x, powers, metric=metric)

    @partial(jax.vmap, in_axes=(None, 0), out_axes=1)
    @partial(jax.vmap, in_axes=(0, None), out_axes=0)
    def f00(x, powers):
        return jnp.prod(g(x, powers))

    pm = f00(x, powers)

    return pm


def evaluate_system(
    coeffs,
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
    a, b = coeffs[: y.shape[0], :], coeffs[y.shape[0] :, :]

    kernel_func = NAME_TO_FUNC[kernel]

    return (
        eval_kernel_matrix(x, y, metric, epsilon, kernel_func) @ a
        + eval_polynomial_matrix(x, metric=metric, powers=powers) @ b
    )
