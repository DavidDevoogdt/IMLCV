from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp

from IMLCV.base.CV import CV, CvMetric
from IMLCV.base.datastructures import jit_decorator, vmap_decorator


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


NAME_TO_FUNC: dict[str, Callable[[jax.Array], jax.Array]] = {
    "linear": linear,
    "thin_plate_spline": thin_plate_spline,
    "cubic": cubic,
    "quintic": quintic,
    "multiquadric": multiquadric,
    "inverse_multiquadric": inverse_multiquadric,
    "inverse_quadratic": inverse_quadratic,
    "gaussian": gaussian,
}


def get_d(x: jax.Array, metric: CvMetric, epsilon: jax.Array | float):
    def wrap_mod(d: jax.Array):
        d = jnp.mod(d, 1.0)
        d = jnp.where(d > 0.5, d - 1.0, d)
        return d

    def scale(val: jax.Array, metric: CvMetric):
        return val / (metric.bounding_box[:, 1] - metric.bounding_box[:, 0])  # val between zero and 1 (usually)

    d = scale(x, metric=metric)

    d = (
        jnp.where(
            metric.periodicities,
            wrap_mod(d),  # norm to dist 1
            d,
        )
        * epsilon
    )

    return d


def cv_norm(x: CV, y: CV, metric: CvMetric, eps: jax.Array | float):
    d = get_d(x.cv - y.cv, metric, eps)

    return jnp.sqrt(jnp.sum(d**2))


def cv_vals(x: CV, power: jax.Array, metric: CvMetric):
    d = get_d(x.cv, metric, epsilon=1.0)

    # both x^n and [sin(x),cos(nx)] are unisolvent
    out = jnp.where(
        metric.periodicities,
        jnp.where(power >= 0, jnp.cos(power * d * 2 * jnp.pi), jnp.sin(power * d * 2 * jnp.pi)),
        d**power,
    )

    return out


@partial(jit_decorator, static_argnums=(4))
def eval_kernel_matrix(
    x: CV, y: CV, metric: CvMetric, eps: jax.Array | float, kernel_func: Callable[[jax.Array], jax.Array]
):
    """Evaluate RBFs, with centers at `x`, at `x`."""

    @partial(vmap_decorator, in_axes=(None, 0), out_axes=1)
    @partial(vmap_decorator, in_axes=(0, None), out_axes=0)
    def f00(x: CV, y: CV):
        return kernel_func(cv_norm(x, y, metric, eps))

    return f00(x, y)


@jit_decorator
def eval_polynomial_matrix(x: CV, metric: CvMetric, powers: jax.Array):
    """Evaluate monomials, with exponents from `powers`, at `x`."""

    @partial(vmap_decorator, in_axes=(None, 0), out_axes=1)
    @partial(vmap_decorator, in_axes=(0, None), out_axes=0)
    def f00(x: CV, power: jax.Array) -> jax.Array:
        return jnp.prod(
            jnp.array(
                cv_vals(x=x, power=power, metric=metric),  # type:ignore
            )
        )

    pm = f00(x, powers)

    return pm


def evaluate_system(
    coeffs: jax.Array,
    x: CV,
    y: CV,
    metric: CvMetric,
    kernel: str,
    epsilon: float | jax.Array,
    powers: jax.Array,
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
