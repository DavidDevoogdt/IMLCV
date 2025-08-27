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


@partial(jit_decorator, static_argnums=(3))
def get_d(
    x: jax.Array,
    metric: CvMetric,
    epsilon: jax.Array | float,
    periodicities: tuple[bool],
    y: jax.Array | None = None,
):
    epsilon = jnp.array(epsilon).reshape((-1,))
    if epsilon.shape[0] == 1:
        epsilon = jnp.full((x.shape[0]), epsilon[0])

    def wrap_mod(d: jax.Array):
        d = jnp.mod(d, 1.0)
        d = jnp.where(d > 0.5, d - 1.0, d)
        return d

    def scale(val: jax.Array, metric: CvMetric):
        return val / (metric.bounding_box[:, 1] - metric.bounding_box[:, 0])  # val between zero and 1 (usually)

    x = scale(x, metric=metric)
    if y is not None:
        y = scale(y, metric=metric)

    _d = []

    for i, pi in enumerate(periodicities):
        if pi:
            if y is None:
                dc = jnp.cos(x[i] * 2 * jnp.pi)
                ds = jnp.sin(x[i] * 2 * jnp.pi)
            else:
                dc = jnp.cos(x[i] * 2 * jnp.pi) - jnp.cos(y[i] * 2 * jnp.pi)
                ds = jnp.sin(x[i] * 2 * jnp.pi) - jnp.sin(y[i] * 2 * jnp.pi)

            # d.append(jnp.sqrt(pi_x**2 + pi_y**2))

            _d.append(dc * epsilon[i])
            _d.append(ds * epsilon[i])

        else:
            if y is None:
                d = x[i]
            else:
                d = x[i] - y[i]

            _d.append(d * epsilon[i])

    return jnp.array(_d)


@partial(jit_decorator, static_argnums=(4))
def cv_norm(
    x: CV,
    y: CV,
    metric: CvMetric,
    eps: jax.Array | float,
    periodicities: tuple[bool],
):
    d = get_d(x.cv, metric, eps, periodicities, y.cv)

    d_sum = jnp.sum(d**2)

    d_sum_safe = jnp.where(d_sum == 0.0, 1.0, d_sum)

    return jnp.where(d_sum == 0.0, 0.0, jnp.sqrt(d_sum_safe))  # avoid division by zero


@partial(jit_decorator, static_argnums=(3))
def cv_vals(
    x: CV,
    power: jax.Array,
    metric: CvMetric,
    periodicities: tuple[bool],
):
    d = get_d(x.cv, metric, epsilon=1.0, periodicities=periodicities)

    return d**power


@partial(jit_decorator, static_argnums=(4, 5, 6))
def eval_kernel_matrix(
    x: CV,
    y: CV,
    metric: CvMetric,
    eps: jax.Array | float,
    kernel_func: Callable[[jax.Array], jax.Array],
    periodicities: tuple[bool],
    norm_jacobian: bool = False,
):
    """Evaluate RBFs, with centers at `x`, at `x`."""

    @partial(vmap_decorator, in_axes=(None, 0), out_axes=1)
    @partial(vmap_decorator, in_axes=(0, None), out_axes=0)
    def f00(x: CV, y: CV):
        r = cv_norm(x, y, metric, eps, periodicities=periodicities)

        if norm_jacobian:
            return jax.jacrev(kernel_func)(r)

        return kernel_func(r)

    return f00(x, y)


@partial(jit_decorator, static_argnums=(3))
def eval_polynomial_matrix(
    x: CV,
    metric: CvMetric,
    powers: jax.Array,
    periodicities: tuple[bool],
):
    """Evaluate monomials, with exponents from `powers`, at `x`."""

    @partial(vmap_decorator, in_axes=(None, 0), out_axes=1)
    @partial(vmap_decorator, in_axes=(0, None), out_axes=0)
    def f00(x: CV, power: jax.Array) -> jax.Array:
        return jnp.prod(
            jnp.array(
                cv_vals(
                    x=x,
                    power=power,
                    metric=metric,
                    periodicities=periodicities,
                ),  # type:ignore
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
    periodicities: tuple[bool],
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
        eval_kernel_matrix(x, y, metric, epsilon, kernel_func, periodicities=periodicities) @ a
        + eval_polynomial_matrix(x, metric=metric, powers=powers, periodicities=periodicities) @ b
    )
