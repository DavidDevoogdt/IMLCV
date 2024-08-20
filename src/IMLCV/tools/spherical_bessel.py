# taken from https://github.com/shashankdholakia/harmonix/blob/main/src/harmonix/utils.py

from functools import partial

import jax
import jax.numpy as jnp
from jax import jit
from jax._src.custom_derivatives import custom_jvp
from jax._src.lax import lax
from jax._src.numpy.util import _where, check_arraylike, promote_dtypes_inexact
from jax._src.typing import Array, ArrayLike
from jax.scipy.special import gamma

# config.update("jax_enable_x64", True)


_lax_const = lax._const


def jint(n):
    return jnp.trunc(n).astype("int")


def spb1(x: ArrayLike, /) -> Array:
    """
    Calculate the spherical Bessel functions j_1(z).
    Follows existing implementation of jnp.sinc for safety around 0,
    using a Maclaurin series to keep continuous derivatives.

    Arguments:
        x: The argument of the function.

    Returns:
        csj: The function j_1(z).
    """

    check_arraylike("spb1", x)
    (x,) = promote_dtypes_inexact(x)

    # not defined at zero
    eq_zero = lax.eq(x, _lax_const(x, 0))

    safe_x = _where(eq_zero, _lax_const(x, 1), x)
    return _where(
        eq_zero, _spb1_maclaurin(0, x), lax.div(lax.sin(safe_x), safe_x**2) - lax.div(lax.cos(safe_x), safe_x)
    )


@partial(custom_jvp, nondiff_argnums=(0,))
def _spb1_maclaurin(k, x):
    # compute the kth derivative of x -> sin(x)/x evaluated at zero (since we
    # compute the monomial term in the jvp rule)
    # TODO(mattjj): see https://github.com/google/jax/issues/10750
    if k % 2:
        return x * 0
    else:
        top = 1.0j * (1.0j / 2) ** k * jnp.sqrt(jnp.pi)
        bottom = (k - 1) * gamma(2 + k / 2) * gamma(0.5 * (k - 1))
        return x * 0 + jnp.real(top / bottom)


@_spb1_maclaurin.defjvp
def _spb1_maclaurin_jvp(k, primals, tangents):
    (x,), (t,) = primals, tangents
    return _spb1_maclaurin(k, x), _spb1_maclaurin(k + 1, x) * t


def envj(n, x):
    """
    Helper function for msta1 and msta2.

    """
    envj = 0.5 * jnp.log10(6.28 * n) - n * jnp.log10(1.36 * x / n)  # always true

    return envj


def msta1(x, mp):
    """
    Calculate the number of terms required for the spherical Bessel function.
    """
    a0 = jnp.abs(x)
    n0 = jint(1.1 * a0) + 1
    f0 = envj(n0, a0) - mp
    n1 = n0 + 5
    f1 = envj(n1, a0) - mp

    nn = jint(n1 - (n1 - n0) / (1.0 - f0 / f1))
    f = envj(nn, a0) - mp
    n0 = n1
    f0 = f1
    n1 = nn
    f1 = f
    diff = jnp.abs(nn - n1)

    def cond_fun(inputs):
        n0, f0, n1, f1, nn, counter, diff = inputs
        return jnp.logical_and(jnp.abs(diff) > 1, counter < 20)

    def body_fun(inputs):
        n0, f0, n1, f1, nn, diff, counter = inputs
        nn = jint(n1 - (n1 - n0) / (1.0 - f0 / f1))
        diff = nn - n1
        f = envj(nn, a0) - mp
        n0 = n1
        f0 = f1
        n1 = nn
        f1 = f
        counter += 1
        return n0, f0, n1, f1, nn, diff, counter

    n0, f0, n1, f1, nn, diff, _ = jax.lax.while_loop(cond_fun, body_fun, (n0, f0, n1, f1, nn, diff, 0))

    return nn


def msta2(x, n, mp):
    """
    Calculate the number of terms required for the spherical Bessel function.
    """
    a0 = jnp.abs(x)
    hmp = 0.5 * mp
    ejn = envj(n, a0)

    obj, n0 = jax.lax.cond(
        ejn <= hmp, lambda _: (mp * 1.0, jint(1.1 * a0) + 1), lambda _: (hmp + ejn, jint(n)), operand=None
    )

    f0 = envj(n0, a0) - obj
    n1 = n0 + 5
    f1 = envj(n1, a0) - obj

    nn = jint(n1 - (n1 - n0) / (1.0 - f0 / f1))

    def cond_fun(inputs):
        n0, f0, n1, f1, nn, diff, counter = inputs
        return jnp.logical_and(jnp.abs(diff) >= 1, counter < 20)

    def body_fun(inputs):
        n0, f0, n1, f1, nn, diff, counter = inputs
        nn = jint(n1 - (n1 - n0) / (1.0 - f0 / f1))
        diff = nn - n1
        f = envj(nn, a0) - obj
        n0 = n1
        f0 = f1
        n1 = nn
        f1 = f
        counter += 1
        return n0, f0, n1, f1, nn, diff, counter

    n0, f0, n1, f1, nn, diff, _ = jax.lax.while_loop(cond_fun, body_fun, (n0, f0, n1, f1, nn, nn - n1, 0))

    return nn + 10


@partial(custom_jvp, nondiff_argnums=(0,))
@partial(jit, static_argnums=0)
def csphjy(n, z):
    """
    Spherical Bessel functions of the first and second kind, and their derivatives.
    Follows the implementation of https://github.com/emsr/maths_burkhardt/blob/master/special_functions.f90, but with the derivatives.
    Arguments:
        n: The order of the spherical Bessel function.
        z: The argument of the function.
    Returns:
        nm: The number of terms used in the calculation.
        csj: The function j_n(z).
        cdj: The derivative of the function j_n(z).
        csy: The function y_n(z).
        cdy: The derivative of the function y_n(z).

    """

    a0 = jnp.abs(z)
    nm = n
    complex = jax.dtypes.canonicalize_dtype(jnp.complex128)
    csj = jnp.zeros(
        int(n + 1),
        dtype=complex,
    )
    csj = csj.at[0].set(jnp.sinc(z / jnp.pi))
    csj = csj.at[1].set(spb1(z))

    if n >= 2:
        csa = csj[0]
        csb = csj[1]
        m = msta1(a0, 200)

        m, nm = jax.lax.cond(m < n, lambda _: (m, m), lambda _: (msta2(a0, n, 15), n), operand=None)

        cf0 = 0.0
        cf1 = 1.0e-100
        cf = (2.0 * m + 3.0) * cf1 / z - cf0

        def body_fun(kk, inputs):
            k = m - kk
            cf, csj, cf0, cf1 = inputs
            cf = (2.0 * k + 3.0) * cf1 / z - cf0

            def true_fun(csj):
                return csj.at[k].set(cf)

            csj = jax.lax.cond(k <= nm, true_fun, lambda csj: csj, csj)
            cf0 = cf1
            cf1 = cf
            return cf, csj, cf0, cf1

        cf, csj, cf0, cf1 = jax.lax.fori_loop(0, m + 1, body_fun, (cf, csj, cf0, cf1))

        cs = jax.lax.cond(jnp.abs(csa) <= jnp.abs(csb), lambda _: csb / cf0, lambda _: csa / cf, operand=None)

        csj = cs * csj

    return csj


@csphjy.defjvp
def csphjy_jvp(n, primals, tangents):
    (z,) = primals
    (z_dot,) = tangents
    csj = csphjy(n, z)

    cdj = jnp.zeros(n + 1, dtype=complex)
    cdj = cdj.at[0].set(-csj[1])  # this is -spb1
    cdj = cdj.at[1:].set(csj[:-1] - (jnp.arange(1, len(csj)) + 1.0) * csj[1:] / z)
    return csj, cdj * z_dot
