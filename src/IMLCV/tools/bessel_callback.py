from functools import partial

import jax
import jax.lax
import jax.numpy as jnp
import numpy as onp
import scipy.special
from jax import custom_jvp, pure_callback

from IMLCV.tools.bessel_jn import bessel_jn

# from jax.scipy.special import bessel_jn

# see https://github.com/google/jax/issues/11002
# see https://jax.readthedocs.io/en/latest/notebooks/external_callbacks.html


def generate_bessel(function, type, sign=1, exp_scaled=False):
    def _function(v, x):
        v = onp.asarray(v)
        x = onp.asarray(x)

        # dims are expanded, ufunc takes
        out = function(v, x)

        return jnp.array(out, dtype=x.dtype)

    def cv_inner(v: jax.Array, z: jax.Array):
        # print(z.sharding)

        res_dtype_shape = jax.ShapeDtypeStruct(
            shape=(),
            dtype=z.dtype,
            # sharding=z.sharding,
        )
        # jax.debug.visualize_array_sharding(v)
        # jax.debug.visualize_array_sharding(z)

        return pure_callback(
            _function,
            res_dtype_shape,
            v,
            z,
            sharding=jax.sharding.SingleDeviceSharding(jax.devices()[0]),
            vmap_decorator_method="expand_dims",
        )

    @custom_jvp
    def cv(v: int, z: jax.Array):
        _v, _z = jnp.asarray(v), jnp.asarray(z)

        # Promote the input to inexact (float/complex).
        # Note that jnp.result_type() accounts for the enable_x64 flag.
        _z = _z.astype(jnp.result_type(float, _z.dtype))

        assert _v.ndim == 0 and _z.ndim == 0, "batch with vmap_decorator"
        return cv_inner(_v, _z)

    @cv.defjvp
    def cv_jvp(primals, tangents):
        v, x = primals
        v, x = jnp.asarray(v), jnp.asarray(x)
        dv, dx = tangents
        primal_out = cv(v, x)

        v_safe = jax.lax.cond(
            v == 0,
            lambda: jnp.ones_like(v),
            lambda: v,
        )

        if type == 0:
            """functions Jv, Yv, Hv_1,Hv_2"""
            # https://dlmf.nist.gov/10.6 formula 10.6.1
            tangents_out = jax.lax.cond(
                v == 0,
                lambda: -cv(v + 1, x),
                # lambda:jax.lax.cond( jnp.abs(x)>=1e-2,
                # lambda: cv(v - 1, x) -  (v/x) *  primal_out ,
                lambda: 0.5 * (cv(v_safe - 1, x) - cv(v_safe + 1, x)),
                # )
            )
        elif type == 1:
            """functions Kv and Iv"""
            # https://dlmf.nist.gov/10.29 formula 10.29.1
            tangents_out = jax.lax.cond(
                v == 0,
                lambda: sign * cv(v + 1, x),
                lambda: 0.5 * (sign * cv(v_safe - 1, x) + sign * cv(v_safe + 1, x)),
            )

        elif type == 2:
            """functions: spherical bessels"""
            # https://dlmf.nist.gov/10.51 formula 10.51.2

            # double where trick
            tangents_out = jax.lax.cond(
                v == 0,
                lambda: -cv(v + 1, x),
                # lambda: (lambda v: cv(v - 1, x) - (v + 1) / x * primal_out)(
                #     jax.lax.cond(v == 0, lambda: jnp.ones_like(v), lambda: v)
                # ),
                lambda: (v * cv(v_safe - 1, x) - (v_safe + 1) * cv(v_safe + 1, x)) / (2 * v_safe + 1),
            )
        else:
            raise ValueError

        # chain rule
        if exp_scaled:
            if sign == -1:
                tangents_out += primal_out
            elif sign == 1:
                tangents_out -= jnp.sign(x) * primal_out

        return primal_out, tangents_out * dx

    return cv


jv = bessel_jn


# jv = generate_bessel(scipy.special.jv, type=0)
yv = generate_bessel(scipy.special.yv, type=0)
hankel1 = generate_bessel(scipy.special.hankel1, type=0)
hankel2 = generate_bessel(scipy.special.hankel2, type=0)

kv = generate_bessel(scipy.special.kv, sign=-1, type=1)
iv = generate_bessel(scipy.special.iv, sign=+1, type=1)


# https://github.com/tpudlik/sbf/blob/master/algos/d_recur_miller.py

ORDER = 100  # Following Jablonski (1994)


@partial(jax.jit, static_argnums=0)
def spherical_jn_recurrence_pattern(n, z):
    jlp1 = 0

    jl = jnp.where(z.dtype == jnp.float32, 1e-30, 1e-200)

    z_safe = jnp.where(z < 1e-12, 1, z)

    # https://dlmf.nist.gov/10.51
    def iter(n, jl, jlp1, norm=True):
        jlm1 = (2 * n + 1) / z_safe * jl - jlp1

        if norm:
            s = jnp.max(jnp.abs(jnp.array([jl, jlm1, 1.0])))
            jlm1 /= s
            jl /= s

        return n - 1, jlm1, jl

    l = n + ORDER

    l, jl, jlp1 = jax.lax.fori_loop(
        0,
        ORDER,
        lambda i, args: iter(*args, norm=True),
        (l, jl, jlp1),
        # unroll=True,
    )

    # scan over other valuess

    def body(carry, xs: None):
        carry = iter(*carry, norm=False)
        return carry, carry[1]

    _, out = jax.lax.scan(
        body,
        init=(l, jl, jlp1),
        xs=None,
        length=n,
        # unroll=True,
    )

    out = out[::-1]

    f0 = jnp.sinc(z / jnp.pi)

    n_safe = jnp.where(out[0] != 0, out[0], 1)

    return out / n_safe * f0


@partial(jax.jit, static_argnums=0)
def spherical_jn_recurrence_pattern_unstable(n, z):
    z_safe: jax.Array = jnp.where(z < 1e-10, 1, z)  # type:ignore

    f0 = jnp.sinc(z_safe / jnp.pi)

    if n == 0:
        return jnp.array([f0])

    f1 = f0 / z_safe + -jnp.cos(z_safe) / z_safe

    if n == 1:
        return jnp.array([f0, f1])

    # https://dlmf.nist.gov/10.51

    def body(carry, xs: None):
        n, fn, fnm1 = carry

        fnp1 = (2 * n + 1) * fn / z_safe - fnm1

        # jax.debug.print("{}", nom)

        # fnp1 = jnp.where(nom < 1e-10, nom / z_safe, 0)

        return (n + 1, fnp1, fn), fnp1

    _, out = jax.lax.scan(
        body,
        init=(1, f1, f0),
        xs=None,
        length=n - 1,
        # unroll=True,
    )

    return jnp.array([f0, f1, *out])


@partial(custom_jvp, nondiff_argnums=(0, 2))
@partial(jax.jit, static_argnums=(0, 2))
def spherical_jn(n, z, forward=True):
    if forward:
        return jnp.where(
            z < 0.5,
            jnp.array([spherical_jn_recurrence(ni, z) for ni in range(n + 1)]),
            spherical_jn_recurrence_pattern_unstable(n, z),
        )

    return spherical_jn_recurrence_pattern(n + 1, z)


@partial(jax.jit, static_argnums=0)
def spherical_jn_recurrence(n, z):
    y = 0

    p0 = 1 / scipy.special.factorial2(2 * n + 1)

    y = p0

    for k in range(1, 6):
        p0 *= (-0.5 * z**2) / (k * (2 * k + 2 * n + 1))

        y += p0

    return z**n * y


@spherical_jn.defjvp
def spherical_jn_jvp(n, forward, primals, tangents):
    (x,) = primals
    (x_dot,) = tangents

    ni = n

    if ni == 0:
        ni = 1

    y = spherical_jn(ni + 1, x, forward=forward)

    # print(f"{y.shape=}")

    # x_safe = jnp.where(x < 1e-10, 1, x)

    n_vec = jnp.arange(0, len(y))

    dy = jnp.zeros(ni + 1)
    dy = dy.at[0].set(-y[1])
    dy = dy.at[1:].set((y[:-2] * n_vec[1:-1] - n_vec[2:] * y[2:]) / (2 * n_vec[1:-1] + 1))

    # dy = dy.at[1:].set(y[:-1] - (jnp.arange(1, len(y)) + 1.0) * y[1:] / x_safe)

    return y[: n + 1], dy[: n + 1] * x_dot


@partial(jax.jit, static_argnums=(0, 2))
def ie_n_recurrence_pattern(n, z, half=False):
    ilp1 = 0.0

    il = 1.0

    z_safe: jax.Array = jnp.where(z < 1e-12, 1, z)  # type:ignore

    # https://dlmf.nist.gov/10.29
    def iter(n, il, ilp1, norm=True):
        if half:
            ilm1 = (2 * (n + 0.5)) * il / z_safe + ilp1
        else:
            ilm1 = (2 * n) * il / z_safe + ilp1

        if norm:
            s = jnp.max(jnp.abs(jnp.array([il, ilm1, 1.0])))
            ilm1 /= s
            il /= s

        return n - 1, ilm1, il

    l = n + ORDER

    l, il, ilp1 = jax.lax.fori_loop(
        0,
        ORDER,
        lambda i, args: iter(*args, norm=True),
        (l, il, ilp1),
        # unroll=True,
    )

    # scan over other valuess

    def body(carry, xs: None):
        carry = iter(*carry, norm=False)
        return carry, carry[1]

    _, out = jax.lax.scan(
        body,
        init=(l, il, ilp1),
        xs=None,
        length=n,
        # unroll=True,
    )

    out = out[::-1]

    n_safe = jnp.where(out[0] != 0, out[0], 1)

    if half:
        # jax.debug.print("{}, {}", z_safe, jnp.sinh(z_safe))

        u = (2.0 / (jnp.pi * z_safe)) ** (1 / 2) * (jnp.exp(z - jnp.abs(z)) - jnp.exp(-z - jnp.abs(z))) / 2

    else:
        u = jax.scipy.special.i0e(z)

    return out / n_safe * u


@partial(jax.jit, static_argnums=(0, 2))
def ie_n_recurrence_pattern_unstalbe(n: int, z: jax.Array, half=False):
    z_safe: jax.Array = jnp.where(z < 1e-12, 1, z)  # mypy: ignore

    if half:
        # jax.debug.print("{}, {}", z_safe, jnp.sinh(z_safe))

        f0 = (
            (2.0 / (jnp.pi * z_safe)) ** (1 / 2)
            * (jnp.exp(z_safe - jnp.abs(z_safe)) - jnp.exp(-z_safe - jnp.abs(z_safe)))
            / 2
        )

    else:
        f0 = jax.scipy.special.i0e(z)

    if n == 0:
        return jnp.array([f0])

    # https://dlmf.nist.gov/10.29
    def iter(n, il, ilm1):
        # print(f"{n=} {il=} {ilm1=}")

        if half:
            ilp1 = ilm1 - (2 * (n + 0.5)) * il / z_safe
        else:
            ilp1 = ilm1 - (2 * n) * il / z_safe

        return n + 1, ilp1, il

    if half:
        # jax.debug.print("{}, {}", z_safe, jnp.sinh(z_safe))

        fm1 = (
            (2.0 / (jnp.pi * z_safe)) ** (1 / 2)
            * (jnp.exp(z_safe - jnp.abs(z_safe)) + jnp.exp(-z_safe - jnp.abs(z_safe)))
            / 2
        )

        f1 = iter(0, f0, fm1)[1]

    else:
        f1 = jax.scipy.special.i1e(z)

    # scan over other valuess

    def body(carry, xs: None):
        carry = iter(*carry)
        return carry, carry[1]

    _, out = jax.lax.scan(
        body,
        init=(1, f1, f0),
        xs=None,
        length=n - 1,
    )

    return jnp.array([f0, f1, *out])


@partial(custom_jvp, nondiff_argnums=(0, 2, 3))
@partial(jax.jit, static_argnums=(0, 2, 3))
def ie_n(n, z, half=False, forward=True):
    if forward:
        return jnp.where(
            z < 0.5,
            jnp.array([spherical_ie_n_recurrence(ni, z, half=half) for ni in range(n + 1)]) * jnp.exp(-jnp.abs(z)),
            ie_n_recurrence_pattern_unstalbe(n, z, half=half),
        )

    return ie_n_recurrence_pattern(n + 1, z, half=half)


@partial(jax.jit, static_argnums=(0, 2))
def spherical_ie_n_recurrence(n, z, half=True):
    assert half

    if half:
        v = n + 0.5
    else:
        v = n

    y = 0

    p0 = 1 / scipy.special.gamma(v + 1)

    y = p0

    for k in range(1, 3):
        p0 *= (0.25 * z**2) / (k * (k + v))

        y += p0

    return (0.5 * z) ** v * y


@ie_n.defjvp
def ie_n_jvp(n, half, forward, primals, tangents):
    (x,) = primals
    (x_dot,) = tangents

    ni = n

    if ni == 0:
        ni = 1

    y = ie_n(ni, x, half=half, forward=forward)

    nu = jnp.arange(1, len(y))

    if half:
        nu = nu + 0.5

    # https://dlmf.nist.gov/10.29
    dy = jnp.zeros(ni + 1)

    x_safe: jax.array = jnp.where(x < 1e-12, 1e-12, x)  # type:ignore

    if half:
        dy = dy.at[0].set(y[1] + 0.5 / x_safe * y[0])

    else:
        dy = dy.at[0].set(y[1])

    dy = dy.at[1:].set(y[:-1] - nu * y[1:] / x_safe)

    dy -= jnp.abs(x_safe) * y  # correction for exp

    return y[: n + 1], dy[: n + 1] * x_dot


spherical_jn_b = generate_bessel(scipy.special.spherical_jn, type=2)
spherical_yn = generate_bessel(scipy.special.spherical_yn, type=2)

ive = generate_bessel(scipy.special.ive, sign=+1, type=1, exp_scaled=True)


kve = generate_bessel(scipy.special.kve, sign=-1, type=1, exp_scaled=True)
