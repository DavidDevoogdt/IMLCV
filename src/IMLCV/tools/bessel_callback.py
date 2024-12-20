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

    def cv_inner(v: int, z: float):
        res_dtype_shape = jax.ShapeDtypeStruct(
            shape=(),
            dtype=z.dtype,
        )

        return pure_callback(
            _function,
            res_dtype_shape,
            v,
            z,
            vmap_method="expand_dims",
        )

    @custom_jvp
    def cv(v, z):
        v, z = jnp.asarray(v), jnp.asarray(z)

        # Promote the input to inexact (float/complex).
        # Note that jnp.result_type() accounts for the enable_x64 flag.
        z = z.astype(jnp.result_type(float, z.dtype))

        assert v.ndim == 0 and z.ndim == 0, "batch with vmap"
        return cv_inner(v, z)

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
def recurrence_pattern(n, z):
    jlp1 = 0

    jl = jnp.where(z.dtype == jnp.float32, 1e-30, 1e-200)

    # https://dlmf.nist.gov/10.51
    def iter(n, jl, jlp1):
        jlm1 = (2 * n + 1) / z * jl - jlp1

        return n - 1, jlm1, jl

    l = n + ORDER

    l, jl, jlp1 = jax.lax.fori_loop(
        0,
        ORDER,
        lambda i, args: iter(*args),
        (l, jl, jlp1),
        unroll=True,
    )

    # scan over other valuess

    def body(carry, xs: None):
        carry = iter(*carry)
        return carry, carry[1]

    _, out = jax.lax.scan(
        body,
        init=(l, jl, jlp1),
        xs=None,
        length=n,
        unroll=True,
    )

    out = out[::-1]

    f0 = jnp.sinc(z / jnp.pi)

    n_safe = jnp.where(out[0] != 0, out[0], 1)

    return out / n_safe * f0


@partial(custom_jvp, nondiff_argnums=(0,))
@partial(jax.jit, static_argnums=0)
def spherical_jn(n, z):
    return recurrence_pattern(n + 1, z)


@spherical_jn.defjvp
def spherical_jn_jvp(n, primals, tangents):
    (x,) = primals
    (x_dot,) = tangents

    ni = n

    if ni == 0:
        ni = 1

    y = spherical_jn(ni, x)

    dy = jnp.zeros(ni + 1)
    dy = dy.at[0].set(-y[1])
    dy = dy.at[1:].set(y[:-1] - (jnp.arange(1, len(y)) + 1.0) * y[1:] / x)

    return y[: n + 1], dy[: n + 1] * x_dot


# spherical_jn = csphjy

spherical_jn_b = generate_bessel(scipy.special.spherical_jn, type=2)
spherical_yn = generate_bessel(scipy.special.spherical_yn, type=2)

ive = generate_bessel(scipy.special.ive, sign=+1, type=1, exp_scaled=True)
kve = generate_bessel(scipy.special.kve, sign=-1, type=1, exp_scaled=True)
