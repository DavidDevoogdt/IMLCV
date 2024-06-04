import jax.lax
import jax.numpy as jnp
import numpy as onp
import scipy.special
from jax import custom_jvp
from jax import pure_callback
from jax.custom_batching import custom_vmap
from IMLCV.tools.bessel_jn import bessel_jn
from functools import partial
import jax


# from jax.scipy.special import bessel_jn

# see https://github.com/google/jax/issues/11002
# see https://jax.readthedocs.io/en/latest/notebooks/external_callbacks.html


def generate_bessel(function, type, sign=1, exp_scaled=False):
    def _function(v, x):
        v, x = onp.asarray(v), onp.asarray(x)

        return function(v, x).astype(x.dtype)

    @custom_vmap
    def cv_inner(v, z):
        res_dtype_shape = jax.ShapeDtypeStruct(
            shape=v.shape,
            dtype=z.dtype,
        )

        return pure_callback(
            _function,
            res_dtype_shape,
            v,
            z,
            vectorized=True,
        )

    @cv_inner.def_vmap
    def _function_vmap(axis_size, in_batched, v, x):
        v_batched, x_batched = in_batched

        if not (v_batched and x_batched):
            a = jax.lax.broadcast(v, [axis_size]) if x_batched else v
            b = jax.lax.broadcast(x, [axis_size]) if v_batched else x
        else:
            a = v
            b = x

        out = cv_inner(a, b)

        return out, True

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


# @partial(custom_jvp, nondiff_argnums=(0,))
# @partial(jax.jit, static_argnums=0)
# def spherical_jn(v, x):
#     out = csphjy(v, x)

#     return jnp.real(out)


# @spherical_jn.defjvp
# def cv_jvp(v, primals, tangents):
#     (x,) = primals
#     (dx,) = tangents

#     o = spherical_jn(v + 1, x)

#     primal_out = o[:-1]

#     v_vec = jnp.arange(1, v + 1)

#     tangents_out = jnp.array([-o[1]])

#     if v != 0:
#         tangents_out = jnp.concatenate(
#             [tangents_out, (v_vec * o[v_vec - 1] - (v_vec + 1) * o[v_vec + 1]) / (2 * v_vec + 1)]
#         )

#     return primal_out, tangents_out * dx


# https://github.com/tpudlik/sbf/blob/master/algos/d_recur_miller.py

ORDER = 100  # Following Jablonski (1994)


@partial(jax.jit, static_argnums=0)
def recurrence_pattern(n, z):
    jlp1 = 0
    jl = 10 ** (-200)

    # https://dlmf.nist.gov/10.51
    def iter(n, jl, jlp1):
        jlm1 = (2 * n + 1) / z * jl - jlp1

        return n - 1, jlm1, jl

    l = n + ORDER

    for _ in range(ORDER):
        l, jl, jlp1 = iter(l, jl, jlp1)

    # scan over other values

    def body(carry, xs):
        carry = iter(*carry)
        return carry, carry[1]

    _, out = jax.lax.scan(body, init=(l, jl, jlp1), xs=None, length=n)

    out = out[::-1]

    f0 = jnp.sinc(z / jnp.pi)

    return out / out[0] * f0


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


# spherical_jn = csphjy  # generate_bessel(scipy.special.spherical_jn, type=2)
spherical_yn = generate_bessel(scipy.special.spherical_yn, type=2)

ive = generate_bessel(scipy.special.ive, sign=+1, type=1, exp_scaled=True)
kve = generate_bessel(scipy.special.kve, sign=-1, type=1, exp_scaled=True)
