from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from jax import closure_convert, custom_jvp, vmap
from jax.tree_util import Partial, tree_map
from scipy.special import roots_laguerre, roots_legendre


def _quad_nd(f, w, x, use_custom_jvp=True):
    x_mg = jnp.array([*jnp.meshgrid(*x, indexing="ij")]).reshape(len(x), -1)
    w_mg = jnp.array([*jnp.meshgrid(*w, indexing="ij")]).reshape(len(w), -1)

    def fun(*args, x_mg=None, w_mg=None, f=None):
        if use_custom_jvp:
            # print("using custom jvp")

            _f_closed, closure_args = closure_convert(
                f,
                *jnp.array([xi[0] for xi in x]),
                *args,
            )

            def f_closed(x, args, closure_args):
                return _f_closed(*x, *args, *closure_args)

            @partial(custom_jvp, nondiff_argnums=(0,))
            def _int(_f2_closed, args, x_mg, w_mg, closure_args):
                @partial(vmap, in_axes=(1, 1))
                def f_int(w, x):
                    y = _f2_closed(x, args, closure_args)
                    w_tot = jnp.prod(w)

                    return tree_map(lambda x: x * w_tot, y)

                out = f_int(w_mg, x_mg)

                return tree_map(partial(jnp.sum, axis=0), out)

            # calculate derivative inside the integral
            # (d/d args int^n_a(x)^b(x)  f(x^n, args) dx^n)  * d args =  int^n_a(x)^b(x)  (d/d args f(x^n, args) * d args) dx^n
            # the latter just carries a single number through the integral

            @_int.defjvp
            def _int_jvp(_f_closed, primals, tangents):
                (args, x_mg, w_mg, closure_args) = primals
                (dargs, dx_mg, dw_mg, dclosure_args) = tangents

                def _f_closed_jvp(x, ada, closure_args):
                    args, dargs = ada
                    y, dy = jax.jvp(lambda args: _f_closed(x, args, closure_args), (args,), (dargs,))

                    return y, dy

                y, dy = _int(_f_closed_jvp, (args, dargs), x_mg, w_mg, closure_args)

                return y, dy

            return _int(f_closed, args, x_mg, w_mg, closure_args)

        else:

            @partial(vmap, in_axes=(1, 1))
            def f_int(w, x):
                y = f(*x, *args)
                w_tot = jnp.prod(w)

                return tree_map(lambda x: x * w_tot, y)

            out = f_int(w_mg, x_mg)

            return tree_map(partial(jnp.sum, axis=0), out)

    return Partial(fun, x_mg=x_mg, w_mg=w_mg, f=f)


def quad_bounds(a, b, scale=1, n=21):
    x_lag, w_lag = roots_laguerre(n)
    x_leg, w_leg = roots_legendre(n)
    x_lag, w_lag = jnp.array(x_lag), jnp.array(w_lag)
    x_leg, w_leg = jnp.array(x_leg), jnp.array(w_leg)

    x = jnp.where(
        b == jnp.inf,
        x_lag,
        x_leg,
    )

    w = jnp.where(
        b == jnp.inf,
        w_lag,
        w_leg,
    )

    def W(x, w, a, b):
        return jnp.where(
            b == jnp.inf,
            jnp.exp(x + jnp.log(w)) / scale,
            0.5 * (b - a) * w,
        )

    def T(x, a, b):
        return jnp.where(
            b == jnp.inf,
            x / scale + a,
            0.5 * (b - a) * x + 0.5 * (b + a),
        )

    return x, w, W, T


def quad(f, a, b, scale=1, n=21):
    x, w, W, T = quad_bounds(a, b, scale, n)

    x = T(x, a, b)
    w = W(x, w, a, b)

    return _quad_nd(f, [w], [x])


# integrates f from 0 to infinity
def dquad(f, a0, b0, a1, b1, scale=1, n=21):
    x0, w0, W0, T0 = quad_bounds(a0, b0, scale, n)

    w0 = W0(x0, w0, a0, b0)
    x0 = T0(x0, a0, b0)

    if isinstance(a1, Callable) or isinstance(b1, Callable):
        a1 = a1(x0) if isinstance(a1, Callable) else jnp.full_like(x0, a1)
        b1 = b1(x0) if isinstance(b1, Callable) else jnp.full_like(x0, b1)

        x1, w1, W1, T1 = quad_bounds(a1[0], b1[0], scale, n)

        w1 = W1(x1, w1, a1, b1)
        x1 = T1(x1, a1, b1)

    else:
        x1, w1, W1, T1 = quad_bounds(a1, b1, scale)

        w1 = W1(x1, w1, a1, b1)
        x1 = T1(x1, a1, b1)

    return _quad_nd(f, [w0, w1], [x0, x1])
