import jax.lax
import jax.numpy as jnp
import numpy as onp
import scipy.special
from jax import custom_jvp, grad, jit, pure_callback, vmap
from jax.custom_batching import custom_vmap

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

        if type == 0:

            """functions Jv, Yv, Hv_1,Hv_2"""
            # https://dlmf.nist.gov/10.6 formula 10.6.1
            tangents_out = jax.lax.cond(
                v == 0,
                lambda: -cv(v + 1, x),
                lambda: 0.5 * (cv(v - 1, x) - cv(v + 1, x)),
            )
        elif type == 1:

            """functions Kv and Iv"""
            # https://dlmf.nist.gov/10.29 formula 10.29.1
            tangents_out = jax.lax.cond(
                v == 0,
                lambda: sign * cv(v + 1, x),
                lambda: 0.5 * (cv(v - 1, x) + cv(v + 1, x)),
            )

        elif type == 2:
            """functions: spherical bessels"""
            # https://dlmf.nist.gov/10.51 formula 10.51.2

            tangents_out = jax.lax.cond(
                v == 0,
                lambda: -cv(v + 1, x),
                lambda: cv(v - 1, x) - (v + 1) / x * primal_out,
            )

        # chain rule
        if exp_scaled:
            if sign == -1:
                tangents_out += primal_out
            elif sign == 1:
                tangents_out -= jnp.sign(x) * primal_out

        return primal_out, tangents_out * dx

    return cv


jv = generate_bessel(scipy.special.jv, type=0)
yv = generate_bessel(scipy.special.yv, type=0)
hankel1 = generate_bessel(scipy.special.hankel1, type=0)
hankel2 = generate_bessel(scipy.special.hankel2, type=0)

kv = generate_bessel(scipy.special.kv, sign=-1, type=1)
iv = generate_bessel(scipy.special.iv, sign=+1, type=1)

spherical_jn = generate_bessel(scipy.special.spherical_jn, type=2)
spherical_yn = generate_bessel(scipy.special.spherical_yn, type=2)

ive = generate_bessel(scipy.special.ive, sign=+1, type=1, exp_scaled=True)
kve = generate_bessel(scipy.special.kve, sign=-1, type=1, exp_scaled=True)

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    print(vmap(kve, in_axes=(0, None))(jnp.array([2, 5]), 2))
    print(vmap(kve, in_axes=(None, 0))(2, jnp.array([2, 5])))
    print(
        vmap(vmap(kve, in_axes=(0, None)), in_axes=(None, 0))(
            jnp.array([2, 5]), jnp.array([2, 5])
        )
    )
    print(
        jit(
            vmap(
                vmap(vmap(kve, in_axes=(0, None)), in_axes=(0, None)), in_axes=(None, 0)
            )
        )(
            jnp.array(
                [
                    [2, 5],
                    [2, 5],
                ]
            ),
            jnp.array(
                [2, 5],
            ),
        )
    )

    # a = jax.grad(lambda z: jv(2.0, z))(jnp.array(5.0))

    for func, name in zip(
        [jv, yv, iv, kv, spherical_jn, spherical_yn, ive, kve],
        ["jv", "yv", "iv", "kv", " spherical_jv", "spherical_yv", "ive", "kve"],
    ):

        plt.figure()

        x = jnp.linspace(0, 20, 1000)
        for i in range(5):

            y = jit(vmap(func, in_axes=(None, 0)))(i, x)
            plt.plot(x, y, label=i)

        plt.ylim([-1.1, 1.1])
        plt.title(name)
        plt.legend()

        plt.draw()
        plt.pause(0.001)

    # sanity chekc to see wither ive and kve behave correctly

    k1e = lambda x: kve(1, x)
    k1e2 = lambda x: kv(1, x) * jnp.exp(x)

    x = jnp.linspace(1, 5, 1000)
    assert jnp.linalg.norm(vmap(k1e)(x) - vmap(k1e2)(x)) < 1e-5
    assert jnp.linalg.norm(vmap(grad(k1e))(x) - vmap(grad(k1e2))(x)) < 1e-5

    i1e = lambda x: ive(1, x)
    i1e2 = lambda x: iv(1, x) * jnp.exp(-jnp.abs(x))

    assert jnp.linalg.norm(vmap(i1e)(x) - vmap(i1e2)(x)) < 1e-5
    assert jnp.linalg.norm(vmap(grad(i1e))(x) - vmap(grad(i1e2))(x)) < 1e-5
