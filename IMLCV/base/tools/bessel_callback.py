from functools import partial

import jax.lax
import jax.numpy as jnp
import scipy.special
from jax import Array, custom_jvp, jit, pure_callback

# see https://github.com/google/jax/issues/11002


def generate_bessel(function):
    """function is Jv, Yv, Hv_1,Hv_2"""

    @partial(custom_jvp, nondiff_argnums=(0))
    @partial(jit, static_argnums=(0))
    def cv(v, x):
        return pure_callback(
            lambda x: function(v, x),
            x,
            x,
            vectorized=True,
        )

    @cv.defjvp
    def cv_jvp(primals, tangents):
        v, x = primals
        dv, dx = tangents
        primal_out = cv(v, x)

        # https://dlmf.nist.gov/10.6 formula 10.6.1
        tangents_out = jax.lax.cond(
            v == 0,
            lambda: -cv(v + 1, x),
            lambda: 0.5 * (cv(v - 1, x) - cv(v + 1, x)),
        )

        return primal_out, tangents_out * dx

    return cv


jv = generate_bessel(scipy.special.jv)
yv = generate_bessel(scipy.special.yv)
hankel1 = generate_bessel(scipy.special.hankel1)
hankel2 = generate_bessel(scipy.special.hankel2)


def generate_modified_bessel(function, sign):
    """function is Kv and Iv"""

    @partial(custom_jvp, nondiff_argnums=(0))
    @partial(jit, static_argnums=(0))
    def cv(v: float, x: Array):
        return pure_callback(
            lambda x: function(v, x),
            x,
            x,
            vectorized=False,
        )

    @cv.defjvp
    def cv_jvp(primals, tangents):
        v, x = primals
        dv, dx = tangents
        primal_out = cv(v, x)

        # https://dlmf.nist.gov/10.6 formula 10.6.1
        tangents_out = jax.lax.cond(
            v == 0,
            lambda: sign * cv(v + 1, x),
            lambda: 0.5 * (cv(v - 1, x) + cv(v + 1, x)),
        )

        return primal_out, tangents_out * dx

    return cv


kv = generate_modified_bessel(scipy.special.kv, sign=-1)
iv = generate_modified_bessel(scipy.special.iv, sign=+1)


def spherical_bessel_genearator(f):
    @custom_jvp
    def cv(v: int, x: Array):
        y = x.astype(jnp.float64)

        out = pure_callback(
            f,
            y,
            v,
            y,
            vectorized=True,
        )

        return out

    @cv.defjvp
    def cv_jvp(primals, tangents):
        v, x = primals
        dv, dx = tangents

        primal_out = cv(v, x)

        # https://dlmf.nist.gov/10.51

        if v == 0:
            tangents_out = -cv(v + 1, x)
        else:
            tangents_out = cv(v - 1, x) - (v + 1) / x * primal_out

        return primal_out, tangents_out * dx

    return cv


spherical_jn = spherical_bessel_genearator(scipy.special.spherical_jn)
spherical_yn = spherical_bessel_genearator(scipy.special.spherical_yn)

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    x = jnp.linspace(0.0, 20.0, num=1000)

    # a = jax.grad(lambda z: jv(2.0, z))(jnp.array(5.0))

    # for func, name in zip(
    #     [jv, yv, iv, kv, spherical_jn, spherical_yn],
    #     ["jv", "yv", "iv", "kv", " spherical_jv", "spherical_yv"],
    # ):

    for func, name in zip(
        [spherical_jn, spherical_yn],
        [" spherical_jv", "spherical_yv"],
    ):

        plt.figure()

        for i in range(5):

            y = func(jnp.array(i), x)
            plt.plot(x, y, label=i)

        plt.ylim([-1.1, 1.1])
        plt.title(name)
        plt.legend()

        plt.draw()
        plt.pause(0.001)

        # plt.show()

    print("done")
