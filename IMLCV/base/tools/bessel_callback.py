import jax.lax
import jax.numpy as jnp
import scipy.special
from jax import custom_jvp, jit, pure_callback, vmap, pmap
from functools import partial

# see https://github.com/google/jax/issues/11002
# see https://jax.readthedocs.io/en/latest/notebooks/external_callbacks.html


def generate_bessel(function, type, sign=1, exp_scaled=False):
    @custom_jvp
    def cv(v, z):
        v, z = jnp.asarray(v), jnp.asarray(z)

        # Promote the input to inexact (float/complex).
        # Note that jnp.result_type() accounts for the enable_x64 flag.
        z = z.astype(jnp.result_type(float, z.dtype))

        fun = lambda v: pure_callback(
            lambda v, z: function(v, z).astype(z.dtype),
            jax.ShapeDtypeStruct(
                shape=z.shape,
                dtype=z.dtype,
            ),
            v,
            z,
            vectorized=True,
        )

        if v.shape == ():
            return fun(v)
        else:
            return vmap(fun)(v)

    @cv.defjvp
    def cv_jvp(primals, tangents):
        v, x = primals
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

    print(kve(0, 2))
    print(kve(0, jnp.array([2, 5])))
    print(kve(jnp.array([2, 5]), 1.0))

    # a = jax.grad(lambda z: jv(2.0, z))(jnp.array(5.0))

    for func, name in zip(
        [jv, yv, iv, kv, spherical_jn, spherical_yn, ive, kve],
        ["jv", "yv", "iv", "kv", " spherical_jv", "spherical_yv", "ive", "kve"],
    ):

        plt.figure()

        x = jnp.linspace(0, 20, 1000)
        for i in range(5):

            y = jit(func)(i, x)
            plt.plot(x, y, label=i)

        plt.ylim([-1.1, 1.1])
        plt.title(name)
        plt.legend()

        plt.draw()
        plt.pause(0.001)

        # plt.show()

    print("done")
