import pytest


@pytest.mark.skip(reason="not supported")
def test_call_tf_batcher():
    import functools

    import jax.numpy as jnp
    import numpy as np
    from jax import jacrev, jit, vmap
    from jax.experimental import jax2tf
    from jax.experimental.jax2tf.call_tf import call_tf_p
    from jax.interpreters import batching

    from IMLCV.external.tf2jax import loop_batcher

    @jit
    def f(x):
        return jnp.array([jnp.sum(x * x), jnp.product(x)])

    x = np.random.random(7)
    y = np.random.random((5, 7))

    # add batcher for call_tf_p
    batching.primitive_batchers[call_tf_p] = functools.partial(loop_batcher, call_tf_p)

    f_t = jax2tf.call_tf(jax2tf.convert(f, with_gradient=True))

    print(f"||f(x) - f_t(x)||_2 = {jnp.linalg.norm(f(x) - f_t(x))}")

    # test vmap
    f_v = jit(vmap(f))
    f_v_t = jit(vmap(f_t))

    print(f"||f(y) - f_t(y)||_2 = {jnp.linalg.norm(f_v(y) - f_v_t(y))}")

    j = jit(jacrev(f))
    j_t = jit(jacrev(f_t))

    print(f"||jac(f)(x) - jac(f_t)(x)||_2 = {jnp.linalg.norm(j(x) - j_t(x))}")


if __name__ == "__main__":
    test_call_tf_batcher()
