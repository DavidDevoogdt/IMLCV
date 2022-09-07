import functools

import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from jax import jacrev, jit, vmap
from jax.experimental import jax2tf
from jax.experimental.jax2tf import call_tf, call_tf_p
from jax.interpreters import batching


def test_call_tf_batcher():
    @jit
    def f(x):
        return jnp.array([jnp.sum(x * x), jnp.product(x)])

    x = np.random.random(7)
    y = np.random.random((5, 7))

    # add batcher for call_tf_p
    batching.primitive_batchers[call_tf_p] = functools.partial(loop_batcher, call_tf_p)

    f_t = jax2tf.call_tf(jax2tf.convert(f, with_gradient=True))

    print(f"||f(x) - f_t(x)||_2 = { jnp.linalg.norm( f(x)-f_t(x) )  }")

    # test vmap
    f_v = jit(vmap(f))
    f_v_t = jit(vmap(f_t))

    print(f"||f(y) - f_t(y)||_2 = { jnp.linalg.norm( f_v(y)-f_v_t(y) )  }")

    j = jit(jacrev(f))
    j_t = jit(jacrev(f_t))

    print(f"||jac(f)(x) - jac(f_t)(x)||_2 = { jnp.linalg.norm( j(x)-j_t(x) )  }")


def loop_batcher(prim, args, dims, **params):

    # do partial application of args on given position"
    def apply(batch_args, f, static_args, static_pos):
        arguments = []

        l = len(batch_args) + len(static_args)
        j = 0
        k = 0
        for i in range(l):
            if i in static_pos:
                arguments.append(static_args[j])
                j += 1
            else:
                arguments.append(batch_args[k])
                k += 1

        return f(*arguments)

    static_pos = []
    static_args = []
    batch_args = []

    # find arguments for partial application
    for i, (arg, batch_axis) in enumerate(zip(args, dims)):
        if batch_axis is None:
            static_pos.append(i)
            static_args.append(arg)
        else:
            assert batch_axis == 0, "other position not yet implemented"
            batch_args.append(arg)

    # vectorize
    def par_fun(batch_args, static_args):
        return tf.vectorized_map(
            fn=functools.partial(
                apply,
                f=params["callable_flat_tf"],
                static_args=static_args,
                static_pos=static_pos,
            ),
            elems=batch_args,
        )

    if len(batch_args) != 1:
        raise NotImplementedError

    # execute with given arguments
    ret = call_tf(par_fun)(batch_args, static_args)
    return (ret, (0,) * len(ret))


if __name__ == "__main__":
    test_call_tf_batcher()
