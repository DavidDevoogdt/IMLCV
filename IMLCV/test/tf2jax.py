import functools
from typing import Iterable

import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jacfwd, jacrev, jit, vmap
from jax.experimental import jax2tf
from jax.experimental.jax2tf import call_tf, call_tf_p
from jax.interpreters import batching


def test_call_tf_batcher():

    @jit
    def f(x):
        return jnp.array([jnp.sum(x * x),  jnp.product(x)])

    f_t = jax2tf.call_tf(jax2tf.convert(f, with_gradient=True))

    x = np.random.random((7))
    y = np.random.random((5, 7))

    print(f"||f(x) - f_t(x)||_2 = { jnp.linalg.norm( f(x)-f_t(x) )  }")

    # test vmap
    f_v = jit(vmap(f))
    f_v_t = jit(vmap(f_t))

    print(f"||f(y) - f_t(y)||_2 = { jnp.linalg.norm( f_v(y)-f_v_t(y) )  }")

    j = jit(jacrev(f))
    j_t = jit(jacrev(f_t))
    j_t = jit(jacrev(f_t))

    print(
        f"||jac(f)(x) - jac(f_t)(x)||_2 = { jnp.linalg.norm( j(x)-j_t(x) )  }")

#new batcher
def loop_batcher(prim, args, dims,  **params):

    # determine new axis size
    for ni, di in enumerate(dims):
        if di is not None:
            axis_size = args[ni].shape[di]
            break

    # generate combination of indices for different arguments
    args_indices = []
    for xi, ia in zip(args, dims):

        xs = xi.shape
        indices = [Ellipsis if i !=
                   ia else None for i, _ in enumerate(xs)]

        l = []
        for i in range(axis_size):
            if ia is not None:
                indices[ia] = i

            l.append(tuple(indices))
        args_indices.append(l)

    # apply function
    out = []
    for inds in list(zip(*args_indices),):
        outp = prim.bind(*[a[b] for a, b in zip(
            args, inds)], **params)
        if not isinstance(outp, Iterable):
            outp = tuple((outp,))
        out.append(outp)

    ret = []

    # collect output in arrays
    for out_args in list(zip(*out)):
        val = jax.numpy.hstack(out_args)
        val = jax.numpy.reshape(val,  (axis_size,  *out_args[0].shape))

        ret.append(val)

    return (ret,  (0,)*len(ret))

#add batcher for call_tf_p
batching.primitive_batchers[call_tf_p] = functools.partial(
    loop_batcher, call_tf_p)


if __name__ == "__main__":
    test_call_tf_batcher()
