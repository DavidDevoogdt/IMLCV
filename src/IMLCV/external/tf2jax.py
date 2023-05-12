import functools

import tensorflow as tf
from jax.experimental.jax2tf import call_tf


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
    ret = call_tf(par_fun)(batch_args, static_args)  # noqa: F821
    return (ret, (0,) * len(ret))
