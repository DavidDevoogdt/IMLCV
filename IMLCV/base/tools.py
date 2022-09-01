from functools import partial

import jax
from jax import jit


class HashableArrayWrapper:
    """#see https://github.com/google/jax/issues/4572"""

    def __init__(self, val):
        self.val = val

    def __hash__(self):
        return hash(self.val.tobytes())

    def __eq__(self, other):
        eq = isinstance(
            other, HashableArrayWrapper) and (self.__hash__()
                                              == other.__hash__())
        return eq

    def __getitem__(self, slice):
        return self.val[slice]


def jit_satic_array(fun, static_array_argnums=(), static_argnums=()):
    """#see https://github.com/google/jax/issues/4572"""

    @partial(jit, static_argnums=static_array_argnums + static_argnums)
    def callee(*args):
        args = list(args)
        for i in static_array_argnums:
            args[i] = args[i].val
        return fun(*args)

    def caller(*args):
        args = list(args)
        for i in static_array_argnums:
            args[i] = HashableArrayWrapper(args[i])
        return callee(*args)

    return caller
