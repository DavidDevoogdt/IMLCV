"""summary IMLCV is still underdevelopement."""

import sys

if sys.version_info[:2] >= (3, 8):
    # TODO: Import directly (no need for conditional) when `python_requires = >= 3.8`
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
else:
    from importlib_metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = "IMLCV"
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError


import logging
import os
import sys
from logging import warning

import jax
import jax._src.tree_util
import jax.numpy as jnp
import jsonpickle
import numpy as np

# from flax.struct import PyTreeNode as MyPyTreeNode
from jax import random
from jsonpickle import tags
from jsonpickle.ext.numpy import register, register_handlers
from jsonpickle.handlers import BaseHandler

logging.getLogger("parsl").setLevel(logging.WARNING)

KEY = random.PRNGKey(0)


if "mpi4py" in sys.modules:
    warning("mpi4py doens't work well with cp2k calc atm")

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_pmap_no_rank_reduction", False)

logging.getLogger("absl").addFilter(
    logging.Filter(
        "call_tf works best with a TensorFlow function that does not capture variables or tensors from the context.",
    ),
)

register_handlers()


moved_functions = {
    "IMLCV.base.rounds.data_loader_output._transform": "IMLCV.base.rounds.DataLoaderOutput._transform",
}


def pytreenode_equal(self, other):
    if not isinstance(other, self.__class__):
        return False

    print(f"{self=}, {other=}")

    self_val, self_tree = jax.tree_flatten(self)
    other_val, other_tree = jax.tree_flatten(other)

    if not self_tree == other_tree:
        return False

    for a, b in zip(self_val, other_val):
        a = jnp.array(a)
        b = jnp.array(b)

        if not a.shape == b.shape:
            return False

        if not a.dtype == b.dtype:
            return False

        if not jnp.allclose(a, b):
            return False

    return True


# @partial(dataclass, frozen=False, field_specifiers=(field,), eq=False)


# class MyPyTreeNode:
#     def __eq__(self, other):
#         return pytreenode_equal(self, other)


class JaxHandler(BaseHandler):
    "flattens the jax array to numpy array, which is already handled by jsonpickle"

    def flatten(self, obj, data):
        self.context: jsonpickle.Unpickler
        data["array"] = self.context.flatten(np.array(obj).copy(), reset=False)
        return data

    def restore(self, data):
        self.context: jsonpickle.Pickler
        return jnp.array(self.context.restore(data["array"], reset=False))


register(jax.Array, JaxHandler, base=True)


class Unpickler(jsonpickle.Unpickler):
    def _restore_object_instance_variables(self, obj, instance):
        update_state = hasattr(instance, "__setstate__") and tags.STATE not in obj

        re_init = update_state
        out = super()._restore_object_instance_variables(obj, instance)

        if update_state:
            # print(f"Warning: {obj[tags.OBJECT]} has no state saved, trying to restore it")

            instance.__setstate__(instance.__dict__)

        if re_init:
            # re-init the object to make sure it's in a consistent state
            # sometimes jax objects are not properly initialized
            instance.__init__(**instance.__dict__)

        return out

    def _restore_function(self, obj):
        # load moved function instead of original
        if obj["py/function"] in moved_functions:
            obj["py/function"] = moved_functions[obj["py/function"]]

        return super()._restore_function(obj)


unpickler = Unpickler(on_missing="warn")
