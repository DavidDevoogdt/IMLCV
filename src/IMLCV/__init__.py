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
from jax import random
import jax._src.tree_util
from flax.struct import PyTreeNode
import jsonpickle
from jsonpickle.handlers import BaseHandler
from jsonpickle.ext.numpy import register_handlers, register
import numpy as np
import jax.numpy as jnp


KEY = random.PRNGKey(0)
LOGLEVEL = logging.CRITICAL


if "mpi4py" in sys.modules:
    warning("mpi4py doens't work well with cp2k calc atm")

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

logging.getLogger("absl").addFilter(
    logging.Filter(
        "call_tf works best with a TensorFlow function that does not capture variables or tensors from the context.",
    ),
)


register_handlers()


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
        update = False

        if isinstance(instance, PyTreeNode):
            update = True

        out = super()._restore_object_instance_variables(obj, instance)

        if update:
            instance.__init__(**instance.__dict__)

        return out
