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
import cloudpickle

import jax
from jax import random
import jax._src.tree_util

from flax.struct import PyTreeNode

# helpr to unpickle class without setstate
import jsonpickle


KEY = random.PRNGKey(0)
LOGLEVEL = logging.CRITICAL


if "mpi4py" in sys.modules:
    warning("mpi4py doens't work well with cp2k calc atm")

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


jax.config.update("jax_enable_x64", True)

# cpu based
jax.config.update("jax_platform_name", "cpu")


logging.getLogger("absl").addFilter(
    logging.Filter(
        "call_tf works best with a TensorFlow function that does not capture variables or tensors from the context.",
    ),
)


class Unpickler(jsonpickle.Unpickler):
    def _restore_object_instance_variables(self, obj, instance):
        update = False

        if isinstance(instance, PyTreeNode):
            update = True

        try:
            out = super()._restore_object_instance_variables(obj, instance)
        except Exception as e:
            print(f"got {e=}\\m{obj=}\n {instance=}")

        if update:
            # print(f"calling init for {instance.__class__}")
            instance.__init__(**instance.__dict__)

        return out
