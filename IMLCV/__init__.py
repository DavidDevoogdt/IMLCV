"""summary IMLCV is still underdevelopement."""

import logging
import os
import sys
from logging import warning

import jax
import tensorflow as tf

# import tensorflow as tf
from jax import random

import yaff

yaff.log.set_level(yaff.log.silent)


# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".50"
tf.config.experimental.set_visible_devices([], "GPU")

KEY = random.PRNGKey(0)
LOGLEVEL = logging.CRITICAL


if "mpi4py" in sys.modules:
    warning("mpi4py doens't work well with cp2k calc atm")

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


jax.config.update("jax_enable_x64", True)

# cpu based
jax.config.update("jax_platform_name", "cpu")

# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "FALSE"
# tf.config.experimental.set_visible_devices([], "GPU")


# from IMLCV.external.tf2jax import call_tf_p, loop_batcher
# batching.primitive_batchers[call_tf_p] = functools.partial(loop_batcher, call_tf_p)
