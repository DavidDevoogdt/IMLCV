"""summary IMLCV is still underdevelopement."""

import functools
import getpass
import os
import sys
from logging import warning
from pathlib import Path


import jax

# import tensorflow as tf
from jax import random
from jax.interpreters import batching

import yaff
from IMLCV.external.tf2jax import call_tf_p, loop_batcher



KEY = random.PRNGKey(0)

if "mpi4py" in sys.modules:
    warning("mpi4py doens't work well with cp2k calc atm")

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

yaff.log.set_level(yaff.log.silent)


jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_array", True)

batching.primitive_batchers[call_tf_p] = functools.partial(loop_batcher, call_tf_p)
