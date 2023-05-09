"""summary IMLCV is still underdevelopement."""

import logging
import os
import sys
from logging import warning

import jax
from jax import random

KEY = random.PRNGKey(0)
LOGLEVEL = logging.CRITICAL


if "mpi4py" in sys.modules:
    warning("mpi4py doens't work well with cp2k calc atm")

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


jax.config.update("jax_enable_x64", True)

# cpu based
jax.config.update("jax_platform_name", "cpu")
