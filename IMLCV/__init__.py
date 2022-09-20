"""summary IMLCV is still underdevelopement."""

import functools
import getpass
import os
from pathlib import Path

import jax
import tensorflow as tf
from jax.interpreters import batching

from IMLCV.external.tf2jax import call_tf_p, loop_batcher

# from yaff.log import log

ROOT_DIR = Path(os.path.dirname(__file__)).parent





name = getpass.getuser()
if name == "vsc43693":
    LOCAL = False
elif name == "david":
    LOCAL = True
else:
    raise ValueError("unknow pc")


DEBUG = True
GPU = False


# SETUP Jax
if not GPU:
    jax.config.update("jax_platform_name", "cpu")
    tf.config.experimental.set_visible_devices([], "GPU")
# jax.config.update('jax_disable_jit', True)


batching.primitive_batchers[call_tf_p] = functools.partial(loop_batcher, call_tf_p)

# parsl.set_stream_logger(level=logging.ERROR)
# os.environ["NUMBA_DISABLE_JIT"] = "1"
# tf.data.experimental.enable_debug_mode()
# tf.config.run_functions_eagerly(True)
