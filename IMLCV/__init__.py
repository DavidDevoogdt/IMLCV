"""summary IMLCV is still underdevelopement."""

import functools
import logging
import os

from jax.interpreters import batching

import parsl
from IMLCV.external.tf2jax import call_tf_p, loop_batcher

# from yaff.log import log

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


LOCAL = True
DEBUG = True

# SETUP Jax

# jax.config.update("jax_platform_name", "cpu")
# tf.config.experimental.set_visible_devices([], "GPU")
# jax.config.update('jax_disable_jit', True)


batching.primitive_batchers[call_tf_p] = functools.partial(loop_batcher, call_tf_p)

parsl.set_file_logger("log.txt", level=logging.WARN)

# parsl.set_stream_logger(level=logging.ERROR)
# os.environ["NUMBA_DISABLE_JIT"] = "1"
# tf.data.experimental.enable_debug_mode()
# tf.config.run_functions_eagerly(True)
