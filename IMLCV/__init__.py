"""summary IMLCV is still underdevelopement."""

import functools
import os

import jax
import tensorflow as tf
from jax.interpreters import batching

from IMLCV.test.tf2jax import call_tf_p, loop_batcher

# from yaff.log import log

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


LOCAL = True
DEBUG = True

# SETUP Jax

jax.config.update('jax_platform_name', 'cpu')
tf.config.experimental.set_visible_devices([], 'GPU')


batching.primitive_batchers[call_tf_p] = functools.partial(
    loop_batcher, call_tf_p)


# os.environ["NUMBA_DISABLE_JIT"] = "1"
