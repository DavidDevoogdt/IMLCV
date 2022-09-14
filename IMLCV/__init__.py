"""summary IMLCV is still underdevelopement."""

import functools
import getpass
import os

import jax
import tensorflow as tf
from ase.calculators.cp2k import Cp2kShell
from jax.interpreters import batching

from IMLCV.external.tf2jax import call_tf_p, loop_batcher


def recv(self):
    """Receive a line from the cp2k_shell"""
    assert self._child.poll() is None  # child process still alive?
    line = self._child.stdout.readline().strip()
    if self._debug:
        print("Received: " + line)
    self.isready = line == "* READY"
    return line


Cp2kShell.recv = recv

# from yaff.log import log

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


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
