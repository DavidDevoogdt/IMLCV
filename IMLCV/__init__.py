"""summary IMLCV is still underdevelopement."""

import os

import jax
from yaff.log import log

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROVIDER = 'local'

# SETUP Jax

jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)
# jax.config.update('jax_disable_jit', True)

log.set_level(log.silent)




