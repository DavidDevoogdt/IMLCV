"""summary IMLCV is still underdevelopement."""
import os

import fireworks.fw_config
import jax
from yaff.log import log

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_TYPE = 'FireWorks'  # 'qsub' or 'direct' or 'FireWorks'
MONGO_HOST = "mongodb+srv://david:david@cluster0.bwzkf.mongodb.net/?retryWrites=true&w=majority"

DEBUG = True

jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)
# jax.config.update('jax_disable_jit', True)

fireworks.fw_config.PRINT_FW_JSON = False


log.set_level(log.silent)
