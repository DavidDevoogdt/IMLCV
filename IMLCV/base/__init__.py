"""this is a test, base."""

import jax
from IMLCV.base.CV import *
from IMLCV.base.CVDiscovery import *
from IMLCV.base.MdEngine import *
from IMLCV.base.Observable import *
from IMLCV.scheme import *

jax.config.update('jax_platform_name', 'cpu')

# from jax.config import config

# config.update('jax_disable_jit', True)
