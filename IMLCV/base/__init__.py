"""this is a test, base."""

from IMLCV.scheme import *
from IMLCV.base.CV import *
from IMLCV.base.CVDiscovery import *
from IMLCV.base.MdEngine import *
from IMLCV.base.Observable import *

import jax

jax.config.update('jax_platform_name', 'cpu')