from functools import partial
from math import floor
from IMLCV.base.MdEngine import MDEngine
from IMLCV.base.bias import NoneBias
from IMLCV.base.rounds import RoundsMd
from IMLCV.base.CV import CV

from molmod.constants import boltzmann
from molmod.units import nanosecond, kjmol

import os
import numpy as np
import scipy as sp
import scipy.interpolate

from thermolib import Histogram2D


class CVDiscovery:
    """convert set of coordinates to good collective variables."""

    def __init__(self) -> None:
        pass

    def compute(self, data) -> CV:
        NotImplementedError


if __name__ == '__main__':
    from IMLCV.test.test_scheme import test_cv_discovery
    test_cv_discovery()