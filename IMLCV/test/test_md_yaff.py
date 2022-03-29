from __future__ import division

import os

from IMLCV.base.CV import CV, CVUtils, CombineCV
from IMLCV.base.MdEngine import YaffEngine, Bias
from yaff.test.common import get_alaninedipeptide_amber99ff
from yaff.log import log
import numpy as np
from molmod import units
from yaff.system import System
from yaff import ForceField
from ase.calculators.cp2k import CP2K
import ase.io
import ase.units
from pathlib import Path
import pytest
from IMLCV.test.common import ala_yaff, mil53_yaff, todo_ASE_yaff
import jax.numpy as jnp

log.set_level(log.medium)
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


def test_yaff_md_ala_dipep():
    yaffmd = ala_yaff()
    yaffmd.run(int(1e3))


def test_yaff_md_mil53():
    yaffmd = mil53_yaff()
    yaffmd.run(int(1e2))


def test_bias_save():
    """save and load bias to disk."""

    yaffmd = ala_yaff()
    yaffmd.run(int(1e3))

    yaffmd.bias.save_bias('output/bias_test.xyz')

    bias = Bias.load_bias('output/bias_test.xyz')

    cvs = np.array([0.0, 0.0])

    [b, db] = yaffmd.bias.compute(cvs=cvs, diff=True)
    [b2, db2] = bias.compute(cvs=cvs, diff=True)

    assert pytest.approx(jnp.sum((b - b2)**2)**(1 / 2))
    assert pytest.approx(jnp.sum((db[0] - db2[0])**2)**(1 / 2))


@pytest.mark.skip(reason="path+files not ready")
def test_yaff_ase():
    yaffmd = todo_ASE_yaff()
    yaffmd.run(1000)


if __name__ == '__main__':
    # test_yaff_md_ala_dipep()
    # test_yaff_md_mil53()
    test_bias_save()
    # test_yaff_ase()
