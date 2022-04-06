from __future__ import division

import os

from yaff.test.common import get_alaninedipeptide_amber99ff
from yaff.log import log
import numpy as np
import ase.units
import pytest
from IMLCV.test.common import ala_yaff, mil53_yaff, todo_ASE_yaff
import jax.numpy as jnp

import cProfile
import pstats
from pstats import SortKey

from IMLCV.base.CV import CV, CVUtils, CombineCV
from IMLCV.base.MdEngine import MDEngine, YaffEngine
from IMLCV.base.bias import BiasMTD, CompositeBias, Bias

import numpy as np
from molmod import units
import ase.io

log.set_level(log.medium)


def change_fold():

    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)


def test_yaff_save_load_func():
    change_fold()

    yaffmd = ala_yaff()

    yaffmd.run(int(761))

    yaffmd.save('output/yaff_save.d')
    yeet = MDEngine.load('output/yaff_save.d', filename='output/output2.h5')

    [coor1, cell1] = yaffmd.get_state()
    [coor2, cell2] = yeet.get_state()

    assert pytest.approx(coor1) == coor2
    assert cell1.shape == cell2.shape
    assert pytest.approx(yaffmd.ener.compute_coor(coor1, cell1)) == yeet.ener.compute_coor(coor2, cell2)


def test_combine_bias():
    change_fold()

    T = 600 * units.kelvin
    ff = get_alaninedipeptide_amber99ff()

    cvs = CombineCV([
        CV(CVUtils.dihedral, numbers=[4, 6, 8, 14], periodicity=[-np.pi, np.pi]),
        CV(CVUtils.dihedral, numbers=[6, 8, 14, 16], periodicity=[-np.pi, np.pi]),
    ])
    bias1 = BiasMTD(cvs=cvs, K=2.0 * units.kjmol, sigmas=np.array([0.35, 0.35]), start=25, step=500)
    bias2 = BiasMTD(cvs=cvs, K=0.5 * units.kjmol, sigmas=np.array([0.1, 0.1]), start=50, step=250)

    bias = CompositeBias(biases=[bias1, bias2])

    yaffmd = YaffEngine(
        ener=ff,
        bias=bias,
        write_step=200,
        T=T,
        P=None,
        timestep=2.0 * units.femtosecond,
        timecon_thermo=100.0 * units.femtosecond,
        filename="output/aladipep.h5",
    )

    yaffmd.run(int(1e2))


def test_yaff_md_mil53():
    change_fold()

    yaffmd = mil53_yaff()
    yaffmd.run(int(1e2))


def bias_save():
    """save and load bias to disk."""

    change_fold()

    yaffmd = ala_yaff()
    yaffmd.run(int(1e3))

    yaffmd.bias.save('output/bias_test_2.xyz')
    bias = Bias.load('output/bias_test_2.xyz')

    cvs = np.array([0.0, 0.0])

    [b, db] = yaffmd.bias.compute(cvs=cvs, diff=True)
    [b2, db2] = bias.compute(cvs=cvs, diff=True)

    assert pytest.approx(b) == b2
    assert pytest.approx(db[0]) == db2[0]


@pytest.mark.skip(reason="path+files not ready")
def test_yaff_ase():
    change_fold()
    yaffmd = todo_ASE_yaff()
    yaffmd.run(1000)


if __name__ == '__main__':

    test_yaff_save_load_func()
    test_combine_bias()
    test_yaff_md_mil53()
    bias_save()
