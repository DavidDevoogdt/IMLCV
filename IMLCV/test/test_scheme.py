from IMLCV.base.CVDiscovery import CVDiscovery
from IMLCV.base.CV import CV, CVUtils, CombineCV
from IMLCV.base.MdEngine import YaffEngine, BiasMTD
from IMLCV.scheme import Scheme

from yaff.test.common import get_alaninedipeptide_amber99ff
from yaff.log import log
import numpy as np
from molmod import units
import ase.io
import ase.units
import os

log.set_level(log.medium)
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


def test_ala_dipep():
    T = 600 * units.kelvin
    ff = get_alaninedipeptide_amber99ff()

    cvs = CombineCV([
        CV(CVUtils.dihedral, numbers=[4, 6, 8, 14], periodicity=2.0 * np.pi),
        CV(CVUtils.dihedral, numbers=[6, 8, 14, 16], periodicity=2.0 * np.pi),
    ])

    bias = BiasMTD(cvs=cvs, K=1.2 * units.kjmol, sigmas=np.array([0.35, 0.35]), start=50, step=50)
    # bias = YaffBiasNone(cvs=None)

    yaffmd = YaffEngine(
        ff=ff,
        bias=bias,
        write_step=5,
        T=T,
        P=None,
        timestep=2.0 * units.femtosecond,
        timecon_thermo=100.0 * units.femtosecond,
        filename="output/aladipepscheme.h5",
    )

    cvd = CVDiscovery()

    scheme = Scheme(md=yaffmd, cvs=cvs, cvd=cvd)
    scheme.run(2, 1e3, 1e3)


if __name__ == "__main__":
    test_ala_dipep()
