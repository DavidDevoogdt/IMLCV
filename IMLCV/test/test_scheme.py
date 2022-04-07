from IMLCV.base.CVDiscovery import CVDiscovery
from IMLCV.base.MdEngine import YaffEngine
from IMLCV.scheme import Scheme
from IMLCV.base.CV import CV, CVUtils, CombineCV
from IMLCV.base.bias import BiasF, BiasMTD, NoneBias

from yaff.log import log
import os

from yaff.test.common import get_alaninedipeptide_amber99ff

log.set_level(log.medium)
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import numpy as np
from molmod import units


def test_ala_dipep():
    T = 600 * units.kelvin

    cvs = CombineCV([
        CV(CVUtils.dihedral, numbers=[4, 6, 8, 14], periodicity=[-np.pi, np.pi]),
        CV(CVUtils.dihedral, numbers=[6, 8, 14, 16], periodicity=[-np.pi, np.pi]),
    ])

    load = False

    if load:
        scheme = Scheme.from_rounds(cvd=CVDiscovery(), filename='output/rounds.p')
    else:
        scheme = Scheme(
            cvd=CVDiscovery(),
            cvs=cvs,
            Engine=YaffEngine,
            ener=get_alaninedipeptide_amber99ff,
            T=T,
            timestep=2.0 * units.femtosecond,
            timecon_thermo=100.0 * units.femtosecond,
        )

        scheme._MTDBias(steps=1e2)
        scheme._grid_umbrella(steps=1e5)
        scheme.rounds.save('rounds.p')

    bias = scheme.get_fes()


if __name__ == "__main__":
    test_ala_dipep()
