from IMLCV.base.CVDiscovery import CVDiscovery
from IMLCV.base.MdEngine import YaffEngine
from IMLCV.base.rounds import Rounds
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

    startround = 0
    rnds = 10

    if startround == 0:

        T = 600 * units.kelvin

        cvs = CombineCV([
            CV(CVUtils.dihedral, numbers=[4, 6, 8, 14], periodicity=[-np.pi, np.pi]),
            CV(CVUtils.dihedral, numbers=[6, 8, 14, 16], periodicity=[-np.pi, np.pi]),
        ])

        scheme = Scheme(cvd=CVDiscovery(),
                        cvs=cvs,
                        Engine=YaffEngine,
                        ener=get_alaninedipeptide_amber99ff,
                        T=T,
                        timestep=2.0 * units.femtosecond,
                        timecon_thermo=100.0 * units.femtosecond,
                        folder='output/ala_B')
    else:
        scheme = Scheme.from_rounds(cvd=CVDiscovery(), folder=f'output/ala_B', round=startround)

    for i in range(startround, rnds + startround):
        #create common bias
        if i != startround:
            scheme._FESBias()
        scheme._MTDBias(steps=1e4)
        scheme.rounds.new_round(scheme.md)

        scheme._grid_umbrella(steps=1e4)
        scheme.rounds.save()

    scheme._FESBias()
    scheme.rounds.save()


if __name__ == "__main__":
    test_ala_dipep()
