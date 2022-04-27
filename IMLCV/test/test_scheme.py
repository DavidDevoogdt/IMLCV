from IMLCV.base.CVDiscovery import CVDiscovery
from IMLCV.base.MdEngine import YaffEngine
from IMLCV.base.rounds import RoundsMd
from IMLCV.scheme import Scheme
from IMLCV.base.CV import CV, CVUtils, CombineCV, Metric, hyperTorus
from IMLCV.base.bias import BiasF, BiasMTD, NoneBias
from IMLCV.base.Observable import Observable

from yaff.log import log
import os

from yaff.test.common import get_alaninedipeptide_amber99ff

log.set_level(log.medium)
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import numpy as np
from molmod import units


def test_ala_dipep_FES():

    if os.path.isfile('output/ala/rounds'):
        if input("recalculate?").strip().lower() != 'true':
            return

        import shutil
        shutil.rmtree('output/ala')

    T = 600 * units.kelvin

    cvs = CombineCV([
        CV(CVUtils.dihedral, numbers=[4, 6, 8, 14], metric=hyperTorus(1)),
        CV(CVUtils.dihedral, numbers=[6, 8, 14, 16], metric=hyperTorus(1)),
    ])

    scheme = Scheme(cvd=CVDiscovery(),
                    cvs=cvs,
                    Engine=YaffEngine,
                    ener=get_alaninedipeptide_amber99ff,
                    T=T,
                    timestep=2.0 * units.femtosecond,
                    timecon_thermo=100.0 * units.femtosecond,
                    folder='output/ala',
                    write_step=20)

    scheme.round(steps=1e4, rnds=4)


def test_ala_dipep_FES_non_per():

    T = 600 * units.kelvin

    #approx boundaries
    cvs = CombineCV([
        CV(CVUtils.dihedral, numbers=[4, 6, 8, 14], metric=Metric(periodicities=[False], boundaries=[-4, 4])),
        CV(CVUtils.dihedral, numbers=[6, 8, 14, 16], metric=Metric(periodicities=[False], boundaries=[-4, 4])),
    ])

    scheme = Scheme(cvd=CVDiscovery(),
                    cvs=cvs,
                    Engine=YaffEngine,
                    ener=get_alaninedipeptide_amber99ff,
                    T=T,
                    timestep=2.0 * units.femtosecond,
                    timecon_thermo=100.0 * units.femtosecond,
                    folder='output/ala_np',
                    write_step=20)

    scheme.round(steps=1e4, rnds=4)


def test_cv_discovery():

    assert os.path.isfile('output/ala/rounds')

    rounds = RoundsMd.load('output/ala')

    rounds2 = rounds.unbias_rounds(calc=False)
    obs = Observable(rounds2, rounds.get_bias().cvs)
    bias = obs.fes_Bias(plot=True)


if __name__ == "__main__":

    # s = Scheme.from_rounds(
    #     cvd=CVDiscovery(),
    #     folder='output/ala/',
    # )
    # s._FESBias(plot=False)

    # test_ala_dipep_FES()
    test_ala_dipep_FES_non_per()
    # test_cv_discovery()
