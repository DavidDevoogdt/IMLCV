from functools import partial
from IMLCV.base.CVDiscovery import CVDiscovery
from IMLCV.base.MdEngine import YaffEngine
from IMLCV.base.rounds import RoundsMd
from IMLCV.scheme import Scheme
from IMLCV.base.CV import CV, CVUtils, CombineCV, Metric, hyperTorus
from IMLCV.base.bias import BiasF, BiasMTD, HarmonicBias, NoneBias
from IMLCV.base.Observable import Observable

from molmod.units import kelvin, kjmol

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

    rerun = False
    phi = partial(CVUtils.dihedral, numbers=[4, 6, 8, 14])
    psi = partial(CVUtils.dihedral, numbers=[6, 8, 14, 16])

    alpha = CVUtils.linear_combination(phi, psi, a=0.5, b=0.5)
    beta = CVUtils.linear_combination(phi, psi, a=0.5, b=-0.5)

    cvs = CombineCV([
        CV(alpha, metric=Metric(periodicities=[False], boundaries=[-3.5, 3.5])),
        CV(beta, metric=Metric(periodicities=[False], boundaries=[-3.5, 3.5])),
    ])

    if rerun == True:

        T = 600 * kelvin

        s = Scheme(cvd=CVDiscovery(),
                   cvs=cvs,
                   Engine=YaffEngine,
                   ener=get_alaninedipeptide_amber99ff,
                   T=T,
                   timestep=2.0 * units.femtosecond,
                   timecon_thermo=100.0 * units.femtosecond,
                   folder='output/ala_np',
                   write_step=20,
                   max_energy=100 * kjmol)

        s.round(steps=1e4, rnds=1, update_metric=True)
    else:
        s = Scheme.from_rounds(
            cvd=CVDiscovery(),
            folder='output/ala_np',
        )

    s.round(steps=1e4, rnds=3)

    # o = Observable(s.rounds)
    # nm = o.new_metric(plot=True)

    # cvs.metric = nm
    # hb = HarmonicBias(cvs, np.array([np.pi, np.pi]), 10 * kjmol)

    # hb.plot("test")


def test_cv_discovery():

    assert os.path.isfile('output/ala/rounds')

    rounds = RoundsMd.load('output/ala')

    rounds2 = rounds.unbias_rounds(calc=False)
    obs = Observable(rounds2, rounds.get_bias().cvs)
    bias = obs.fes_Bias(plot=True)


if __name__ == "__main__":

    # nm.distance(np.array([np.pi / 2, np.pi / 2]), np.array([-np.pi / 2, -np.pi / 2]))

    # s._FESBias(plot=False)

    test_ala_dipep_FES_non_per()

    # test_ala_dipep_FES()
    # test_ala_dipep_FES_non_per()
    # test_cv_discovery()
