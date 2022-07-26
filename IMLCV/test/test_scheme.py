import os
from cmath import nan
from functools import partial

import numpy as np
from IMLCV.base.bias import BiasF, GridBias
from IMLCV.base.CV import CV, CVUtils, cvflow
from IMLCV.base.CVDiscovery import CVDiscovery, TranformerUMAP
from IMLCV.base.MdEngine import YaffEngine
from IMLCV.base.metric import Metric, hyperTorus
from IMLCV.base.Observable import Observable
from IMLCV.base.rounds import RoundsMd
from IMLCV.launch.parsl_conf.config import config
from IMLCV.scheme import Scheme
from molmod import units
from molmod.constants import boltzmann
from molmod.units import kelvin, kjmol
from yaff.test.common import get_alaninedipeptide_amber99ff

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


def test_ala_dipep_FES(name='ala6'):

    if os.path.isfile(f'output/{name}/rounds'):
        # if input("recalculate?").strip().lower() != 'true':
        #     return

        import shutil
        shutil.rmtree(f'output/{name}')

    T = 600 * units.kelvin

    cvs = CV(
        f=[CVUtils.dihedral(numbers=[4, 6, 8, 14]),
           CVUtils.dihedral(numbers=[6, 8, 14, 16])],
        metric=Metric(
            periodicities=[True, True],
            bounding_box=[[-np.pi,  np.pi],
                          [-np.pi, np.pi]])
    )

    scheme = Scheme(cvd=CVDiscovery(transformer=TranformerUMAP),
                    cvs=cvs,
                    Engine=YaffEngine,
                    ener=get_alaninedipeptide_amber99ff,
                    T=T,
                    timestep=2.0 * units.femtosecond,
                    timecon_thermo=100.0 * units.femtosecond,
                    folder=f'output/{name}',
                    write_step=20,
                    # max_energy=70*kjmol,
                    )

    scheme.round(steps=1e4, rnds=10, n=4)


def test_ala_dipep_FES_non_per(name="ala_np", restart=True):

    d = np.sqrt(2)*np.pi*1.05

    cvs = CV(
        f=cvflow(cvs=[CVUtils.dihedral(numbers=[4, 6, 8, 14]),
                      CVUtils.dihedral(numbers=[6, 8, 14, 16])], tranf=CVUtils.rotate(alpha=np.pi/4)),
        metric=Metric(
            periodicities=[False, False],
            bounding_box=[[-d, d],
                          [-d, d]])
    )

    if restart:

        if os.path.isfile(f'output/{name}/rounds'):
            import shutil
            shutil.rmtree(f'output/{name}')

        T = 600 * kelvin

        s = Scheme(cvd=CVDiscovery(),
                   cvs=cvs,
                   Engine=YaffEngine,
                   ener=get_alaninedipeptide_amber99ff,
                   T=T,
                   timestep=2.0 * units.femtosecond,
                   timecon_thermo=100.0 * units.femtosecond,
                   folder=f'output/{name}',
                   write_step=10)
    else:
        s = Scheme.from_rounds(cvd=CVDiscovery(), folder=f"output/{name}")

    s.round(steps=1e4, K=3.0 * T * boltzmann, rnds=10, update_metric=True)


def test_cv_discovery():

    assert os.path.isfile('output/ala6/rounds')

    rounds = RoundsMd.load('output/ala6')

    rounds2 = rounds.unbias_rounds(calc=False)
    obs = Observable(rounds2, rounds.get_bias().cvs)
    bias = obs.fes_bias(plot=True)
    print(bias)


# def test_grid_bias():

#     cvs = CV(f=lambda x: nan,  n=2, metric=Metric(
#         periodicities=[False, False], bounding_box=[[0, 10], [-5, 5]]))

#     a = np.linspace(0, 10, endpoint=True)
#     b = np.linspace(-5, 5, endpoint=True)

#     a1 = (a[1:]+a[:-1])/2
#     b1 = (b[1:]+b[:-1])/2

#     x, y = np.meshgrid(a1, b1, indexing='ij')
#     x, y = x*kjmol, y*kjmol

#     gb = GridBias(cvs=cvs,  vals=y)
#     gb.plot('test', vmin=None, vmax=None)


if __name__ == "__main__":
    config(cluster='doduo', max_blocks=20)

    # test_ala_dipep_FES(name='test_cv_001')
    test_ala_dipep_FES_non_per(name='test_cv_002')
