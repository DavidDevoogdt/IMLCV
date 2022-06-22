import os
from cmath import nan
from functools import partial

import numpy as np
from IMLCV.base.bias import GridBias
from IMLCV.base.CV import CV, CombineCV, CVUtils
from IMLCV.base.CVDiscovery import CVDiscovery
from IMLCV.base.MdEngine import YaffEngine
from IMLCV.base.metric import Metric, hyperTorus
from IMLCV.base.Observable import Observable
from IMLCV.base.rounds import RoundsMd
from IMLCV.launch.parsl_conf.config import config
from IMLCV.scheme import Scheme
from molmod import units
from molmod.units import kelvin, kjmol
from yaff.test.common import get_alaninedipeptide_amber99ff

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


def test_ala_dipep_FES(name='ala2'):

    if os.path.isfile(f'output/{name}/rounds'):
        # if input("recalculate?").strip().lower() != 'true':
        #     return

        import shutil
        shutil.rmtree(f'output/{name}')

    T = 300 * units.kelvin

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
                    folder=f'output/{name}',
                    write_step=30)

    scheme.round(steps=2e4, rnds=10, n=4)


def test_ala_dipep_FES_non_per():

    phi = partial(CVUtils.dihedral, numbers=[4, 6, 8, 14])
    psi = partial(CVUtils.dihedral, numbers=[6, 8, 14, 16])

    d = np.sqrt(2)*np.pi*1.05
    cvs = CV(CVUtils.rotate(np.pi/4, phi, psi), n=2, metric=Metric(periodicities=[False, False], bounding_box=[
             [-d, d], [-d, d]]))

    a = True
    if a:

        if os.path.isfile('output/ala_np/rounds'):
            import shutil
            shutil.rmtree('output/ala_np')

        T = 300 * kelvin

        s = Scheme(cvd=CVDiscovery(),
                   cvs=cvs,
                   Engine=YaffEngine,
                   ener=get_alaninedipeptide_amber99ff,
                   T=T,
                   timestep=2.0 * units.femtosecond,
                   timecon_thermo=100.0 * units.femtosecond,
                   folder='output/ala_np',
                   write_step=10)
    else:
        s = Scheme.from_rounds(cvd=CVDiscovery(), folder="output/ala_np")

    s.round(steps=1e3, rnds=10, update_metric=True)


def test_cv_discovery():

    assert os.path.isfile('output/ala/rounds')

    rounds = RoundsMd.load('output/ala')

    rounds2 = rounds.unbias_rounds(calc=False)
    obs = Observable(rounds2, rounds.get_bias().cvs)
    bias = obs.fes_bias(plot=True)
    print(bias)


def test_grid_bias():

    cvs = CV(f=lambda x: nan,  n=2, metric=Metric(
        periodicities=[False, False], bounding_box=[[0, 10], [-5, 5]]))

    a = np.linspace(0, 10, endpoint=True)
    b = np.linspace(-5, 5, endpoint=True)

    a1 = (a[1:]+a[:-1])/2
    b1 = (b[1:]+b[:-1])/2

    x, y = np.meshgrid(a1, b1, indexing='ij')
    x, y = x*kjmol, y*kjmol

    gb = GridBias(cvs=cvs,  vals=y)
    gb.plot('test', vmin=None, vmax=None)

    print('a')


if __name__ == "__main__":
    config(cluster='doduo', max_blocks=20)

    test_ala_dipep_FES()
