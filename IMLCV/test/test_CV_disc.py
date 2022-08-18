import os
import shutil
from cmath import nan
from functools import partial

import numpy as np
from IMLCV.base.bias import Bias, BiasF, GridBias, NoneBias
from IMLCV.base.CV import CV, CvFlow, Volume, dihedral, rotate_2d
from IMLCV.base.CVDiscovery import CVDiscovery, TranformerUMAP
from IMLCV.base.MdEngine import YaffEngine
from IMLCV.base.metric import Metric, MetricUMAP, hyperTorus
from IMLCV.base.Observable import Observable
from IMLCV.base.rounds import RoundsMd
from IMLCV.launch.parsl_conf.config import config
from IMLCV.scheme import Scheme
from molmod import units
from molmod.constants import boltzmann
from molmod.units import kelvin, kjmol
from yaff.test.common import get_alaninedipeptide_amber99ff

# parsl.load()

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


def cleancopy(base):

    if not os.path.exists(f"{base}_orig"):
        assert os.path.exists(f"{base}"), "folder not found"
        shutil.copytree(f"{base}", f"{base}.orig")

    if os.path.exists(f"{base}"):
        shutil.rmtree(f"{base}")
    shutil.copytree(f"{base}_orig", f"{base}")


def test_cv_discovery(name="test_cv_003", recalc=False):
    # make copy and restore orig

    config(cluster='doduo', max_blocks=15)

    cleancopy(f'output/{name}')

    if recalc:

        T = 600*kelvin

        cv0 = CV(
            f=(
                dihedral(numbers=[4, 6, 8, 14]) +
                dihedral(numbers=[6, 8, 14, 16])
            ),
            metric=Metric(
                periodicities=[True, True],
                bounding_box=[[- np.pi, np.pi],
                              [-np.pi, np.pi]])
        )

        scheme0: Scheme = Scheme(cvd=CVDiscovery(transformer=TranformerUMAP()),
                                 cvs=cv0,
                                 Engine=YaffEngine,
                                 ener=get_alaninedipeptide_amber99ff,
                                 T=T,
                                 timestep=2.0 * units.femtosecond,
                                 timecon_thermo=100.0 * units.femtosecond,
                                 folder=f'output/{name}',
                                 write_step=20,
                                 )

        scheme0.round(rnds=2, steps=1e4, n=3)
    else:
        scheme0 = Scheme.from_rounds(
            cvd=CVDiscovery(transformer=TranformerUMAP()),
            folder=f'output/{name}',
        )

    scheme0.update_CV(
        samples=2e3,
        # parametric_reconstruction=True,
        global_correlation_loss_weight=0.6,
        decoder=False,
    )
    scheme0.round(rnds=2, steps=3e4, n=4)

    # base = f"output/{name}"

    # cleancopy(base=base)

    # rounds = RoundsMd.load(base)

    # cvd = CVDiscovery()
    # newCV = cvd.compute(rounds=rounds)

    # # update the cv
    # mde = rounds.get_engine()
    # mde.bias = NoneBias(cvs=newCV)
    # rounds.new_round(mde)

    # do new umbrella
    # scheme = Scheme.from_rounds(cvd=CVDiscovery( ), folder=f"output/{name}")

    # scheme._grid_umbrella(steps=2e4, n=3)
    # scheme._FESBias()

    # scheme.rounds.new_round(scheme.md)
    # scheme.rounds.save()


if __name__ == "__main__":

    test_cv_discovery()
