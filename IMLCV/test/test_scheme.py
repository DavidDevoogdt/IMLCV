import os

import numpy as np

from IMLCV.base.CV import CV, dihedral, rotate_2d
from IMLCV.base.CVDiscovery import CVDiscovery, TranformerUMAP
from IMLCV.base.MdEngine import YaffEngine
from IMLCV.base.metric import Metric
from IMLCV.base.Observable import Observable
from IMLCV.base.rounds import RoundsMd
from IMLCV.launch.parsl_conf.config import config
from IMLCV.scheme import Scheme
from molmod import units
from molmod.units import kjmol
from yaff.test.common import get_alaninedipeptide_amber99ff

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


def test_ala_dipep_FES(
    name="ala6", find_metric=False, restart=True, max_energy=70 * kjmol
):

    if restart:

        if os.path.isfile(f"output/{name}/rounds"):
            import shutil

            shutil.rmtree(f"output/{name}")

        T = 600 * units.kelvin

        if not find_metric:
            cvs = CV(
                f=(dihedral(numbers=[4, 6, 8, 14]) + dihedral(numbers=[6, 8, 14, 16])),
                metric=Metric(
                    periodicities=[True, True],
                    bounding_box=[[-np.pi, np.pi], [-np.pi, np.pi]],
                ),
            )
        else:
            d = np.sqrt(2) * np.pi * 1.05

            cvs = CV(
                f=(dihedral(numbers=[4, 6, 8, 14]) + dihedral(numbers=[6, 8, 14, 16]))
                * rotate_2d(alpha=np.pi / 4)
                * rotate_2d(alpha=np.pi / 8),
                metric=Metric(
                    periodicities=[False, False], bounding_box=[[-d, d], [-d, d]]
                ),
            )

        scheme = Scheme(
            cvd=CVDiscovery(transformer=TranformerUMAP),
            cvs=cvs,
            Engine=YaffEngine,
            ener=get_alaninedipeptide_amber99ff,
            T=T,
            timestep=2.0 * units.femtosecond,
            timecon_thermo=100.0 * units.femtosecond,
            folder=f"output/{name}",
            write_step=20,
            max_energy=max_energy,
        )
    else:
        scheme = Scheme.from_rounds(cvd=CVDiscovery(), folder=f"output/{name}")

    scheme.round(steps=2e4, rnds=10, n=4, update_metric=find_metric)


def test_unbiasing():

    assert os.path.isfile("output/ala6/rounds")

    rounds = RoundsMd.load("output/ala6")

    rounds2 = rounds.unbias_rounds(calc=False)
    obs = Observable(rounds2, rounds.get_bias().cvs)
    bias = obs.fes_bias(plot=True)
    print(bias)


if __name__ == "__main__":
    config(cluster="doduo", max_blocks=20)
    test_ala_dipep_FES(name="test_cv_004", find_metric=True)
