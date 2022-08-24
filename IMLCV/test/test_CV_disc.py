import os
import shutil
from importlib import import_module
from math import sqrt

import numpy as np
from IMLCV.base.CV import CV, CvFlow, Volume, dihedral, rotate_2d
from IMLCV.base.CVDiscovery import CVDiscovery, TranformerUMAP
from IMLCV.base.MdEngine import YaffEngine
from IMLCV.base.metric import Metric
from IMLCV.launch.parsl_conf.config import config
from IMLCV.scheme import Scheme
from keras.api._v2 import keras as KerasAPI
from molmod import units
from molmod.constants import boltzmann
from molmod.units import kelvin, kjmol
from yaff.test.common import get_alaninedipeptide_amber99ff

keras: KerasAPI = import_module("tensorflow.keras")
# parsl.load()


abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


def cleancopy(base):

    if not os.path.exists(f"{base}"):
        os.mkdir(base)
        return

    if not os.path.exists(f"{base}_orig"):
        assert os.path.exists(f"{base}"), "folder not found"
        shutil.copytree(f"{base}", f"{base}_orig")

    if os.path.exists(f"{base}"):
        shutil.rmtree(f"{base}")
    shutil.copytree(f"{base}_orig", f"{base}")


def test_cv_discovery(name="test_cv_disc_004", recalc=False):
    # make copy and restore orig

    config(cluster='doduo', max_blocks=10)

    full_name = f'output/{name}'
    pe = os.path.exists(full_name)

    if recalc or not pe:
        if pe:
            shutil.rmtree(full_name)
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

        cvd = CVDiscovery

        scheme0: Scheme = Scheme(cvd=cvd,
                                 cvs=cv0,
                                 Engine=YaffEngine,
                                 ener=get_alaninedipeptide_amber99ff,
                                 T=T,
                                 timestep=2.0 * units.femtosecond,
                                 timecon_thermo=100.0 * units.femtosecond,
                                 folder=full_name,
                                 write_step=10,
                                 )

        scheme0.round(rnds=3, steps=5e3, n=10)

        del scheme0  # close roundsobject

    cleancopy(full_name)

    cvd = CVDiscovery(
        transformer=TranformerUMAP(
            outdim=3,
            periodicity=[False, False, False],
            # bounding_box=np.array([
            #     [0.0, 1.0],
            #     [0.0, 1.0],
            # ]),
        )
    )

    scheme0 = Scheme.from_rounds(
        cvd=cvd,
        folder=full_name,
    )

    scheme0.update_CV(
        samples=5e3,

        n_neighbors=40,
        min_dist=0.6,

        nunits=100,
        nlayers=4,

        metric='l2',
        densmap=True,
        parametric_reconstruction=True,
        parametric_reconstruction_loss_fcn=keras.losses.MSE,


        # global_correlation_loss_weight=0.6,
        decoder=True,
        # run_eagerly=True,
    )
    scheme0.round(rnds=4, steps=1e4, n=5)


if __name__ == "__main__":

    test_cv_discovery()
