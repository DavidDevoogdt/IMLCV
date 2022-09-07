import os
import shutil
from importlib import import_module

import numpy as np
from keras.api._v2 import keras as KerasAPI

from IMLCV.base.bias import NoneBias
from IMLCV.base.CV import CV, dihedral
from IMLCV.base.CVDiscovery import CVDiscovery, TranformerAutoEncoder
from IMLCV.base.MdEngine import YaffEngine
from IMLCV.base.metric import Metric
from IMLCV.launch.parsl_conf.config import config
from IMLCV.scheme import Scheme
from molmod import units
from molmod.units import kelvin
from yaff.test.common import get_alaninedipeptide_amber99ff

keras: KerasAPI = import_module("tensorflow.keras")  # type: ignore


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


def test_cv_discovery(name="test_cv_disc_14", recalc=True):
    # make copy and restore orig

    config(cluster="doduo", max_blocks=10)

    full_name = f"output/{name}"
    full_name_orig = f"output/{name}_orig"
    pe = os.path.exists(full_name)
    pe_orig = os.path.exists(full_name_orig)

    if recalc or (not pe and not pe_orig):
        if pe:
            shutil.rmtree(full_name)
        if pe_orig:
            shutil.rmtree(full_name_orig)
        T = 300 * kelvin

        cv0 = CV(
            f=(dihedral(numbers=[4, 6, 8, 14]) + dihedral(numbers=[6, 8, 14, 16])),
            metric=Metric(
                periodicities=[True, True],
                bounding_box=[[-np.pi, np.pi], [-np.pi, np.pi]],
            ),
        )

        scheme0: Scheme = Scheme(
            cvd=None,
            cvs=cv0,
            Engine=YaffEngine,
            ener=get_alaninedipeptide_amber99ff,
            T=T,
            timestep=2.0 * units.femtosecond,
            timecon_thermo=100.0 * units.femtosecond,
            folder=full_name,
            write_step=1,
        )

        scheme0.round(rnds=3, steps=1e4, n=4)

        scheme0.rounds.run(NoneBias(scheme0.rounds.get_bias().cvs), steps=1e5)
        scheme0.rounds.save()

        del scheme0  # close roundsobject

    cleancopy(full_name)

    cvd = CVDiscovery(
        transformer=TranformerAutoEncoder(
            outdim=3,
            # periodicity=[True, True],
            # periodicity=[False, False, False],
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
        samples=1e3,
        n_neighbors=60,
        min_dist=0.8,
        nunits=200,
        nlayers=4,
        # metric=None,
        metric="l2",
        densmap=True,
        parametric_reconstruction=True,
        parametric_reconstruction_loss_fcn=keras.losses.MSE,
        # random_state=np.random.randint(0, 1000),
        decoder=True,
    )

    # scheme0.round(rnds=4, steps=1e4, n=3)


if __name__ == "__main__":

    test_cv_discovery()
