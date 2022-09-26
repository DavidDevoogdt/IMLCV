import os
from importlib import import_module

import numpy as np
import pytest
from keras.api._v2 import keras as KerasAPI
from molmod import units
from molmod.units import kelvin, kjmol

from IMLCV.base.bias import Bias, BiasF, BiasMTD, CompositeBias, GridBias, HarmonicBias
from IMLCV.base.CV import CV, CvFlow, Metric, SystemParams, Volume, dihedral, rotate_2d
from IMLCV.base.CVDiscovery import CVDiscovery, TranformerAutoEncoder, TranformerUMAP
from IMLCV.base.MdEngine import MDEngine, YaffEngine
from IMLCV.base.metric import Metric
from IMLCV.external.parsl_conf.config import config
from IMLCV.scheme import Scheme
from IMLCV.test.common import alanine_dipeptide_yaff, ase_yaff, get_FES
from yaff.test.common import get_alaninedipeptide_amber99ff

keras: KerasAPI = import_module("tensorflow.keras")


def do_conf():
    config(cluster="doduo", max_blocks=10)


def test_cv_discovery(name="test_cv_disc", md=None, recalc=False):
    do_conf()
    if md is None or md == "al":
        md = alanine_dipeptide_yaff()
    elif md == "perov":
        md = ase_yaff()
    else:
        raise ValueError("unknown system")

    cvd = CVDiscovery(
        transformer=TranformerAutoEncoder(
            outdim=3,
        )
    )

    scheme0 = get_FES(name=name, engine=md, cvd=cvd, recalc=recalc)

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


def test_harmonic():

    cvs = CV(
        f=(dihedral(numbers=[4, 6, 8, 14]) + dihedral(numbers=[6, 8, 14, 16])),
        metric=Metric(
            periodicities=[True, True], bounding_box=[[0, 2 * np.pi], [0, 2 * np.pi]]
        ),
    )

    bias = HarmonicBias(cvs, q0=np.array([np.pi, -np.pi]), k=1.0)

    x = np.random.rand(2)

    a1, _ = bias.compute(np.array([np.pi, np.pi]) + x)
    a2, _ = bias.compute(np.array([-np.pi, -np.pi] + x))
    a3, _ = bias.compute(np.array([np.pi, -np.pi]) + x)
    a4, _ = bias.compute(np.array([-np.pi, np.pi] + x))
    a5, _ = bias.compute(np.array([np.pi, np.pi]) + x.T)

    assert pytest.approx(a1, abs=1e-5) == a2
    assert pytest.approx(a1, abs=1e-5) == a3
    assert pytest.approx(a1, abs=1e-5) == a4
    assert pytest.approx(a1, abs=1e-5) == a5


def test_virial():
    # virial for volume based CV is V*I(3)

    metric = Metric(periodicities=[False], bounding_box=[0, 4])
    cv0 = CV(f=Volume, metric=metric)
    coordinates = np.random.random((10, 3))
    cell = np.random.random((3, 3))
    vir = np.zeros((3, 3))

    bias = BiasF(cvs=cv0, f=lambda x: x)  # simply take volume as lambda

    vol, _, vir = bias.compute_coor(
        SystemParams(coordinates=coordinates, cell=cell), vir=True
    )
    assert pytest.approx(vir, abs=1e-7) == vol * np.eye(3)


def test_grid_bias():

    # bounds = [[0, 3], [0, 3]]
    n = [4, 6]

    cv = CV(
        CvFlow(func=lambda x: x.coordinates),
        Metric(
            periodicities=[False, False],
            bounding_box=np.array([[-2, 2], [1, 5]]),
        ),
    )

    bins = [
        np.linspace(a, b, ni, endpoint=True, dtype=np.double)
        for ni, (a, b) in zip(n, cv.metric.bounding_box)
    ]

    def f(x, y):
        return x**3 + y

    # reevaluation of thermolib histo
    bin_centers1, bin_centers2 = 0.5 * (bins[0][:-1] + bins[0][1:]), 0.5 * (
        bins[1][:-1] + bins[1][1:]
    )
    xc, yc = np.meshgrid(bin_centers1, bin_centers2, indexing="ij")
    xcf = np.reshape(xc, (-1))
    ycf = np.reshape(yc, (-1))
    val = np.array([f(x, y) for x, y in zip(xcf, ycf)]).reshape(xc.shape)

    bias = GridBias(cvs=cv, vals=val)

    def c(x, y):
        return bias.compute(cvs=np.array([x, y]))[0]

    val2 = np.array([c(x, y) for x, y in zip(xcf, ycf)]).reshape(xc.shape)
    assert np.allclose(val, val2)


def test_yaff_save_load_func(full_name):

    yaffmd = alanine_dipeptide_yaff(full_name=full_name)

    yaffmd.run(int(761))

    yaffmd.save("output/yaff_save.d")
    yeet = MDEngine.load("output/yaff_save.d", filename="output/output2.h5")

    sp1 = yaffmd.sp
    sp2 = yeet.sp

    assert pytest.approx(sp1.coordinates) == sp2.coordinates
    assert pytest.approx(sp1.cell) == sp2.cell
    assert pytest.approx(yaffmd.ener.compute_coor(sp1)) == yeet.energy.compute_coor(sp2)


def test_combine_bias(full_name):

    T = 300 * kelvin

    cv0 = CV(
        f=(dihedral(numbers=[4, 6, 8, 14]) + dihedral(numbers=[6, 8, 14, 16])),
        metric=Metric(
            periodicities=[True, True],
            bounding_box=[[-np.pi, np.pi], [-np.pi, np.pi]],
        ),
    )

    bias1 = BiasMTD(
        cvs=cv0, K=2.0 * units.kjmol, sigmas=np.array([0.35, 0.35]), start=25, step=500
    )
    bias2 = BiasMTD(
        cvs=cv0, K=0.5 * units.kjmol, sigmas=np.array([0.1, 0.1]), start=50, step=250
    )

    bias = CompositeBias(biases=[bias1, bias2])

    mde = YaffEngine(
        ener=get_alaninedipeptide_amber99ff,
        T=T,
        timestep=2.0 * units.femtosecond,
        timecon_thermo=100.0 * units.femtosecond,
        write_step=1,
        bias=bias,
    )

    mde.run(int(1e2))


def test_bias_save(full_name):
    """save and load bias to disk."""

    yaffmd = alanine_dipeptide_yaff(full_name)
    yaffmd.run(int(1e3))

    yaffmd.bias.save("output/bias_test_2.xyz")
    bias = Bias.load("output/bias_test_2.xyz")

    cvs = np.array([0.0, 0.0])

    [b, db] = yaffmd.bias.compute(cvs=cvs, diff=True)
    [b2, db2] = bias.compute(cvs=cvs, diff=True)

    assert pytest.approx(b) == b2
    assert pytest.approx(db[0]) == db2[0]


def test_ala_dipep_FES(
    name="ala6", find_metric=False, restart=True, max_energy=70 * kjmol
):
    do_conf()
    if restart:

        if os.path.isfile(f"output/{name}/rounds"):
            import shutil

            shutil.rmtree(f"output/{name}")

        T = 300 * units.kelvin

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


if __name__ == "__main__":

    # test_virial()
    # with tempfile.TemporaryDirectory() as tmp:
    #     test_yaff_save_load_func(full_name=f"{tmp}/load_save.h5")
    #     test_combine_bias(full_name=f"{tmp}/combine.h5")
    #     test_bias_save(full_name=f"{tmp}/bias_save.h5")
    # test_unbiasing()
    # test_cv_discovery( md=alanine_dipeptide_yaff() ,recalc=True)
    test_cv_discovery(name="test_cv_disc_perov", md="perov", recalc=True)
