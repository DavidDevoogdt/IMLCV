import os
import shutil
import tempfile
from importlib import import_module
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from keras.api._v2 import keras as KerasAPI
from molmod import units
from molmod.units import kelvin, kjmol

from IMLCV import LOCAL
from IMLCV.base.bias import (
    Bias,
    BiasF,
    BiasMTD,
    CompositeBias,
    EnergyResult,
    HarmonicBias,
    RbfBias,
    YaffEnergy,
)
from IMLCV.base.CV import (
    CollectiveVariable,
    CvFlow,
    Metric,
    SystemParams,
    Volume,
    dihedral,
    rotate_2d,
)
from IMLCV.base.CVDiscovery import CVDiscovery, TranformerAutoEncoder, TranformerUMAP
from IMLCV.base.MdEngine import MDEngine, StaticTrajectoryInfo, YaffEngine
from IMLCV.base.metric import Metric
from IMLCV.external.parsl_conf.config import config
from IMLCV.scheme import Scheme
from IMLCV.test.common import alanine_dipeptide_yaff, ase_yaff, get_FES
from yaff.test.common import get_alaninedipeptide_amber99ff

keras: KerasAPI = import_module("tensorflow.keras")


def do_conf():
    config(cluster="doduo", max_blocks=10)


def test_cv_discovery(
    name="test_cv_disc",
    md=alanine_dipeptide_yaff(),
    recalc=False,
    steps=5e3,
    k=5 * kjmol,
    cvd="AE",
    n=4,
    init=500,
):
    # do_conf()

    if cvd == "AE":
        cvd = CVDiscovery(
            transformer=TranformerAutoEncoder(
                outdim=3,
            )
        )
        kwargs = {}
    elif cvd == "UMAP":

        cvd = CVDiscovery(
            transformer=TranformerUMAP(
                outdim=3,
            )
        )

        kwargs = dict(
            n_neighbors=60,
            min_dist=0.8,
            nunits=200,
            nlayers=4,
            # metric=None,
            metric="l2",
            densmap=True,
            parametric_reconstruction=True,
            parametric_reconstruction_loss_fcn=keras.losses.MSE,
            decoder=True,
        )
    else:
        raise ValueError

    scheme0 = get_FES(
        name=name,
        engine=md,
        cvd=cvd,
        recalc=recalc,
        steps=steps,
        K=k,
        n=n,
        init=init,
    )

    scheme0.update_CV(
        samples=1e3,
        **kwargs,
    )


def test_harmonic():

    cvs = CollectiveVariable(
        f=(dihedral(numbers=[4, 6, 8, 14]) + dihedral(numbers=[6, 8, 14, 16])),
        metric=Metric(
            periodicities=[True, True], bounding_box=[[0, 2 * np.pi], [0, 2 * np.pi]]
        ),
    )

    bias = HarmonicBias(cvs, q0=np.array([np.pi, -np.pi]), k=1.0)

    x = np.random.rand(2)

    a1, _ = bias.compute_from_cv(np.array([np.pi, np.pi]) + x)
    a2, _ = bias.compute_from_cv(np.array([-np.pi, -np.pi] + x))
    a3, _ = bias.compute_from_cv(np.array([np.pi, -np.pi]) + x)
    a4, _ = bias.compute_from_cv(np.array([-np.pi, np.pi] + x))
    a5, _ = bias.compute_from_cv(np.array([np.pi, np.pi]) + x.T)

    assert pytest.approx(a1, abs=1e-5) == a2
    assert pytest.approx(a1, abs=1e-5) == a3
    assert pytest.approx(a1, abs=1e-5) == a4
    assert pytest.approx(a1, abs=1e-5) == a5


def test_virial():
    # virial for volume based CV is V*I(3)

    metric = Metric(periodicities=[False])
    cv0 = CollectiveVariable(f=Volume, metric=metric)
    coordinates = np.random.random((10, 3))
    cell = np.random.random((3, 3))
    vir = np.zeros((3, 3))

    def fun(x):
        return x[0]

    bias = BiasF(cvs=cv0, g=fun)

    e_r: EnergyResult = bias.compute_from_system_params(
        SystemParams(coordinates=coordinates, cell=cell), vir=True
    )
    vol = e_r.energy
    vir = e_r.vtens
    assert pytest.approx(vir, abs=1e-7) == vol * np.eye(3)


def test_grid_bias():

    # bounds = [[0, 3], [0, 3]]
    n = [4, 6]

    cv = CollectiveVariable(
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

    bias = RbfBias(cvs=cv, vals=val)

    def c(x, y):
        return bias.compute_from_cv(cvs=np.array([x, y]))[0]

    val2 = np.array([c(x, y) for x, y in zip(xcf, ycf)]).reshape(xc.shape)
    assert np.allclose(val, val2)


def test_yaff_save_load_func(full_name):

    yaffmd = alanine_dipeptide_yaff()

    yaffmd.run(int(761))

    yaffmd.save("output/yaff_save.d")
    yeet = MDEngine.load("output/yaff_save.d")

    sp1 = yaffmd.sp
    sp2 = yeet.sp

    assert pytest.approx(sp1.coordinates) == sp2.coordinates
    assert pytest.approx(sp1.cell) == sp2.cell
    assert (
        pytest.approx(yaffmd.energy.compute_from_system_params(sp1).energy)
        == yeet.energy.compute_from_system_params(sp2).energy
    )

    # yeet.run(100)


def test_combine_bias(full_name):

    T = 300 * kelvin

    cv0 = CollectiveVariable(
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

    stic = StaticTrajectoryInfo(
        T=T,
        timestep=2.0 * units.femtosecond,
        timecon_thermo=100.0 * units.femtosecond,
        write_step=1,
        atomic_numbers=np.array(
            [1, 6, 1, 1, 6, 8, 7, 1, 6, 1, 6, 1, 1, 1, 6, 8, 7, 1, 6, 1, 1, 1],
            dtype=int,
        ),
    )

    mde = YaffEngine(
        energy=YaffEnergy(f=get_alaninedipeptide_amber99ff),
        bias=bias,
        static_trajectory_info=stic,
    )

    mde.run(int(1e2))


def test_bias_save(full_name):
    """save and load bias to disk."""

    yaffmd = alanine_dipeptide_yaff()
    yaffmd.run(int(1e3))

    yaffmd.bias.save("output/bias_test_2.xyz")
    bias = Bias.load("output/bias_test_2.xyz")

    cvs = np.array([0.0, 0.0])

    [b, db] = yaffmd.bias.compute_from_cv(cvs=cvs, diff=True)
    [b2, db2] = bias.compute_from_cv(cvs=cvs, diff=True)

    assert pytest.approx(b) == b2
    assert pytest.approx(db[0]) == db2[0]


def test_ala_dipep_FES(
    name="ala6", find_metric=False, restart=True, max_energy=70 * kjmol
):
    # do_conf()
    if restart:

        if os.path.isfile(f"output/{name}/rounds"):
            import shutil

            shutil.rmtree(f"output/{name}")

        T = 300 * units.kelvin

        if not find_metric:
            cvs = CollectiveVariable(
                f=(dihedral(numbers=[4, 6, 8, 14]) + dihedral(numbers=[6, 8, 14, 16])),
                metric=Metric(
                    periodicities=[True, True],
                    bounding_box=[[-np.pi, np.pi], [-np.pi, np.pi]],
                ),
            )
        else:
            d = np.sqrt(2) * np.pi * 1.05

            cvs = CollectiveVariable(
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


def test_grid_selection(name="point_selection", recalc=False):
    do_conf()

    md = alanine_dipeptide_yaff()
    scheme = get_FES(
        name=name,
        engine=md,
        cvd=None,
        recalc=recalc,
        steps=5000,
    )

    scheme.grid_umbrella(n=5)


def test_copy(name):
    fold = Path("output").resolve()
    old = fold / name
    new = fold / f"{name}_test_copy"

    if new.exists():
        shutil.rmtree(new)

    shutil.copytree(old, new)
    s = Scheme.from_rounds(folder=new)

    # s.grid_umbrella(n=4)
    # s.FESBias()

    s.cvd = CVDiscovery(
        transformer=TranformerAutoEncoder(
            outdim=50,
        )
    )

    s.update_CV(samples=1e5, num_rounds=2)


def test_neigh():
    rng = jax.random.PRNGKey(42)
    key1, key2, rng = jax.random.split(rng, 3)

    n = 10

    sp = SystemParams(
        coordinates=jax.random.uniform(key1, (n, 3)),
        cell=jax.random.uniform(key2, (3, 3)),
    )

    # should convege to 1
    r_cut = 20

    neihg_ij = sp.neighbourghs(r_cut)

    new_shapes = jnp.array([[a.shape[0] for a in b] for b in neihg_ij])

    neigh_calc = jnp.mean(jnp.sum(new_shapes, axis=0))
    neigh_exp = n / jnp.abs(jnp.linalg.det(sp.cell)) * (4 / 3 * jnp.pi * r_cut**3)

    print(f"err neigh density {jnp.abs(neigh_calc / neigh_exp - 1)}")


if __name__ == "__main__":

    if LOCAL:
        md = alanine_dipeptide_yaff
        # md = mil53_yaff
        k = 10 * kjmol / (6.14**2)
        name = "test_cv_disc_ala_restart"
    else:
        md = ase_yaff
        k = 10 * kjmol
        name = "test_cv_disc_perov"
    a = False

    do_conf()

    if a:
        test_virial()
        with tempfile.TemporaryDirectory() as tmp:
            test_yaff_save_load_func(full_name=f"{tmp}/load_save.h5")
            test_combine_bias(full_name=f"{tmp}/combine.h5")
            test_bias_save(full_name=f"{tmp}/bias_save.h5")
        # test_unbiasing()ct (object 'round_0' doesn't
        test_cv_discovery(md=md(), recalc=True)

        test_grid_selection(recalc=True)

    test_cv_discovery(
        name=name,
        md=md(),
        recalc=True,
        k=k,
        steps=5e3,
        n=6,
    )

    scheme0.FESBias(plot=True)
    scheme0.rounds.new_round(scheme0.md)
    scheme0.rounds.save()
    scheme0.round(rnds=5, init=None, steps=1e3, K=k, update_metric=False, n=8)
