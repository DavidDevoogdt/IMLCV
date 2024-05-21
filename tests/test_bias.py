from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from IMLCV.base.bias import Bias
from IMLCV.base.bias import BiasF
from IMLCV.base.bias import CompositeBias
from IMLCV.base.CV import CollectiveVariable, NeighbourListInfo
from IMLCV.base.CV import CV
from IMLCV.base.CV import CvFlow
from IMLCV.base.CV import CvMetric
from IMLCV.base.CV import CvTrans
from IMLCV.base.CV import SystemParams
from IMLCV.base.rounds import Rounds
from IMLCV.configs.config_general import ROOT_DIR
from IMLCV.implementations.bias import BiasMTD
from IMLCV.implementations.bias import HarmonicBias
from IMLCV.implementations.bias import RbfBias
from IMLCV.implementations.CV import dihedral
from IMLCV.implementations.CV import Volume
from IMLCV.implementations.energy import YaffEnergy
from molmod import units
from molmod.units import kelvin

######################################
#              test                  #
######################################


def test_harmonic():
    cvs = CollectiveVariable(
        f=(dihedral(numbers=(4, 6, 8, 14)) + dihedral(numbers=(6, 8, 14, 16))),
        metric=CvMetric.create(
            periodicities=[True, True],
            bounding_box=[[0, 2 * np.pi], [0, 2 * np.pi]],
        ),
    )

    bias = HarmonicBias.create(cvs, q0=CV(jnp.array([np.pi, -np.pi])), k=1.0)

    x = np.random.rand(2)

    a1, _ = bias.compute_from_cv(CV(jnp.array([np.pi, np.pi]) + x))
    a2, _ = bias.compute_from_cv(CV(jnp.array([-np.pi, -np.pi] + x)))
    a3, _ = bias.compute_from_cv(CV(jnp.array([np.pi, -np.pi]) + x))
    a4, _ = bias.compute_from_cv(CV(jnp.array([-np.pi, np.pi] + x)))
    a5, _ = bias.compute_from_cv(CV(jnp.array([np.pi, np.pi]) + x.T))

    assert pytest.approx(a1, abs=1e-5) == a2
    assert pytest.approx(a1, abs=1e-5) == a3
    assert pytest.approx(a1, abs=1e-5) == a4
    assert pytest.approx(a1, abs=1e-5) == a5


def test_virial():
    # virial for volume based CV is V*I(3)

    metric = CvMetric.create(periodicities=[False])
    cv0 = CollectiveVariable(f=Volume, metric=metric)
    coordinates = np.random.random((10, 3))
    cell = np.random.random((3, 3))
    vir = np.zeros((3, 3))

    def fun(x):
        return x.cv[0]

    bias = BiasF.create(cvs=cv0, g=fun)

    _, e_r = bias.compute_from_system_params(
        sp=SystemParams(coordinates=coordinates, cell=cell),
        vir=True,
    )
    vol = e_r.energy
    vir = e_r.vtens
    assert pytest.approx(vir, abs=1e-7) == vol * np.eye(3)


@pytest.mark.parametrize("kernel", ["linear", "thin_plate_spline"])
def test_RBF_bias(kernel):
    # bounds = [[0, 3], [0, 3]]
    n = 5

    def a(sp, nl, _, shmap):
        return CV(cv=sp.coordinates)

    collective_variable = CollectiveVariable(
        f=CvFlow.from_function(f=a),
        metric=CvMetric.create(
            periodicities=[False, False],
            bounding_box=jnp.array([[-2, 2], [1, 5]]),
        ),
    )

    _, _, center_cvs = collective_variable.metric.grid(n=n)

    def f(
        cv: CV,
        _nl,
        _cond,
        shmap,
    ):
        return cv.replace(cv=cv.cv[0] ** 3 + cv.cv[1])

    val, _, _ = CvTrans.from_cv_function(f).compute_cv_trans(center_cvs)

    bias = RbfBias.create(cvs=collective_variable, cv=center_cvs, vals=val.cv, kernel=kernel)

    val2, _ = bias.compute_from_cv(center_cvs)
    assert jnp.allclose(val.cv, val2)


def test_combine_bias():
    from yaff.test.common import get_alaninedipeptide_amber99ff

    from IMLCV.base.MdEngine import StaticMdInfo
    from IMLCV.implementations.MdEngine import YaffEngine

    T = 300 * kelvin

    cv0 = CollectiveVariable(
        f=(dihedral(numbers=(4, 6, 8, 14)) + dihedral(numbers=(6, 8, 14, 16))),
        metric=CvMetric.create(
            periodicities=[True, True],
            bounding_box=[[-np.pi, np.pi], [-np.pi, np.pi]],
        ),
    )

    bias1 = BiasMTD.create(
        cvs=cv0,
        K=2.0 * units.kjmol,
        sigmas=jnp.array([0.35, 0.35]),
        start=25,
        step=500,
    )
    bias2 = BiasMTD.create(
        cvs=cv0,
        K=0.5 * units.kjmol,
        sigmas=jnp.array([0.1, 0.1]),
        start=50,
        step=250,
    )

    bias = CompositeBias.create(biases=[bias1, bias2])

    stic = StaticMdInfo(
        T=T,
        timestep=2.0 * units.femtosecond,
        timecon_thermo=100.0 * units.femtosecond,
        write_step=1,
        atomic_numbers=jnp.array(
            [1, 6, 1, 1, 6, 8, 7, 1, 6, 1, 6, 1, 1, 1, 6, 8, 7, 1, 6, 1, 1, 1],
            dtype=int,
        ),
    )

    mde = YaffEngine.create(
        energy=YaffEnergy(f=get_alaninedipeptide_amber99ff),
        bias=bias,
        static_trajectory_info=stic,
    )

    mde.run(int(1e2))


def test_bias_save(tmpdir):
    """save and load bias to disk."""
    from IMLCV.examples.example_systems import alanine_dipeptide_yaff

    yaffmd = alanine_dipeptide_yaff(
        bias=lambda cv0: BiasMTD.create(
            cvs=cv0,
            K=2.0 * units.kjmol,
            sigmas=jnp.array([0.35, 0.35]),
            start=25,
            step=500,
        ),
    )

    yaffmd.run(int(1e2))

    tmpdir = Path(tmpdir)

    yaffmd.bias.save(tmpdir / "bias_test_2.json")
    bias = Bias.load(tmpdir / "bias_test_2.json")

    from IMLCV.base.CV import CV

    cvs = CV(cv=jnp.array([0.0, 0.0]))

    [b2, db2] = bias.compute_from_cv(cvs=cvs, diff=True)
    [b, db] = yaffmd.bias.compute_from_cv(cvs=cvs, diff=True)

    # print(f"{bias=}")
    # print(f"{yaffmd.bias=}")

    print(f"{db=}\n{db2=}")

    assert pytest.approx(b) == b2
    assert jnp.allclose(db.cv, db2.cv)


# @pytest.mark.skip(reason="file_outdated")
@pytest.mark.parametrize("choice", ["rbf"])
def test_FES_bias(tmpdir, config_test, choice):
    import zipfile

    folder = tmpdir / "alanine_dipeptide"

    print(folder)

    with zipfile.ZipFile(
        ROOT_DIR / "data" / "alanine_dipeptide.zip",
        "r",
    ) as zip_ref:
        zip_ref.extractall(folder)

    from IMLCV.scheme import Scheme

    rnds = Rounds.create(folder=folder, new_folder=False)

    scheme0 = Scheme.from_rounds(rnds)

    sp = scheme0.md.sp

    info = NeighbourListInfo.create(
        r_cut=scheme0.md.static_trajectory_info.r_cut,
        z_array=scheme0.md.static_trajectory_info.atomic_numbers,
    )

    nl = sp.get_neighbour_list(info=info)

    scheme0.FESBias(
        plot=False,
        choice=choice,
        cv_round=0,
    )

    _ = scheme0.md.bias.compute_from_system_params(sp, nl)


def test_reparametrize():
    cvs = CollectiveVariable(
        f=(dihedral(numbers=(4, 6, 8, 14)) + dihedral(numbers=(6, 8, 14, 16))),
        metric=CvMetric.create(
            periodicities=[True, True],
            bounding_box=[[0, 2 * np.pi], [0, 2 * np.pi]],
        ),
    )

    bias = CompositeBias.create(
        biases=[
            HarmonicBias.create(cvs, q0=CV(jnp.array([np.pi, -np.pi])), k=1.0 / 6**2),
            HarmonicBias.create(cvs, q0=CV(jnp.array([0.0, 0.5])), k=1.0 / 6**2),
        ],
    )

    n = 20
    margin = 0.1

    _, cv_grid, _ = bias.collective_variable.metric.grid(n=n, margin=margin)

    new_bias = bias.resample(cv_grid=cv_grid)

    b = bias.compute_from_cv(cv_grid)[0]
    b2 = new_bias.compute_from_cv(cv_grid)[0]

    assert jnp.allclose(b, b2)
