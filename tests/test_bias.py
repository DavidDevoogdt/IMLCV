from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from IMLCV.base.bias import Bias
from IMLCV.base.bias import BiasF
from IMLCV.base.bias import CompositeBias
from IMLCV.base.CV import CollectiveVariable
from IMLCV.base.CV import CV
from IMLCV.base.CV import CvFlow
from IMLCV.base.CV import CvMetric
from IMLCV.base.CV import CvTrans
from IMLCV.base.CV import SystemParams
from IMLCV.base.rounds import Rounds
from IMLCV.configs.config_general import config
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
        f=(dihedral(numbers=[4, 6, 8, 14]) + dihedral(numbers=[6, 8, 14, 16])),
        metric=CvMetric(
            periodicities=[True, True],
            bounding_box=[[0, 2 * np.pi], [0, 2 * np.pi]],
        ),
    )

    bias = HarmonicBias(cvs, q0=CV(jnp.array([np.pi, -np.pi])), k=1.0)

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

    metric = CvMetric(periodicities=[False])
    cv0 = CollectiveVariable(f=Volume, metric=metric)
    coordinates = np.random.random((10, 3))
    cell = np.random.random((3, 3))
    vir = np.zeros((3, 3))

    def fun(x):
        return x.cv[0]

    bias = BiasF(cvs=cv0, g=fun)

    _, e_r = bias.compute_from_system_params(
        SystemParams(coordinates=coordinates, cell=cell),
        vir=True,
    )
    vol = e_r.energy
    vir = e_r.vtens
    assert pytest.approx(vir, abs=1e-7) == vol * np.eye(3)


@pytest.mark.parametrize("kernel", ["linear", "thin_plate_spline"])
def test_RBF_bias(kernel):
    # bounds = [[0, 3], [0, 3]]
    n = 5

    cv = CollectiveVariable(
        CvFlow(func=lambda x: x.coordinates),
        CvMetric(
            periodicities=[False, False],
            bounding_box=np.array([[-2, 2], [1, 5]]),
        ),
    )

    bins = cv.metric.grid(n=n)

    @CvTrans.from_cv_function
    def f(cv: CV, _nl, _cond):
        return cv.cv[0] ** 3 + cv.cv[1]

    # reevaluation of thermolib histo
    bin_centers1, bin_centers2 = 0.5 * (bins[0][:-1] + bins[0][1:]), 0.5 * (bins[1][:-1] + bins[1][1:])
    xc, yc = np.meshgrid(bin_centers1, bin_centers2, indexing="ij")
    xcf = np.reshape(xc, (-1))
    ycf = np.reshape(yc, (-1))

    center_cvs = CV(jnp.stack([xcf, ycf], axis=1))

    val, _ = f.compute_cv_trans(center_cvs)

    # with jax.disable_jit():
    bias = RbfBias(cvs=cv, cv=center_cvs, vals=val, kernel=kernel)

    val2, _ = bias.compute_from_cv(center_cvs)
    assert np.allclose(val, val2)


def test_combine_bias():
    from yaff.test.common import get_alaninedipeptide_amber99ff

    from IMLCV.base.MdEngine import StaticMdInfo
    from IMLCV.implementations.MdEngine import YaffEngine

    T = 300 * kelvin

    cv0 = CollectiveVariable(
        f=(dihedral(numbers=[4, 6, 8, 14]) + dihedral(numbers=[6, 8, 14, 16])),
        metric=CvMetric(
            periodicities=[True, True],
            bounding_box=[[-np.pi, np.pi], [-np.pi, np.pi]],
        ),
    )

    bias1 = BiasMTD(
        cvs=cv0,
        K=2.0 * units.kjmol,
        sigmas=np.array([0.35, 0.35]),
        start=25,
        step=500,
    )
    bias2 = BiasMTD(
        cvs=cv0,
        K=0.5 * units.kjmol,
        sigmas=np.array([0.1, 0.1]),
        start=50,
        step=250,
    )

    bias = CompositeBias(biases=[bias1, bias2])

    stic = StaticMdInfo(
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


def test_bias_save(tmpdir):
    """save and load bias to disk."""
    from IMLCV.examples.example_systems import alanine_dipeptide_yaff

    yaffmd = alanine_dipeptide_yaff(
        bias=lambda cv0: BiasMTD(
            cvs=cv0,
            K=2.0 * units.kjmol,
            sigmas=np.array([0.35, 0.35]),
            start=25,
            step=500,
        ),
    )

    yaffmd.run(int(1e2))

    tmpdir = Path(tmpdir)

    yaffmd.bias.save(tmpdir / "bias_test_2.xyz")
    bias = Bias.load(tmpdir / "bias_test_2.xyz")

    from IMLCV.base.CV import CV

    cvs = CV(cv=jnp.array([0.0, 0.0]))

    [b, db] = yaffmd.bias.compute_from_cv(cvs=cvs, diff=True)
    [b2, db2] = bias.compute_from_cv(cvs=cvs, diff=True)

    assert pytest.approx(b) == b2
    assert pytest.approx(db.cv) == db2.cv


@pytest.mark.parametrize("choice", ["gridbias", "rbf"])
def test_FES_bias(tmpdir, choice):
    import zipfile

    folder = tmpdir / "alanine_dipeptide"

    with zipfile.ZipFile(
        ROOT_DIR / "data" / "alanine_dipeptide.zip",
        "r",
    ) as zip_ref:
        zip_ref.extractall(folder)

    from IMLCV.scheme import Scheme

    rnds = Rounds(folder=folder, new_folder=False)
    scheme0 = Scheme.from_rounds(rnds)

    config()

    scheme0.FESBias(
        plot=False,
        choice=choice,
    )

    sp = scheme0.md.sp
    nl = sp.get_neighbour_list(
        scheme0.md.static_trajectory_info.r_cut,
        z_array=scheme0.md.static_trajectory_info.atomic_numbers,
    )

    cv, _ = scheme0.md.bias.compute_from_system_params(sp, nl)
    assert jnp.allclose(cv.cv, jnp.array([-2.85656026, 2.79090329]))
    # assert jnp.allclose(energy_result.energy, er)


if __name__ == "__main__":
    test_bias_save("tmp")
