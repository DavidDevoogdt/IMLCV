import shutil
import zipfile
from importlib import import_module
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from IMLCV.base.bias import NoneBias
from IMLCV.base.CV import SystemParams
from IMLCV.base.CVDiscovery import CVDiscovery
from IMLCV.base.MdEngine import StaticTrajectoryInfo
from IMLCV.base.rounds import Rounds
from IMLCV.configs.config_general import config
from IMLCV.configs.config_general import ROOT_DIR
from IMLCV.implementations.CV import NoneCV
from IMLCV.implementations.CvDiscovery import TranformerAutoEncoder
from IMLCV.implementations.CvDiscovery import TransoformerLDA
from IMLCV.implementations.energy import YaffEnergy
from IMLCV.implementations.MdEngine import YaffEngine
from IMLCV.implementations.tensorflow.CvDiscovery import TranformerUMAP
from IMLCV.scheme import Scheme
from molmod import units
from molmod.units import angstrom
from molmod.units import kelvin
from yaff.test.common import get_alaninedipeptide_amber99ff

try:
    TF_INSTALLED = True
except ImportError:
    TF_INSTALLED = False


def _cv_discovery_asserts(scheme0: Scheme, out_dim, r_cut):
    b = scheme0.md.bias

    sp = scheme0.md.sp
    nl = scheme0.md.sp.get_neighbour_list(
        r_cut=r_cut,
        z_array=scheme0.md.static_trajectory_info.atomic_numbers,
    )

    cv, dcv = b.collective_variable.compute_cv(sp, nl, jacobian=True)

    assert cv.shape == (out_dim,)
    assert dcv.shape == (out_dim, sp.shape[0], 3)

    assert ~jnp.any(cv.cv == jnp.nan)
    assert ~jnp.any(dcv.cv.coordinates == jnp.nan)
    if sp.cell is not None:
        assert ~jnp.any(dcv.cv.cell == jnp.nan)


@pytest.mark.skipif(not TF_INSTALLED, reason="tensorflow not installed")
@pytest.mark.parametrize("cvd", ["AE", "UMAP"])
def test_cv_discovery(
    tmpdir,
    cvd,
    out_dim=3,
):
    # tmpdir = Path("tmp")
    folder = tmpdir / "alanine_dipeptide"

    chunk_size = 200

    with zipfile.ZipFile(
        ROOT_DIR / "data" / "alanine_dipeptide.zip",
        "r",
    ) as zip_ref:
        zip_ref.extractall(tmpdir)

    from IMLCV.scheme import Scheme

    rnds = Rounds(folder=folder, new_folder=False)
    scheme0 = Scheme.from_rounds(rnds)

    r_cut = 3 * angstrom

    tf_kwargs = {
        "outdim": out_dim,
        "descriptor": "sb",
        "descriptor_kwargs": {
            "r_cut": r_cut,
            "n_max": 2,
            "l_max": 2,
            "reshape": True,
        },
    }

    if cvd == "AE":
        tf = TranformerAutoEncoder(**tf_kwargs)

        kwargs = {"num_epochs": 20}
    elif cvd == "UMAP":
        tf = TranformerUMAP(**tf_kwargs)

        from keras.api._v2 import keras as KerasAPI

        keras: KerasAPI = import_module("tensorflow.keras")

        kwargs = dict(
            n_neighbors=40,
            min_dist=0.8,
            nunits=200,
            nlayers=2,
            metric="l2",
            densmap=True,
            parametric_reconstruction=True,
            parametric_reconstruction_loss_fcn=keras.losses.MSE,
            decoder=True,
            jac=jax.jacrev,  # calltf only supports jacrev, fixed with custom loop batchter  # https://github.com/google/jax/issues/14150
        )
    else:
        raise ValueError

    cvd = CVDiscovery(
        transformer=tf,
    )

    config()

    scheme0.update_CV(
        samples=1e3,
        cvd=cvd,
        new_r_cut=r_cut,
        chunk_size=chunk_size,
        **kwargs,
    )

    _cv_discovery_asserts(scheme0, out_dim, r_cut)


def get_LDA_CV_round(folder, r_cut, lda_steps=1000, T=300 * kelvin) -> Rounds:
    tic = StaticTrajectoryInfo(
        T=T,
        timestep=2.0 * units.femtosecond,
        timecon_thermo=100.0 * units.femtosecond,
        write_step=10,
        atomic_numbers=jnp.array(
            [1, 6, 1, 1, 6, 8, 7, 1, 6, 1, 6, 1, 1, 1, 6, 8, 7, 1, 6, 1, 1, 1],
            dtype=int,
        ),
        screen_log=10,
        equilibration=0 * units.femtosecond,
        r_cut=r_cut,
        # max_grad=max_grad,
    )

    sp0 = SystemParams(
        coordinates=jnp.array(
            [
                [26.77741932, 35.69692667, 0.15117809],
                [26.90970015, 33.69381697, -0.30235618],
                [25.20894663, 32.76785116, 0.43463701],
                [28.66714545, 33.01351556, 0.51022606],
                [26.68293301, 33.05131008, -3.09915086],
                [25.9081453, 34.69537182, -4.55423998],
                [27.26874811, 30.7458442, -3.93063036],
                [28.21361118, 29.61200852, -2.72120563],
                [27.00418645, 30.06554279, -6.55734968],
                [25.19004937, 30.84033051, -7.14316479],
                [28.95060437, 31.21827573, -8.35258951],
                [30.78363872, 30.27341267, -8.25810321],
                [28.23250844, 31.12378943, -10.28011017],
                [29.36634412, 33.20248817, -7.95574702],
                [27.06087824, 27.19315907, -6.82191134],
                [28.38368653, 25.92704256, -5.40461674],
                [26.07822065, 26.26719326, -8.93840461],
                [25.37902198, 27.51441251, -10.22341838],
                [25.6624809, 23.64047394, -9.59980876],
                [25.83255625, 23.43260406, -11.64071298],
                [26.85300836, 22.29876838, -8.59825391],
                [23.73496024, 23.13024788, -9.07068544],
            ],
        ),
        cell=None,
    )

    sp1 = SystemParams(
        coordinates=jnp.array(
            [
                [23.931, 32.690, -5.643],
                [24.239, 31.818, -3.835],
                [22.314, 31.227, -3.153],
                [25.100, 33.275, -2.586],
                [25.835, 29.525, -3.858],
                [27.425, 29.164, -2.258],
                [25.638, 27.991, -5.861],
                [24.292, 28.473, -7.216],
                [27.221, 25.765, -6.438],
                [26.509, 24.957, -8.255],
                [29.991, 26.660, -6.699],
                [30.753, 27.301, -4.872],
                [30.920, 25.078, -7.447],
                [30.233, 28.236, -8.053],
                [26.856, 23.398, -4.858],
                [27.483, 21.402, -5.810],
                [25.732, 23.673, -2.608],
                [25.785, 25.535, -1.850],
                [25.227, 21.564, -0.916],
                [26.860, 20.494, -0.570],
                [24.444, 22.298, 0.859],
                [23.648, 20.454, -1.497],
            ],
        ),
        cell=None,
    )

    refs = sp0 + sp1

    mde = YaffEngine(
        energy=YaffEnergy(f=get_alaninedipeptide_amber99ff),
        static_trajectory_info=tic,
        bias=NoneBias(cvs=NoneCV()),
    )

    pre_round = Rounds(folder=folder)

    pre_round.add_round_from_md(mde)

    biases = []
    for _ in refs:
        biases.append(NoneBias(cvs=NoneCV()))

    pre_round.run_par(biases=biases, steps=lda_steps, sp0=refs)

    return pre_round


def test_LDA_CV(tmpdir, out_dim=3, r_cut=3 * angstrom):
    config()

    folder = tmpdir / "alanine_dipeptide_LDA"

    if (p := ROOT_DIR / "data" / "alanine_dipeptide_LDA.zip").exists():
        import zipfile

        with zipfile.ZipFile(p, "r") as zip_ref:
            zip_ref.extractall(folder)

        rnds = Rounds(folder=folder, new_folder=False)
    else:
        rnds = get_LDA_CV_round(folder=folder, r_cut=r_cut)
        shutil.make_archive(p.parent / p.stem, "zip", folder)

    scheme0 = Scheme.from_rounds(rnds)

    tf_kwargs = {
        "outdim": out_dim,
        "descriptor": "sb",
        "descriptor_kwargs": {
            "r_cut": r_cut,
            "n_max": 2,
            "l_max": 2,
            "reshape": True,
        },
    }

    tf = TransoformerLDA(**tf_kwargs)

    scheme0.update_CV(
        samples=1e3,
        split_data=True,
        cvd=CVDiscovery(transformer=tf),
        new_r_cut=r_cut,
        chunk_size=500,
        max_iterations=10,
        plot=False,
    )

    _cv_discovery_asserts(scheme0, out_dim, r_cut)


if __name__ == "__main__":
    shutil.rmtree("tmp", ignore_errors=True)
    # test_cv_discovery(tmpdir=Path("tmp"), cvd="UMAP")
    test_LDA_CV(tmpdir=Path("tmp"))
