import shutil
from importlib import import_module
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from IMLCV.base.bias import NoneBias
from IMLCV.base.rounds import Rounds
from IMLCV.configs.config_general import ROOT_DIR
from IMLCV.examples.example_systems import alanine_dipeptide_refs
from IMLCV.examples.example_systems import alanine_dipeptide_yaff
from IMLCV.implementations.CV import NoneCV
from IMLCV.implementations.CV import sb_descriptor
from IMLCV.implementations.CvDiscovery import TranformerAutoEncoder
from IMLCV.implementations.CvDiscovery import TransoformerLDA
from IMLCV.implementations.tensorflow.CvDiscovery import TranformerUMAP
from IMLCV.scheme import Scheme
from molmod.units import angstrom
from molmod.units import kjmol


def get_rounds_ala(tmpdir) -> Rounds:
    loops = 2
    steps = 1000
    folder = tmpdir / "alanine_dipeptide"

    if (p := ROOT_DIR / "data" / "alanine_dipeptide.zip").exists():
        import zipfile

        with zipfile.ZipFile(p, "r") as zip_ref:
            zip_ref.extractall(folder)

        rnds = Rounds(folder=folder, new_folder=False)
    else:
        mde = alanine_dipeptide_yaff()

        scheme = Scheme(folder=folder, Engine=mde)
        scheme.inner_loop(
            K=2 * kjmol,
            rnds=loops,
            n=4,
            init=500,
            steps=steps,
            plot=False,
        )

        rnds = scheme.rounds
        shutil.make_archive(p.parent / p.stem, "zip", folder)

    return rnds


def get_LDA_CV_round(tmpdir, lda_steps=1000) -> Rounds:
    folder = tmpdir / "alanine_dipeptide_LDA"

    if (p := ROOT_DIR / "data" / "alanine_dipeptide_LDA.zip").exists():
        import zipfile

        with zipfile.ZipFile(p, "r") as zip_ref:
            zip_ref.extractall(folder)

        rnds = Rounds(folder=folder, new_folder=False)
    else:
        mde = alanine_dipeptide_yaff()
        refs = alanine_dipeptide_refs()

        rnds = Rounds(folder=folder)
        rnds.add_cv_from_cv(cv=NoneCV())
        rnds.add_round_from_md(mde)

        biases = []
        for _ in refs:
            biases.append(NoneBias.create(collective_variable=NoneCV()))

        rnds.run_par(biases=biases, steps=lda_steps, sp0=refs, plot=False)

        for i in range(len(biases)):
            print("finsihing")
            rnds.finish_data(i=i)

        shutil.make_archive(p.parent / p.stem, "zip", folder)

    return rnds


def _cv_discovery_asserts(scheme0: Scheme, out_dim, r_cut, pdf_file: Path):
    b = scheme0.md.bias

    # test single datapoint
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

    # test batch
    dlo = scheme0.rounds.data_loader(
        out=1e2,
        split_data=False,
        num=1,
        stop=0,
        only_finished=False,
        ignore_invalid=True,
        new_r_cut=r_cut,
    )

    cv_recomputed, _ = b.collective_variable.compute_cv(dlo.sp[0], dlo.nl[0], jacobian=False)

    assert jnp.allclose(cv_recomputed.cv, dlo.cv[0].cv, atol=1e-5)

    assert pdf_file.exists()


# @pytest.mark.skipif(not TF_INSTALLED, reason="tensorflow not installed")
@pytest.mark.parametrize(
    "cvd",
    [
        "AE",
        "UMAP",
    ],
)
def test_cv_discovery(
    tmpdir,
    config_test,
    cvd,
    out_dim=2,
):
    # tmpdir = Path("tmp")

    chunk_size = 200

    rnds = get_rounds_ala(tmpdir / "alanine_dipeptide")
    scheme0 = Scheme.from_rounds(rnds)

    r_cut = 4 * angstrom

    descriptor = sb_descriptor(r_cut=r_cut, n_max=2, l_max=2, reshape=True)

    if cvd == "AE":
        kwargs = {
            "num_epochs": 2,
            "batch_size": 50,
        }

        tf = TranformerAutoEncoder(outdim=out_dim, descriptor=descriptor, **kwargs)
    elif cvd == "UMAP":
        from keras.api._v2 import keras as KerasAPI

        keras: KerasAPI = import_module("tensorflow.keras")

        kwargs = dict(
            n_neighbors=20,
            min_dist=0.5,
            nunits=50,
            nlayers=2,
            metric="l2",
            densmap=False,
            parametric_reconstruction=True,
            parametric_reconstruction_loss_fcn=keras.losses.MSE,
            decoder=True,
            n_training_epochs=1,
            batch_size=50,
            n_epochs=5,  # only for test cases
        )

        tf = TranformerUMAP(outdim=out_dim, descriptor=descriptor, **kwargs)
    else:
        raise ValueError

    dlo = scheme0.rounds.data_loader(out=2e3, split_data=True, new_r_cut=r_cut)

    scheme0.update_CV(
        transformer=tf,
        new_r_cut=r_cut,
        dlo=dlo,
        chunk_size=chunk_size,
        plot=True,
    )

    pdf_file = rnds.path(rnds.cv) / "cvdiscovery.pdf"

    _cv_discovery_asserts(scheme0, out_dim, r_cut, pdf_file)


def test_LDA_CV(tmpdir, config_test, out_dim=1, r_cut=3 * angstrom):
    rnds = get_LDA_CV_round(tmpdir)

    scheme0 = Scheme.from_rounds(rnds)

    descriptor = sb_descriptor(r_cut=r_cut, n_max=2, l_max=2, reshape=True)

    tf = TransoformerLDA(
        outdim=out_dim,
        descriptor=descriptor,
        max_iterations=20,
        alpha_rematch=1e-1,
        sort="rematch",
    )

    dlo = scheme0.rounds.data_loader(out=5e2, new_r_cut=r_cut, split_data=True, chunk_size=200)

    scheme0.update_CV(
        transformer=tf,
        dlo=dlo,
        new_r_cut=r_cut,
        chunk_size=200,
        plot=True,
    )

    pdf_file = rnds.path(rnds.cv) / "cvdiscovery.pdf"

    _cv_discovery_asserts(scheme0, out_dim, r_cut, pdf_file)


if __name__ == "__main__":
    shutil.rmtree("tmp", ignore_errors=True)

    from IMLCV.configs.config_general import config

    config(env="local", path_internal="tmp", local_ref_threads=32)

    # (ROOT_DIR / "data" / "alanine_dipeptide.zip").unlink(missing_ok=True)
    # (ROOT_DIR / "data" / "alanine_dipeptide_LDA.zip").unlink(missing_ok=True)

    test_cv_discovery(tmpdir=Path("tmp") / "UMAP", config_test=None, cvd="UMAP")
    test_cv_discovery(tmpdir=Path("tmp") / "AE", config_test=None, cvd="AE")
    test_LDA_CV(tmpdir=Path("tmp"))
