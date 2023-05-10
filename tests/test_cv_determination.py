from importlib import import_module
from pathlib import Path

import jax.numpy as jnp
import pytest
from IMLCV.base.CVDiscovery import CVDiscovery
from IMLCV.base.rounds import Rounds
from IMLCV.configs.config_general import config
from IMLCV.configs.config_general import ROOT_DIR
from IMLCV.implementations.CvDiscovery import TranformerAutoEncoder
from IMLCV.implementations.tensorflow.CvDiscovery import TranformerUMAP
from molmod.units import angstrom

try:
    pass

    TF_INSTALLED = True
except ImportError:
    TF_INSTALLED = False


@pytest.mark.skipif(not TF_INSTALLED, reason="tensorflow not installed")
@pytest.mark.parametrize(
    "cvd",
    [
        "AE",
    ],
)
def test_cv_discovery(
    tmpdir,
    cvd,
    out_dim=3,
):
    # tmpdir = Path("tmp")
    folder = tmpdir / "alanine_dipeptide"

    chunk_size = 200

    import zipfile

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
        r_cut=r_cut,
        chunk_size=chunk_size,
        **kwargs,
    )

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


if __name__ == "__main__":
    # test_cv_discovery(tmpdir="tmp", cvd="UMAP")
    test_cv_discovery(tmpdir=Path("tmp"), cvd="AE")
