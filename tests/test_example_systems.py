import keras
import pytest
from molmod.units import angstrom, kjmol
from tensorflow import keras

from configs.config_general import ROOT_DIR
from examples.example_systems import alanine_dipeptide_yaff
from IMLCV.base.CVDiscovery import CVDiscovery, TranformerAutoEncoder, TranformerUMAP
from IMLCV.base.rounds import Rounds


def test_ala():
    # config(env="local")

    folder = ROOT_DIR / "IMLCV" / "examples" / "output" / "ala_1d_soap3"

    sys = alanine_dipeptide_yaff(cv="backbone_dihedrals", folder=folder / "LDA")

    sys.run(100)

    assert sys.get_trajectory().sp.shape == (100, 22, 3)


@pytest.mark.skip(reason="not implemented yet")
def test_cv_discovery(
    name="test_cv_disc",
    md=None,
    recalc=False,
    steps=5e3,
    k=5 * kjmol,
    cvd="AE",
    n=4,
    init=500,
    out_dim=3,
):
    from IMLCV.scheme import Scheme

    # do_conf()
    if md is None:
        from examples.example_systems import alanine_dipeptide_yaff

        md = alanine_dipeptide_yaff()

    tf_kwargs = {
        "outdim": out_dim,
        "descriptor": "sb",
        "descriptor_kwargs": {
            "r_cut": 5 * angstrom,
            "sti": md.static_trajectory_info,
        },
    }

    if cvd == "AE":
        tf = TranformerAutoEncoder(**tf_kwargs)

        kwargs = {}
    elif cvd == "UMAP":
        tf = TranformerUMAP(**tf_kwargs)

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

    cvd = CVDiscovery(
        transformer=tf,
    )

    from pathlib import Path

    if Path(name).exists():
        rnds = Rounds(folder=Path(name))
        scheme0 = Scheme.from_rounds(rnds)

    else:
        scheme0 = Scheme(md=md, folder=name)
        scheme0.inner_loop(rnds=5, K=k, n=n, init=init, steps=steps)

    scheme0.update_CV(
        samples=1e3,
        cvd=cvd,
        **kwargs,
    )
