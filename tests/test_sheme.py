import pytest
from IMLCV.base.rounds import Rounds
from IMLCV.configs.config_general import config
from IMLCV.configs.config_general import ROOT_DIR
from IMLCV.examples.example_systems import alanine_dipeptide_yaff
from IMLCV.examples.example_systems import CsPbI3
from IMLCV.scheme import Scheme
from molmod.units import kjmol


@pytest.mark.skip(reason="run on HPC")
def test_perov(tmpdir, steps=500, recalc=False):
    config()

    scheme = Scheme(folder=tmpdir, Engine=CsPbI3())
    scheme.inner_loop(K=1 * kjmol, n=2, init=100, steps=steps)


# @pytest.mark.skip(reason="run on HPC")
def test_ala(tmpdir, steps=10):
    # tmpdir = Path("tmp")
    folder = tmpdir / "alanine_dipeptide"

    import zipfile

    with zipfile.ZipFile(
        ROOT_DIR / "data" / "alanine_dipeptide.zip",
        "r",
    ) as zip_ref:
        zip_ref.extractall(tmpdir)

    from IMLCV.scheme import Scheme

    rnds = Rounds(folder=folder, new_folder=False)
    scheme0 = Scheme.from_rounds(rnds)

    config()

    scheme0.FESBias()
    scheme0.rounds.add_round_from_md(scheme0.md)
    scheme0.grid_umbrella(steps=steps, n=2, k=1 * kjmol)

    for r, t in scheme0.rounds.iter(num=1):
        assert t.ti.CV.shape == (steps, 2)
