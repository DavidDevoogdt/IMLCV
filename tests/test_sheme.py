import zipfile
from pathlib import Path

import pytest
from IMLCV.base.rounds import Rounds
from IMLCV.configs.config_general import ROOT_DIR
from IMLCV.examples.example_systems import CsPbI3
from IMLCV.scheme import Scheme
from molmod.units import kjmol


@pytest.mark.skip(reason="run on HPC")
def test_perov(tmpdir, config_test, steps=500, recalc=False):
    scheme = Scheme(folder=tmpdir, Engine=CsPbI3())
    scheme.inner_loop(K=1 * kjmol, n=2, init=100, steps=steps)


# @pytest.mark.skip(reason="run on HPC")
@pytest.mark.skip(reason="file_outdated")
def test_ala(tmpdir, config_test, steps=100):
    assert (p := ROOT_DIR / "data" / "alanine_dipeptide.zip").exists()
    folder = Path(tmpdir) / "alanine_dipeptide"

    with zipfile.ZipFile(p, "r") as zip_ref:
        zip_ref.extractall(folder)

    rnds = Rounds(folder=folder, new_folder=False)
    scheme0 = Scheme.from_rounds(rnds)

    scheme0.FESBias()
    scheme0.rounds.add_round_from_md(scheme0.md)
    scheme0.grid_umbrella(steps=steps, n=2, k=1 * kjmol)

    for r, t in scheme0.rounds.iter(num=1):
        assert t.ti.CV.shape == (steps, 2)


if __name__ == "__main__":
    test_ala("tmp")
