from IMLCV.examples.example_systems import alanine_dipeptide_yaff
from configs.config_general import ROOT_DIR, config


def test_ala():
    config(env="local")

    folder = ROOT_DIR / "IMLCV" / "examples" / "output" / "ala_1d_soap3"

    sys = alanine_dipeptide_yaff(cv="backbone_dihedrals", folder=folder / "LDA")

    sys.run(100)
