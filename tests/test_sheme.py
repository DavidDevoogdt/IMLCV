import pytest
from IMLCV.configs.config_general import config
from IMLCV.examples.example_systems import alanine_dipeptide_yaff
from IMLCV.examples.example_systems import CsPbI3
from IMLCV.scheme import Scheme
from molmod.units import kjmol


@pytest.mark.skip(reason="run on HPC")
def test_perov(tmpdir, steps=500, recalc=False):
    config()

    # if path.exists() and not recalc:
    #     scheme = Scheme.from_rounds(path, copy=True)
    # else:
    scheme = Scheme(folder=tmpdir, Engine=CsPbI3())
    scheme.inner_loop(K=1 * kjmol, n=2, init=100, steps=steps)


@pytest.mark.skip(reason="run on HPC")
def test_ala(tmpdir, steps=1000):
    config()

    scheme = Scheme(folder=tmpdir, Engine=alanine_dipeptide_yaff())
    scheme.inner_loop(K=5 * kjmol, rnds=1, n=3, init=100, steps=steps)
