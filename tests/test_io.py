import pytest

from IMLCV.base.MdEngine import MDEngine

# from pytest import tm


@pytest.fixture(scope="session")
def io_dir(tmp_path_factory):
    return tmp_path_factory.getbasetemp()


def test_yaff_save_load_func(io_dir):
    from IMLCV.examples.example_systems import alanine_dipeptide_yaff

    yaffmd = alanine_dipeptide_yaff()

    yaffmd.run(int(761))

    yaffmd.save(io_dir / "yaff_save.d")
    yeet = MDEngine.load(io_dir / "yaff_save.d")

    sp1 = yaffmd.sp
    sp2 = yeet.sp

    assert pytest.approx(sp1.coordinates) == sp2.coordinates
    assert pytest.approx(sp1.cell) == sp2.cell
    assert (
        pytest.approx(yaffmd.energy.compute_from_system_params(sp1).energy, abs=1e-6)
        == yeet.energy.compute_from_system_params(sp2).energy
    )
