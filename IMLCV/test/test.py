from molmod.units import kjmol

from IMLCV import ROOT_DIR
from IMLCV.base.hwsetup import do_conf
from IMLCV.examples.example_systems import alanine_dipeptide_yaff
from IMLCV.scheme import Scheme


def f(recalc=True):

    do_conf()

    path = ROOT_DIR / "IMLCV" / "examples" / "output" / "ala"
    if path.exists() and not recalc:
        scheme = Scheme.from_rounds(path)
    else:
        scheme = Scheme(folder=path, Engine=alanine_dipeptide_yaff())
        scheme.round(K=10 * kjmol / 6.14**2, n=6, steps=2000)


if __name__ == "__main__":
    f()
