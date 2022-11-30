from molmod.units import angstrom, kjmol

from IMLCV import ROOT_DIR
from IMLCV.base.CV import CV, sb_descriptor
from IMLCV.base.hwsetup import do_conf
from IMLCV.base.rounds import Rounds
from IMLCV.examples.example_systems import alanine_dipeptide_yaff
from IMLCV.scheme import Scheme

path = ROOT_DIR / "IMLCV" / "examples" / "output" / "ala"


def f(recalc=False):

    do_conf()

    if path.exists() and not recalc:
        scheme = Scheme.from_rounds(path, copy=True)
    else:
        scheme = Scheme(folder=path, Engine=alanine_dipeptide_yaff())
        scheme.round(K=10 * kjmol / 6.14**2, n=6, steps=2000)


def test_recon():
    rounds = Rounds(folder=path, copy=False)

    desc = None
    cv = []

    for a, b in rounds.iter(num=2):
        if desc is None:
            desc = sb_descriptor(r_cut=5 * angstrom, sti=a.tic, n_max=5, l_max=5)
        cv += [desc.compute_cv_flow(b.ti.sp)]
    cv_tot = CV.stack(*cv)


if __name__ == "__main__":
    f(recalc=True)
    test_recon()
