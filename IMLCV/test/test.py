from molmod.units import angstrom, kjmol

from IMLCV import ROOT_DIR
from IMLCV.base.CV import CV, sb_descriptor
from IMLCV.base.hwsetup import do_conf
from IMLCV.base.rounds import Rounds
from IMLCV.examples.example_systems import alanine_dipeptide_yaff
from IMLCV.scheme import Scheme
from jax import jit

path = ROOT_DIR / "IMLCV" / "examples" / "output" / "ala"


def f(recalc=False):

    do_conf()

    if path.exists() and not recalc:
        scheme = Scheme.from_rounds(path, copy=True)
    else:
        scheme = Scheme(folder=path, Engine=alanine_dipeptide_yaff())
        scheme.round(K=10 * kjmol / 6.14**2, n=8, steps=2000)


def test_recon():
    rounds = Rounds(folder=path, copy=False)

    desc = None
    cv = []

    for a, b in rounds.iter(num=2):
        if desc is None:
            desc = sb_descriptor(r_cut=3 * angstrom, sti=a.tic, n_max=5, l_max=5)

        out = desc.compute_cv_flow(b.ti.sp[0:100])
        cv += [out]
    cv_tot = CV.stack(*cv)


if __name__ == "__main__":
    test_recon()
