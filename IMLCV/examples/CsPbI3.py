from molmod.units import angstrom, kjmol

from configs.config_general import ROOT_DIR
from IMLCV.base.CV import CV, sb_descriptor
from configs.config_general import config
from IMLCV.base.rounds import Rounds
from IMLCV.examples.example_systems import alanine_dipeptide_yaff, CsPbI3
from IMLCV.scheme import Scheme
from jax import jit




def test_perov(path,steps=500,recalc=False):
    config()
    

    if path.exists() and not recalc:
        scheme = Scheme.from_rounds(path, copy=True)
    else:
        scheme = Scheme(folder=path, Engine=CsPbI3())
        scheme.inner_loop(K=1 * kjmol , n=8, init=100, steps=steps)

def test_ala(path,recalc=False, steps=500):
    config(singlepoint_nodes=1)
    

    if path.exists() and not recalc:
        scheme = Scheme.from_rounds(path, copy=True)
    else:
        scheme = Scheme(folder=path, Engine=alanine_dipeptide_yaff())
        scheme.inner_loop(K=5 * kjmol , n=8, init=100, steps=steps)



if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='CLI interface to set simulation params')
    parser.add_argument('s',"system",   nargs='+', choices=['ala','CsPbI3'])
    parser.add_argument('-n','--steps'  , type=int, default=500 )

    args = parser.parse_args()

    
    # if args.system

    print(args)




    # config()
    # path = ROOT_DIR / "IMLCV" / "examples" / "output" / "CsPbI3_500_1kjmol"
    # scheme = Scheme.from_rounds(path, copy=True)
    # scheme.FESBias(plot=True)
    # scheme.rounds.add_round_from_md(scheme.md)
    # scheme.inner_loop(K=1 * kjmol , n=8, init=0, steps=500)


    # path = ROOT_DIR / "IMLCV" / "examples" / "output" / "CsPbI3_500_1kjmol"
    # test_perov(recalc=True,path=path)
    # path = ROOT_DIR / "IMLCV" / "examples" / "output" / "ala_500"
    # test_ala(recalc=True,path=path)
