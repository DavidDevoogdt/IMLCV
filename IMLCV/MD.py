# this is a helper function to perform md simulations. Executed by parsl on HPC infrastructure, but

import argparse
import os
import sys

import dill

import IMLCV
from IMLCV.base.MdEngine import MDEngine


def do_MD(md: MDEngine, steps: int):
    md.run(steps=args.steps)
    d = md.get_trajectory()
    return d


if __name__ == "__main__":
    # this is used as part of the rounds run_par method and shouldn't be called manually
    parser = argparse.ArgumentParser(
        description='Perform an (enhanced) biased MD')
    parser.add_argument('--MDEngine', type=str,
                        help='path to MD engine')
    parser.add_argument('--bias', type=str,
                        help='path to bias')
    parser.add_argument('--steps', type=int,
                        help='number of MD steps to perform')
    parser.add_argument('--temp_traj', type=str,
                        help='file to save temporary traject')
    parser.add_argument('--outfile', type=str,
                        help='file to write the params to')

    args = parser.parse_args()

    print(f"dir {os.path.curdir} args {sys.argv}")
        
    ##
    md = MDEngine.load(args.MDEngine, filename=args.temp_traj)
    d = do_MD(md, args.steps)

    with open(args.outfile, 'wb') as f:
        dill.dump(d, f)
