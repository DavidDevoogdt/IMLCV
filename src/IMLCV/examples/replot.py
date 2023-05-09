from __future__ import annotations

import argparse

from IMLCV.configs.bash_app_python import bash_app_python
from IMLCV.configs.config_general import config
from IMLCV.configs.config_general import ROOT_DIR


@bash_app_python(executors=["default"])
def f(args):
    from IMLCV.configs.config_general import config
    from IMLCV.base.rounds import Rounds

    config()
    rounds = Rounds(folder=args.folder, new_folder=False)

    # folder = ROOT_DIR / "IMLCV" / "examples" / "output" / "CsPbI3_plot"
    # rounds = Rounds(folder=  folder, copy=True)

    rounds.write_xyz(r=args.round, num=5, repeat=args.repeat)

    # for i in range(1, rounds.round):
    #     obs = ThermoLIB(rounds, rnd=i)
    #     fesBias = obs.fes_bias(plot=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CLI interface to set simulation params",
        epilog="Question? ask David.Devoogdt@ugent.be",
    )
    parser.add_argument("-f", "--folder", type=str)
    parser.add_argument("-r", "--round", type=int, default=None)
    parser.add_argument("-rep", "--repeat", type=int, default=None)

    args = parser.parse_args()

    folder = ROOT_DIR / "IMLCV" / "examples" / "output" / args.folder
    args.folder = folder

    config()

    f(
        args,
        stdout=str(folder / "replot.stdout"),
        stderr=str(folder / "replot.stderr"),
        execution_folder=folder,
    ).result()
