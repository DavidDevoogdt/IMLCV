import argparse

from configs.bash_app_python import bash_app_python
from configs.config_general import ROOT_DIR, config


@bash_app_python(executors=["default"])
def f():
    from configs.config_general import ROOT_DIR, config
    from IMLCV.base.Observable import ThermoLIB
    from IMLCV.base.rounds import Rounds

    config()
    rounds = Rounds(folder=folder, new_folder=False)

    folder = ROOT_DIR / "IMLCV" / "examples" / "output" / "CsPbI3_plot"
    rounds = Rounds(folder=folder, copy=True)

    for i in range(1, rounds.round):
        obs = ThermoLIB(rounds, rnd=i)
        fesBias = obs.fes_bias(plot=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CLI interface to set simulation params",
        epilog="Question? ask David.Devoogdt@ugent.be",
    )
    parser.add_argument("-f", "--folder", type=str)

    args = parser.parse_args()

    folder = ROOT_DIR / "IMLCV" / "examples" / "output" / args.folder

    config()

    f(
        stdout=str(folder / "replot.stdout"),
        stderr=str(folder / "replot.stderr"),
    ).result()
