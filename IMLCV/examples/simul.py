import argparse

from configs.bash_app_python import bash_app_python
from configs.config_general import ROOT_DIR, config

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="CLI interface to set simulation params",
        epilog="Question? ask David.Devoogdt@ugent.be",
    )

    group = parser.add_argument_group("simulation")

    parser.add_argument("-s", "--system", type=str, choices=["ala", "CsPbI3"])
    parser.add_argument("-n", "--n_steps", type=int, default=500)
    parser.add_argument("-ni", "--n_steps_init", type=int, default=100)
    parser.add_argument("-r", "--rounds", type=int, default=20)
    parser.add_argument("-nu", "--n_umbrellas", type=int, default=8)
    parser.add_argument("-spb", "--samples_per_bin", type=int, default=400)
    parser.add_argument(
        "-K",
        "--K_umbrellas",
        type=float,
        default=2.0,
        help="force constant of umbrella in [kjmol]. Value is corrected by CV domain and number of umbrellas: K = k/[ (x_0-x_1)/2n  **2",
    )

    parser.add_argument("-c", "--cont", action="store_true")

    group = parser.add_argument_group("Parsl")

    parser.add_argument("--nodes", default=None, type=int)
    parser.add_argument(
        "-wt",
        "--walltime",
        default="48:00:00",
        type=str,
        help="walltime of the singlepoint workers",
    )
    parser.add_argument("-f", "--folder", type=str, default=None)

    args = parser.parse_args()

    if args.folder is None:
        args.folder = ROOT_DIR / "IMLCV" / "examples" / "output" / args.system
    else:
        args.folder = ROOT_DIR / "IMLCV" / "examples" / "output" / args.folder

    if args.cont:
        assert args.folder.exists()
        args.n_steps_init = 0
    else:
        assert (
            not args.folder.exists()
        ), "this path already exists, please provide name with --folder"

    config(bootstrap=True, walltime=args.walltime)

    @bash_app_python(executors=["default"])
    def app(args):

        from molmod.units import kjmol

        from configs.config_general import config
        from IMLCV.examples.example_systems import CsPbI3, alanine_dipeptide_yaff
        from IMLCV.scheme import Scheme

        if args.nodes is None:
            if args.system == "ala":
                args.nodes = 1
            else:
                args.nodes = 16

        if args.system == "ala":
            engine = alanine_dipeptide_yaff()

        elif args.system == "CsPbI3":
            engine = CsPbI3()

        config(singlepoint_nodes=args.nodes, walltime=args.walltime)

        if not args.cont:
            scheme = Scheme(folder=args.folder, Engine=engine)
        else:
            scheme = Scheme.from_rounds(folder=args.folder, new_folder=False)
            scheme.FESBias(plot=True, samples_per_bin=args.samples_per_bin)
            scheme.rounds.add_round_from_md(scheme.md)

        scheme.inner_loop(
            K=args.K_umbrellas * kjmol,
            n=args.n_umbrellas,
            init=args.n_steps_init,
            steps=args.n_steps,
            samples_per_bin=args.samples_per_bin,
        )

    app(
        args=args,
        stdout=str(args.folder / "IMLCV.stdout"),
        stderr=str(args.folder / "IMLCV.stderr"),
    ).result()