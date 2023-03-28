import argparse
import os

from configs.bash_app_python import bash_app_python
from configs.config_general import ROOT_DIR, config

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="CLI interface to set simulation params",
        epilog="Question? ask David.Devoogdt@ugent.be",
    )

    subparsers = parser.add_subparsers(title="system", dest="system", required=True)
    CsPbI3 = subparsers.add_parser("CsPbI3")

    CsPbI3.add_argument("--unit_cells", nargs="+", type=int)
    CsPbI3.add_argument(
        "--cv", type=str, choices=["cell_vec", "soap_dist"], required=True
    )
    CsPbI3.add_argument("--input_atoms", nargs="+", type=str, default=None)
    CsPbI3.add_argument("--input_atoms", action="store_true")

    ala = subparsers.add_parser("alanine_dipeptide")
    ala.add_argument("--unit_cells", nargs="+", type=int)
    ala.add_argument("--cv", type=str, choices=["backbone_dihedrals", "soap_dist"])
    ala.add_argument("--project", action="store_true")

    group = parser.add_argument_group("simulation")

    group.add_argument("-n", "--n_steps", type=int, default=5000)
    group.add_argument("-ni", "--n_steps_init", type=int, default=100)

    group.add_argument("-r", "--rounds", type=int, default=20)
    group.add_argument("-nu", "--n_umbrellas", type=int, default=8)
    group.add_argument("-spb", "--samples_per_bin", type=int, default=400)
    group.add_argument(
        "-K",
        "--K_umbrellas",
        type=float,
        default=2.0,
        help="force constant of umbrella in [kjmol]. Value is corrected by CV domain and number of umbrellas: K = k/[ (x_0-x_1)/2n  **2",
    )

    group.add_argument("-c", "--cont", action="store_true")

    group = parser.add_argument_group("Parsl")

    group.add_argument("--nodes", default=None, type=int)
    group.add_argument("-mpc", "--memory_per_core", default=None, type=int)
    group.add_argument("-mmpn", "--min_memery_per_node", default=None, type=int)
    group.add_argument(
        "-wt",
        "--walltime",
        default="48:00:00",
        type=str,
        help="walltime of the singlepoint workers",
    )

    group = parser.add_argument_group("General")
    group.add_argument("-f", "--folder", type=str, default=None)
    group.add_argument("-b", "--bootstrap", action="store_true")

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

    # if args.sys=="CsPbI3":

    def app(args):
        print("loading mdoules")

        from molmod.units import kjmol

        from configs.config_general import config
        from IMLCV.examples.example_systems import CsPbI3, alanine_dipeptide_yaff
        from IMLCV.scheme import Scheme

        print("loading parsl config")

        if args.nodes is None:
            if args.system == "ala":
                args.nodes = 1
            else:
                args.nodes = 16

        config(
            singlepoint_nodes=args.nodes,
            walltime=args.walltime,
            memory_per_core=args.memory_per_core,
            min_memery_per_node=args.min_memery_per_node,
            path_internal=args.folder / "parsl_info",
        )

        print("Loading system")

        if args.system == "ala":
            engine = alanine_dipeptide_yaff(cv=args.cv)

        elif args.system == "CsPbI3":
            engine = CsPbI3(
                cv=args.cv, unit_cells=args.unit_cells, input_atoms=args.input_atoms
            )

        if not args.cont:
            scheme = Scheme(folder=args.folder, Engine=engine)
        else:
            scheme = Scheme.from_rounds(folder=args.folder, new_folder=False)
            scheme.FESBias(plot=True, samples_per_bin=args.samples_per_bin)
            scheme.rounds.add_round_from_md(scheme.md)

        print("starting inner loop")

        scheme.inner_loop(
            K=args.K_umbrellas * kjmol,
            n=args.n_umbrellas,
            init=args.n_steps_init,
            steps=args.n_steps,
            samples_per_bin=args.samples_per_bin,
        )

    if args.bootstrap:
        config(bootstrap=True, walltime=args.walltime)
        bash_app_python(executors=["default"], function=app)(
            args=args,
            execution_folder=args.folder,
            stdout="IMLCV.stdout",
            stderr="IMLCV.stderr",
        ).result()
    else:
        os.environ["XLA_FLAGS"] = (
            "--xla_cpu_multi_thread_eigen=false " "intra_op_parallelism_threads=1"
        )

        app(args=args)
