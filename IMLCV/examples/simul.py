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
        "--cv", type=str, choices=["cell_vec", "soap_dist", "soap_lda"], required=True
    )
    CsPbI3.add_argument("--input_atoms", nargs="+", type=str, default=None)
    CsPbI3.add_argument("--project", action="store_true")
    CsPbI3.add_argument("--lda_steps", type=int, default=500)

    ala = subparsers.add_parser("alanine_dipeptide")
    ala.add_argument(
        "--cv", type=str, choices=["backbone_dihedrals", "soap_dist", "soap_lda"]
    )
    ala.add_argument("--lda_steps", type=int, default=500)
    ala.add_argument(
        "--kernel_type", choices=["rematch", "average", "none"], default="rematch"
    )
    ala.add_argument("--kernel_LDA", action="store_true")
    ala.add_argument("--arithmic", action="store_true")
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
    group.add_argument(
        "--init_max_grad",
        type=float,
        default=500,
        help="max value of gradient wrt atomic positions of bias during initialisation,ink Kjmol/Ang",
    )
    group.add_argument(
        "--max_grad",
        type=float,
        default=500,
        help="max value of gradient wrt atomic positions of bias during initialisation,ink Kjmol/Ang",
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
    group.add_argument(
        "-wtb",
        "--walltime_bootstrap",
        default="72:00:00",
        type=str,
        help="walltime of the singlepoint workers",
    )

    args = parser.parse_args()

    if args.folder is None:
        folder = ROOT_DIR / "IMLCV" / "examples" / "output" / args.system
    else:
        folder = ROOT_DIR / "IMLCV" / "examples" / "output" / args.folder

    if args.cont:
        assert folder.exists()

    else:
        # look for first avaialble folder
        i = 0
        while True:
            p = folder.parent / (f"{folder.name}_{i:0>3}")
            if p.exists():
                i += 1
            else:
                break

        folder = p

    args.folder = folder

    if not args.folder.exists():
        args.folder.mkdir(parents=True)

    with open(folder / "cmd.txt", "a") as f:
        f.write(f"{args}")

    def app(args):
        print("loading mdoules")

        from molmod.units import angstrom, kjmol

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

        if args.system == "alanine_dipeptide":

            engine = alanine_dipeptide_yaff(
                cv=args.cv,
                kernel=args.kernel_LDA,
                harmonic=not args.arithmic,
                folder=args.folder / "LDA",
                kernel_type=args.kernel_type,
                lda_steps=args.lda_steps,
            )

        elif args.system == "CsPbI3":
            engine = CsPbI3(
                cv=args.cv,
                unit_cells=args.unit_cells,
                input_atoms=args.input_atoms,
                lda_steps=args.lda_steps,
                folder=args.folder / "LDA",
            )

        if not args.cont:
            scheme = Scheme(folder=args.folder, Engine=engine)
        else:

            from IMLCV.base.rounds import Rounds

            rnds = Rounds(folder=args.folder, new_folder=False)

            if rnds.round != -1:
                args.n_steps_init = 0
                scheme = Scheme.from_rounds(rounds=rnds)

                scheme.FESBias(plot=True, samples_per_bin=args.samples_per_bin)
                scheme.rounds.add_round_from_md(scheme.md)
            else:
                print(
                    f"there is no round data in {args.folder} to continue form, starting from init"
                )
                rnds.add_round_from_md(engine)
                scheme = Scheme.from_rounds(rounds=rnds)

        print("starting inner loop")

        scheme.inner_loop(
            K=args.K_umbrellas * kjmol,
            n=args.n_umbrellas,
            init=args.n_steps_init,
            steps=args.n_steps,
            samples_per_bin=args.samples_per_bin,
            init_max_grad=args.init_max_grad * kjmol / angstrom,
            max_grad=args.max_grad * kjmol / angstrom,
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
