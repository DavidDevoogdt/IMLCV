# this is a helper function to perform md simulations. Executed by parsl on HPC infrastructure, but

import os
import sys
from datetime import datetime
from pathlib import Path

import jsonpickle
from parsl import AUTO_LOGNAME, File, bash_app, python_app
from parsl.dataflow.futures import AppFuture


# @typeguard.typechecked
def bash_app_python(
    function=None,
    executors=None,
    uses_mpi=False,  # used in jax.distributed.initialize()
    pickle_extension="json",
    pass_files=False,
    auto_log=False,
    profile=False,
    remove_stdout=True,
    remove_stderr=True,
):
    from IMLCV.configs.config_general import PARSL_DICT, REFERENCE_COMMANDS, Executors

    if executors is None:
        executors = Executors.default

    labels, precommand = PARSL_DICT[executors.value]

    def decorator(func):
        def wrapper(
            *args,
            execution_folder=None,
            stdout=None,
            stderr=None,
            inputs=[],
            outputs=[],
            **kwargs,
        ):
            from IMLCV import unpickler

            # merge in and outputsdirector
            inp = [*inputs, *kwargs.pop("inputs", [])]
            outp = [*outputs, *kwargs.pop("outputs", [])]

            if execution_folder is None:
                p = Path.cwd() / func.__name__

                i = 0
                while p.exists():
                    p = Path.cwd() / (f"{func.__name__}_{i:0>3}")
                    i += 1

                execution_folder = p

            if isinstance(execution_folder, str):
                execution_folder = Path(execution_folder)

            execution_folder.mkdir(exist_ok=True)

            def rename_num(name, i):
                stem = name.name.split(".")
                stem[0] = f"{stem[0]}_{i:0>3}"
                return name.parent / ".".join(stem)

            def find_num(name):
                i = 0
                while rename_num(name, i).exists():
                    i += 1
                return i, rename_num(name, i)

            i, lockfile = find_num(execution_folder / f"{func.__name__}.lock")

            with open(lockfile, "w+"):
                pass

            file_in = rename_num(execution_folder / f"{func.__name__}.inp.{pickle_extension}", i)
            with open(file_in, "w+"):
                pass

            file_out = rename_num(execution_folder / f"{func.__name__}.outp.{pickle_extension}", i)

            if not auto_log:
                stdout = str(
                    rename_num(execution_folder / (f"{ func.__name__}.stdout" if stdout is None else Path(stdout)), i)
                )

                stderr = str(
                    rename_num(execution_folder / (f"{ func.__name__}.stderr" if stderr is None else Path(stderr)), i)
                )
            else:
                stdout = AUTO_LOGNAME
                stderr = AUTO_LOGNAME

            if not execution_folder.exists():
                execution_folder.mkdir(exist_ok=True, parents=True)

            def fun(*args, stdout, stderr, inputs, outputs, **kwargs):
                from pathlib import Path

                from parsl import File

                execution_folder = Path(inputs[-1].filepath)

                op = [str(os.path.relpath(i.filepath, execution_folder)) for i in outputs]
                ip = [str(os.path.relpath(o.filepath, execution_folder)) for o in inputs[:-1]]

                file_in = ip[-1]
                file_out = op[-1]
                if pass_files:
                    if len(ip) > 1:
                        kwargs["inputs"] = [File(i) for i in ip[:-1]]
                    if len(op) > 1:
                        kwargs["outputs"] = [File(o) for o in op[:-1]]

                filename = execution_folder / file_in

                if filename.suffix == ".json":
                    with open(filename, "r+") as f:
                        f.writelines(
                            jsonpickle.encode((func, args, kwargs, REFERENCE_COMMANDS), indent=1, use_base85=True)
                        )
                else:
                    import cloudpickle

                    with open(filename, "rb+") as f:
                        cloudpickle.dump((func, args, kwargs, REFERENCE_COMMANDS), f)

                return f"{precommand} python  -u { os.path.realpath( __file__ ) } --folder { str(execution_folder) } --file_in { file_in  }  --file_out  { file_out  }  {'--uses_mpi' if uses_mpi else ''}   {'--profile' if profile else ''} "

            fun.__name__ = func.__name__

            future: AppFuture = bash_app(function=fun, executors=labels)(
                inputs=[*inp, File(str(file_in)), File(str(execution_folder))],
                outputs=[*outp, File(str(file_out))],
                stdout=stdout,
                stderr=stderr,
                *args,
                **kwargs,
            )

            def load(inputs=[], outputs=[]):
                if not auto_log:
                    stdout, stderr, lockfile = inputs[-3].filepath, inputs[-2].filepath, inputs[-1].filepath
                    inputs = inputs[:-3]

                filename = Path(inputs[-1].filepath)
                if filename.suffix == ".json":
                    with open(filename) as f:
                        result = jsonpickle.decode(f.read(), context=unpickler)
                else:
                    import cloudpickle

                    with open(filename, "rb") as f:
                        result = cloudpickle.load(f)
                import os

                os.remove(inputs[-1].filepath)

                if remove_stderr and not auto_log:
                    os.remove(stderr)
                if remove_stdout and not auto_log:
                    os.remove(stdout)

                if remove_stdout and remove_stderr and not auto_log:
                    os.remove(lockfile)

                return result

            load.__name__ = f"{func.__name__}_load"

            load_inp = [*future.outputs]

            if not auto_log:
                load_inp = [*load_inp, File(stdout), File(stderr), File(lockfile)]

            return python_app(load, executors=PARSL_DICT["threadpool"][0])(
                inputs=load_inp,
                outputs=outp,
            )

        return wrapper

    if function is not None:
        return decorator(function)
    return decorator


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Execute python function in a bash app",
    )
    parser.add_argument(
        "--file_in",
        type=str,
        help="path to file f containing pickle.dump((func, args, kwargs), f)",
    )

    parser.add_argument(
        "--file_out",
        type=str,
        help="path to file f containing pickle.dump((func, args, kwargs), f)",
    )

    parser.add_argument(
        "--uses_mpi",
        action="store_true",
        help="Wether or not this is launchded with mpi",
    )

    parser.add_argument(
        "--profile",
        action="store_true",
        help="Wether or not tho profile program",
    )

    parser.add_argument("--folder", type=str, help="working directory")
    args = parser.parse_args()

    cwd = os.getcwd()
    os.chdir(args.folder)

    # import jax

    rank = 0

    # only run program once
    if args.uses_mpi:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        num_ranks = comm.Get_size()

        assert num_ranks > 1, "MPI is not installed"

        print(f"print hello from {rank=}/{num_ranks}")

    import os

    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={len(os.sched_getaffinity(0))}"
    import jax

    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_pmap_no_rank_reduction", False)

    import IMLCV  # noqa: F401

    if rank == 0:
        print("#" * 20)
        import platform

        my_system = platform.uname()
        print(f"got input {sys.argv}")
        print(f"task started at {datetime.now():%d/%m/%Y %H:%M:%S }")
        print(f"working in folder {os.getcwd()}")
        print(f"System: {my_system.system}")
        print(f"Node Name: {my_system.node}")
        print(f"Release: {my_system.release}")
        print(f"Version: {my_system.version}")
        print(f"Machine: {my_system.machine}")
        print(f"Processor: {my_system.processor}")
        print(f"working with {jax.local_device_count()=},   {jax.device_count()=}  {os.sched_getaffinity(0)=}")

        file_in = Path(args.file_in)

        from IMLCV import unpickler

        if file_in.suffix == ".json":
            with open(file_in) as f1:
                func, fargs, fkwargs, ref_com = jsonpickle.decode(f1.read(), context=unpickler)
        else:
            import cloudpickle

            with open(file_in, "rb") as f2:
                func, fargs, fkwargs, ref_com = cloudpickle.load(f2)

        print(f"loaded  {ref_com=}")

        print("#" * 20)
    else:
        func = None
        fargs = None
        fkwargs = None
        ref_com = None

    if args.uses_mpi:
        func = comm.bcast(func, root=0)
        fargs = comm.bcast(fargs, root=0)
        fkwargs = comm.bcast(fkwargs, root=0)
        ref_com = comm.bcast(ref_com, root=0)

    from IMLCV.configs.config_general import REFERENCE_COMMANDS

    REFERENCE_COMMANDS.update(ref_com)

    if args.profile:
        import cProfile
        import pstats

        with cProfile.Profile() as pr:
            a = func(*fargs, **fkwargs)

        ps = pstats.Stats(pr)
        ps.sort_stats(pstats.SortKey.CUMULATIVE).print_stats()

    else:
        a = func(*fargs, **fkwargs)

    if args.uses_mpi:
        a = comm.gather(a, root=0)

    if rank == 0:
        file_out = Path(args.file_out)

        if file_out.suffix == ".json":
            with open(file_out, "w+") as f3:
                f3.writelines(jsonpickle.encode(a, use_base85=True))
        else:
            import cloudpickle

            with open(file_out, "wb+") as f4:
                cloudpickle.dump(a, f4)

        os.remove(args.file_in)

        print("#" * 20)
        print(f"task finished at {datetime.now():%d/%m/%Y %H:%M:%S}")
        print("#" * 20)
    # change back to starting directory
    os.chdir(cwd)
