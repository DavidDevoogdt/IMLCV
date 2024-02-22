# this is a helper function to perform md simulations. Executed by parsl on HPC infrastructure, but
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import cloudpickle
import jsonpickle
from IMLCV import unpickler
from parsl import bash_app
from parsl import File
from parsl import python_app
from parsl.dataflow.futures import AppFuture


# @typeguard.typechecked
def bash_app_python(
    function=None,
    executors="all",
    precommand="",  # command to run before the python command
    pickle_extension="cloudpickle",
    pass_files=False,
):
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
            # merge in and outputs
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

            file_in = rename_num(
                execution_folder / f"{func.__name__}.inp.{pickle_extension}",
                i,
            )
            with open(file_in, "w+"):
                pass

            file_out = rename_num(
                execution_folder / f"{func.__name__}.outp.{pickle_extension}",
                i,
            )

            stdout = rename_num(
                execution_folder / (f"{ func.__name__}.stdout" if stdout is None else Path(stdout)),
                i,
            )

            stderr = rename_num(
                execution_folder / (f"{ func.__name__}.stderr" if stderr is None else Path(stderr)),
                i,
            )

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
                        f.writelines(jsonpickle.encode((func, args, kwargs), indent=1, use_base85=True))
                else:
                    with open(filename, "rb+") as f:
                        cloudpickle.dump((func, args, kwargs), f)

                return f"{precommand} python  -u { os.path.realpath( __file__ ) } --folder { str(execution_folder) } --file_in { file_in  }  --file_out  { file_out  }"

            fun.__name__ = func.__name__

            future: AppFuture = bash_app(function=fun, executors=executors)(
                inputs=[*inp, File(str(file_in)), File(str(execution_folder))],
                outputs=[*outp, File(str(file_out))],
                stdout=str(stdout),
                stderr=str(stderr),
                *args,
                **kwargs,
            )

            def load(inputs=[], outputs=[]):
                filename = Path(inputs[-1].filepath)
                if filename.suffix == ".json":
                    with open(filename) as f:
                        result = jsonpickle.decode(f.read(), context=unpickler)
                else:
                    with open(filename, "rb") as f:
                        result = cloudpickle.load(f)
                import os

                os.remove(inputs[-1].filepath)
                return result

            load.__name__ = f"{func.__name__}_load"

            return python_app(load, executors=["default"])(
                inputs=future.outputs,
                outputs=outp,
            )

        return wrapper

    if function is not None:
        return decorator(function)
    return decorator


if __name__ == "__main__":
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
    parser.add_argument("--folder", type=str, help="working directory")
    args = parser.parse_args()

    cwd = os.getcwd()
    os.chdir(args.folder)

    rank = 0
    use_mpi = False

    try:
        from mpi4py import MPI

        # MPI.pickle.__init__(pickle.dumps, pickle.loads)

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        num_ranks = comm.Get_size()

        if num_ranks > 1:
            use_mpi = True
    except ImportError:
        pass

    if rank == 0:
        print("#" * 20)
        print(f"got input {sys.argv}")
        print(f"task started at {datetime.now():%d/%m/%Y %H:%M:%S }")
        print(f"working in folder {os.getcwd()}")
        if use_mpi:
            print(f"using mpi with {num_ranks} ranks")

        file_in = Path(args.file_in)

        if file_in.suffix == ".json":
            with open(file_in) as f1:
                func, fargs, fkwargs = jsonpickle.decode(f1.read(), context=unpickler)
        else:
            with open(file_in, "rb") as f2:
                func, fargs, fkwargs = cloudpickle.load(f2)

        print("#" * 20)
    else:
        func = None
        fargs = None
        fkwargs = None

    if use_mpi:
        func = comm.bcast(func, root=0)
        fargs = comm.bcast(fargs, root=0)
        fkwargs = comm.bcast(fkwargs, root=0)

    a = func(*fargs, **fkwargs)

    if use_mpi:
        a = comm.gather(a, root=0)

    if rank == 0:
        file_out = Path(args.file_out)

        if file_out.suffix == ".json":
            with open(file_out, "w+") as f3:
                f3.writelines(jsonpickle.encode(a, use_base85=True))
        else:
            with open(file_out, "wb+") as f4:
                cloudpickle.dump(a, f4)

        os.remove(args.file_in)

        print("#" * 20)
        print(f"task finished at {datetime.now():%d/%m/%Y %H:%M:%S}")
        print("#" * 20)
    # change back to starting directory
    os.chdir(cwd)
