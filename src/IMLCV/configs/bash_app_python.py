from __future__ import annotations

import os
import sys
from asyncio import Future
from datetime import datetime
from pathlib import Path
from typing import Callable, TypeVar

import jsonpickle
from parsl import AUTO_LOGNAME, File, bash_app, python_app
from parsl.dataflow.futures import AppFuture
from typing_extensions import ParamSpec

T = TypeVar("T")
P = ParamSpec("P")


def fun(
    stdout,
    stderr,
    precommand,
    inputs: list[File],
    outputs: list[File],
    pass_files_in=0,
    pass_files_out=0,
    uses_mpi=False,
    profile=False,
):
    from pathlib import Path

    execution_folder = Path(inputs[-1].filepath)

    op = [str(os.path.relpath(i.filepath, execution_folder)) for i in outputs]
    ip = [str(os.path.relpath(o.filepath, execution_folder)) for o in inputs[:-1]]

    file_in = ip[-1]
    file_out = op[-1]

    out = f"{precommand} python  -u {os.path.realpath(__file__)} --folder {str(execution_folder)} --file_in {file_in}  --file_out  {file_out} "

    # print(f"{pass_files=}")

    if pass_files_in > 0:
        # if len(ip[:-1]) > 0:
        out += "--inputs "

        for f in ip[:pass_files_in]:
            out += f"{f} "

    if pass_files_out > 0:
        # if len(op[:-1]) > 0:
        out += "--outputs "

        for f in op[:pass_files_out]:
            out += f"{f} "

    if uses_mpi:
        out += "--uses_mpi "

    if profile:
        out += "--profile "

    print(f"{out=}")

    return out


def load(inputs: list[File] = [], outputs: list[File] = [], auto_log=False, remove_stderr=True, remove_stdout=True):
    if not auto_log:
        stdout, stderr, lockfile = inputs[-3].filepath, inputs[-2].filepath, inputs[-1].filepath
        inputs = inputs[:-3]

    filename = Path(inputs[-1].filepath)
    if filename.suffix == ".json":
        from IMLCV import unpickler

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


def bash_app_python(
    function: Callable[P, T],
    executors=None,
    uses_mpi=False,  # used in jax.distributed.initialize()
    pickle_extension="json",
    auto_log=False,
    profile=False,
    remove_stdout=True,
    remove_stderr=True,
    execution_folder: Path | str | None = None,
    stdout: str | Path | None = None,
    stderr: str | Path | None = None,
    inputs: list[str | Path] = [],  # inputs that need te be present but not passed as arguments
    outputs: list[str | Path] = [],  # outputs that need te be present but not passed as arguments
) -> Callable[P, Future[T]]:
    from IMLCV.configs.config_general import PARSL_DICT, REFERENCE_COMMANDS, RESOURCES_DICT, Executors

    if executors is None:
        executors = Executors.default

    labels, precommand = PARSL_DICT[executors.value]

    resources = RESOURCES_DICT[executors.value]

    # def decorator(func):
    def wrapper(
        *args: P.args,
        **kwargs: P.kwargs,
    ):
        # from IMLCV import unpickler

        _files_in_pass: list[str | Path] = kwargs.pop("inputs", [])  # type:ignore
        _files_out_pass: list[str | Path] = kwargs.pop("outputs", [])  # type:ignore

        # merge in and outputs
        inp = [*_files_in_pass, *inputs]  # type:ignore
        outp = [*_files_out_pass, *outputs]  # type: ignore

        if execution_folder is None:
            p = Path.cwd() / function.__name__

            i = 0
            while p.exists():
                p = Path.cwd() / (f"{function.__name__}_{i:0>3}")
                i += 1

            _execution_folder = p
        else:
            _execution_folder = execution_folder

        if isinstance(_execution_folder, str):
            _execution_folder = Path(_execution_folder)

        _execution_folder.mkdir(exist_ok=True, parents=True)

        def rename_num(name: Path, i: int):
            stem = name.name.split(".")
            stem[0] = f"{stem[0]}_{i:0>3}"
            return name.parent / ".".join(stem)

        def find_num(name) -> tuple[int, Path]:
            i = 0
            while rename_num(name, i).exists():
                i += 1
            return i, rename_num(name, i)

        i, lockfile = find_num(_execution_folder / f"{function.__name__}.lock")

        with open(lockfile, "w+"):
            pass

        file_in = rename_num(_execution_folder / f"{function.__name__}.inp.{pickle_extension}", i)
        with open(file_in, "w+"):
            pass

        file_out = rename_num(_execution_folder / f"{function.__name__}.outp.{pickle_extension}", i)

        if not auto_log:
            _stdout = str(
                rename_num(_execution_folder / (f"{function.__name__}.stdout" if stdout is None else Path(stdout)), i)
            )

            _stderr = str(
                rename_num(_execution_folder / (f"{function.__name__}.stderr" if stderr is None else Path(stderr)), i)
            )
        else:
            _stdout = AUTO_LOGNAME
            _stderr = AUTO_LOGNAME

        if not _execution_folder.exists():
            _execution_folder.mkdir(exist_ok=True, parents=True)

        if file_in.suffix == ".json":
            with open(file_in, "r+") as f:
                f.writelines(jsonpickle.encode((function, args, kwargs, REFERENCE_COMMANDS), indent=1, use_base85=True))  # type: ignore
        else:
            import cloudpickle

            with open(file_in, "rb+") as f:
                cloudpickle.dump((function, args, kwargs, REFERENCE_COMMANDS), f)

        def get_file(x: str | Path | File):
            if isinstance(x, Path):
                x = str(x)

            if isinstance(x, str):
                x = File(x)

            return x

        inp = [get_file(x) for x in inp]
        outp = [get_file(x) for x in outp]

        future: AppFuture = bash_app(
            function=fun,
            executors=labels,
        )(
            precommand=precommand,
            profile=profile,
            uses_mpi=uses_mpi,
            # pass_files=pass_files,
            pass_files_in=len(_files_in_pass),
            pass_files_out=len(_files_out_pass),
            inputs=[*inp, File(str(file_in)), File(str(execution_folder))],
            outputs=[*outp, File(str(file_out))],
            stdout=_stdout,
            stderr=_stderr,
        )

        load_inp = [*future.outputs]

        if not auto_log:
            load_inp = [
                *load_inp,
                File(_stdout),  # type: ignore
                File(_stderr),  # type: ignore
                File(lockfile),
            ]

        fut: Future[T] = python_app(load, executors=PARSL_DICT["threadpool"][0])(
            inputs=load_inp,
            outputs=outp,
            remove_stderr=remove_stderr,
            remove_stdout=remove_stdout,
            auto_log=auto_log,
        )

        return fut

    return wrapper


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

    parser.add_argument(
        "--inputs",
        action="extend",
        nargs="+",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--outputs",
        action="extend",
        nargs="+",
        type=str,
        default=None,
    )

    parser.add_argument("--folder", type=str, help="working directory")
    args = parser.parse_args()

    cwd = os.getcwd()
    os.chdir(args.folder)

    # import jax

    rank = 0

    # only run program once
    if args.uses_mpi:
        from mpi4py import MPI  # type: ignore

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        num_ranks = comm.Get_size()

        assert num_ranks > 1, "MPI is not installed"

        print(f"print hello from {rank=}/{num_ranks}")

    import os

    # os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={len(os.sched_getaffinity(0))}"
    import jax

    # jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_enable_x64", True)
    # jax.config.update("jax_pmap_no_rank_reduction", False)

    import torch

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
        print(f"working with {jax.local_devices()=},   {jax.device_count()=}  {os.sched_getaffinity(0)=}")
        print(f"{torch.get_num_threads()=}")
        print(f"{torch.cuda.is_available()=}")

        file_in = Path(args.file_in)

        from IMLCV import unpickler

        if file_in.suffix == ".json":
            with open(file_in) as f1:
                func, fargs, fkwargs, ref_com = jsonpickle.decode(f1.read(), context=unpickler)  # type: ignore
        else:
            import cloudpickle

            with open(file_in, "rb") as f2:
                func, fargs, fkwargs, ref_com = cloudpickle.load(f2)

        if args.inputs is not None:
            inputs = [Path(a) for a in args.inputs]

            fkwargs["inputs"] = inputs  # type: ignore

        if args.outputs is not None:
            outputs = [Path(a) for a in args.outputs]

            fkwargs["outputs"] = outputs  # type: ignore

        print(f"loaded  {ref_com=}  ")

        # print("#" * 20)
        # print("ENVIRONMENT VARIABLES:")
        # for k in sorted(os.environ):
        #     print(f"{k}={os.environ[k]}")
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

    REFERENCE_COMMANDS.update(ref_com)  # type: ignore

    if args.profile:
        import cProfile
        import pstats

        with cProfile.Profile() as pr:
            a = func(*fargs, **fkwargs)  # type: ignore

        ps = pstats.Stats(pr)
        ps.sort_stats(pstats.SortKey.CUMULATIVE).print_stats()

    else:
        a = func(*fargs, **fkwargs)  # type: ignore

    if args.uses_mpi:
        a = comm.gather(a, root=0)

    if rank == 0:
        file_out = Path(args.file_out)

        if file_out.suffix == ".json":
            with open(file_out, "w+") as f3:
                f3.writelines(jsonpickle.encode(a, use_base85=True))  # type: ignore
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
