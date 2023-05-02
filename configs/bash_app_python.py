# this is a helper function to perform md simulations. Executed by parsl on HPC infrastructure, but

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import cloudpickle
from parsl import File, bash_app, python_app
from parsl.dataflow.dflow import AppFuture


# @typeguard.typechecked
def bash_app_python(
    function=None,
    executors="all",
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
            inputs = [*inputs, *kwargs.pop("inputs", [])]
            outputs = [*outputs, *kwargs.pop("outputs", [])]

            @bash_app(executors=executors)
            def fun(*args, stdout, stderr, inputs, outputs, **kwargs):
                if len(inputs) > 1:
                    kwargs["inputs"] = inputs[:-1]
                if len(outputs) > 1:
                    kwargs["outputs"] = outputs[:-1]

                filename_in = inputs[-1].filepath
                filename_out = outputs[-1].filepath
                fold = os.path.dirname(filename_in)
                if not os.path.exists(fold):
                    os.mkdir(fold)

                with open(filename_in, "wb+") as f:
                    cloudpickle.dump((func, args, kwargs), f)

                return f"python  -u { os.path.realpath( __file__ ) }    --folder {execution_folder} --file_in {  os.path.relpath(filename_in,execution_folder)  }  --file_out  {  os.path.relpath(filename_out,execution_folder)  }"

            fun.__name__ = func.__name__

            def rename(name):
                path, name = os.path.split(name.filepath)
                return os.path.join(path, f"bash_app_{name}")

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

            def rename_num(stdout, i):
                
                stem = stdout.name.split(".")
                stem[0] = f"{stem[0]}_{i:0>3}"
                return stdout.parent /  ".".join(stem)

            fl = execution_folder / f"{func.__name__}.lock"

            if fl.exists():
                with open(fl) as f:
                    i = int(f.readline()) + 1
            else:
                i = 0

            with open(fl, "w+") as f:
                f.write(f"{i}")

            file_in = str(
                rename_num(execution_folder / f"{func.__name__}.inp.cloudpickle", i)
            )
            file_out = str(
                rename_num(execution_folder / f"{func.__name__}.outp.cloudpickle", i)
            )

            stdout = str(
                rename_num(
                    execution_folder / f"{ func.__name__}.stdout"
                    if stdout is None
                    else Path(stdout),
                    i,
                )
            )
            stderr = str(
                rename_num(
                    execution_folder
                    / (f"{ func.__name__}.stderr" if stderr is None else Path(stderr)),
                    i,
                )
            )

            future: AppFuture = fun(
                inputs=[*inputs, File(file_in)],
                outputs=[*[File(rename(o)) for o in outputs], File(file_out)],
                stdout=stdout,
                stderr=stderr,
                *args,
                **kwargs,
            )

            @python_app(executors=["default"])
            def load(inputs=[], outputs=[]):
                with open(inputs[-1].filepath, "rb") as f:
                    result = cloudpickle.load(f)
                import os
                import shutil

                os.remove(inputs[-1].filepath)
                for i, o in zip(inputs[:-1], outputs):
                    shutil.move(i.filepath, o.filepath)

                return result

            return load(inputs=future.outputs, outputs=outputs)

        return wrapper

    if function is not None:
        return decorator(function)
    return decorator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Execute python function in a bash app"
    )
    parser.add_argument(
        "--file_in",
        type=str,
        help="path to file f containing cloudpickle.dump((func, args, kwargs), f)",
    )

    parser.add_argument(
        "--file_out",
        type=str,
        help="path to file f containing cloudpickle.dump((func, args, kwargs), f)",
    )
    parser.add_argument("--folder", type=str, help="working directory")
    args = parser.parse_args()

    print("#" * 20)
    print(f"got input {sys.argv}")
    print(f"task started at {datetime.now():%d/%m/%Y %H:%M:%S }")

    os.chdir(args.folder)

    print(f"working in folder {os.getcwd()}")

    with open(args.file_in, "rb") as f:
        func, fargs, fkwargs = cloudpickle.load(f)
    print("#" * 20)

    a = func(*fargs, **fkwargs)

    with open(args.file_out, "wb+") as f:
        cloudpickle.dump(a, f)

    os.remove(args.file_in)

    print("#" * 20)
    print(f"task finished at {datetime.now():%d/%m/%Y %H:%M:%S}")
    print("#" * 20)
