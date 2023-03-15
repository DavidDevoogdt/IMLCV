# this is a helper function to perform md simulations. Executed by parsl on HPC infrastructure, but

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import dill
from parsl import File, bash_app, python_app


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

                if len(inputs) > 0:
                    kwargs["inputs"] = inputs
                if len(outputs) > 1:
                    kwargs["outputs"] = outputs[:-1]

                filename = outputs[-1].filepath
                fold = os.path.dirname(filename)
                if not os.path.exists(fold):
                    os.mkdir(fold)

                with open(filename, "wb+") as f:
                    dill.dump((func, args, kwargs), f)

                return f"""python  -u { os.path.realpath( __file__ ) }  --folder {execution_folder} --file {filename}"""

            fun.__name__ = func.__name__

            from parsl.dataflow.dflow import AppFuture

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
            file = File(str(execution_folder / "function.dill"))

            future: AppFuture = fun(
                inputs=inputs,
                outputs=[*[File(rename(o)) for o in outputs], file],
                stdout=str(execution_folder / ("stdout" if stdout is None else stdout)),
                stderr=str(execution_folder / ("stderr" if stderr is None else stderr)),
                *args,
                **kwargs,
            )

            @python_app(executors=["default"])
            def load(inputs=[], outputs=[]):
                with open(inputs[-1].filepath, "rb") as f:
                    result = dill.load(f)
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
        "--file",
        type=str,
        help="path to file f containing dill.dump((func, args, kwargs), f)",
    )
    parser.add_argument("--folder", type=str, help="working directory")
    args = parser.parse_args()

    print("#" * 20)
    print(f"got input {sys.argv}")
    print(f"task started at {datetime.now():%d/%m/%Y %H:%M:%S }")

    os.chdir(args.folder)

    print(f"working in folder {os.getcwd()}")

    with open(args.file, "rb") as f:
        func, fargs, fkwargs = dill.load(f)
    print("#" * 20)

    a = func(*fargs, **fkwargs)

    with open(args.file, "wb+") as f:
        dill.dump(a, f)

    print("#" * 20)
    print(f"task finished at {datetime.now():%d/%m/%Y %H:%M:%S}")
    print("#" * 20)
