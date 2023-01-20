# this is a helper function to perform md simulations. Executed by parsl on HPC infrastructure, but

import argparse
import os
import sys
import uuid

import dill
from parsl import File, bash_app, python_app

from configs.config_general import ROOT_DIR


# @typeguard.typechecked
def bash_app_python(
    function=None,
    executors="all",
):
    def decorator(func):
        def wrapper(
            *args,
            folder=None,
            stdout=None,
            stderr=None,
            inputs=[],
            outputs=[],
            **kwargs,
        ):

            # merge in and outputs
            inputs = [*inputs, *kwargs.pop("inputs", [])]
            outputs = [*outputs, *kwargs.pop("outputs", [])]

            if folder is None:
                folder = os.getcwd()

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

                return f"""python -u { os.path.realpath( __file__ ) } --folder {folder} --file {filename}"""

            fun.__name__ = func.__name__

            from parsl.dataflow.dflow import AppFuture

            def rename(name):
                path, name = os.path.split(name.filepath)
                return os.path.join(path, f"bash_app_{name}")

            fold = ROOT_DIR / "IMLCV" / "bash_python_app"
            if not os.path.exists(fold):

                os.mkdir(fold)

            filename = str(f"{fold}/{str(uuid.uuid4())}")

            file = File(filename)

            future: AppFuture = fun(
                inputs=inputs,
                outputs=[*[File(rename(o)) for o in outputs], file],
                stdout=stdout,
                stderr=stderr,
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
            # return load(inputs=[future.outputs[-1]])

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

    os.chdir(args.folder)

    print(f"working in folder {os.getcwd()}")

    with open(args.file, "rb") as f:
        func, fargs, fkwargs = dill.load(f)

    print(f"calling {func} with args {fargs} and  kwargs {fkwargs}")

    print("#" * 20)

    a = func(*fargs, **fkwargs)

    with open(args.file, "wb+") as f:
        dill.dump(a, f)

    print("#" * 20)
    print(f"function finished properly, results dumped in {args.file}")
