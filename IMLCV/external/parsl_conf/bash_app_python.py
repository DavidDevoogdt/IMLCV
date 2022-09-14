# this is a helper function to perform md simulations. Executed by parsl on HPC infrastructure, but

import argparse
import os
import sys
import uuid

import dill
from parsl import File, bash_app, python_app


# @typeguard.typechecked
def bash_app_python(
    function=None,
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

            @bash_app
            def fun(*args, inputs, outputs, stdout, stderr, **kwargs):

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

            fun.__name__ == func.__name__

            from parsl.dataflow.dflow import AppFuture

            def rename(name):
                path, name = os.path.split(name.filepath)
                return os.path.join(path, f"bash_app_{name}")

            fold = ".bash_python_app"
            if not os.path.exists(fold):
                os.mkdir(fold)

            filename = f"{fold}/{str(uuid.uuid4())}"

            file = File(filename)

            future: AppFuture = fun(
                inputs=inputs,
                outputs=[*[File(rename(o)) for o in outputs], file],
                stdout=stdout,
                stderr=stderr,
                *args,
                **kwargs,
            )

            @python_app
            def load(inputs=[], outputs=[]):
                with open(inputs[-1].filepath, "rb") as f:
                    result = dill.load(f)
                import shutil

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
    parser = argparse.ArgumentParser(description="Perform an (enhanced) biased MD")
    parser.add_argument("--file", type=str, help="path to bias")
    parser.add_argument("--folder", type=str, help="working directory")

    print(f"got input {sys.argv}")

    args = parser.parse_args()
    os.chdir(args.folder)

    with open(args.file, "rb") as f:
        func, fargs, fkwargs = dill.load(f)

    print(f"calling {func} with args {fargs} and  kwargs {fkwargs}")

    a = func(*fargs, **fkwargs)

    with open(args.file, "wb+") as f:
        dill.dump(a, f)
