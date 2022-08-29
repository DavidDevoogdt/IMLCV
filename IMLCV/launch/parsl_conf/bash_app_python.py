# this is a helper function to perform md simulations. Executed by parsl on HPC infrastructure, but

import argparse
import os
import sys
import uuid
from typing import List, Literal, Optional, Union

import dill
from parsl import bash_app
from parsl.app.bash import BashApp
from parsl.data_provider.files import File
from parsl.dataflow.dflow import DataFlowKernel


# @typeguard.typechecked
def bash_app_python(
        function=None,
        data_flow_kernel: Optional[DataFlowKernel] = None,
        cache: bool = False,
        executors:   Union[List[str], Literal['all']] = 'all',
        ignore_for_cache: Optional[List[str]] = None):

    def decorator(func):
        def wrapper(*args,  stdout=None, stderr=None, inputs=[], outputs=[], **kwargs):
            fold = ".bash_python_app"
            if not os.path.exists(fold):
                os.mkdir(fold)

            filename = f"{fold}/{str(uuid.uuid4())}"

            # merge in and outputs
            inputs = [*inputs,  *kwargs.pop("inputs", [])]
            outputs = [*outputs,  *kwargs.pop("outputs", [])]

            def fun(*args, inputs, outputs, stdout, stderr, **kwargs):

                if len(inputs) > 0:
                    kwargs['inputs'] = inputs
                if len(outputs) > 1:
                    kwargs['outputs'] = outputs[1:]

                filename = outputs[0].filepath
                fold = os.path.dirname(filename)
                if not os.path.exists(fold):
                    os.mkdir(fold)

                with open(outputs[0].filepath, 'wb+') as f:
                    dill.dump((func, args, kwargs), f)

                return f'''python -u { os.path.realpath( __file__ ) } --cwd {os.getcwd()} --file {filename}'''
            fun.__name__ = func.__name__

            bash_app_fun = bash_app(fun, data_flow_kernel=data_flow_kernel,
                                    cache=cache, executors=executors, ignore_for_cache=ignore_for_cache)

            future: BashApp = bash_app_fun(
                inputs=inputs, outputs=[File(filename), *outputs], stdout=stdout, stderr=stderr, *args, **kwargs)

            # modify the future such that the output is recovered
            _res = future.result

            def result():
                _res()  # wait for result to finish
                with open(filename, 'rb') as f:
                    ret = dill.load(f)

                os.remove(filename)

                return ret

            future.result = result

            # cleanup inputs and outputs
            future._outputs = future._outputs[1:]

            # future.task_def['kwargs']['inputs'] = future.task_def['kwargs']['inputs']
            future.task_def['kwargs']['outputs'] = future.task_def['kwargs']['outputs'][1:]

            return future

        return wrapper
    if function is not None:
        return decorator(function)
    return decorator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Perform an (enhanced) biased MD')
    parser.add_argument('--file', type=str,
                        help='path to bias')
    parser.add_argument('--cwd', type=str,
                        help='working directory')

    print(f"got input {sys.argv}")

    args = parser.parse_args()
    os.chdir(args.cwd)

    with open(args.file, 'rb') as f:
        func, fargs, fkwargs = dill.load(f)

    print(f"calling {func} with {fargs} and {fkwargs}")

    a = func(*fargs, **fkwargs)

    with open(args.file, 'wb+') as f:
        dill.dump(a, f)
