# this is a helper function to perform md simulations. Executed by parsl on HPC infrastructure, but

import argparse
import os
import uuid

import dill
from parsl import bash_app


def bash_app_python(func):
    def wrapper(bash_app_kwargs={}, *args, **kwargs):

        stdout = kwargs.pop('stdout', None)
        stderr = kwargs.pop('stderr', None)

        inputs = kwargs.pop('inputs', [])
        outputs = kwargs.pop('outputs', [])

        @bash_app(**bash_app_kwargs)
        def python_func(inputs, outputs, stdout, stderr, *args, **kwargs):

            filename = str(uuid.uuid4())

            with open(filename, 'wb+') as f:
                dill.dump((func, args, kwargs), f)

            return f'''python -u { os.path.realpath(  __file__ ) } --cwd {os.getcwd()} --file {filename}'''

        return f2(*args, **kwargs, inputs=inputs, outputs=outputs, stdout=stdout, stderr=stderr)

    return wrapper


if __name__ == "__main__":
    # this function is called by the wrapper
    parser = argparse.ArgumentParser(
        description='Perform an (enhanced) biased MD')
    parser.add_argument('--file', type=str,
                        help='path to bias')
    parser.add_argument('--cwd', type=str,
                        help='working directory')

    args = parser.parse_args()
    os.chdir(args.cwd)

    with open(args.file, 'rb') as f:
        func, fargs, fkwargs = dill.load(f)

    func(*fargs, **fkwargs)
    os.remove(args.file)
