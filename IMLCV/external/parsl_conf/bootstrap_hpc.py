from sys import stderr, stdout

from IMLCV.external.parsl_conf.bash_app_python import bash_app_python
from IMLCV.external.parsl_conf.config import config


def bootstrap_hpc(function):
    def f(*args, **kwargs):

        config(cluster="slaking", spawnjob=True)

        future = bash_app_python(function=function)(
            stdout=stdout,
            stderr=stderr,
            *args,
            **kwargs,
        )

        future.outputs[0].result()

    return f
