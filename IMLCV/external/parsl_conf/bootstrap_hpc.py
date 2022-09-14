import parsl

from IMLCV.external.parsl_conf.bash_app_python import bash_app_python
from IMLCV.external.parsl_conf.config import config
from IMLCV.test.test import test_cv_discovery


def bootstrap_hpc(function):
    def f(*args, **kwargs):

        config(cluster="slaking", spawnjob=True)

        future = bash_app_python(function=function)(
            stdout=parsl.AUTO_LOGNAME,
            stderr=parsl.AUTO_LOGNAME,
            *args,
            **kwargs,
        )

        return future.result()

    return f


if __name__ == "__main__":
    out = bootstrap_hpc(test_cv_discovery)(
        name="hpc_perovskite", md="perov", recalc=True
    )
    print(out)
