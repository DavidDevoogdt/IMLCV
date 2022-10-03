import parsl
from molmod.units import kjmol

from IMLCV.external.parsl_conf.bash_app_python import bash_app_python
from IMLCV.external.parsl_conf.config import config
from IMLCV.test.common import ase_yaff
from IMLCV.test.test import test_cv_discovery


def bootstrap_hpc(function):
    def f(*args, **kwargs):

        config(
            cluster="slaking",
            spawnjob=True,
            time="24:00:00",
        )

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
        name="hpc_perovskite",
        md=ase_yaff(),
        recalc=True,
        steps=100,
        K=50 * kjmol,
    )
