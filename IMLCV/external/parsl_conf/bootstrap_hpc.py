import parsl

from IMLCV.external.parsl_conf.bash_app_python import bash_app_python
from IMLCV.external.parsl_conf.config import config


def bootstrap_hpc(function):
    def f(*args, **kwargs):

        config(
            cluster="slaking",
            spawnjob=True,
            time="72:00:00",
        )

        future = bash_app_python(function=function)(
            stdout=parsl.AUTO_LOGNAME,
            stderr=parsl.AUTO_LOGNAME,
            *args,
            **kwargs,
        )

        return future.result()

    return f


def func(name):
    import os
    import shutil

    from IMLCV.base.rounds import RoundsMd
    from IMLCV.external.parsl_conf.config import config
    from IMLCV.test.common import ase_yaff

    config(cluster="doduo", max_blocks=10, mem_per_node=20)

    if os.path.exists(f"output/{name}"):
        shutil.rmtree(f"output/{name}")

    engine = ase_yaff()
    round = RoundsMd(folder=f"output/{name}")
    round.new_round(md=engine)
    round.run_par([None for _ in range(10)], steps=1000)

    round.write_xyz()


def f():
    from molmod.units import kjmol

    from IMLCV.external.parsl_conf.config import config
    from IMLCV.test.common import ase_yaff
    from IMLCV.test.test import test_cv_discovery

    config(cluster="doduo", time="48:00:00", mem_per_node=20)
    test_cv_discovery(
        name="hpc_perovskite_biased_04",
        md=ase_yaff(small=True),
        recalc=True,
        steps=1000,
        k=10 * kjmol,
    )


if __name__ == "__main__":
    out = bootstrap_hpc(f)()

    # out = bootstrap_hpc(func)(
    #     name="hpc_perovskite_unbiased",
    # )
