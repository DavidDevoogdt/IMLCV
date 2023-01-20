import argparse

import parsl

from configs.bash_app_python import bash_app_python
from configs.vsc_stevin import config


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

    from configs.vsc_stevin import config
    from IMLCV.base.rounds import Rounds
    from IMLCV.test.common import ase_yaff

    config(cluster="doduo", max_blocks=10, mem_per_node=20)

    if os.path.exists(f"output/{name}"):
        shutil.rmtree(f"output/{name}")

    engine = ase_yaff(file)
    round = Rounds(folder=f"output/{name}")
    round.add_round(md=engine)
    round.run_par([None for _ in range(10)], steps=1000)

    round.write_xyz()


def f(name):
    from molmod.units import kjmol

    from configs.vsc_stevin import config
    from IMLCV.test.common import ase_yaff
    from IMLCV.test.test import test_cv_discovery

    config(cluster="doduo", time="72:00:00", mem_per_node=10)
    test_cv_discovery(
        name=name,
        md=ase_yaff(small=True),
        recalc=True,
        steps=500,
        k=10 * kjmol,
        n=8,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute python on hpc")
    parser.add_argument("--name", type=str, help="name of simulation")
    args = parser.parse_args()

    out = bootstrap_hpc(f)(name=args.name)
