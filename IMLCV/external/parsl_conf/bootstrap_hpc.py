import parsl
from molmod.units import kjmol

from IMLCV.external.parsl_conf.bash_app_python import bash_app_python
from IMLCV.external.parsl_conf.config import config

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


def func(name):
    from IMLCV.base.rounds import RoundsMd
    from IMLCV.test.common import ase_yaff
    from IMLCV.external.parsl_conf.config import config
    import shutil
    import os

    config(cluster="doduo", max_blocks=10)

    if os.path.exists(f"output/{name}" ):
        shutil.rmtree( f"output/{name}" )

    engine = ase_yaff()
    round = RoundsMd( folder=f"output/{name}"  )
    round.new_round(md=engine)
    round.run_par( [ None  for _ in range(10) ] ,steps=1000 )
    
    round.write_xyz()

if __name__ == "__main__":
    # out = bootstrap_hpc(test_cv_discovery)(
    #     name="hpc_perovskite_6",
    #     md=ase_yaff(),
    #     recalc=True,
    #     steps=500,
    #     k=0.5 * kjmol,
    # )

    out = bootstrap_hpc(func)(
        name="hpc_perovskite_unbiased",
    )
