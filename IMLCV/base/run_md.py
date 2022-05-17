import sys

import dill
from IMLCV.base.MdEngine import MDEngine
from yaff.log import log

log.set_level(log.medium)


def run_md(folder_name):
    """method used to perform md runs. arguments are constructed in rounds.run_par and shouldn't be done manually"""

    with open(f"{folder_name}/inp", mode='rb') as f:
        kwargs = dill.load(f)

    common_md_name = kwargs["common_md_name"]
    steps = kwargs["steps"]
    i = kwargs["i"]
    folder = kwargs["folder"]
    round = kwargs["round"]

    b = kwargs["bias"]
    md = MDEngine.load(
        common_md_name, filename=f"{folder_name}/traj.h5", bias=b)

    md.run(steps=steps)

    from IMLCV.base.rounds import RoundsMd
    d, attr = RoundsMd._add(
        md, f'{folder}/round_{round}/bias_{i}')

    with open(f"{folder_name}/out", mode='wb') as f:
        dill.dump([d, attr, i], f)


if __name__ == '__main__':
    print(sys.argv[1])
    run_md(folder_name=sys.argv[1])
