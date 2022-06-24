import os
import shutil

import dill
import IMLCV
import matplotlib.pyplot as plt
import numpy as np
import parsl
import umap
from IMLCV.base.bias import Bias, NoneBias
from IMLCV.base.rounds import RoundsMd
from IMLCV.launch.parsl_conf.config import config
from molmod.units import kjmol

# parsl.load()

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


def cleancopy(base):

    if not os.path.exists(f"{base}_orig"):
        assert os.path.exists(f"{base}"), "folder not found"
        shutil.copytree(f"{base}", f"{base}_orig")

    if os.path.exists(f"{base}"):
        shutil.rmtree(f"{base}")
    shutil.copytree(f"{base}_orig", f"{base}")


if __name__ == "__main__":

    config()

    # make copy and restore orig
    name = "ala2"
    base = f"output/{name}"

    # cleancopy(base=base)
    with open(f"{base}/rounds", 'rb') as f:
        rounds: RoundsMd = dill.load(f)

    md = rounds.get_engine(r=6)

    md.run(int(1e5))

    rounds.new_round(md=md)
    rounds.max_energy = 60*kjmol
    rounds.run(bias=None, steps=1e5)

    rounds.new_round()

    pos_arr = []
    cvs_arr = []
    energies = []

    for dictionary in rounds.iter(num=1):
        bias = Bias.load(dictionary['attr']["name_bias"])

        pos = dictionary["positions"]
        if 'cell' in dictionary:
            cell = dictionary["cell"]
            cvs = np.array([bias.cvs.compute(coordinates=x, cell=y)[0]
                           for (x, y) in zip(pos, cell)], dtype=np.double)
        else:
            cvs = np.array([bias.cvs.compute(coordinates=p, cell=None)[0]
                           for p in pos], dtype=np.double)
        biases = np.array(np.apply_along_axis(lambda x:  bias.compute(cvs=x)[0],
                                              arr=cvs,
                                              axis=1),
                          dtype=np.double)

        pos_arr.append(pos)
        cvs_arr.append(cvs)
        energies.append(biases)

    # reschape
    x = np.vstack(pos_arr)
    x = np.reshape(x, (x.shape[0], -1))
    y = np.vstack(cvs_arr)
    e = np.vstack(energies)

    reducer = umap.UMAP(n_neighbors=20, n_components=2, min_dist=1)
    trans = reducer.fit(x, y)
    c = trans.transform(x)

    fig = plt.figure()
    ax = fig.add_subplot(projection='2d')
    # ax.scatter(c[:, 0], c[:, 1], c[:, 2], c=e)
    ax.scatter(c[:, 0], c[:, 1], c[:, 2], c=e)

    plt.show()

    print('done')
