import os
import shutil

import dill
import matplotlib.pyplot as plt
import numpy as np
import umap
from IMLCV.base.bias import Bias
from IMLCV.base.rounds import RoundsMd

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


def cleancopy(base):

    if not os.path.exists(f"{base}_orig"):
        assert os.path.exists(f"{base}"), "folder not found"
        shutil.copytree(f"{base}", f"{base}_orig")

    shutil.rmtree(f"{base}")
    shutil.copytree(f"{base}_orig", f"{base}")


if __name__ == "__main__":

    # make copy and restore orig
    name = "ala2"
    base = f"output/{name}"

    # cleancopy(name=base)
    with open(f"{base}/rounds", 'rb') as f:
        rounds: RoundsMd = dill.load(f)

    pos_arr = []
    cvs_arr = []

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

        pos_arr.append(pos)
        cvs_arr.append(cvs)

    # reschape
    x = np.vstack(pos_arr)
    x = np.reshape(x, (x.shape[0], -1))
    y = np.vstack(cvs_arr)

    reducer = umap.UMAP(n_neighbors=20, n_components=3)
    trans = reducer.fit(x)
    c = trans.transform(x)

    plt.scatter(c[:, 0], c[:, 1])
