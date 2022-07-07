from functools import partial
from nis import match
from random import choices, random
from typing import Any, List, Tuple

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import umap
from IMLCV.base.bias import Bias
from IMLCV.base.CV import CV
from IMLCV.base.rounds import Rounds, RoundsMd
from matplotlib.axes import Axes
from matplotlib.axis import Axis
from molmod.constants import boltzmann
from numpy.random import choice

plt.rcParams['text.usetex'] = True


class Transformer:
    def __init__(self) -> None:
        pass

    def fit(self, x, **kwargs):
        pass


class UMAP(Transformer):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, x, **kwargs):

        shape = x.shape
        x = np.reshape(x, (shape[0], -1))

        # reducer = umap.ParametricUMAP(
        #     n_neighbors=50, min_dist=0.2, n_components=2)
        reducer = umap.UMAP(densmap=True, n_neighbors=20, min_dist=0.9)

        trans = reducer.fit(x)

        return lambda x:  trans.transform(np.reshape(x, (shape[0], -1)))


class CVDiscovery:
    """convert set of coordinates to good collective variables."""

    def __init__(self, rounds: RoundsMd, transformer: Transformer = UMAP()) -> None:
        self.rounds = rounds
        self.transformer = transformer

    def _get_data(self, num=3, out=1e4):

        pos_arr = []
        cell_arr = []
        cvs_arr = []
        energies = []
        times = []
        taus = []
        weights = []

        def get_cvs(pos, cell, CV: CV):
            if cell is not None:
                cell = dictionary["cell"]
                cvs_mapped = np.array([CV.compute(coordinates=x, cell=y)[0]
                                       for (x, y) in zip(pos, cell)], dtype=np.double)
            else:
                cvs_mapped = np.array([CV.compute(coordinates=p, cell=None)[0]
                                       for p in pos], dtype=np.double)
                cell = None

            cvs_mapped = np.array(np.apply_along_axis(lambda x:  CV.metric.map(x),
                                                      arr=cvs_mapped,
                                                      axis=1),
                                  dtype=np.double)

            return cvs_mapped

        cv = None

        for dictionary in self.rounds.iter(num=num):
            bias = Bias.load(dictionary['attr']["name_bias"])

            pos = dictionary["positions"]
            cell = dictionary.get("cell", None)
            cvs_mapped = get_cvs(pos, cell, bias.cvs)

            if cv is None:
                cv = bias.cvs
            biases = np.array(np.apply_along_axis(lambda x:  bias.compute(cvs=x, map=False)[0],
                                                  arr=cvs_mapped,
                                                  axis=1),
                              dtype=np.double)

            beta = 1/(dictionary['round']['T']*boltzmann)
            weight = np.exp(beta * biases)

            time = dictionary["t"]

            # def integr(fx, x):
            #     y = np.array(
            #         [0, *np.cumsum((fx[1:] + fx[:-1]) * (x[1:] - x[:-1]) / 2)])
            #     return y

            # beta = 1/(dictionary['round']['T']*boltzmann)
            # tau = integr(np.exp(beta * biases), time)

            times.append(time)
            # taus.append(tau)
            pos_arr.append(pos)
            cvs_arr.append(cvs_mapped)
            # energies.append(biases)
            cell_arr.append(cell)
            weights.append(weight)

        # start_tuas = [0, * np.cumsum([tau[-1] for tau in taus[:-1]])]
        # taus = [tau+st for tau, st in zip(taus, start_tuas)]

        # taus = np.hstack(taus)
        # new_taus = np.linspace(
        #     start=taus.min(), stop=taus.max(), num=int(out))

        # def _interp(x_new, x, y):
        #     if y is not None:
        #         return jnp.apply_along_axis(
        #             lambda yy: jnp.interp(x_new, x, yy), arr=y, axis=0)
        #     return None

        # pos = np.vstack(pos_arr)
        # new_pos = _interp(new_taus, taus, pos)

        # if cell is not None:
        #     cell = np.vstack(cell_arr)
        #     new_cell = _interp(new_taus, taus, cell)
        # else:
        #     new_cell = None

        # #
        # cvs = get_cvs(new_pos, new_cell, cv)

        probs = np.hstack(weights)
        probs = probs/np.sum(probs)

        indices = choice(len(probs), int(out), p=probs, replace=False)

        pos = np.vstack(pos_arr)[indices]
        if cell is not None:
            cell = np.vstack(cell_arr)[indices]
        else:
            cell = None
        cvs = np.vstack(cvs_arr)[indices]

        return [pos, cell, cvs]

    def compute(self,  plot=True) -> CV:

        x, _, cv = self._get_data(num=6, out=1e4)

        fit = True

        if fit:
            t = self.transformer.fit(x)

        if plot:

            if fit:

                c = t(x)
                c = (c[:]-c.min(axis=0))/(c.max(axis=0) - c.min(axis=0))

                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

                # plot setting
                kwargs = {'s': 2,
                          'cmap': 'jet', }

                ax1.scatter(c[:, 0], c[:, 1], c=cv[:, 0],
                            label='phi', **kwargs)
                ax2.scatter(c[:, 0], c[:, 1], c=cv[:, 1],
                            label='psi', **kwargs)
                for ax in [ax1, ax2]:
                    ax.set_xlabel('umap cv1')
                    ax.set_ylabel('umap cv2')

                ax3.scatter(cv[:, 0], cv[:, 1], c=cv[:, 0],
                            label='umap CV1', **kwargs)
                ax4.scatter(cv[:, 0], cv[:, 1], c=cv[:, 1],
                            label='umap CV2', **kwargs)

                for ax in [ax3, ax4]:
                    ax.set_xlabel(r' $\Psi$')
                    ax.set_ylabel(r' $\Phi$')

                plt.show()

            else:
                fig, ax = plt.subplots(2, 2)
                ax.scatter(cv[:, 0], cv[:, 1])

            plt.show()

        raise NotImplementedError
