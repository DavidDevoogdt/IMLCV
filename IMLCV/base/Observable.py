from IMLCV.base.MdEngine import MDEngine

import os
from yaff.log import log
import yaff.analysis.biased_sampling
import numpy as np
import matplotlib.pyplot as plt
from molmod import units


class Observable:
    """class to convert data and CVs to different thermodynamic/ kinetic observables."""

    def __init__(self, mde: MDEngine) -> None:
        pass

    def get_fes():
        npoints = 51
        # Construct a regular 2D grid, spanning from -pi to +pi in both dimensions
        grid0 = np.linspace(-np.pi, np.pi, npoints, endpoint=False)
        grid1 = np.linspace(-np.pi, np.pi, npoints, endpoint=False)
        grid = np.zeros((grid0.shape[0] * grid1.shape[0], 2))
        grid[:, 0] = np.repeat(grid0, grid1.shape[0])
        grid[:, 1] = np.tile(grid1, grid0.shape[0])
        mtd = yaff.analysis.biased_sampling.SumHills(grid)
        mtd.load_hdf5('traj.h5')
        fes = mtd.compute_fes()
        # Reshape to rectangular grids
        grid = grid.reshape((grid0.shape[0], grid1.shape[0], 2))
        fes = fes.reshape((grid0.shape[0], grid1.shape[0]))
        return grid, fes

    def make_plot_2D(grid, fes):
        fes -= np.amin(fes)
        plt.clf()
        plt.contourf(grid[:, :, 0], grid[:, :, 1], fes / units.kjmol)
        plt.xlabel("$\phi\,[\mathrm{rad}]$")
        plt.ylabel("$\psi\,[\mathrm{rad}]$")
        plt.title("$F\,[\mathrm{kJ}\,\mathrm{mol}^{-1}]$")
        plt.savefig('ala_dipep.png')
