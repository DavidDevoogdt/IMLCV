from __future__ import division

import os, sys
from IMLCV.base.CV import CV, CVUtils, CombineCV
from IMLCV.base.MdEngine import YaffEngine
from yaff.pes import CVInternalCoordinate, DihedAngle
from yaff.test.common import get_alaninedipeptide_amber99ff
from yaff.log import log
from yaff.analysis.biased_sampling import SumHills
import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
from molmod import units, constants
import ase.io

log.set_level(log.medium)
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# np.random.seed(3)


def get_fes():
    npoints = 51
    # Construct a regular 2D grid, spanning from -pi to +pi in both dimensions
    grid0 = np.linspace(-np.pi, np.pi, npoints, endpoint=False)
    grid1 = np.linspace(-np.pi, np.pi, npoints, endpoint=False)
    grid = np.zeros((grid0.shape[0] * grid1.shape[0], 2))
    grid[:, 0] = np.repeat(grid0, grid1.shape[0])
    grid[:, 1] = np.tile(grid1, grid0.shape[0])
    mtd = SumHills(grid)
    mtd.load_hdf5('traj.h5')
    fes = mtd.compute_fes()
    # Reshape to rectangular grids
    grid = grid.reshape((grid0.shape[0], grid1.shape[0], 2))
    fes = fes.reshape((grid0.shape[0], grid1.shape[0]))
    return grid, fes


def make_plot(grid, fes, T):
    # Free energy as a function of DihedAngle(4,6,8,14), by integrating over
    # other collective variable
    beta = 1.0 / constants.boltzmann / T
    fes_phi = -1. / beta * np.log(np.sum(np.exp(-beta * fes), axis=1))
    fes_phi -= np.amin(fes_phi)
    plt.clf()
    plt.plot(grid[:, 0, 0], fes_phi / units.kjmol)
    plt.xlabel("$\phi\,[\mathrm{rad}]$")
    plt.ylabel("$F\,[\mathrm{kJ}\,\mathrm{mol}^{-1}]$")
    plt.savefig('fes_phi.png')


def make_plot_2D(grid, fes):
    fes -= np.amin(fes)
    plt.clf()
    plt.contourf(grid[:, :, 0], grid[:, :, 1], fes / units.kjmol)
    plt.xlabel("$\phi\,[\mathrm{rad}]$")
    plt.ylabel("$\psi\,[\mathrm{rad}]$")
    plt.title("$F\,[\mathrm{kJ}\,\mathrm{mol}^{-1}]$")
    plt.savefig('ala_dipep.png')


def test_yaff_md():
    T = 1000 * units.kelvin
    ff = get_alaninedipeptide_amber99ff()

    cvs = CombineCV([
        CV(CVUtils.dihedral, numbers=[4, 6, 8, 14]),
        CV(CVUtils.dihedral, numbers=[6, 8, 14, 16]),
    ])

    sigmas = np.array([0.35, 0.35])
    periodicities = np.array([2.0 * np.pi, 2.0 * np.pi])
    K = 5 * units.kjmol

    yaffmd = YaffEngine(
        ff=ff,
        ES="MTD",
        sigmas=sigmas,
        periodicities=periodicities,
        K=K,
        step_hills=25,
        cv=cvs,
        T=T,
        P=None,
        timestep=2.0 * units.femtosecond,
        timecon_thermo=100.0 * units.femtosecond,
    )

    yaffmd.run(int(1e2))

    # aseSys = yaffmd.to_ASE_traj()
    # ase.io.write('md_ext.xyz', aseSys, format='extxyz', append=False)

    # grid, fes = get_fes()
    # make_plot_2D(grid, fes)


if __name__ == '__main__':
    test_yaff_md()
