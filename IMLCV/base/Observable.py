import numpy as np

from IMLCV.base.MdEngine import MDEngine


class Observable:
    """class to convert data and CVs to different thermodynamic/ kinetic
    observables."""

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
        mtd = SumHills(grid)
        mtd.load_hdf5('traj.h5')
        fes = mtd.compute_fes()
        # Reshape to rectangular grids
        grid = grid.reshape((grid0.shape[0], grid1.shape[0], 2))
        fes = fes.reshape((grid0.shape[0], grid1.shape[0]))
        return grid, fes
