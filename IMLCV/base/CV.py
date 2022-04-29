from __future__ import annotations

from abc import ABC
from dbm import ndbm
from functools import partial
import itertools

import jax
# import numpy as np
import jax.numpy as jnp
from jax import jit, grad
import dill
import alphashape
from matplotlib import pyplot as plt
from matplotlib.colors import Colormap
import numpy as np
from descartes import PolygonPatch
from shapely.geometry import Point, MultiPoint
from sklearn.cluster import DBSCAN


class Metric:

    def __init__(self, periodicities, boundaries=None) -> None:
        if boundaries is None:
            boundaries = jnp.zeros(len(periodicities))

        if isinstance(periodicities, list):
            periodicities = jnp.array(periodicities)

        self.boundaries = boundaries
        self.periodicities = periodicities
        self.type = periodicities

    # @partial(jnp.vectorize)
    def distance(self, x1, x2):
        return self._periodic_wrap(x1 - x2, min=True)

    # @partial(jnp.vectorize)
    def wrap(self, x1):
        return self._periodic_wrap(x1, min=False)

    def _update_boundaries(self, x1):
        self.boundaries[:, 1] = jnp.where(
            jnp.logical_and(self.perodicities == 0, x1 < self.boundaries[:, 0]),
            x1,
            self.boundaries,
        )

        self.boundaries[:, 1] = jnp.where(
            jnp.logical_and(self.perodicities == 0, x1 > self.boundaries[:, 1]),
            x1,
            self.boundaries,
        )

    @partial(jit, static_argnums=(0, 2))
    def _periodic_wrap(self, xs, min=False):
        """Translate cvs such over periodic vector

        Args:
            cvs: array of cvs
            min (bool): if False, translate to cv range. I true, minimises norm of vector
        """
        per = self.boundaries

        if min:
            o = 0.0
        else:
            o = per[:, 0]

        coor = (xs - o) / (per[:, 1] - per[:, 0])
        coor = jnp.mod(coor, 1)
        if min:
            coor = jnp.where(coor > 0.5, coor - 1, coor)

        coor = (coor) * (per[:, 1] - per[:, 0]) + o
        return jnp.where(self.periodicities, coor, xs)

    def __add__(self, other):
        assert isinstance(self, Metric)
        if other is None:
            return self

        assert isinstance(other, Metric)

        periodicities = jnp.hstack((self.periodicities, other.periodicities))
        boundaries = jnp.vstack((self.boundaries, other.boundaries))

        return Metric(periodicities=periodicities, boundaries=boundaries)

    def grid(self, n, endpoints=True):

        assert not (jnp.abs(self.boundaries) < 1e-12).any(), 'give proper boundaries'
        grid = [
            jnp.linspace(row[0], row[1], n, endpoint=endpoints) for per, row in zip(self.periodicities, self.boundaries)
        ]

        return grid

    @property
    def ndim(self):
        return len(self.periodicities)

    def update_metric(self, trajs, convex=True, plot=True, acc=10):
        trajs = np.array(trajs)

        points = np.vstack([trajs[:, 0, :], trajs[:, 1, :]])
        mpoints = MultiPoint(points)

        if convex:
            a = mpoints.convex_hull

        else:
            if len(points) > 30:
                np.random.shuffle(points)
                points = points[1:30]

            a = alphashape.alphashape(points)
            raise NotImplementedError

        bound = a.boundary
        dist_avg = bound.length / len(trajs)

        if convex == True:
            assert np.array([p.distance(bound) for p in mpoints
                            ]).max() < acc * dist_avg, "metrix boundaries are not convex"

        proj = np.array([[bound.project(Point(tr[0, :])), bound.project(Point(tr[1, :]))] for tr in trajs])

        #sort pair
        as1 = proj.argsort(axis=1)
        proj = np.take_along_axis(proj, as1, axis=1)
        trajs = np.array([pair[argsort, :] for argsort, pair in zip(as1, trajs)])

        clustering = DBSCAN(eps=20 * dist_avg, min_samples=10).fit(proj).labels_

        #cylically join begin and end clusters
        centers = [np.average(proj[clustering == i, 0]) for i in range(0, clustering.max() + 1)]
        proj[clustering == np.argmin(centers), 0] += bound.length
        proj[clustering == np.argmin(centers), :] = proj[clustering == np.argmin(centers), ::-1]
        trajs[clustering == np.argmin(centers), :, :] = trajs[clustering == np.argmin(centers), ::-1, :]

        offset = proj[clustering != -1].min()
        proj -= offset

        clustering = DBSCAN(eps=acc * dist_avg, min_samples=10).fit(proj).labels_

        if plot == True:
            for i in range(-1, clustering.max() + 1):
                plt.scatter(proj[clustering == i, 0], proj[clustering == i, 1])
            plt.show()

        assert clustering.max() + 1 >= self.ndim, "number of new periodicities do not correspond wiht original number"

        if plot == True:
            for i in range(0, clustering.max() + 1):
                vmax = proj[:, 0].max()

                plt.scatter(
                    trajs[clustering == i, 0, 0],
                    trajs[clustering == i, 0, 1],
                    c=proj[clustering == i, 0],
                    vmin=0,
                    vmax=vmax,
                )
                plt.scatter(
                    trajs[clustering == i, 1, 0],
                    trajs[clustering == i, 1, 1],
                    c=proj[clustering == i, 0],
                    vmin=0,
                    vmax=vmax,
                )
            plt.show()

        #make 2 meshgrids to make interpolation
        n = 10

        lrange = []
        xrange = []
        for i in range(0, clustering.max() + 1):
            pi = proj[clustering == i, :]
            a1 = pi[:, 0].argmin()
            a2 = pi[:, 1].argmin()
            l1 = [pi[a1, 0], pi[a2, 0]]
            l2 = [pi[a1, 1], pi[a2, 1]]

            def f(x):
                x += offset
                if x >= bound.length:
                    x -= bound.length

                p = bound.interpolate(x)
                return [p.x, p.y]

            xr1 = np.array([f(l) for l in np.linspace(l1[0], l1[1], num=n)])
            xr2 = np.array([f(l) for l in np.linspace(l2[0], l2[1], num=n)])

            lrange.append([l1, l2])
            xrange.append([xr1, xr2])

        mgrid = [np.zeros([n for _ in range(self.ndim)])] * self.ndim
        mgrid2 = [a[:] for a in mgrid]
        #make meshgrid to map to
        # for tup in itertools.product(range(0, n), repeat=self.ndim):
        #     for nd in range(self.ndim):

        #         up =  xrange[nd][  tup[n]  ] [0]
        #         down = xrange[nd][  tup[n]  ] [1]

        #         mgrid[nd][tup] = xrange[nd][  tup[n]  ]

        #         mgrid[i][tup] = xrange[]
        #     print(tup)


class hyperTorus(Metric):

    def __init__(self, n) -> None:
        periodicities = [True for _ in range(n)]
        boundaries = jnp.zeros((n, 2))
        boundaries = boundaries.at[:, 0].set(-jnp.pi)
        boundaries = boundaries.at[:, 1].set(jnp.pi)

        super().__init__(periodicities, boundaries)


class CV:
    """base class for CVs.

    args:
        f: function f(coordinates, cell, **kwargs) that returns a single CV
        **kwargs: arguments of custom function f
    """

    def __init__(self, f, metric: Metric, n=1, **kwargs) -> None:

        self.f = f
        self.kwargs = kwargs
        self._update_params(**kwargs)
        self.n = n

        self.metric = metric

    def compute(self, coordinates, cell, jac_p=False, jac_c=False):
        """
        args:
            coodinates: cartesian coordinates, as numpy array of form (number of atoms,3)
            cell: cartesian coordinates of cell vectors, as numpy array of form (3,3)
            grad: if not None, is set to the to gradient of CV wrt coordinates
            vir: if not None, is set to the to gradient of CV wrt cell params
        """
        val = self.cv(coordinates, cell)
        jac_p_val = self.jac_p(coordinates, cell) if jac_p else None
        jac_c_val = self.jac_c(coordinates, cell) if jac_c else None

        return [val, jac_p_val, jac_c_val]

    def _update_params(self, **kwargs):
        """update the CV functions."""

        self.cv = jit(lambda x, y: (jnp.ravel(partial(self.f, **kwargs)(x, y))))
        self.jac_p = jit(jax.jacfwd(self.cv, argnums=(0)))
        self.jac_c = jit(jax.jacfwd(self.cv, argnums=(1)))

    def __eq__(self, other):
        if not isinstance(other, CV):
            return NotImplemented
        return dill.dumps(self.cv) == dill.dumps(other.cv)

    # def find_periodicity(self, coordinates, cell):
    #     # object.interpolate
    #     from IMLCV.base.bias import BiasMTD

    #     x = coordinates[:] + np.random.rand(*coordinates.shape) * 1e-6
    #     # c = cell[:]
    #     px = jnp.zeros(x.shape)

    #     # pc = jnp.zeros(c.shape)

    #     sigmas = jnp.zeros((self.n)) + 1
    #     bias = BiasMTD(self, K=0.1, sigmas=sigmas)
    #     dt = 1e-2

    #     cv1 = jnp.zeros(self.n) * jnp.nan
    #     cv2 = jnp.zeros(self.n) * jnp.nan

    #     bias.update_bias(x + np.random.rand(*x.shape) * 1e-6, None)
    #     pairs = jnp.zeros((0, self.n, 2))

    #     for i in range(int(1e5)):

    #         ener, gpos, vir = bias.compute_coor(x, None, gpos=jnp.zeros(x.shape))

    #         px = dt * gpos
    #         # px /= jnp.linalg.norm(px)

    #         x += px * dt

    #         if i % 100 == 0:
    #             print(f"\n{i}", end="")

    #         if i % 100 == 0:
    #             bias.update_bias(x, None)

    #         if i % 2 == 0:
    #             cv1, _, _ = self.compute(x, None)
    #         else:
    #             cv2, _, _ = self.compute(x, None)

    #         n = jnp.linalg.norm(cv1 - cv2)

    #         if n > 1:
    #             pairs = jnp.vstack((pairs, jnp.array([[cv1, cv2]])))
    #             print('.', end="")
    #     print()

    #     alphashape.alphashape(pairs)


class CombineCV(CV):
    """combine multiple CVs into one CV."""

    def __init__(self, cvs) -> None:
        self.n = 0
        self.cvs = cvs
        metric = None

        for cv in cvs:
            self.n += cv.n
            if metric is None:
                metric = cv.metric
            else:
                metric += cv.metric

        self.metric = metric
        self._update_params()

    def _update_params(self):
        """function selects on ouput according to index."""

        def f(x, y, cvs):
            return jnp.array([cv.cv(x, y) for cv in cvs])

        self.f = partial(f, cvs=self.cvs)

        super()._update_params()


class CVUtils:
    """collection of predifined CVs. Intended to be used as argument to CV class.

    args:
        coordinates (np.array(n_atoms,3)): cartesian coordinates
        cell (np.array((3,3)): cartesian coordinates of cell vectors
    """

    @staticmethod
    def dihedral(coordinates, cell, numbers):
        """from https://stackoverflow.com/questions/20305272/dihedral-torsion-angle-from-four-points-in-cartesian-
        coordinates-in-python.

        args:
            numbers: list with index of 4 atoms that form dihedral
        """
        p0 = coordinates[numbers[0]]
        p1 = coordinates[numbers[1]]
        p2 = coordinates[numbers[2]]
        p3 = coordinates[numbers[3]]

        b0 = -1.0 * (p1 - p0)
        b1 = p2 - p1
        b2 = p3 - p2

        b1 /= jnp.linalg.norm(b1)

        v = b0 - jnp.dot(b0, b1) * b1
        w = b2 - jnp.dot(b2, b1) * b1

        x = jnp.dot(v, w)
        y = jnp.dot(jnp.cross(b1, v), w)
        return jnp.arctan2(y, x)

    @staticmethod
    def Volume(coordinates, cell):
        return jnp.abs(jnp.dot(cell[0], jnp.cross(cell[1], cell[2])))
