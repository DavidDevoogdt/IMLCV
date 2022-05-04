from __future__ import annotations

from abc import ABC
from dbm import ndbm
from functools import partial
import itertools

import jax
# import numpy as np
import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit, grad
import dill
import alphashape
from matplotlib import pyplot as plt
from matplotlib.cbook import ls_mapper
from matplotlib.colors import Colormap
import numpy as np
from descartes import PolygonPatch
from scipy.interpolate import griddata, LinearNDInterpolator, Rbf

from shapely.geometry import Point, MultiPoint, LineString
from sklearn.cluster import DBSCAN


class Metric:

    def __init__(
        self,
        periodicities,
        boundaries=None,
        wrap_meshgrids=None,
        wrap_boundaries=None,
    ) -> None:
        if boundaries is None:
            boundaries = jnp.zeros(len(periodicities))

        if isinstance(boundaries, list):
            boundaries = jnp.array(boundaries)

        if isinstance(periodicities, list):
            periodicities = jnp.array(periodicities)

        self.wrap_meshgrids = wrap_meshgrids

        self.boundaries = boundaries
        self.periodicities = periodicities
        self.type = periodicities

        if wrap_boundaries is None:
            self.wrap_boundaries = self.boundaries
        else:
            if isinstance(wrap_boundaries, list):
                wrap_boundaries = jnp.array(wrap_boundaries)

            self.wrap_boundaries = wrap_boundaries

    @partial(jit, static_argnums=(0, 3))
    def distance(self, x1, x2, wrap_x2=True):

        xs = self.grid_wrap(x1)
        if wrap_x2 == False:
            xs -= x2
        else:
            xs -= self.grid_wrap(x2)

        return self._periodic_wrap(xs, min=True)

    def wrap(self, x1):
        xs = self.grid_wrap(x1)
        return self._periodic_wrap(xs, min=False)

    @partial(jit, static_argnums=(0, 2))
    def _periodic_wrap(self, xs, min=False):
        """Translate cvs such over periodic vector

        Args:
            cvs: array of cvs
            min (bool): if False, translate to cv range. I true, minimises norm of vector
        """

        per = self.wrap_boundaries

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

    @partial(jit, static_argnums=(0))
    def grid_wrap(self, x):
        if self.wrap_meshgrids is None:
            return x

        x = (x - self.boundaries[:, 0]) / (self.boundaries[:, 1] -
                                           self.boundaries[:, 0]) * (jnp.array(self.wrap_meshgrids[0].shape) - 1)
        wrapped = jnp.array([jsp.ndimage.map_coordinates(wp, x, order=1) for wp in self.wrap_meshgrids])
        return wrapped

    def __add__(self, other):
        assert isinstance(self, Metric)
        if other is None:
            return self

        assert isinstance(other, Metric)

        periodicities = jnp.hstack((self.periodicities, other.periodicities))
        boundaries = jnp.vstack((self.boundaries, other.boundaries))
        wrap_boundaries = jnp.vstack((self.wrap_boundaries, other.wrap_boundaries))

        if self.wrap_meshgrids is None and other.wrap_meshgrids is None:
            wrap_meshgrids = None
        elif self.wrap_meshgrids is not None and other.wrap_meshgrids is not None:
            wrap_meshgrids = [*self.wrap_meshgrids, *other.wrap_meshgrids]
        else:
            raise NotImplementedError

        return Metric(
            periodicities=periodicities,
            boundaries=boundaries,
            wrap_boundaries=wrap_boundaries,
            wrap_meshgrids=wrap_meshgrids,
        )

    def grid(self, n, endpoints=True, wrap=False):

        if wrap == True:
            b = self.wrap_boundaries
        else:
            b = self.boundaries

        assert not (jnp.abs(b[:, 1] - b[:, 0]) < 1e-12).any(), 'give proper boundaries'
        grid = [jnp.linspace(row[0], row[1], n, endpoint=endpoints) for per, row in zip(self.periodicities, b)]

        return grid

    @property
    def ndim(self):
        return len(self.periodicities)

    def update_metric(self, trajs, convex=True, plot=True, acc=30) -> Metric:
        """find best fitting bounding box and get affine tranformation+ new boundaries"""

        trajs = np.array(trajs)

        points = np.vstack([trajs[:, 0, :], trajs[:, 1, :]])
        mpoints = MultiPoint(points)

        if convex:
            a = mpoints.convex_hull
            # a = mpoints.minimum_rotated_rectangle

        else:
            if len(points) > 30:
                np.random.shuffle(points)
                points = points[1:30]

            a = alphashape.alphashape(points)
            raise NotImplementedError

        bound = a.boundary
        bound_type = type(bound)

        dist_avg = bound.length / len(trajs)

        if convex == True:
            assert np.array([p.distance(bound) for p in mpoints]).max() < acc * dist_avg, "boundaries are not convex"

        proj = np.array([[bound.project(Point(tr[0, :])), bound.project(Point(tr[1, :]))] for tr in trajs])

        def get_gaps(proj):

            projgaps = np.sort(np.hstack([proj[:, 0], proj[:, 1]]))

            gaps = (projgaps[1:] - projgaps[:-1]) > 10 * dist_avg
            gaps = [projgaps[0:-1][gaps], projgaps[1:][gaps]]
            return gaps

        gaps = get_gaps(proj)

        def get_lengths(proj, gaps):
            a = np.copy(proj[:])
            for i, j in list(zip(*gaps))[::-1]:
                a[a > i] -= j - i

            return a

        #sort pair
        as1 = proj.argsort(axis=1)
        proj = np.take_along_axis(proj, as1, axis=1)
        trajs = np.array([pair[argsort, :] for argsort, pair in zip(as1, trajs)])

        clustering = DBSCAN(eps=acc * dist_avg, min_samples=10).fit(get_lengths(proj, gaps)).labels_

        #cylically join begin and end clusters
        centers = [np.average(proj[clustering == i, 0]) for i in range(0, clustering.max() + 1)]
        proj[clustering == np.argmin(centers), 0] += bound.length
        proj[clustering == np.argmin(centers), :] = proj[clustering == np.argmin(centers), ::-1]
        trajs[clustering == np.argmin(centers), :, :] = trajs[clustering == np.argmin(centers), ::-1, :]

        offset = proj[clustering != -1].min()
        proj -= offset

        gaps = get_gaps(proj)
        clustering = DBSCAN(eps=acc * dist_avg, min_samples=10).fit(get_lengths(proj, gaps)).labels_

        if plot == True:
            for i in range(-1, clustering.max() + 1):
                plt.scatter(proj[clustering == i, 0], proj[clustering == i, 1])
            plt.show()

        ndim = clustering.max() + 1
        assert ndim <= self.ndim, "number of new periodicities do not correspond wiht original number"

        if self.ndim != 2:
            raise NotImplementedError("only tested for n == 2")

        boundaries = []
        periodicities = []

        n = 30

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
                return np.array(p)

            lin1 = np.linspace(l1[0], l1[1], num=n)
            lin2 = np.linspace(l2[0], l2[1], num=n)

            xr1 = np.array([np.array([f(l1), f(l2)]) for l1, l2 in zip(lin1, lin2)])

            lrange.append([l1, l2])
            xrange.append(xr1)

            avg_len = (abs(l1[1] - l1[0]) + abs(l2[1] - l2[0])) / 2

            boundaries.append([0, avg_len])
            periodicities.append(True)

        interps = []
        for i in range(self.ndim):

            points = []
            z = []

            #add boundaries
            lin = np.linspace(0, boundaries[i][1], num=n)
            z.append(lin)
            z.append(lin)

            range_high = np.array([lrange[i][0][1], lrange[i][1][1]])
            range_low = np.array([lrange[i][0][0], lrange[i][1][0]])

            points.append(xrange[i][:, 0, :])
            points.append(xrange[i][:, 1, :])

            #append other boundaries
            for j in range(self.ndim):
                if i == j:
                    continue

                z.append(np.zeros(n))
                z.append(np.ones(n) * boundaries[i][1])

                #potentially differently ordered
                arr = np.array(lrange[j])
                arr.sort(axis=1)
                range_high.sort()
                range_low.sort()

                high = abs(arr - range_high).sum(axis=1)
                high_amin = np.argmin(high)
                err1 = high[high_amin]

                low = abs(arr - range_low).sum(axis=1)
                low_amin = np.argmin(low)
                err2 = low[low_amin]

                if err1 < err2:
                    order = high_amin == 0
                else:
                    order = low_amin == 0

                if order:
                    points.append(xrange[j][:, 1, :])
                    points.append(xrange[j][:, 0, :])
                else:
                    points.append(xrange[j][:, 0, :])
                    points.append(xrange[j][:, 1, :])

            # interps.append(LinearNDInterpolator(np.vstack(points), np.hstack(z)))
            interps.append(Rbf(np.vstack(points)[:, 0], np.vstack(points)[:, 1], np.hstack(z)))

        #get the boundaries from most distal points in traj + some margin
        num = 50

        old_boundaries = []
        lspaces = []

        for i in range(ndim):
            a = trajs[:, :, i].min()
            b = trajs[:, :, i].max()
            d = (b - a) * 0.01
            a = a - d
            b = b + d
            old_boundaries.append([a, b])

            lspaces.append(np.linspace(a, b, num=num))

        interp_meshgrid = np.meshgrid(*lspaces)
        interp_mg = []
        for i in range(ndim):
            interp_mg.append(interps[i](*interp_meshgrid))

        if plot == True:

            for j in [0, 1]:
                # plt.contourf(interp_meshgrid[0], interp_meshgrid[1], c=interp_mg[i], cmap=plt.get_cmap('plasma'), s=2)
                plt.pcolor(interp_meshgrid[0],
                           interp_meshgrid[1],
                           interp_mg[j],
                           cmap=plt.get_cmap('Greys'),
                           vmax=interp_mg[j][~np.isnan(interp_mg[i])].max() * 2)

                for i in range(0, clustering.max() + 1):
                    vmax = proj[clustering != -1, 0].max()
                    vmin = proj[clustering != -1, 0].min()

                    plt.scatter(trajs[clustering == i, 0, 0],
                                trajs[clustering == i, 0, 1],
                                c=proj[clustering == i, 0],
                                vmax=vmax,
                                vmin=vmin,
                                s=5,
                                cmap=plt.get_cmap('plasma'))
                    plt.scatter(trajs[clustering == i, 1, 0],
                                trajs[clustering == i, 1, 1],
                                c=proj[clustering == i, 0],
                                vmax=vmax,
                                vmin=vmin,
                                s=5,
                                cmap=plt.get_cmap('plasma'))

                plt.show()

        return Metric(periodicities=periodicities,
                      boundaries=old_boundaries,
                      wrap_meshgrids=interp_mg,
                      wrap_boundaries=boundaries)


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

    @staticmethod
    def linear_combination(cv1, cv2, a=1, b=1):
        return lambda x, y: a * cv1(x, y) + b * cv2(x, y)
