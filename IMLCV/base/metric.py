from __future__ import annotations

from functools import partial

import alphashape

# import numpy as np
import jax.numpy as jnp
import jax.scipy as jsp
import numba
import numpy as np
from jax import jit
from matplotlib import pyplot as plt
from scipy.interpolate import RBFInterpolator
from shapely.geometry import MultiPoint, Point
from sklearn.cluster import DBSCAN


class Metric:
    """class to keep track of topology of given CV. Identifies the periodicitie of CVs and maps to unit square with correct peridicities"""

    def __init__(
        self,
        periodicities,
        bounding_box=None,
        map_meshgrids=None,
    ) -> None:
        if bounding_box is None:
            bounding_box = jnp.zeros((len(periodicities), 2), jnp.float32)
            bounding_box = bounding_box.at[:, 1].set(1.0)

        if isinstance(bounding_box, list):
            bounding_box = jnp.array(bounding_box, dtype=jnp.float32)

        if bounding_box.ndim == 1:
            bounding_box = jnp.reshape(bounding_box, (1, 2))

        if isinstance(periodicities, list):
            periodicities = jnp.array(periodicities)
        assert periodicities.ndim == 1

        self.map_meshgrids = map_meshgrids

        self.bounding_box = bounding_box
        self.periodicities = periodicities
        self.type = periodicities

        # self._boundaries = np.zeros(self.bounding_box.shape)
        # self._boundaries[:, 1] = 1

    @partial(jit, static_argnums=(0))
    def difference(self, x1, x2):
        """x1 and x2 should already be mapped."""
        xs = x2 - x1
        return self._periodic_wrap(xs, min=True)

    @partial(jit, static_argnums=(0, 2))
    def _periodic_wrap(self, xs, min=False):
        """Translate cvs such over unit cell.

        min=True calculates distances, False translates one vector inside box
        """

        coor = jnp.mod(xs, 1)  # between 0 and 1
        if min:
            coor = jnp.where(coor > 0.5, coor - 1, coor)  # between [-0.5,0.5]

        return jnp.where(self.periodicities, coor, xs)

    @partial(jit, static_argnums=(0))
    def map(self, y):
        """transform CVs to lie in unit square."""

        y = (y - self.bounding_box[:, 0]) / (
            self.bounding_box[:, 1] - self.bounding_box[:, 0]
        )

        if self.map_meshgrids is not None:

            y = y * (jnp.array(self.map_meshgrids[0].shape) - 1)
            y = jnp.array(
                [
                    jsp.ndimage.map_coordinates(wp, y, order=1)
                    for wp in self.map_meshgrids
                ]
            )

        return self._periodic_wrap(y, min=False)

    def __add__(self, other):
        assert isinstance(self, Metric)
        if other is None:
            return self

        assert isinstance(other, Metric)

        periodicities = jnp.hstack((self.periodicities, other.periodicities))
        bounding_box = jnp.vstack((self.bounding_box, other.bounding_box))

        if self.map_meshgrids is None and other.map_meshgrids is None:
            map_meshgrids = None
        elif (self.map_meshgrids is not None) and (other.map_meshgrids is not None):
            map_meshgrids = [*self.map_meshgrids, *other.map_meshgrids]
        else:
            raise NotImplementedError

        return Metric(
            periodicities=periodicities,
            bounding_box=bounding_box,
            map_meshgrids=map_meshgrids,
        )

    def grid(self, n, map=True, endpoints=None):
        """forms regular grid in mapped space. If coordinate is periodic, last rows are ommited.

        Args:
            n: number of points in each dim
            map: boolean. True: work in mapped space (default), False: calculate grid in space without metric
            endpoints: if

        Returns:
            meshgrid and vector with distances between points

        """

        if endpoints is None:
            endpoints = np.array(~self.periodicities)
        elif isinstance(endpoints, bool):
            endpoints = np.full(self.periodicities.shape, endpoints)

        if map:
            b = np.zeros(self.bounding_box.shape)
            b[:, 1] = 1
        else:
            b = self.bounding_box

        assert not (jnp.abs(b[:, 1] - b[:, 0]) < 1e-12).any(), "give proper boundaries"
        grid = [
            jnp.linspace(row[0], row[1], n, endpoint=endpoints[i])
            for i, row in enumerate(b)
        ]

        return grid

    @property
    def ndim(self):
        return len(self.periodicities)

    def _get_mask(self, tol=0.1, interp_mg=None):
        if interp_mg is None:
            assert self.map_meshgrids is not None
            interp_mg = self.map_meshgrids
        else:
            interp_mg = jnp.apply_along_axis(self.map, axis=0, arr=np.array(interp_mg))

        m = np.logical_or(
            *[np.logical_or(ip < (-tol), ip > (1 + tol)) for ip in interp_mg]
        )
        mask = np.ones(m.shape)
        mask[m] = np.nan
        return mask

    def update_metric(
        self, trajs, convex=True, fn=None, acc=30, trim=False, tol=0.1
    ) -> Metric:
        """find best fitting bounding box and get affine tranformation+ new
        boundaries."""

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

        dist_avg = bound.length / len(trajs)

        if convex:
            assert (
                np.array([bound.distance(p) for p in mpoints]).max() < acc * dist_avg
            ), "boundaries are not convex"

        proj = np.array(
            [
                [bound.project(Point(tr[0, :])), bound.project(Point(tr[1, :]))]
                for tr in trajs
            ]
        )

        def get_lengths(proj):
            b = np.copy(proj[:])
            projgaps = np.sort(np.hstack([b[:, 0], b[:, 1]]))

            # remove the gapps along boundary
            gaps = (projgaps[1:] - projgaps[:-1]) > 1 * dist_avg
            gaps = [projgaps[0:-1][gaps], projgaps[1:][gaps]]

            a = np.copy(proj[:])
            for i, j in list(zip(*gaps))[::-1]:
                a[a > i] -= j - i

            return a

        # sort pair
        as1 = proj.argsort(axis=1)
        proj = np.take_along_axis(proj, as1, axis=1)
        trajs = np.array([pair[argsort, :] for argsort, pair in zip(as1, trajs)])

        clustering = DBSCAN(eps=5 * dist_avg).fit(get_lengths(proj)).labels_

        if fn is not None:
            plt.clf()
            for i in range(-1, clustering.max() + 1):
                plt.scatter(proj[clustering == i, 0], proj[clustering == i, 1])
            plt.savefig(f"{fn}/coord_cluster_pre")

        # look for largest cluster and take it as starting point of indexing, shift other points cyclically
        index = np.argmax(np.bincount(clustering[clustering >= 0]))

        offset = proj[clustering == index, 0].min()
        proj[proj < offset] += bound.length

        as1 = proj.argsort(axis=1)
        proj = np.take_along_axis(proj, as1, axis=1)
        trajs = np.array([pair[argsort, :] for argsort, pair in zip(as1, trajs)])

        proj -= offset

        clustering = DBSCAN(eps=3 * dist_avg).fit(get_lengths(proj)).labels_

        if fn is not None:
            plt.clf()
            for i in range(-1, clustering.max() + 1):
                plt.scatter(proj[clustering == i, 0], proj[clustering == i, 1])
            plt.savefig(f"{fn}/coord_cluster")

        ndim = clustering.max() + 1
        assert (
            ndim <= self.ndim
        ), """number of new periodicities do not
        correspond wiht original number"""

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

            # boundaries.append([0, avg_len])
            boundaries.append([0, 1])

            periodicities.append(True)

        interps = []
        for i in range(self.ndim):

            points_arr = []
            z = []

            range_high = np.array([lrange[i][0][1], lrange[i][1][1]])
            range_low = np.array([lrange[i][0][0], lrange[i][1][0]])

            # # append other boundaries
            for j in range(self.ndim):
                if i == j:
                    continue

                z.append(np.zeros(n))
                z.append(np.ones(n) * boundaries[i][1])

                # potentially differently ordered
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
                    points_arr.append(xrange[j][:, 1, :])
                    points_arr.append(xrange[j][:, 0, :])
                else:
                    points_arr.append(xrange[j][:, 0, :])
                    points_arr.append(xrange[j][:, 1, :])

            interps.append(RBFInterpolator(np.vstack(points_arr), np.hstack(z)))

        # get the boundaries from most distal points in traj + some margin
        num = 100

        old_boundaries = []
        lspaces = []

        for i in range(ndim):
            if trim:
                a = trajs[:, :, i].min()
                b = trajs[:, :, i].max()
                d = (b - a) * 0.05
                a = a - d
                b = b + d

            else:
                a = self.bounding_box[i][0]
                b = self.bounding_box[i][1]

            old_boundaries.append([a, b])
            lspaces.append(np.linspace(a, b, num=num))

        dims = [len(l) for l in lspaces]
        interp_meshgrid = np.array(np.meshgrid(*lspaces, indexing="ij"))
        imflat = interp_meshgrid.reshape(ndim, -1).T
        interp_mg = []
        for i in range(ndim):

            arr_flat = interps[i](imflat)
            arr = arr_flat.reshape(dims)
            interp_mg.append(arr)

        new_metric = Metric(
            periodicities=periodicities,
            bounding_box=old_boundaries,
            map_meshgrids=interp_mg,
        )

        mask = new_metric._get_mask(tol=tol)
        interp_mg = [im * mask for im in interp_mg]

        if fn is not None:

            for j in [0, 1]:

                plt.clf()
                plt.pcolor(
                    interp_meshgrid[0, :],
                    interp_meshgrid[1, :],
                    interp_mg[j] * mask,
                    # cmap=plt.get_cmap('Greys')
                )

                plt.colorbar()

                for i in range(0, clustering.max() + 1):
                    vmax = proj[clustering != -1, 0].max()
                    vmin = proj[clustering != -1, 0].min()

                    plt.scatter(
                        trajs[clustering == i, 0, 0],
                        trajs[clustering == i, 0, 1],
                        c=proj[clustering == i, 0],
                        vmax=vmax,
                        vmin=vmin,
                        s=5,
                        cmap=plt.get_cmap("plasma"),
                    )
                    plt.scatter(
                        trajs[clustering == i, 1, 0],
                        trajs[clustering == i, 1, 1],
                        c=proj[clustering == i, 0],
                        vmax=vmax,
                        vmin=vmin,
                        s=5,
                        cmap=plt.get_cmap("plasma"),
                    )

                plt.savefig(f"{fn}/coord{j}")

        return new_metric


class MetricUMAP(Metric):
    def __init__(self, periodicities, bounding_box=None) -> None:
        super().__init__(periodicities=periodicities, bounding_box=bounding_box)

        bb = np.array(self.bounding_box)
        per = np.array(self.periodicities)

        # @numba.njit
        # def map(y):

        #     return (y - bb[:, 0]) / (
        #         bb[:, 1] - bb[:, 0])

        @numba.njit
        def _periodic_wrap(xs, min=False):
            coor = np.mod(xs, 1)  # between 0 and 1
            if min:
                # between [-0.5,0.5]
                coor = np.where(coor > 0.5, coor - 1, coor)

            return np.where(per, coor, xs)

        @numba.njit
        def g(x, y):
            # r1 = map(x)
            # r2 = map(y)

            return _periodic_wrap(x - y, min=True)

        @numba.njit
        def val_and_grad(x, y):
            r = g(x, y)
            d = np.sqrt(np.sum(r**2))

            return d, r / (d + 1e-6)

        self.umap_f = val_and_grad


class hyperTorus(Metric):
    def __init__(self, n) -> None:
        periodicities = [True for _ in range(n)]
        boundaries = jnp.zeros((n, 2))
        boundaries = boundaries.at[:, 0].set(-jnp.pi)
        boundaries = boundaries.at[:, 1].set(jnp.pi)

        super().__init__(periodicities, boundaries)
