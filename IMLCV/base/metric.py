from __future__ import annotations

import jax.numpy as jnp
import numba
import numpy as np

from IMLCV.base.CV import Metric


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
