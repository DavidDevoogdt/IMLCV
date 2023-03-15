from __future__ import annotations

import dataclasses
import tempfile
from abc import abstractmethod
from collections.abc import Callable

# from IMLCV.base.CV import CV, CvTrans
from functools import partial
from importlib import import_module
from typing import TYPE_CHECKING

import jax
import jax.flatten_util
import jax.lax

# import numpy as np
import jax.numpy as jnp
import jax.scipy.optimize
import jax_dataclasses as jdc
import jaxopt
import jaxopt.objective
import numpy as np
import tensorflow
import tensorflow as tfl
from flax import linen as nn
from flax.linen.linear import Dense
from jax import Array, jacfwd, jit, vmap
from jax.experimental.jax2tf import call_tf
from keras.api._v2 import keras as KerasAPI
from molmod.units import angstrom


if TYPE_CHECKING:
    pass

keras: KerasAPI = import_module("tensorflow.keras")

import dataclasses

import distrax
import jax.numpy as jnp
import numba
import numpy as np
from tensorflow_probability.substrates import jax as tfp

######################################
#        Data types                  #
######################################


@jdc.pytree_dataclass
class SystemParams:
    coordinates: Array
    cell: Array | None = None

    def __post_init__(self):
        if isinstance(self.coordinates, Array):
            self.__dict__["coordinates"] = jnp.array(self.coordinates)
        if isinstance(self.cell, Array):
            self.__dict__["cell"] = jnp.array(self.cell)

    def __getitem__(self, slices):
        return SystemParams(
            coordinates=self.coordinates[slices],
            cell=(self.cell[slices] if self.cell is not None else None),
        )

    def __iter__(self):
        if not self.batched:
            yield self
            return
        for i in range(self.coordinates.shape[0]):
            yield SystemParams(
                coordinates=self.coordinates[i, :, :],
                cell=self.cell[i, :, :] if self.cell is not None else None,
            )
        return

    @property
    def batched(self):
        return len(self.coordinates.shape) == 3

    @property
    def shape(self):
        return self.coordinates.shape

    def __add__(self, other):
        assert isinstance(other, SystemParams)
        if self.cell is None:
            assert other.cell is None

        s = self.batch()
        o = other.batch()

        assert s.shape[1:] == o.shape[1:]

        return SystemParams(
            coordinates=jnp.vstack([s.coordinates, o.coordinates]),
            cell=None if self.cell is None else jnp.vstack([s.cell, o.cell]),
        )

    def __str__(self):
        string = f"coordinates shape: \n{self.coordinates.shape}"
        if self.cell is not None:
            string += f"\n cell [Angstrom]:\n{self.cell/angstrom }"

        return string

    def batch(self) -> SystemParams:
        if self.batched:
            return self

        return SystemParams(
            coordinates=jnp.array([self.coordinates]),
            cell=self.cell if self.cell is None else jnp.array([self.cell]),
        )

    def unbatch(self) -> SystemParams:
        if not self.batched:
            return self
        assert self.shape[0] == 1
        return SystemParams(
            coordinates=self.coordinates[0, :],
            cell=self.cell if self.cell is None else self.cell[0, :],
        )

    def angles(self, deg=True) -> Array:
        @partial(vmap, in_axes=(None, 0))
        @partial(vmap, in_axes=(0, None))
        def ang(x1, x2):
            a = jnp.arccos(jnp.dot(x1, x2) / (jnp.dot(x1, x1) * jnp.dot(x2, x2)) ** 0.5)
            if deg:
                a *= 180 / jnp.pi
            return a

        return ang(self.cell, self.cell)

    @staticmethod
    def apply_g_inner(
        sp: SystemParams,
        func,
        r_cut,
        ijk=(None, None, None),
        exclude_self=True,
    ):

        i, j, k = ijk

        if ijk != (None, None, None):
            assert sp.cell is not None
            pos = (
                sp.coordinates
                + i * sp.cell[0, :]
                + j * sp.cell[1, :]
                + k * sp.cell[2, :]
            )
        else:
            pos = sp.coordinates

        norm2 = jnp.sum(pos**2, axis=1)
        index_j = jnp.ones_like(norm2, dtype=jnp.int32).cumsum() - 1

        bools = norm2 < r_cut**2

        if exclude_self:
            bools = jnp.logical_and(
                bools,
                jnp.logical_not(vmap(lambda x: jnp.allclose(x, jnp.zeros((3,))))(pos)),
            )

        # def _g(bools, pos, index_j):

        true_val = vmap(lambda a, b: func(a, b, i, j, k))(pos, index_j)
        false_val = jax.tree_map(
            lambda a: jnp.zeros_like(a),
            true_val,
        )

        val = vmap(
            lambda b, x, y: jax.tree_map(
                lambda t, f: jnp.where(b, t, f),
                x,
                y,
            )
        )(
            bools,
            true_val,
            false_val,
        )

        return bools, val

    @staticmethod
    def _get_num_per_images(cell, r_cut):

        if cell is None or r_cut is None:
            return None

        # orthogonal distance for number of blocks
        v1, v2, v3 = cell[0, :], cell[1, :], cell[2, :]

        # sorted from short to long
        e3 = v3 / jnp.linalg.norm(v3)
        e1 = jnp.cross(v2, e3)
        e1 /= jnp.linalg.norm(e1)
        e2 = jnp.cross(e3, e1)

        proj = vmap(vmap(jnp.dot, in_axes=(0, None)), in_axes=(None, 0))(
            jnp.array([e1, e2, e3]), jnp.array([v1, v2, v3])
        )

        bounds = jnp.ceil(jnp.sum(jnp.abs(jnp.linalg.inv(proj)) * r_cut, axis=1))

        return bounds

    # @partial(jit, static_argnums=(1, 2, 3, 7, 8, 9))
    # # @jit
    def _get_neighbour_list(
        self,
        r_cut,
        r_skin: float = 0.0,
        z_array: Array | None = None,
        z_unique: Array | None = None,
        num_neighs: int | None = None,
        nxyz: list[int] | None = None,
    ) -> tuple[bool, SystemParams, NeighbourList | None]:

        if r_cut is None:
            return self, None

        sp = self.canoncialize()

        b = True

        # cannot be jitted
        if sp.cell is not None:
            if nxyz is None:

                if sp.batched:
                    nxyz = [
                        int(i)
                        for i in jnp.max(
                            vmap(
                                lambda sp: SystemParams._get_num_per_images(
                                    sp.cell, r_cut + r_skin
                                )
                            )(sp),
                            axis=0,
                        ).tolist()
                    ]
                else:

                    nxyz = [
                        int(i)
                        for i in SystemParams._get_num_per_images(
                            sp.cell, r_cut + r_skin
                        ).tolist()
                    ]
            else:
                b = (
                    b
                    and (
                        SystemParams._get_num_per_images(sp.cell, r_cut)
                        <= jnp.array(nxyz)
                    ).all()
                )  # only check r_cut, because we are retracing
            nx, ny, nz = nxyz
            bx = jnp.arange(-nx, nx + 1)
            by = jnp.arange(-ny, ny + 1)
            bz = jnp.arange(-nz, nz + 1)
        else:
            bnds = None
            bx, by, bz = None, None, None

        @jit
        def func(r_ij, index, i, j, k):
            return (
                jnp.linalg.norm(r_ij),
                jnp.array(index),
                jnp.array([i, j, k]),
            )

        @jit
        def func2(r_ij, index, i, j, k):
            return (
                jnp.linalg.norm(r_ij),
                jnp.array(index),
                None,
            )

        @partial(vmap, in_axes=(None, 0))
        def res(sp, center_coordinates):

            if center_coordinates is not None:
                sp_center = SystemParams(sp.coordinates - center_coordinates, sp.cell)
            else:
                sp_center = SystemParams(sp.coordinates, sp.cell)

            sp_center = sp_center.wrap_positions(True)

            # sp_center = SystemParams.canoncialize(sp_center, b)

            if sp_center.cell is None:
                _, (r, atoms, _) = SystemParams.apply_g_inner(
                    sp=sp_center,
                    func=func2,
                    r_cut=jnp.inf,
                    exclude_self=False,
                )
                idx = jnp.argsort(r)

                return r[idx] < r_cut + r_skin, r[idx], atoms[idx], None

            _, (r, atoms, indices) = (
                vmap(
                    vmap(
                        vmap(
                            lambda i, j, k: SystemParams.apply_g_inner(
                                sp=sp_center,
                                func=func,
                                r_cut=jnp.inf,
                                ijk=(i, j, k),
                                exclude_self=False,
                            ),
                            in_axes=(0, None, None),
                            out_axes=0,
                        ),
                        in_axes=(None, 0, None),
                        out_axes=1,
                    ),
                    in_axes=(None, None, 0),
                    out_axes=2,
                )
            )(bx, by, bz)

            r, atoms, indices = (
                jnp.reshape(r, (-1,)),
                jnp.reshape(atoms, (-1)),
                jnp.reshape(indices, (-1, 3)),
            )
            idx = jnp.argsort(r)

            return r[idx] < r_cut + r_skin, r[idx], atoms[idx], indices[idx, :]

        @jit
        def _f(sp):
            bools, r, a, ijk = res(sp, sp.coordinates)
            num_neighs = jnp.max(jnp.sum(bools, axis=1))

            return num_neighs, bools, r, a, ijk

        @partial(jit, static_argnums=0)
        def take(num_neighs, r, a, ijk):
            r, a, = (
                r[:, 0:num_neighs],
                a[:, 0:num_neighs],
            )
            if ijk is not None:
                ijk = ijk[:, 0:num_neighs]
            return r, a, ijk

        if sp.batched:
            _f = vmap(_f)
            take = vmap(take, in_axes=(None, 0, 0, 0))

        nn, _, r, a, ijk = _f(sp)

        if sp.batched:
            nn = jnp.max(nn)  # ingore: type

        if num_neighs is None:
            num_neighs = int(nn)
        else:
            b = jnp.logical_and(b, nn <= num_neighs)

        r, a, ijk = take(num_neighs, r, a, ijk)

        return (
            b,
            sp,
            NeighbourList(
                r_cut=r_cut,
                r_skin=r_skin,
                atom_indices=a,
                ijk_indices=ijk,
                z_array=z_array,
                z_unique=z_unique,
                sp_orig=sp,
                nxyz=nxyz,
            ),
        )

    def get_neighbour_list(
        self,
        r_cut,
        r_skin=0.0,
        z_array: Array | None = None,
    ) -> tuple[SystemParams, NeighbourList | None]:

        b, sp, nl = self._get_neighbour_list(
            r_cut=r_cut,
            r_skin=r_skin,
            z_array=z_array,
            z_unique=jnp.unique(z_array) if z_array is not None else None,
        )
        return sp, nl

    def apply_fun_neighbour(
        self,
        r_cut,
        center_coordinates: Array | None = None,
        func=lambda r_ij, atom_index_j: (r_ij, atom_index_j),
        exclude_self=False,
        reduce_neigh=lambda x: jnp.sum(x, axis=0),
    ):
        """
        the function g should be jax jttable, and takes as arguments the relative vector r_ij and its atom index
        """

        func2 = lambda a, b, i, j, k: func(a, b)

        assert not self.batched

        if center_coordinates is not None:
            sp = SystemParams(
                self.coordinates - center_coordinates, self.cell
            ).canoncialize(min=True)
        else:
            sp = SystemParams(self.coordinates, self.cell).canoncialize(min=True)

        if sp.cell is None:
            return SystemParams.apply_g_inner(
                sp=sp, func=func2, r_cut=r_cut, exclude_self=exclude_self
            )

        bounds = SystemParams._get_num_per_images(sp.cell, r_cut)

        def reduction(bools, vals):
            return jnp.sum(bools), jax.tree_map(reduce_neigh, vals)

        init_val = jax.tree_map(
            lambda o: jnp.zeros(shape=o.shape, dtype=o.dtype),
            jax.eval_shape(
                lambda x: reduction(
                    *SystemParams.apply_g_inner(
                        sp=x, func=func2, r_cut=r_cut, exclude_self=exclude_self
                    )
                ),
                sp,
            ),
        )

        def k_loop(k, ifarg):

            (i, j), (n_prev, tree_prev) = ifarg

            n, vals = reduction(
                *SystemParams.apply_g_inner(
                    sp, func2, r_cut, (i, j, k), exclude_self=exclude_self
                )
            )

            farg = (
                n_prev + n,
                jax.tree_map(
                    lambda x, y: reduce_neigh(jnp.stack([x, y])), tree_prev, vals
                ),
            )
            return (i, j), farg

        def j_loop(j, ifarg):

            i, farg = ifarg

            (i, j), farg = jax.lax.fori_loop(
                lower=-bounds[2],
                upper=bounds[2] + 1,
                body_fun=k_loop,
                init_val=((i, j), farg),
            )

            return i, farg

        def i_loop(i, farg):

            i, farg = jax.lax.fori_loop(
                lower=-bounds[1],
                upper=bounds[1] + 1,
                body_fun=j_loop,
                init_val=(i, farg),
            )

            return farg

        return jax.lax.fori_loop(
            lower=-bounds[0],
            upper=bounds[0] + 1,
            body_fun=i_loop,
            init_val=init_val,
        )

    def apply_fun_neighbour_pairs(
        self,
        r_cut,
        center_coordinates: Array | None = None,
        disable_center_map=False,
        func=lambda r_ij, r_ik, atom_index_j, atom_index_k: 1,
        exclude_self=True,
        mask1=None,
        mask2=None,
        unique=True,
    ):
        """
        function that loops over all pairs of atoms within cutoff radius.

        if sp2 is none, all pairs with itself are made, otherwise all pairs with sp2

        center_coordinates

        """

        def h(r_ij, r_ik, atom_index_j, atom_index_k):
            nonlocal func

            n1 = jnp.linalg.norm(r_ik) != 0
            n2 = jnp.linalg.norm(r_ij) != 0
            n12 = jnp.linalg.norm(r_ij - r_ik) != 0.0

            bools = jnp.zeros_like(n1) == 0
            if exclude_self:
                bools = jnp.logical_and(bools, n1)
                bools = jnp.logical_and(bools, n2)
            if unique:
                bools = jnp.logical_and(bools, n12)

            true_val = func(r_ij, r_ik, atom_index_j, atom_index_k)
            false_val = jax.tree_map(
                lambda a: jnp.zeros_like(a),
                true_val,
            )

            val = jax.tree_map(
                lambda t, f: jnp.where(bools, t, f),
                true_val,
                false_val,
            )

            return (bools * 1, val)

        def apply(center_coordinates: Array | None, sp: SystemParams):

            if center_coordinates is not None:
                sp = SystemParams(
                    sp.coordinates - center_coordinates, sp.cell
                ).canoncialize()
            else:
                sp = SystemParams(sp.coordinates, sp.cell).canoncialize()

            if mask1 is not None:
                sp1 = sp[mask1]
            else:
                sp1 = sp

            if mask2 is not None:
                sp2 = sp[mask2]
            else:
                sp2 = sp

            i, (j, (k, val)) = sp1.apply_fun_neighbour(
                r_cut,
                func=lambda r_ij, atom_index_j: sp2.apply_fun_neighbour(
                    r_cut,
                    func=lambda r_ik, atom_index_k: h(
                        r_ij, r_ik, atom_index_j, atom_index_k
                    ),
                ),
            )

            return k, val

        assert not self.batched

        out = apply(center_coordinates, self)

        return out

    @jit
    def minkowski_reduce(self) -> SystemParams:
        """base on code from ASE: https://wiki.fysik.dtu.dk/ase/_modules/ase/geometry/minkowski_reductsp.cellion.html#minkowski_reduce"""
        if self.cell is None:
            return self

        import itertools

        TOL = 1e-12

        @partial(jit, static_argnums=(0,))
        def cycle_checker(d):
            assert d in [2, 3]
            max_cycle_length = {2: 60, 3: 3960}[d]
            return jnp.zeros((max_cycle_length, 3 * d), dtype=int)

        @jit
        def add_site(visited, H):
            # flatten array for simplicity
            H = H.ravel()

            # check if site exists
            found = (visited == H).all(axis=1).any()

            # shift all visited sites down and place current site at the top
            visited = jnp.roll(visited, 1, axis=0)
            visited = visited.at[0].set(H)
            return visited, found

        @jit
        def reduction_gauss(B, hu, hv):
            """Calculate a Gauss-reduced lattice basis (2D reduction)."""
            visited = cycle_checker(2)
            u = hu @ B
            v = hv @ B

            def body(vals):
                u, v, hu, hv, visited, found, i = vals

                # for it in range(MAX_IT):
                x = jnp.array(jnp.round(jnp.dot(u, v) / jnp.dot(u, u)), dtype=jnp.int32)
                hu, hv = hv - x * hu, hu
                u = hu @ B
                v = hv @ B
                site = jnp.array([hu, hv])

                visited, found = add_site(visited=visited, H=site)

                return (u, v, hu, hv, visited, found, i + 1)

            def cond_fun(vals):
                u, v, hu, hv, visited, found, i = vals

                return jnp.logical_not(
                    jnp.logical_and(
                        jnp.logical_or(jnp.dot(u, u) >= jnp.dot(v, v), found), i != 0
                    )
                )

            u, v, hu, hv, visited, found, i = jax.lax.while_loop(
                cond_fun=cond_fun,
                body_fun=body,
                init_val=(u, v, hu, hv, visited, False, 0),
            )

            return hv, hu

        @jit
        def relevant_vectors_2D(u, v):
            cs = jnp.array(list(itertools.product([-1, 0, 1], repeat=2)))
            vs = cs @ jnp.array([u, v])
            indices = jnp.argsort(jnp.linalg.norm(vs, axis=1))[:7]
            return vs[indices], cs[indices]

        @jit
        def closest_vector(t0, u, v):
            t = t0
            a = jnp.zeros(2, dtype=int)
            rs, cs = relevant_vectors_2D(u, v)

            dprev = float("inf")
            ds = jnp.linalg.norm(rs + t, axis=1)
            index = jnp.argmin(ds)

            def body_fun(vals):
                ds, index, a, _, t, i = vals

                dprev = ds[index]

                r = rs[index]
                kopt = jnp.array(
                    jnp.round(-jnp.dot(t, r) / jnp.dot(r, r)), dtype=jnp.int32
                )
                a += kopt * cs[index]
                t = t0 + a[0] * u + a[1] * v

                ds = jnp.linalg.norm(rs + t, axis=1)
                index = jnp.argmin(ds)

                return ds, index, a, dprev, t, i + 1

            def cond_fun(vals):
                ds, index, a, dprev, t, i = vals

                return jnp.logical_not(jnp.logical_or(index == 0, ds[index] >= dprev))

            ds, index, a, dprev, t, i = jax.lax.while_loop(
                cond_fun=cond_fun,
                body_fun=body_fun,
                init_val=(ds, index, a, dprev, t0, 0),
            )

            return a

        @jit
        def reduction_full(B):
            """Calculate a Minkowski-reduced lattice basis (3D reduction)."""
            # init
            visited = cycle_checker(d=3)
            H = jnp.eye(3, dtype=int)
            norms = jnp.linalg.norm(B, axis=1)

            def body(vals):
                H, norms, visited, _, i = vals
                # for it in range(MAX_IT):
                # Sort vectors by norm
                H = H[jnp.argsort(norms)]

                # Gauss-reduce smallest two vectors
                hw = H[2]
                hu, hv = reduction_gauss(B, H[0], H[1])
                H = jnp.array([hu, hv, hw])
                R = H @ B

                # Orthogonalize vectors using Gram-Schmidt
                u, v, _ = R
                X = u / jnp.linalg.norm(u)
                Y = v - X * jnp.dot(v, X)
                Y /= jnp.linalg.norm(Y)

                # Find closest vector to last element of R
                pu, pv, pw = R @ jnp.array([X, Y]).T
                nb = closest_vector(pw, pu, pv)

                # Update basis
                H = H.at[2].set(jnp.array([nb[0], nb[1], 1]) @ H)
                R = H @ B

                norms = jnp.linalg.norm(R, axis=1)

                visited, found = add_site(visited, H)

                return H, norms, visited, found, i + 1

            def cond_fun(vals):
                _, norms, _, found, i = vals

                return jnp.logical_not(
                    jnp.logical_and(jnp.logical_or(norms[2] >= norms[1], found), i != 0)
                )

            H, norms, visited, found, i = jax.lax.while_loop(
                cond_fun, body, (H, norms, visited, False, 0)
            )

            return H @ B, H

        @jit
        def is_minkowski_reduced(cell):
            """Tests if a cell is Minkowski-reduced.

            Parameters:

            cell: array
                The lattice basis to test (in row-vector format).
            pbc: array, optional
                The periodic boundary conditions of the cell (Default `True`).
                If `pbc` is provided, only periodic cell vectors are tested.

            Returns:

            is_reduced: bool
                True if cell is Minkowski-reduced, False otherwise.
            """

            """These conditions are due to Minkowski, but a nice description in English
            can be found in the thesis of Carine Jaber: "Algorithmic approaches to
            Siegel's fundamental domain", https://www.theses.fr/2017UBFCK006.pdf
            This is also good background reading for Minkowski reduction.

            0D and 1D cells are trivially reduced. For 2D cells, the conditions which
            an already-reduced basis fulfil are:
            |b1| ≤ |b2|
            |b2| ≤ |b1 - b2|
            |b2| ≤ |b1 + b2|

            For 3D cells, the conditions which an already-reduced basis fulfil are:
            |b1| ≤ |b2| ≤ |b3|

            |b1 + b2|      ≥ |b2|
            |b1 + b3|      ≥ |b3|
            |b2 + b3|      ≥ |b3|
            |b1 - b2|      ≥ |b2|
            |b1 - b3|      ≥ |b3|
            |b2 - b3|      ≥ |b3|
            |b1 + b2 + b3| ≥ |b3|
            |b1 - b2 + b3| ≥ |b3|
            |b1 + b2 - b3| ≥ |b3|
            |b1 - b2 - b3| ≥ |b3|
            """

            A = jnp.array(
                [
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 1, 0],
                    [1, 0, 1],
                    [0, 1, 1],
                    [1, -1, 0],
                    [1, 0, -1],
                    [0, 1, -1],
                    [1, 1, 1],
                    [1, -1, 1],
                    [1, 1, -1],
                    [1, -1, -1],
                ]
            )
            lhs = jnp.linalg.norm(A @ cell, axis=1)
            norms = jnp.linalg.norm(cell, axis=1)
            rhs = norms[jnp.array([0, 1, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2])]

            return (lhs >= rhs - TOL).all()

        @jit
        def minkowski_reduce(cell):
            """Calculate a Minkowski-reduced lattice basis.  The reduced basis
            has the shortest possible vector lengths and has
            norm(a) <= norm(b) <= norm(c).

            Implements the method described in:

            Low-dimensional Lattice Basis Reduction Revisited
            Nguyen, Phong Q. and Stehlé, Damien,
            ACM Trans. Algorithms 5(4) 46:1--46:48, 2009
            :doi:`10.1145/1597036.1597050`

            Parameters:

            cell: array
                The lattice basis to reduce (in row-vector format).
            pbc: array, optional
                The periodic boundary conditions of the cell (Default `True`).
                If `pbc` is provided, only periodic cell vectors are reduced.

            Returns:

            rcell: array
                The reduced lattice basis.
            op: array
                The unimodular matrix transformation (rcell = op @ cell).
            """

            reduced = jax.lax.cond(
                is_minkowski_reduced(cell=cell),
                lambda: cell,
                lambda: reduction_full(cell)[1] @ cell,
            )

            # make righ handed
            # reduced = jnp.diag(jnp.sign(jnp.sum(jnp.sign(reduced), axis=1))) @ reduced
            return reduced

        if self.batched:
            cell = vmap(minkowski_reduce)(self.cell)
        else:
            cell = minkowski_reduce(self.cell)

        return SystemParams(coordinates=self.coordinates, cell=cell)

    @partial(jit, static_argnames=["min"])
    def wrap_positions(self, min=False) -> SystemParams:
        """wrap pos to lie within unit cell"""

        f = lambda x, y: SystemParams._wrap_pos(cell=x, coordinates=y, min=min)
        if self.batched:
            f = vmap(f)

        return SystemParams(
            coordinates=f(self.cell, self.coordinates),
            cell=self.cell,
        )

    @staticmethod
    @partial(jit, static_argnames=["min"])
    def _wrap_pos(cell: Array | None, coordinates: Array, min=False) -> Array:
        if cell is None:
            return coordinates

        trans = vmap(vmap(jnp.dot, in_axes=(0, None)), in_axes=(None, 0))(
            vmap(lambda x: x / jnp.linalg.norm(x))(cell), jnp.eye(3)
        )

        @partial(vmap)
        def proj(x):
            return jnp.linalg.inv(trans) @ x

        @partial(vmap)
        def deproj(x):
            return trans @ x

        scaled = proj(coordinates)

        norms = jnp.linalg.norm(cell, axis=1)

        reduced = jax.lax.cond(
            min,
            lambda: (jnp.mod(scaled / norms + 0.5, 1) - 0.5) * norms,
            lambda: jnp.mod(scaled, norms),
        )

        return deproj(reduced)

    @partial(jit, static_argnames=["min"])
    def canoncialize(self, min=False) -> SystemParams:

        mr: SystemParams = self.minkowski_reduce()

        return mr.wrap_positions(min)

    def min_distance(self, index_1, index_2):
        assert self.batched is False

        sp = self.canoncialize()  # necessary if cell is skewed
        coor1 = sp.coordinates[index_1, :]
        coor2 = sp.coordinates[index_2, :]

        @partial(vmap, in_axes=(None, None, 0))
        @partial(vmap, in_axes=(None, 0, None))
        @partial(vmap, in_axes=(0, None, None))
        def dist(n0, n1, n2):
            return jnp.linalg.norm(
                coor2
                - coor1
                + n0 * sp.cell[0, :]
                + n1 * sp.cell[1, :]
                + n2 * sp.cell[2, :]
            )

        ind = jnp.array([-1, 0, 1])
        return jnp.min(dist(ind, ind, ind))


@jdc.pytree_dataclass
class NeighbourList:
    r_cut: jdc.Static[jnp.floating]
    atom_indices: Array
    z_array: Array | None
    z_unique: Array
    r_skin: jdc.Static[jnp.floating]
    sp_orig: SystemParams | None = None
    ijk_indices: Array | None = None
    nxyz: jdc.Static[list[int] | None] = None

    def _pos(self, sp: SystemParams):
        @partial(vmap, in_axes=(0, 0, None, None))
        def _transform(ijk, a, spi, sp):
            out = spi[a].coordinates
            if ijk is not None:
                (i, j, k) = ijk
                out = out + sp.cell[0, :] * i + sp.cell[1, :] * j + sp.cell[2, :] * k
            return out

        @partial(vmap, in_axes=(0, 0, 0, None))
        def _recover(center, ijk, a, sp):

            return _transform(
                ijk,
                a,
                SystemParams(sp.coordinates - center, sp.cell).canoncialize(min=True),
                sp,
            )

        if sp.batched:
            assert self.batched
            _recover = vmap(_recover, in_axes=(0, 0, 0, 0))

        return _recover(sp.coordinates, self.ijk_indices, self.atom_indices, sp)

    def apply_fun_neighbour(
        self,
        sp: SystemParams,
        r_cut=None,
        func=lambda r_ij, atom_index_j: (jnp.linalg.norm(r_ij), r_ij, atom_index_j),
        fill_value=0,
        reduce="full",  # or 'z' or 'none'
        split_z=False,  #
        exclude_self=False,
    ):
        @NeighbourList.batch_sp_nl
        def _apply_fun_neighbour(sp: SystemParams, nl: NeighbourList):

            if r_cut is None:
                r_cut is nl.r_cut

            pos = nl._pos(sp)
            ind = nl.atom_indices
            r = jnp.linalg.norm(pos, axis=-1)

            bools = r**2 < nl.r_cut**2

            if exclude_self:
                bools = jnp.logical_and(bools, r**2 != 0.0)

            true_val = vmap(vmap(func))(pos, ind)
            false_val = jax.tree_map(
                lambda a: jnp.zeros_like(a) + fill_value,
                true_val,
            )

            def _get(bools):
                def _red(bools):

                    val = vmap(
                        lambda b, x, y: jax.tree_map(
                            lambda t, f: vmap(jnp.where)(b, t, f), x, y
                        )
                    )(bools, true_val, false_val)

                    if reduce == "full":
                        return jax.tree_map(
                            lambda x: jnp.sum(x, axis=(1)), (bools, val)
                        )
                    elif reduce == "z":
                        return jax.tree_map(
                            lambda x: jnp.sum(x, axis=(0, 1)), (bools, val)
                        )
                    elif reduce == "none":
                        return bools, val
                    else:
                        raise ValueError(
                            f"unknown value {reduce} for reduce argument of neighbourghfunction, try 'none','z' or 'full'"
                        )

                if not reduce == "z":
                    return _red(bools=bools)

                @partial(vmap, in_axes=(0, None))
                def _f(u, a):

                    b = vmap(
                        lambda t, f: jnp.where(a == u, t, f),
                        in_axes=(1, 1),
                        out_axes=1,
                    )(bools, jnp.full_like(bools, False))

                    return _red(bools=b)

                return _f(self.z_unique, self.z_array)

            if not split_z:
                return _get(bools)

            assert nl.z_array is not None, "provide z_array to neighbourlist"

            @partial(vmap, in_axes=(0, None), out_axes=1)
            def sel(u1, at):
                @vmap
                def _b(at):
                    bools_z1 = vmap(lambda x: x == u1)(at)
                    return bools_z1

                return _get(jnp.logical_and(_b(at), bools))

            return sel(nl.z_unique, nl.z_array[ind])

        return _apply_fun_neighbour(sp, self)

    def apply_fun_neighbour_pair(
        self,
        sp: SystemParams,
        func_single=lambda r_ij, atom_index_j: (1,),
        func_double=lambda r_ij, atom_index_j, data_j, r_ik, atom_index_k, data_k: (
            r_ij,
            atom_index_j,
            r_ik,
            atom_index_k,
        ),
        r_cut=None,
        fill_value=0,
        reduce="full",  # or 'z' or 'none'
        split_z=False,  #
        exclude_self=True,
        unique=True,
    ):
        @NeighbourList.batch_sp_nl
        def _apply_fun_neighbour_pair(sp: SystemParams, nl: NeighbourList):
            pos = nl._pos(sp)
            ind = nl.atom_indices

            bools, data_single = nl.apply_fun_neighbour(
                sp=sp,
                func=func_single,
                r_cut=r_cut,
                reduce="none",
                fill_value=fill_value,
                exclude_self=exclude_self,
            )

            out_ijk = vmap(
                lambda x, y, z: vmap(
                    vmap(func_double, in_axes=(0, 0, 0, None, None, None)),
                    in_axes=(None, None, None, 0, 0, 0),
                )(x, y, z, x, y, z)
            )(pos, ind, data_single)

            bools = vmap(
                lambda b: vmap(
                    vmap(jnp.logical_and, in_axes=(0, None)),
                    in_axes=(None, 0),
                )(b, b)
            )(bools)

            if unique:
                bools = vmap(lambda b: b.at[jnp.diag_indices_from(b)].set(False))(bools)

            out_ijk_f = jax.tree_map(
                lambda o: jnp.full_like(o, fill_value),
                out_ijk,
            )

            def get(bools):
                def _red(bools):
                    val = jax.tree_map(
                        lambda t, f: vmap(
                            vmap(vmap(lambda x, y, z: jnp.where(x, y, z)))
                        )(bools, t, f),
                        out_ijk,
                        out_ijk_f,
                    )

                    if reduce == "full":
                        return jax.tree_map(
                            lambda x: jnp.sum(x, axis=(1, 2)), (bools, val)
                        )
                    elif reduce == "z":
                        return jax.tree_map(
                            lambda x: jnp.sum(x, axis=(0, 1, 2)), (bools, val)
                        )
                    elif reduce == "none":
                        return bools, val
                    else:
                        raise ValueError(
                            f"unknown value {reduce} for reduce argument of neighbourghfunction, try 'none','z' or 'full'"
                        )

                if not reduce == "z":
                    return _red(bools=bools)

                @partial(vmap, in_axes=(0, None))
                def _f(u, a):

                    b = vmap(
                        vmap(
                            lambda t, f: jnp.where(a == u, t, f),
                            in_axes=(1, 1),
                            out_axes=1,
                        ),
                        in_axes=(2, 2),
                        out_axes=2,
                    )(bools, jnp.full_like(bools, False))

                    return _red(bools=b)

                return _f(self.z_unique, self.z_array)

            if not split_z:
                return get(bools)

            assert nl.z_array is not None, "provide z_array to neighbourlist"

            @partial(vmap, in_axes=(None, 0, None), out_axes=2)
            @partial(vmap, in_axes=(0, None, None), out_axes=1)
            def sel(u1, u2, at):
                @partial(vmap, in_axes=(0))
                def _b(at):
                    bools_z1_z2 = vmap(
                        vmap(
                            lambda x, y: jnp.logical_and(x == u1, y == u2),
                            in_axes=(0, None),
                        ),
                        in_axes=(None, 0),
                    )(at, at)
                    return bools_z1_z2

                return get(jnp.logical_and(_b(at), bools))

            return sel(nl.z_unique, nl.z_unique, nl.z_array[nl.atom_indices])

        return _apply_fun_neighbour_pair(sp, self)

    @property
    def batched(self):
        return len(self.atom_indices.shape) == 3

    @property
    def shape(self):
        return self.atom_indices.shape

    @property
    def num_neigh(self):
        return self.shape[-1]

    @staticmethod
    def batch_sp_nl(f):
        """wrapper to vmap over neighbourlist and systemparams if needed"""

        def fun(sp: SystemParams, nl: NeighbourList, *args, **kwargs):
            if nl.batched:
                assert sp.batched
                assert nl.shape[0] == sp.shape[0]

            return NeighbourList.batch_x_nl(f)(sp, nl, *args, **kwargs)

        return fun

    @staticmethod
    def batch_x_nl(f):
        """wrapper to vmap over neighbourlist and systemparams if needed"""

        def fun(x, nl: NeighbourList, *args, **kwargs):
            if nl.batched:
                return vmap(
                    lambda x, atom_indices, ijk_indices, sp_orig: f(
                        x,
                        NeighbourList(
                            r_cut=nl.r_cut,
                            atom_indices=atom_indices,
                            z_array=nl.z_array,
                            r_skin=nl.r_skin,
                            sp_orig=sp_orig,
                            ijk_indices=ijk_indices,
                            z_unique=nl.z_unique,
                        ),
                        *args,
                        **kwargs,
                    )
                )(x, nl.atom_indices, nl.ijk_indices, nl.sp_orig)

            return f(x, nl, *args, **kwargs)

        return fun

    @jit
    def update(self, sp: SystemParams) -> tuple[bool, SystemParams, NeighbourList]:

        max_displacement = jnp.max(
            jnp.linalg.norm(self._pos(self.sp_orig) - self._pos(sp), axis=-1)
        )

        def _f(sp):
            return sp._get_neighbour_list(
                self.r_cut,
                self.r_skin,
                self.z_array,
                self.z_unique,
                self.num_neigh,
                self.nxyz,
            )

        return jax.lax.cond(
            max_displacement > self.r_skin,
            _f,
            lambda sp: (True, sp, self),
            sp,
        )


@jdc.pytree_dataclass
class CV:
    cv: Array
    mapped: jdc.Static[bool] = False
    _combine_dims: jdc.Static[list | None] = None
    _stack_dims: jdc.Static[list | None] = None

    @property
    def batched(self):
        return len(self.cv.shape) == 2

    @property
    def batch_dim(self):
        if self.batched:
            return self.shape[0]
        return 1

    @property
    def dim(self):
        if self.cv.shape == ():
            return 1
        return self.cv.shape[-1]

    @property
    def size(self):
        return self.cv.size

    @property
    def shape(self):
        return self.cv.shape

    @property
    def combine_dims(self):
        if self._combine_dims is None:
            return self.dim
        return self._combine_dims

    @property
    def stack_dims(self):
        if self._stack_dims is None:
            return [self.batch_dim]
        return self._stack_dims

    def __add__(self, other) -> CV:
        assert isinstance(other, Array)
        return CV(cv=self.cv + other)

    def __radd__(self, other) -> CV:
        return other + self

    def __sub__(self, other) -> CV:
        assert isinstance(other, Array)
        return CV(cv=self.cv - other)

    def __rsub__(self, other) -> CV:
        return other - self

    def __mul__(self, other) -> CV:
        assert isinstance(other, Array)
        return CV(cv=self.cv * other)

    def __rmul__(self, other) -> CV:
        return other * self

    def __matmul__(self, other) -> CV:
        assert isinstance(other, Array)
        return CV(cv=self.cv @ other)

    def __rmatmul__(self, other) -> CV:
        return other @ self

    def __div__(self, other) -> CV:
        assert isinstance(other, Array)
        return CV(cv=self.cv / other)

    def __rdiv__(self, other) -> CV:
        return other / self

    def batch(self) -> CV:
        if self.batched:
            return self
        return CV(cv=jnp.array([self.cv]), mapped=self.mapped)

    def __iter__(self):
        if not self.batched:
            yield self
            return

        for i in range(self.cv.shape[0]):
            yield self[i]
        return

    def __getitem__(self, idx):
        assert self.batched
        return CV(cv=self.cv[idx, :], mapped=self.mapped)

    def unbatch(self) -> CV:
        if not self.batched:
            return self
        assert self.cv.shape[0] == 1
        return CV(cv=self.cv[0, :], mapped=self.mapped)

    @staticmethod
    def stack(*cvs: CV) -> CV:
        in_dims = None
        mapped = None

        cv_arr = []
        stack_dims = []

        for cv in cvs:
            assert isinstance(cv, CV)
            if in_dims is None:
                in_dims = cv._combine_dims
                mapped = cv.mapped
            else:
                assert cv._combine_dims == in_dims
                assert cv.mapped == mapped

            cv_arr.append(cv.batch().cv)
            stack_dims += cv.stack_dims

        assert mapped is not None

        return CV(
            cv=jnp.vstack(cv_arr),
            _combine_dims=in_dims,
            _stack_dims=stack_dims,
        )

    def unstack(self) -> list[CV]:
        i = 0

        out: list[CV] = []
        for j in self.stack_dims:
            out += [
                CV(
                    cv=self.cv[i : i + j, :],
                    mapped=self.mapped,
                    _combine_dims=self._combine_dims,
                )
            ]
            i += j
        return out

    def split(self, flatten=False) -> list[CV]:
        """inverse operation of combine"""
        if self._combine_dims is None:
            return [CV(cv=self.cv, mapped=self.mapped)]

        def broaden_tree(subtree):
            if isinstance(subtree, int):
                return [subtree]
            num = []
            for leaf in subtree:
                num += broaden_tree(leaf)

            return num

        if not flatten:
            sz = [sum(broaden_tree(a)) for a in self._combine_dims]
            out_dim = self._combine_dims
        else:
            sz = broaden_tree(self._combine_dims)
            out_dim = sz

        end = jnp.cumsum(jnp.array(sz))
        start = jnp.hstack([0, end[:-1]])

        return [
            CV(
                cv=jax.lax.dynamic_slice_in_dim(
                    self.cv, start_index=s, slice_size=e, axis=-1
                ),
                _combine_dims=out_dim[i] if isinstance(out_dim[i], list) else None,
            )
            for i, (s, e) in enumerate(zip(start, sz))
        ]

    @staticmethod
    def combine(*cvs: CV, flatten=False) -> CV:
        """merges a list of CVs into a single CV. The dimenisions are stored such that it can later be split into separated CVs"""

        out_cv: list[Array] = []
        out_dim: list[int] = []

        mapped = None
        batched = None
        bdim = None

        assert len(cvs) != 0
        if len(cvs) == 1:
            return cvs[0]

        def inner(cv: CV) -> tuple[list[Array], list[int]]:
            nonlocal mapped
            nonlocal batched
            nonlocal bdim

            if mapped is None:
                mapped = cv.mapped
            else:
                assert mapped == cv.mapped

            if batched is None:
                batched = cv.batched
                if batched:
                    bdim = cv.batch_dim
            else:
                assert batched == cv.batched
                if batched:
                    assert bdim == cv.batch_dim

            def simple(cv: CV):
                return [cv.cv], [cv.combine_dims]

            if cv._combine_dims is not None and flatten:
                cv_split = cv.split()

                cvi: list[Array] = []
                dimi: list[int] = []

                for ii in cv_split:
                    if ii._combine_dims is None:
                        a, b = simple(ii)
                    else:
                        a, b = a, b = inner(ii)

                    cvi += a
                    dimi += b

            else:
                cvi, dimi = simple(cv)

            return cvi, dimi

        for cv in cvs:
            a, b = inner(cv)
            out_cv += a
            out_dim += b

        return CV(cv=jnp.hstack(out_cv), mapped=mapped, _combine_dims=out_dim)  # type: ignore


class CvMetric:
    """class to keep track of topology of given CV. Identifies the periodicitie of CVs and maps to unit square with correct peridicities"""

    def __init__(
        self,
        periodicities,
        bounding_box=None,
        # map_meshgrids=None,
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

        self.bounding_box = bounding_box
        self.periodicities = periodicities

    @partial(jit, static_argnums=(0))
    def norm(self, x1: CV, x2: CV, k=1.0):
        diff = self.difference(x1=x1, x2=x2) * k
        return jnp.linalg.norm(diff)

    @partial(jit, static_argnums=(0, 2))
    def periodic_wrap(self, x: CV, min=False) -> CV:
        out = CV(cv=self.__periodic_wrap(self.__map(x.cv), min=min), mapped=True)

        if x.mapped:
            return out
        return self.__unmap(out)

    @partial(jit, static_argnums=(0))
    def difference(self, x1: CV, x2: CV) -> Array:
        assert not x1.mapped
        assert not x2.mapped

        return self.min_cv(
            x2.cv - x1.cv,
        )

    def min_cv(self, cv: Array):
        mapped = self.__map(cv, displace=False)
        wrapped = self.__periodic_wrap(mapped, min=True)

        return self.__unmap(
            wrapped,
            displace=False,
        )

    @partial(jit, static_argnums=(0, 2))
    def __periodic_wrap(self, xs: Array, min=False):
        """Translate cvs such over unit cell.

        min=True calculates distances, False translates one vector inside box
        """

        coor = jnp.mod(xs, 1)  # between 0 and 1
        if min:
            coor = jnp.where(coor > 0.5, coor - 1, coor)  # between [-0.5,0.5]

        return jnp.where(self.periodicities, coor, xs)

    @partial(jit, static_argnums=(0, 2))
    def __map(self, x: Array, displace=True) -> Array:
        """transform CVs to lie in unit square."""

        if displace:
            x -= self.bounding_box[:, 0]

        y = x / (self.bounding_box[:, 1] - self.bounding_box[:, 0])

        return y

    @partial(jit, static_argnums=(0, 2))
    def __unmap(self, x: Array, displace=True) -> Array:
        """transform CVs to lie in unit square."""

        y = x * (self.bounding_box[:, 1] - self.bounding_box[:, 0])

        if displace:
            x += self.bounding_box[:, 0]

        return y

    def __add__(self, other):
        assert isinstance(self, CvMetric)
        if other is None:
            return self

        assert isinstance(other, CvMetric)

        periodicities = jnp.hstack((self.periodicities, other.periodicities))
        bounding_box = jnp.vstack((self.bounding_box, other.bounding_box))

        return CvMetric(
            periodicities=periodicities,
            bounding_box=bounding_box,
        )

    def grid(self, n, endpoints=None, margin=None):
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

        b = self.bounding_box

        if margin is not None:
            diff = (b[:, 1] - b[:, 0]) * margin
            b = b.at[:, 0].set(b[:, 0] - diff)
            b = b.at[:, 1].set(b[:, 1] + diff)

        assert not (jnp.abs(b[:, 1] - b[:, 0]) < 1e-12).any(), "give proper boundaries"
        grid = [
            jnp.linspace(row[0], row[1], n, endpoint=endpoints[i])
            for i, row in enumerate(b)
        ]

        return grid

    @property
    def ndim(self):
        return len(self.periodicities)


######################################
#       CV tranformations            #
######################################


@dataclasses.dataclass(kw_only=True)
class CvFunInput:
    input: int
    conditioners: list[int] | None = None

    def split(self, x: CV):
        cvs = x.split()
        if self.conditioners is not None:
            cond = [cvs[i] for i in self.conditioners]
        else:
            cond = []

        return cvs[self.input], cond

    def combine(self, x: CV, res: CV):
        cvs = [*x.split()]
        cvs[self.input] = res
        return CV.combine(*cvs)


@dataclasses.dataclass(kw_only=True)
class CvFunBase:
    _: dataclasses.KW_ONLY
    cv_input: CvFunInput | None = None

    def calc(self, x: CV, reverse=False, log_det=False) -> tuple[CV, Array | None]:
        if self.cv_input is not None:
            y, cond = self.cv_input.split(x)
        else:
            y, cond = x, []

        if log_det:
            out, log_det = self._log_Jf(y, *cond, reverse=reverse)
        else:
            out, log_det = self._calc(y, *cond, reverse=reverse), None

        if self.cv_input:
            out = self.cv_input.combine(x, out)

        return out, log_det

    @abstractmethod
    def _calc(self, x: CV, *conditioners: CV, reverse=False) -> CV:
        pass

    def _log_Jf(
        self, x: CV, *conditioners: CV, reverse=False
    ) -> tuple[CV, Array | None]:
        """naive automated implementation, overrride this"""
        f = lambda x: self._calc(x, *conditioners, reverse=reverse)

        a = f(x)
        b = jacfwd(f)(x)
        log_det = jnp.log(jnp.abs(jnp.linalg.det(b.cv.cv)))

        return a, log_det


@dataclasses.dataclass(kw_only=True)
class CvFun(CvFunBase):
    forward: Callable[[CV, CV | None], CV] | None = None
    backward: Callable[[CV, CV | None], CV] | None = None

    def _calc(self, x: CV, *conditioners: CV, reverse=False) -> CV:
        if len(conditioners) == 0:
            c = None
        else:
            c = CV.combine(*conditioners)

        if reverse:
            assert self.backward is not None
            return self.backward(x, c)
        else:
            assert self.forward is not None
            return self.forward(x, c)


class CvFunNn(nn.Module, CvFunBase):
    """used to instantiate flax linen CvTrans"""

    @abstractmethod
    def setup(self):
        pass

    def _calc(self, x: CV, *y: CV, reverse=False) -> CV:
        if reverse:
            return self.backward(x, *y)
        else:
            return self.forward(x, *y)

    @abstractmethod
    def forward(self, x: CV, *y: CV) -> CV:
        pass

    @abstractmethod
    def backward(self, x: CV, *y: CV) -> CV:
        pass


class CvFunDistrax(nn.Module, CvFunBase):
    """
    creates bijective CV function based on a distrax flow. The seup function should initialize the bijector

    class RealNVP(CvFunDistrax):
        _: dataclasses.KW_ONLY
        latent_dim: int

        def setup(self):
            self.s = Dense(features=self.latent_dim)
            self.t = Dense(features=self.latent_dim)

            # Alternating binary mask.
            self.bijector = distrax.as_bijector(
                tfp.bijectors.RealNVP(
                    fraction_masked=0.5,
                    shift_and_log_scale_fn=self.shift_and_scale,
                )
            )

        def shift_and_scale(self, x0, input_depth, **condition_kwargs):
            return self.s(x0), self.t(x0)
    """

    bijector: distrax.Bijector = dataclasses.field(init=False)

    @abstractmethod
    def setup(self):
        """setups self.bijector"""

    def _calc(self, x: CV, *y: CV, reverse=False, log_det=False) -> CV:
        z = CV.combine(*y, x).cv

        if reverse:
            assert self.bijector is not None
            return CV(self.bijector.inverse(z))
        else:
            assert self.bijector is not None
            return CV(self.bijector.forward(z))

    def _log_Jf(self, x: CV, *y: CV, reverse=False) -> tuple[CV, Array | None]:
        """naive implementation, overrride this"""
        assert self.bijector is not None
        f = (
            self.bijector.inverse_and_log_det
            if reverse
            else self.bijector.forward_and_log_det
        )
        z, jac = f(CV.combine(*y, x).cv)
        return CV(cv=z), jac


@dataclasses.dataclass(kw_only=True)
class CvTrans:
    """f can either be a single CV tranformation or a list of transformations"""

    trans: list[CvFunBase]

    @staticmethod
    def from_array_function(f: Callable[[Array], Array]):
        def f2(x: CV, cond: CV | None = None):
            assert cond is None, "implement this"

            if x.batched:
                out = vmap(f)(x.cv)
            else:
                out = f(x.cv)

            return CV(cv=out)

        return CvTrans.from_cv_function(f=f2)

    @staticmethod
    def from_cv_function(f: Callable[[CV, CV | None], CV]) -> CvTrans:
        return CvTrans.from_cv_fun(proto=CvFun(forward=f))

    @staticmethod
    def from_cv_fun(proto: CvFunBase):
        return CvTrans(trans=[proto])

    def compute_cv_trans(
        self, x: CV, reverse=False, log_Jf=False
    ) -> tuple[CV, Array | None]:
        """
        result is always batched
        arg: CV
        """
        if x.batched:
            return vmap(self.compute_cv_trans)(x, reverse, log_Jf)

        ordered = reversed(self.trans) if reverse else self.trans

        if log_Jf:
            log_det = jnp.array(0.0)
        else:
            log_det = None

        for tr in ordered:
            x, log_det_i = tr.calc(x=x, reverse=reverse, log_det=log_Jf)
            if log_Jf:
                assert log_det_i is not None
                log_det += log_det_i
        return x, log_det

    def __mul__(self, other):
        assert isinstance(other, CvTrans), "can only multiply by CvTrans object"
        return CvTrans(trans=[*self.trans, *other.trans])


class CvTransNN(nn.Module, CvTrans):
    trans: list[CvFunBase]

    def setup(self) -> None:
        pass

    @nn.compact
    def compute_cv_trans(
        self, x: CV, reverse=False, log_Jf=False
    ) -> tuple[CV, Array | None]:
        return super().compute_cv_trans(x, reverse, log_Jf)

    @nn.nowrap
    def __mul__(self, other):
        assert isinstance(other, CvTrans | CvTransNN)
        return CvTransNN(trans=[s + o for s, o in zip(self.trans, other.trans)])


class NormalizingFlow(nn.Module):
    """normalizing flow. _ProtoCvTransNN are stored separately because they need to be initialized by this module in setup"""

    flow: CvTransNN | CvTransNN

    def setup(self) -> None:
        if isinstance(self.flow, CvTrans):
            self.nn_flow = CvTransNN(trans=self.flow.trans)
        else:
            self.nn_flow = self.flow

    def calc(self, x: CV, reverse: bool, test_log_det=False):
        a, b = self.nn_flow.compute_cv_trans(x, reverse=reverse, log_Jf=True)

        if test_log_det:
            b2 = jnp.log(
                jnp.abs(
                    jnp.linalg.det(
                        jacfwd(
                            lambda x: self.nn_flow.compute_cv_trans(
                                x, reverse=reverse, log_Jf=False
                            )[0]
                        )(x).cv.cv
                    )
                )
            )
            assert jnp.abs(b - b2) < 1e-5

        return a, b


class CvFlow:
    def __init__(
        self,
        func: Callable[[SystemParams, NeighbourList | None], CV],
        trans: CvTrans | None = None,
        # batched=False,
    ) -> None:
        self.f0 = func
        self.f1 = trans
        # self.batched = batched

    @staticmethod
    def from_function(
        f: Callable[[SystemParams, NeighbourList | None], Array]
    ) -> CvFlow:
        def f2(
            sp: SystemParams,
            nl: NeighbourList | None = None,
        ):
            return CV(cv=f(sp, nl))

        return CvFlow(func=f2)

    @partial(jit, static_argnums=(0,))
    def compute_cv_flow(
        self,
        x: SystemParams,
        nl: NeighbourList | None = None,
    ) -> CV:
        @NeighbourList.batch_sp_nl
        def _compute_cv_flow(sp, nl):

            # if x.batched:
            #     return vmap(jit(self.compute_cv_flow))(x, nl)

            out = self.f0(sp, nl)

            if self.f1 is not None:
                out, _ = self.f1.compute_cv_trans(out)

            return out

        return _compute_cv_flow(x, nl)

    def __add__(self, other):
        assert isinstance(other, CvFlow)

        def f_add(x: SystemParams, nl: NeighbourList):
            cv1: CV = self.compute_cv_flow(x, nl)
            cv2: CV = other.compute_cv_flow(x, nl)

            return CV.combine(cv1, cv2)

        return CvFlow(func=f_add)

    def __mul__(self, other):
        assert isinstance(other, CvTrans), "can only multiply by CvTrans object"

        if self.f1 is None:
            self.f1 = other
        else:
            self.f1 *= other

        return self

    def find_sp(
        self,
        x0: SystemParams,
        target: CV,
        nl0: NeighbourList | None = None,
        maxiter=10000,
        tol=1e-4,
        norm=lambda cv1, cv2, nl2: jnp.linalg.norm(cv1 - cv2),
        solver=jaxopt.GradientDescent,
    ) -> SystemParams:

        # @partial(jit, static_argnums=(1, 2))
        def loss(sp, nl, norm):
            @jit
            def _f(sp: SystemParams, nl: NeighbourList):
                # sp = f(sp_flat)
                bool, sp, nl = nl.update(sp)

                cvi = self.compute_cv_flow(sp, nl)
                nn = norm(cvi.cv, target.cv, nl)

                return nn, (bool, nn, sp, nl)

            return _f(sp, nl), jacfwd(lambda s, n: _f(s, n)[0])(sp, nl)

        _l = jit(partial(loss, norm=norm))

        slvr = solver(_l, value_and_grad=True, has_aux=True, tol=tol, maxiter=10)
        state = slvr.init_state(x0, nl=nl0)

        for _ in range(maxiter):
            x0, state = jit(slvr.update)(x0, state, nl=nl0)

            b, nn, x0, nl0 = state.aux
            print(
                f"step:{state.iter_num} norm {nn:.4f} err {state.error:.4f} update nl={not b}"
            )

            if not b:
                x0, nl0 = x0.get_neighbour_list(
                    r_cut=nl0.r_cut, r_skin=nl0.r_skin, z_array=nl0.z_array
                )

        sp0 = _l(x0, nl0)
        return sp0


######################################
#       Collective variable          #
######################################


class CollectiveVariable:
    def __init__(self, f: CvFlow, metric: CvMetric, jac=jacfwd) -> None:
        "jac: kind of jacobian. Default is jacfwd (more efficient for tall matrices), but functions with custom jvp's only support jacrev"

        self.metric = metric
        self.f = f
        self.jac = jac

    @partial(jit, static_argnums=(0, 3))
    def compute_cv(
        self,
        sp: SystemParams,
        nl: NeighbourList | None = None,
        jacobian=False,
    ) -> tuple[CV, CV]:
        @NeighbourList.batch_sp_nl
        def _compute_cv(sp, nl):
            cvf = self.f.compute_cv_flow
            dcv = self.jac(cvf)

            cv = cvf(sp, nl)
            dcv = dcv(sp, nl) if jacobian else None

            return (cv, dcv)

        return _compute_cv(sp, nl)

    @property
    def n(self):
        return self.metric.ndim


######################################
#       CV transformations           #
######################################


class PeriodicLayer(keras.layers.Layer):
    def __init__(self, bbox, periodicity, **kwargs):
        super().__init__(**kwargs)

        self.bbox = tfl.Variable(np.array(bbox, dtype=np.float32))
        self.periodicity = np.array(periodicity)

    def call(self, inputs):
        # maps to periodic box
        bbox = self.bbox

        inputs_mod = (
            tfl.math.mod(inputs - bbox[:, 0], bbox[:, 1] - bbox[:, 0]) + bbox[:, 0]
        )
        return tfl.where(self.periodicity, inputs_mod, inputs)

    def metric(self, r):
        # maps difference
        a = self.bbox[:, 1] - self.bbox[:, 0]

        r = tfl.math.mod(r, a)
        r = tfl.where(r > a / 2, r - a, r)
        return tfl.norm(r, axis=1)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "bbox": np.array(self.bbox),
                "periodicity": self.periodicity,
            }
        )
        return config


class KerasTrans(CvTrans):
    def __init__(self, encoder) -> None:
        self.encoder = encoder

    @partial(jit, static_argnums=(0,))
    def compute_cv_trans(self, cc: Array):
        out = call_tf(self.encoder.call)(cc)
        return out

    def __getstate__(self):
        # https://stackoverflow.com/questions/48295661/how-to-pickle-keras-model
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as fd:
            tensorflow.keras.models.save_model(self.encoder, fd.name, overwrite=True)
            model_str = fd.read()
        d = {"model_str": model_str}
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as fd:
            fd.write(state["model_str"])
            fd.flush()

            custom_objects = {"PeriodicLayer": PeriodicLayer}
            with keras.utils.custom_object_scope(custom_objects):
                model = keras.models.load_model(fd.name)

        self.encoder = model


@CvFlow.from_function
def Volume(sp: SystemParams, _):
    assert sp.cell is not None, "can only calculate volume if there is a unit cell"

    vol = jnp.abs(jnp.dot(sp.cell[0], jnp.cross(sp.cell[1], sp.cell[2])))
    return vol


def distance_descriptor():
    @CvFlow.from_function
    def h(x: SystemParams, _):

        x = x.canoncialize()

        n = x.shape[-2]

        out = vmap(vmap(x.min_distance, in_axes=(0, None)), in_axes=(None, 0))(
            jnp.arange(n), jnp.arange(n)
        )

        return out[jnp.triu_indices_from(out, k=1)]

        # return out

    return h


def dihedral(numbers):
    """from https://stackoverflow.com/questions/20305272/dihedral-torsion-
    angle-from-four-points-in-cartesian- coordinates-in-python.

    args:
        numbers: list with index of 4 atoms that form dihedral
    """

    @CvFlow.from_function
    def f(sp: SystemParams, _):
        coor = sp.coordinates
        p0 = coor[numbers[0]]
        p1 = coor[numbers[1]]
        p2 = coor[numbers[2]]
        p3 = coor[numbers[3]]

        b0 = -1.0 * (p1 - p0)
        b1 = p2 - p1
        b2 = p3 - p2

        b1 /= jnp.linalg.norm(b1)

        v = b0 - jnp.dot(b0, b1) * b1
        w = b2 - jnp.dot(b2, b1) * b1

        x = jnp.dot(v, w)
        y = jnp.dot(jnp.cross(b1, v), w)
        return jnp.arctan2(y, x)

    return f


def sb_descriptor(
    r_cut,
    n_max: int,
    l_max: int,
    references: SystemParams | None = None,
    references_nl: NeighbourList | None = None,
):
    from IMLCV.base.tools.soap_kernel import Kernel, p_i, p_inl_sb

    @jit
    def f(sp: SystemParams, nl: NeighbourList):
        assert nl is not None, "provide neighbourlist for sb describport"

        return p_i(
            sp=sp,
            nl=nl,
            p=p_inl_sb(r_cut=r_cut, n_max=n_max, l_max=l_max),
            r_cut=r_cut,
        )

    if references is not None:
        assert references_nl is not None

        refs = NeighbourList.batch_sp_nl(f)(references, references_nl)

        @NeighbourList.batch_x_nl
        def _f(refs, references_nl, val, nl):
            return Kernel(val, refs, nl, references_nl)

        _f = partial(_f, refs, references_nl)

        # @jit
        def sb_descriptor_distance(sp: SystemParams, nl: NeighbourList):
            assert nl is not None, "provide neighbourlist for sb describport"

            val = f(sp=sp, nl=nl)
            com = _f(val, nl)

            return 1 - com

        return CvFlow.from_function(sb_descriptor_distance)  # type: ignore

    else:
        return CvFlow.from_function(f)  # type: ignore


######################################
#           CV trans                 #
######################################


# class MeshGrid(CvTrans):
#     def __init__(self, meshgrid) -> None:
#         self.map_meshgrids = meshgrid
#         super().__init__(f)

#     def _f(self, x: CV):
#         #  if self.map_meshgrids is not None:
#         y = x.cv

#         y = y * (jnp.array(self.map_meshgrids[0].shape) - 1)
#         y = jnp.array(
#             [jsp.ndimage.map_coordinates(wp, y, order=1) for wp in self.map_meshgrids]
#         )


def rotate_2d(alpha):
    @CvTrans.from_cv_function
    def f(cv: CV):
        return (
            jnp.array(
                [[jnp.cos(alpha), jnp.sin(alpha)], [-jnp.sin(alpha), jnp.cos(alpha)]]
            )
            @ cv
        )

    return f


def scale_cv_trans(array: CV):
    "axis 0 is batch axis"
    maxi = jnp.max(array.cv, axis=0)
    mini = jnp.min(array.cv, axis=0)
    diff = (maxi - mini) / 2
    # mask = jnp.abs(diff) > 1e-6

    @CvTrans.from_array_function
    def f0(x):
        return (x - (mini + maxi) / 2) / diff

    return f0


######################################
#           CV Fun                   #
######################################


class RealNVP(CvFunNn):
    """use in combination with swaplink"""

    _: dataclasses.KW_ONLY
    features: int
    cv_input: CvFunInput

    def setup(self) -> None:
        self.s = Dense(features=self.features)
        self.t = Dense(features=self.features)

    def forward(self, x: CV, *cond: CV):
        y = CV.combine(*cond).cv
        return CV(cv=x.cv * self.s(y) + self.t(y))

    def backward(self, z: CV, *cond: CV):
        y = CV.combine(*cond).cv
        return CV(cv=(z.cv - self.t(y)) / self.s(y))


class DistraxRealNVP(CvFunDistrax):
    _: dataclasses.KW_ONLY
    latent_dim: int

    def setup(self):
        """Creates the flow model."""
        self.s = Dense(features=self.latent_dim)
        self.t = Dense(features=self.latent_dim)

        # Alternating binary mask.
        self.bijector = distrax.as_bijector(
            tfp.bijectors.RealNVP(
                fraction_masked=0.5,
                shift_and_log_scale_fn=self.shift_and_scale,
            )
        )

    def shift_and_scale(self, x0, input_depth, **condition_kwargs):
        return self.s(x0), self.t(x0)


######################################
#           Test                     #
######################################


class MetricUMAP(CvMetric):
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


class hyperTorus(CvMetric):
    def __init__(self, n) -> None:
        periodicities = [True for _ in range(n)]
        boundaries = jnp.zeros((n, 2))
        boundaries = boundaries.at[:, 0].set(-jnp.pi)
        boundaries = boundaries.at[:, 1].set(jnp.pi)

        super().__init__(periodicities, boundaries)


######################################
#           Test                     #
######################################


def test_cv_split_combine():
    import jax.random

    prng = jax.random.PRNGKey(42)
    k1, k2, k3, prng = jax.random.split(prng, 4)

    a = CV(cv=jax.random.uniform(k1, (5, 2)))
    b = CV(cv=jax.random.uniform(k1, (5, 2)))
    c = CV(cv=jax.random.uniform(k1, (5, 1)))

    d = CV.combine(a, b)
    e = CV.combine(d, c, flatten=False)

    ab2, c2 = e.split()
    a2, b2 = ab2.split()

    a3, b3, c3 = e.split(flatten=True)

    for x, y, z in zip([a, b, c], [a2, b2, c2], [a3, b3, c3]):
        assert ((x.cv - y.cv) == 0).all()
        assert ((x.cv - z.cv) == 0).all()


def test_nf():
    def test(mf, x, k2):
        var_a = mf.init(k2, x, False, method=NormalizingFlow.calc)

        y, Jy = mf.apply(
            variables=var_a,
            method=NormalizingFlow.calc,
            x=x,
            test_log_det=True,
            reverse=False,
        )
        x2, Jx = mf.apply(
            variables=var_a,
            method=NormalizingFlow.calc,
            x=y,
            test_log_det=True,
            reverse=True,
        )

        assert (jnp.abs(x.cv - x2.cv) < 1e-5).all()
        assert (jnp.abs(Jx + Jy) < 1e-5).all()

    def get_chain(input, cond):
        chain = CvTrans.from_cv_fun(
            RealNVP(
                features=features,
                cv_input=CvFunInput(input=input, conditioners=cond),
            )
        )
        return chain

    features = 5

    prng = jax.random.PRNGKey(seed=42)
    ##test 0

    key_1, key_2, prng = jax.random.split(prng, 3)
    x = CV(cv=jax.random.uniform(key=key_1, shape=(10,)), _combine_dims=[5, 5])

    test(NormalizingFlow(flow=get_chain(input=0, cond=[1])), x, key_2)

    ##test 1

    key_1, key_2, prng = jax.random.split(prng, 3)
    x = CV(cv=jax.random.uniform(key=key_1, shape=(10,)), _combine_dims=[5, 5])
    test(
        NormalizingFlow(
            flow=get_chain(input=0, cond=[1]) * get_chain(input=1, cond=[0])
        ),
        x,
        key_2,
    )

    ## test with distrax

    # https://www.tensorflow.org/probability/api_docs/python/tfp/bijectors/RealNVP
    key_1, key_2, key3, prng = jax.random.split(prng, 4)
    x2 = CV(cv=jax.random.uniform(key=key_1, shape=(10,)), _combine_dims=[5, 5])

    chain = CvTrans.from_cv_fun(
        DistraxRealNVP(
            latent_dim=5,
        )
    )

    test(NormalizingFlow(chain), x2, key_2)


def _get_sp_rand(
    prng, n=15, r_cut=3
) -> tuple[jax.random.KeyArray, SystemParams, NeighbourList]:
    k1, k2, k3, prng = jax.random.split(prng, 4)

    r_side = 6 * (n / 5) ** (1 / 3)

    sp0 = SystemParams(
        coordinates=jax.random.uniform(k1, shape=(n, 3)) * r_side,
        cell=jnp.array(
            [
                [n, 0, 0],
                [0, n, 0],
                [0, 0, n],
            ],
            dtype=jnp.float64,
        ),
    ).canoncialize()

    # r_cut = 4 * angstrom
    z_array = jax.random.randint(k2, (n,), 0, 5)

    sp0, nl0 = sp0.get_neighbour_list(r_cut=r_cut, z_array=z_array)

    return prng, sp0, nl0


def _permute_sp_rand(
    prng, sp0, nl0, eps
) -> tuple[jax.random.KeyArray, SystemParams, NeighbourList]:
    k1, k2, k3, prng = jax.random.split(prng, 4)

    sp1 = SystemParams(
        coordinates=sp0.coordinates
        + jax.random.uniform(k1, shape=sp0.coordinates.shape) * eps,
        cell=sp0.cell + jax.random.normal(k3, (3, 3)) * eps,
    )

    sp1, nl1 = sp1.get_neighbour_list(r_cut=nl0.r_cut, z_array=nl0.z_array)
    return prng, sp1, nl1


def _get_equival_sp(sp, rng) -> tuple[jax.random.KeyArray, SystemParams]:
    # check rotational and translationa invariance
    from scipy.spatial.transform import Rotation as R

    key, rng = jax.random.split(rng)

    rot_mat = jnp.array(
        R.random(random_state=int(jax.random.randint(key, (), 0, 100))).as_matrix()
    )
    pos2 = (
        vmap(lambda a: rot_mat @ a, in_axes=0)(sp.coordinates)
        + jax.random.normal(key, (3,)) * 5
    )
    cell_r = vmap(lambda a: rot_mat @ a, in_axes=0)(sp.cell)
    sp2 = SystemParams(coordinates=pos2, cell=cell_r)
    return rng, sp2


def test_reconstruction():

    prng = jax.random.PRNGKey(seed=42)

    r_cut = 5
    prng, sp0, nl0 = _get_sp_rand(prng=prng, n=10, r_cut=r_cut)
    k1, k2, k3, prng = jax.random.split(prng, 4)

    # with jax.debug_nans():

    cv = sb_descriptor(r_cut=r_cut, n_max=3, l_max=3, references=sp0, references_nl=nl0)

    cv0 = cv.compute_cv_flow(sp0, nl0)  # should be close to 0

    prng, sp1, nl1 = _permute_sp_rand(prng, sp0, nl0, eps=0.5)

    tol = 1e-6

    from IMLCV.base.tools.soap_kernel import Kernel

    sp01 = cv.find_sp(
        x0=sp1,
        nl0=nl1,
        target=cv0,
        tol=tol,
        # norm=lambda cv1, cv2, nl2: jnp.sqrt(
        #     1 - Kernel(cv1, cv2, nl1, nl2, matching="REMatch")
        # ),
        # solver=jaxopt.GradientDescent
        # solver=jaxopt.NonlinearCG
        solver=jaxopt.LBFGS,
        # solver=partial(jaxopt.ScipyMinimize, method="l-bfgs-b"),
    )

    sp01, nl01 = sp01.get_neighbour_list(r_cut=r_cut, z_array=nl1.z_array)

    assert (
        1
        - Kernel(
            cv.compute_cv_flow(sp0, nl0).cv,
            cv.compute_cv_flow(sp01, nl01).cv,
            nl0,
            nl01,
        )
    ) < tol


def test_neigh():
    rng = jax.random.PRNGKey(42)

    r_cut = 200
    n = 15
    rng, sp, nl = _get_sp_rand(prng=rng, n=n, r_cut=r_cut)

    func = lambda r_ij, index: (jnp.linalg.norm(r_ij), index)

    neigh_calc, r_exp = sp.apply_fun_neighbour(
        func=func,
        r_cut=r_cut,
        center_coordinates=sp.coordinates[0],
    )
    neigh_exp = n / jnp.abs(jnp.linalg.det(sp.cell)) * (4 / 3 * jnp.pi * r_cut**3)

    print(f"err neigh density {jnp.abs(  (neigh_calc - neigh_exp) /neigh_exp)}")

    rng, sp2 = _get_equival_sp(sp, rng)

    neigh_calc_2, r_exp_2 = sp2.apply_fun_neighbour(
        func=func,
        r_cut=r_cut,
        center_coordinates=sp2.coordinates[0],
    )

    assert neigh_calc == neigh_calc_2, f"{neigh_calc} != {neigh_calc_2}"
    assert jnp.abs(r_exp_2[0] - r_exp[0]) < 1e-5
    assert r_exp_2[1] - r_exp[1] == 0

    ##method 4: with neighbourlist

    sp3, nl = sp.get_neighbour_list(r_cut=r_cut, z_array=nl.z_array)
    neigh_calc_4, r_exp_4 = nl.apply_fun_neighbour(sp=sp3, r_cut=r_cut, func=func)

    assert neigh_calc == neigh_calc_4[0], f"{neigh_calc} != {neigh_calc_4[0]}"
    assert jnp.abs((r_exp_4[0][0] - r_exp[0]) / r_exp[0]) < 1e-6
    assert r_exp_4[1][0] - r_exp[1] == 0

    ##method 5: with neighbourlist, split z

    neigh_calc_5_z, r_exp_5_z = nl.apply_fun_neighbour(
        sp=sp3, r_cut=r_cut, func=func, split_z=True
    )

    neigh_calc_5, r_exp_5 = jax.tree_map(
        lambda x: jnp.sum(x, axis=1), (neigh_calc_5_z, r_exp_5_z)
    )

    assert (neigh_calc_4 == neigh_calc_5).all(), f"{neigh_calc_4} != {neigh_calc_5}"
    assert ((jnp.abs(r_exp_4[0] - r_exp_5[0]) / (r_exp_5[0])) < 1e-5).all()
    assert (jnp.abs(r_exp_4[1] - r_exp_5[1]) < 1e-9).all()

    ##method 6: with neighbourlist, no reduction

    neigh_calc_6_zz, r_exp_6_zz = nl.apply_fun_neighbour(
        sp=sp3,
        r_cut=r_cut,
        func=func,
        reduce="none",
        split_z=False,
    )

    ##method 7: with neighbourlist, sort_z_self

    neigh_calc_7_zz, r_exp_7_zz = nl.apply_fun_neighbour(
        sp=sp3,
        r_cut=r_cut,
        func=func,
        reduce="z",
        split_z=False,
    )

    # check nl update code

    key1, key2, key3, rng = jax.random.split(rng, 4)
    sp4 = SystemParams(
        sp3.coordinates + 0.1 * jax.random.normal(key1, sp3.coordinates.shape),
        sp3.cell + 0.1 * jax.random.normal(key2, sp3.cell.shape),
    )
    bool, sp4, nl4 = nl.update(sp4)


def test_neigh_pair():
    rng = jax.random.PRNGKey(42)

    rng, sp, nl = _get_sp_rand(rng, 15, 3)

    z_array = nl.z_array
    r_cut = nl.r_cut

    func = lambda r_ij, r_ik, atom_index_j, atom_index_k: (
        jnp.linalg.norm(r_ij - r_ik),
        z_array[atom_index_j],
        z_array[atom_index_k],
    )
    n, (pair_dist, index_j, index_k) = sp.apply_fun_neighbour_pairs(
        r_cut=r_cut,
        center_coordinates=sp.coordinates[0],
        func=func,
        exclude_self=True,
    )

    # same up to physcial symmetries
    rng, sp2 = _get_equival_sp(sp, rng)

    k2, (pair_dist2, index_j2, index_k2) = sp2.apply_fun_neighbour_pairs(
        r_cut=r_cut,
        center_coordinates=sp2.coordinates[0],
        func=func,
        exclude_self=True,
    )

    assert n == k2
    assert jnp.abs(pair_dist - pair_dist2) < 0.1

    # neighbourghlist

    func = lambda r_ij, atom_index_j, data_j, r_ik, atom_index_k, data_k: (
        jnp.linalg.norm(r_ij - r_ik),
        z_array[atom_index_j],
        z_array[atom_index_k],
    )

    sp3, nl = sp.get_neighbour_list(r_cut=r_cut, z_array=z_array)

    k3, (pair_dist3, index_j3, index_k3) = nl.apply_fun_neighbour_pair(
        sp=sp3,
        r_cut=r_cut,
        func_double=func,
        exclude_self=True,
        unique=True,
    )

    assert n == k3[0]
    assert jnp.abs(pair_dist - pair_dist3[0]) < 0.2

    ## tst z split, resummations should yield orriginal result

    k4_zz, (pair_dist4_zz, index_j4_zz, index_k4_zz) = nl.apply_fun_neighbour_pair(
        sp=sp3,
        r_cut=r_cut,
        func_double=func,
        exclude_self=True,
        unique=True,
        split_z=True,
    )

    # reduce to non split version
    k4, pair_dist4 = jax.tree_map(
        lambda x: jnp.sum(jnp.sum(x, axis=2), axis=1), (k4_zz, pair_dist4_zz)
    )

    assert jnp.linalg.norm(k4 - k3) == 0
    assert jnp.linalg.norm(pair_dist4[0] - pair_dist3[0]) < 0.2
    assert jnp.linalg.norm(pair_dist4[1] - pair_dist3[1]) < 0.2

    ## same but without reduction

    k5_zz, (pair_dist5_zz, index_j5_zz, index_k5_zz) = nl.apply_fun_neighbour_pair(
        sp=sp3,
        r_cut=r_cut,
        func_double=func,
        reduce="none",
        exclude_self=True,
        unique=True,
        split_z=True,
    )

    # test if j and k indices are correctly fragmented
    for j, zj in enumerate(jnp.unique(z_array)):
        for k, zk in enumerate(jnp.unique(z_array)):
            assert (index_j5_zz[:, j, k, :][k5_zz[:, j, k, :]] == zj).all()
            assert (index_k5_zz[:, j, k, :][k5_zz[:, j, k, :]] == zk).all()

    ## same but with reduction per z

    k6_zz, (pair_dist6_zz, index_j6_zz, index_k6_zz) = nl.apply_fun_neighbour_pair(
        sp=sp3,
        r_cut=r_cut,
        func_double=func,
        reduce="z",
        exclude_self=True,
        unique=True,
        split_z=True,
    )

    pair_dist_avg = pair_dist / n
    # https://math.stackexchange.com/questions/167932/mean-distance-between-2-points-in-a-ball
    pair_dist_exact = 36 / 35 * r_cut

    print(
        f"err neigh density {jnp.abs( (pair_dist_avg -pair_dist_exact)/pair_dist_exact  )}"
    )


def test_minkowski_reduce():
    prng = jax.random.PRNGKey(42)
    key1, key2, prng = jax.random.split(prng, 3)

    sp = SystemParams(
        coordinates=jnp.zeros((22, 3)),
        cell=jnp.array(
            [
                [2, 0, 0],
                [6, 1, 0],
                [8, 9, 1],
            ]
        ),
    ).minkowski_reduce()

    assert jnp.linalg.norm(sp.cell - jnp.array([[0, 1, 0], [0, 0, 1], [2, 0, 0]])) == 0


def test_canoncicalize():
    prng = jax.random.PRNGKey(42)
    k1, k2, prng = jax.random.split(prng, 3)

    cell = jax.random.uniform(k1, (3, 3))
    coordinates = jax.random.uniform(k2, (2, 3))

    # make scramled cell

    cell2 = cell
    for i in range(10):
        k1, prng = jax.random.split(prng, 2)

        k1, prng = jax.random.split(prng, 2)

        new_combo = jax.random.randint(k1, (3,), minval=-2, maxval=2)
        new_combo = new_combo.at[i % 3].set(1)

        cell2 = cell2.at[i, :].set(new_combo @ cell2)

    k1, prng = jax.random.split(prng, 2)
    coordinates2 = (
        coordinates + jax.random.randint(k1, (3,), minval=-3, maxval=3) @ cell
    )

    sp0 = SystemParams(cell=cell, coordinates=coordinates)
    sp1 = SystemParams(cell=cell2, coordinates=coordinates2)

    # test distance
    assert jnp.abs(sp0.min_distance(0, 1) - sp1.min_distance(0, 1)) < 1e-6

    # test minkowski reduction
    sp0, sp1 = sp0.minkowski_reduce(), sp1.minkowski_reduce()

    assert (jnp.abs(sp0.cell - sp1.cell) < 1e-6).all()

    sp0, sp1 = sp0.wrap_positions(), sp1.wrap_positions()

    assert (jnp.abs(sp0.coordinates - sp1.coordinates) < 1e-6).all()


if __name__ == "__main__":
    pass

    # jax.config.update('jax_platforms', 'cpu')
    # test_cv_split_combine()
    # test_nf()

    # test_reconstruction()

    # test_neigh()
    # test_neigh_pair()

    test_reconstruction()

    # test_minkowski_reduce()
    # test_canoncicalize()
