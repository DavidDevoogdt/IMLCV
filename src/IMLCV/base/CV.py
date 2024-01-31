from __future__ import annotations

import dataclasses
import itertools
from abc import abstractmethod
from dataclasses import KW_ONLY
from functools import partial
from pathlib import Path
from typing import Callable

import cloudpickle
import distrax
import jax.flatten_util
import jax.lax
import jax.numpy as jnp
import jax.scipy.optimize
import jaxopt.objective
import jsonpickle
from flax import linen as nn
from flax.struct import field
from flax.struct import PyTreeNode
from IMLCV import Unpickler
from jax import Array
from jax import jacfwd
from jax import jit
from jax import pmap
from jax import vmap
from jax.tree_util import Partial
from jax.tree_util import tree_flatten
from jax.tree_util import tree_unflatten

######################################
#        Data types                  #
######################################


def padded_pmap(f, n_devices: int | None = None):
    # helper function to pad pytree, apply pmap and unpad the result

    def get_shape(n_devices, batch_dim):
        _, b = jnp.divmod(batch_dim, n_devices)

        if b == 0:
            return 0

        return n_devices - b

    def n_pad(x, n_devices, p):
        if x is None:
            return x

        if p != 0:
            pad_shape = jnp.zeros((len(x.shape), 2), dtype=jnp.int16)
            pad_shape = pad_shape.at[0, 1].set(p)
            pad_shape = jnp.array(pad_shape)
            x_padded = jnp.pad(x, pad_shape)
        else:
            x_padded = x

        return jnp.reshape(x_padded, (n_devices, -1, *x.shape[1:]))

    def n_unpad(y, p):
        y = jnp.reshape(y, (-1, *y.shape[2:]))
        if p != 0:
            y = y[:-p]
        return y

    def apply_pmap_fn(
        *args: PyTreeNode,
        n_devices: int | None = None,
    ):
        if n_devices is None:
            n_devices = jax.device_count("cpu")

        if n_devices == 1:
            # print(f"only 1 device found, skipping parallel mapping")
            # still vmap it
            return f(*args)

        p = None

        # pad the input args
        in_tree, tree_def = tree_flatten(args)

        for i in range(len(in_tree)):
            if p is None:
                p = get_shape(n_devices, in_tree[i].shape[0])
            else:
                assert p == get_shape(n_devices, in_tree[i].shape[0]), "inconsisitent batch dims"

            in_tree[i] = n_pad(in_tree[i], n_devices, p)

        in_tree = tree_unflatten(tree_def, in_tree)

        out_tree = pmap(f)(*in_tree)

        del in_tree  # for memory purposes

        out_tree, tree_def = tree_flatten(out_tree)

        for i in range(len(out_tree)):
            out_tree[i] = n_unpad(out_tree[i], p)

        out_tree = tree_unflatten(tree_def, out_tree)

        return out_tree

    # def apply_pmap_fn(
    #     *args: PyTreeNode,
    #     n_devices: int | None = None,
    # ):
    #     if n_devices is None:
    #         n_devices = jax.device_count("cpu")

    #     if n_devices == 1:
    #         # print(f"only 1 device found, skipping parallel mapping")
    #         # still vmap it
    #         return f(*args)

    #     p = None

    #     # pad the input args
    #     in_tree, tree_def = tree_flatten(args)

    #     for i in range(len(in_tree)):
    #         if p is None:
    #             p = get_shape(n_devices, in_tree[i].shape[0])
    #         else:
    #             assert p == get_shape(n_devices, in_tree[i].shape[0]), "inconsisitent batch dims"

    #         in_tree[i] = n_pad(in_tree[i], n_devices, p)

    #     in_tree = tree_unflatten(tree_def, in_tree)

    #     out_tree = pmap(f)(*in_tree)

    #     del in_tree  # for memory purposes

    #     out_tree, tree_def = tree_flatten(out_tree)

    #     for i in range(len(out_tree)):
    #         out_tree[i] = n_unpad(out_tree[i], p)

    #     out_tree = tree_unflatten(tree_def, out_tree)

    #     return out_tree

    return partial(apply_pmap_fn, n_devices=n_devices)


def chunk_map(f, chunk_size):
    # helper function to pad pytree, apply function in chunks of chunk_size and unpad the result

    def get_shape(chunk_size, batch_dim):
        a = batch_dim // chunk_size
        b = batch_dim - a * chunk_size

        # print(f"{a=}\n{b=}\n{chunk_size=}\n{batch_dim=}\n")

        if b == 0:
            return a, 0

        # print(f"working with {a,b=}")

        return a + 1, chunk_size - b

    def n_pad(x, a, p):
        if x is None:
            return x

        if p != 0:
            pad_shape = [[0, 0] for a in range(x.ndim)]
            pad_shape[0][1] = p
            x_padded = jnp.pad(x, pad_shape)
        else:
            x_padded = x

        return jnp.reshape(x_padded, (a, -1, *x.shape[1:]))

    def n_unpad(y, p):
        out = jnp.reshape(y, (-1, *y.shape[2:]))
        if p != 0:
            out = out[:-p]
        return out

    def _f(*args, **kwargs):
        if chunk_size is None:
            return f(*args, **kwargs)

        tree_padded, tree_def = tree_flatten((args, kwargs))
        leaves_padded = []

        a = None
        p = None

        for leaf_padded in tree_padded:
            if p is None:
                a, p = get_shape(chunk_size, leaf_padded.shape[0])
            else:
                a_2, p_2 = get_shape(chunk_size, leaf_padded.shape[0])

                assert a == a_2 and p == p_2, "inconsisitent batch dims"

            leaves_padded.append(n_pad(leaf_padded, a, p))

        def __f(x):
            (args, kwargs) = tree_unflatten(tree_def, x)
            return f(*args, **kwargs)

        out_padded = jax.lax.map(__f, leaves_padded)

        # remove padding from output leaves
        tree_padded, tree_def = tree_flatten(out_padded)
        leaves = []

        for leaf_padded in tree_padded:
            leaves.append(n_unpad(leaf_padded, p))

        tree_unpadded = tree_unflatten(tree_def, leaves)

        return tree_unpadded

    return _f


class SystemParams(PyTreeNode):
    coordinates: Array
    cell: Array | None = field(default=None)

    def __getitem__(self, slices) -> SystemParams:
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
    def batch_dim(self):
        if self.batched:
            return self.shape[0]
        return 1

    @property
    def shape(self):
        return self.coordinates.shape

    def __add__(self, other):
        assert isinstance(other, SystemParams), f"{other=} is not { SystemParams} "
        if self.cell is None:
            assert other.cell is None

        s = self.batch()
        o = other.batch()

        assert s.shape[1:] == o.shape[1:]

        return SystemParams(
            coordinates=jnp.vstack([s.coordinates, o.coordinates]),
            cell=None if self.cell is None else jnp.vstack([s.cell, o.cell]),
        )

    @staticmethod
    def stack(*sps: SystemParams) -> SystemParams:
        return sum(sps[1:], sps[0])

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

    def _get_neighbour_list(
        self,
        r_cut,
        r_skin: float = 0.0,
        z_array: tuple[int] | None = None,
        z_unique: tuple[int] | None = None,
        num_z_unique: tuple[int] | None = None,
        num_neighs: int | None = None,
        nxyz: tuple[int] | None = None,
        chunk_size=None,
        max_neighs=None,
    ) -> tuple[bool, NeighbourList | None]:
        if r_cut is None:
            return False, None, None, None

        sp, (op_cell, op_coor, _) = self.canonicalize()

        b = True

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
                jnp.array([e1, e2, e3]),
                jnp.array([v1, v2, v3]),
            )

            bounds = jnp.ceil(jnp.sum(jnp.abs(jnp.linalg.inv(proj)) * r_cut, axis=1))

            bounds = jnp.nan_to_num(bounds)

            return bounds

        # cannot be jitted
        if sp.cell is not None:
            # if nxyz is None:
            if sp.batched:
                new_nxyz = jnp.max(
                    vmap(
                        lambda sp: _get_num_per_images(sp.cell, r_cut + r_skin),
                    )(sp),
                    axis=0,
                )

            else:
                new_nxyz = _get_num_per_images(sp.cell, r_cut + r_skin)

            if nxyz is None:
                nxyz = [int(i) for i in new_nxyz.tolist()]
            else:
                b = b and (jnp.array(new_nxyz) <= jnp.array(nxyz)).all()  # only check r_cut, because we are retracing

            nx, ny, nz = nxyz
            bx = jnp.arange(-nx, nx + 1)
            by = jnp.arange(-ny, ny + 1)
            bz = jnp.arange(-nz, nz + 1)
        else:
            bx, by, bz = None, None, None
            new_nxyz = None

        # this is batchable

        @jax.jit
        def func(r_ij, index, i, j, k):
            return (
                jnp.linalg.norm(r_ij),
                jnp.array(index),
                jnp.array([i, j, k]),
            )

        @jax.jit
        def func2(r_ij, index, i, j, k):
            return (
                jnp.linalg.norm(r_ij),
                jnp.array(index),
                None,
            )

        def _apply_g_inner(
            sp: SystemParams,
            func,
            r_cut,
            ijk=(None, None, None),
            exclude_self=True,
        ):
            i, j, k = ijk

            if ijk != (None, None, None):
                assert sp.cell is not None
                pos = sp.coordinates + i * sp.cell[0, :] + j * sp.cell[1, :] + k * sp.cell[2, :]
            else:
                pos = sp.coordinates

            norm2 = jnp.sum(pos**2, axis=1)
            index_j = jnp.ones_like(norm2, dtype=jnp.int32).cumsum() - 1

            bools = norm2 < r_cut**2

            if exclude_self:
                bools = jnp.logical_and(
                    bools,
                    jnp.logical_not(
                        vmap(jnp.allclose, in_axes=(0, None))(pos, jnp.zeros((3,))),
                    ),
                )

            true_val = vmap(func, in_axes=(0, 0, None, None, None))(pos, index_j, i, j, k)
            false_val = jax.tree_map(
                jnp.zeros_like,
                true_val,
            )

            val = vmap(
                lambda b, x, y: jax.tree_map(lambda t, f: jnp.where(b, t, f), x, y),
            )(bools, true_val, false_val)

            return bools, val

        @partial(vmap, in_axes=(None, 0, None))
        @partial(jax.jit, static_argnames=["take_num"])
        def res(sp, center_coordinates, take_num=0):
            if center_coordinates is not None:
                sp_center = SystemParams(sp.coordinates - center_coordinates, sp.cell)
            else:
                sp_center = SystemParams(sp.coordinates, sp.cell)

            sp_center, center_op = sp_center.wrap_positions(min=True)

            if sp_center.cell is None:
                _, (r, atoms, _) = _apply_g_inner(
                    sp=sp_center,
                    func=func2,
                    r_cut=jnp.inf,
                    exclude_self=False,
                )

                idx = jnp.argsort(r)[0:take_num]
                n = jnp.sum(r < r_cut + r_skin)
                return n, r[idx], atoms[idx], None, center_op

            _, (r, atoms, indices) = vmap(
                vmap(
                    vmap(
                        lambda i, j, k: _apply_g_inner(
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
            )(bx, by, bz)

            r = jnp.reshape(r, (-1,))
            atoms = jnp.reshape(atoms, (-1))
            indices = jnp.reshape(indices, (-1, 3))

            idx = jnp.argsort(r)[0:take_num]

            n = jnp.sum(r < r_cut + r_skin)

            return n, r[idx], atoms[idx], indices[idx, :], center_op

        @partial(jax.jit, static_argnames=["take_num"])
        def _f(sp: SystemParams, take_num):
            n, r, a, ijk, center_op = res(sp, sp.coordinates, take_num)
            num_neighs = jnp.max(n)

            return num_neighs, r, a, ijk, center_op

        def get_f(take_num):
            f = Partial(_f, take_num=take_num)
            if sp.batched:
                f = padded_pmap(chunk_map(vmap(f), chunk_size=chunk_size))
            return f

        # not jittable
        if num_neighs is None:
            nn, _, _, _, _ = get_f(1)(sp)
            if sp.batched:
                nn = jnp.max(nn)  # ingore: type

            num_neighs = int(nn)

        nn, r, a, ijk, center_op = get_f(num_neighs)(sp)

        if sp.batched:
            nn = jnp.max(nn)  # ingore: type

        b = jnp.logical_and(b, nn <= num_neighs)

        return (
            b,
            nn,
            new_nxyz,
            NeighbourList(
                r_cut=r_cut,
                r_skin=r_skin,
                atom_indices=a,
                ijk_indices=ijk,
                z_array=z_array,
                z_unique=z_unique,
                num_z_unique=num_z_unique,
                sp_orig=self,
                nxyz=nxyz,
                op_cell=op_cell,
                op_coor=op_coor,
                op_center=center_op,
            ),
        )

    def get_neighbour_list(
        self,
        r_cut,
        z_array: list[int] | Array,
        r_skin=1.0,
        chunk_size=None,
    ) -> NeighbourList | None:
        def to_tuple(a):
            if a is None:
                return None
            return tuple([int(ai) for ai in a])

        zu = jnp.unique(jnp.array(z_array)) if z_array is not None else None
        nzu = vmap(lambda zu: jnp.sum(jnp.array(z_array) == zu))(zu) if zu is not None else None

        b, _, _, nl = self._get_neighbour_list(
            r_cut=r_cut,
            r_skin=r_skin,
            z_array=to_tuple(z_array),
            z_unique=to_tuple(zu),
            num_z_unique=to_tuple(nzu),
            chunk_size=chunk_size,
        )
        return nl

    @jax.jit
    def minkowski_reduce(self) -> tuple[SystemParams, Array]:
        """base on code from ASE: https://wiki.fysik.dtu.dk/ase/_modules/ase/geometry/minkowski_reduction.html"""
        if self.cell is None:
            return self, jnp.eye(3)

        import itertools

        TOL = 1e-12

        @partial(jit, static_argnums=(0,))
        def cycle_checker(d):
            assert d in [2, 3]
            max_cycle_length = {2: 60, 3: 3960}[d]
            return jnp.zeros((max_cycle_length, 3 * d), dtype=int)

        @jax.jit
        def add_site(visited, H):
            # flatten array for simplicity
            H = H.ravel()

            # check if site exists
            found = (visited == H).all(axis=1).any()

            # shift all visited sites down and place current site at the top
            visited = jnp.roll(visited, 1, axis=0)
            visited = visited.at[0].set(H)
            return visited, found

        @jax.jit
        def reduction_gauss(B, hu, hv):
            """Calculate a Gauss-reduced lattice basis (2D reduction)."""
            visited = cycle_checker(2)
            u = hu @ B
            v = hv @ B

            def body(vals):
                u, v, hu, hv, visited, found, i = vals

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
                        jnp.logical_or(jnp.dot(u, u) >= jnp.dot(v, v), found),
                        i != 0,
                    ),
                )

            u, v, hu, hv, visited, found, i = jax.lax.while_loop(
                cond_fun=cond_fun,
                body_fun=body,
                init_val=(u, v, hu, hv, visited, False, 0),
            )

            return hv, hu

        @jax.jit
        def relevant_vectors_2D(u, v):
            cs = jnp.array(list(itertools.product([-1, 0, 1], repeat=2)))
            vs = cs @ jnp.array([u, v])
            indices = jnp.argsort(jnp.linalg.norm(vs, axis=1))[:7]
            return vs[indices], cs[indices]

        @jax.jit
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
                    jnp.round(-jnp.dot(t, r) / jnp.dot(r, r)),
                    dtype=jnp.int32,
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

        @jax.jit
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
                    jnp.logical_and(
                        jnp.logical_or(
                            norms[2] >= norms[1],
                            found,
                        ),
                        i != 0,
                    ),
                )

            H, norms, visited, found, i = jax.lax.while_loop(
                cond_fun,
                body,
                (H, norms, visited, False, 0),
            )

            return H @ B, H

        @jax.jit
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
                ],
            )
            lhs = jnp.linalg.norm(A @ cell, axis=1)
            norms = jnp.linalg.norm(cell, axis=1)
            rhs = norms[jnp.array([0, 1, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2])]

            return (lhs >= rhs - TOL).all()

        @jax.jit
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

            reduced, op = jax.lax.cond(
                is_minkowski_reduced(cell=cell),
                lambda: (cell, jnp.eye(3, dtype=int)),
                lambda: reduction_full(cell),
            )

            change_handedness = jnp.sign(jnp.linalg.det(cell)) * jnp.sign(jnp.linalg.det(reduced))
            # jax.debug.print("handedness change {}", change_handedness)

            op = change_handedness * op

            return op @ cell, op

        if self.batched:
            cell, op = vmap(minkowski_reduce)(self.cell)
        else:
            cell, op = minkowski_reduce(self.cell)

        return SystemParams(coordinates=self.coordinates, cell=cell), op

    @jax.jit
    def apply_minkowski_reduction(self, op):
        assert not self.batched, "apply vamp"

        if self.cell is None:
            return self

        if op is None:
            return self

        return SystemParams(self.coordinates, op @ self.cell)

    @jax.jit
    def rotate_cell(self) -> tuple[SystemParams, tuple[Array, Array] | None]:
        if self.cell is None:
            return self, None

        if self.batched:
            return vmap(SystemParams.rotate_cell)(self)

        q, r = jnp.linalg.qr(self.cell.T)

        # make diagonals positive
        signs = jnp.sign(jnp.diag(r))

        # do not flip sign of new coordinates, as these are absolute
        new_cell = jnp.diag(signs) @ self.cell @ q
        new_coordinates = self.coordinates @ q

        return SystemParams(coordinates=new_coordinates, cell=new_cell), (signs, q)

    @jax.jit
    def apply_rotation(self, op):
        signs, q = op
        sp = self
        if sp.cell is None:
            return sp

        assert not sp.batched, "apply vmap"

        new_cell = jnp.diag(signs) @ self.cell @ q
        new_coordinates = self.coordinates @ q

        return SystemParams(coordinates=new_coordinates, cell=new_cell)

    @partial(jit, static_argnames=["min"])
    def wrap_positions(self, min=False) -> tuple[SystemParams, Array]:
        """wrap pos to lie within unit cell"""

        if self.batched:
            return vmap(Partial(SystemParams.wrap_positions, min=min))(self)

        cell = self.cell
        coordinates = self.coordinates

        if self.cell is None:
            return self, jnp.zeros_like(coordinates, dtype=jnp.int64)

        trans = vmap(vmap(jnp.dot, in_axes=(0, None)), in_axes=(None, 0))(
            vmap(lambda x: x / jnp.linalg.norm(x))(cell),
            jnp.eye(3),
        )

        @partial(vmap)
        def proj(x):
            return jnp.linalg.inv(trans) @ x

        @partial(vmap)
        def deproj(x):
            return trans @ x

        scaled = proj(coordinates)

        norms = jnp.linalg.norm(cell, axis=1)

        def _f1():
            x0 = scaled / norms + 0.5
            a = jnp.mod(x0, 1)
            b = (a - 0.5) * norms
            return a - x0, b

        def _f2():
            x0 = scaled / norms
            a = jnp.mod(x0, 1)
            b = a * norms
            return a - x0, b

        op, reduced = jax.lax.cond(min, _f1, _f2)

        return SystemParams(deproj(reduced), cell), op

    @jax.jit
    def apply_wrap(sp: SystemParams, wrap_op: Array) -> SystemParams:
        if sp.cell is None:
            return sp

        assert not sp.batched, "apply vmap"

        return SystemParams(sp.coordinates + wrap_op @ sp.cell, sp.cell)

    @partial(jit, static_argnames=["min", "qr"])
    def canonicalize(self, min=False, qr=False) -> tuple[SystemParams, Array, Array]:
        if self.batched:
            return vmap(lambda sp: sp.canonicalize(min=min, qr=qr))(self)

        mr, op_cell = self.minkowski_reduce()

        op_qr = None

        if qr:
            mr, op_qr = mr.rotate_cell()

        mr, op_coor = mr.wrap_positions(min)

        return mr, (op_cell, op_coor, op_qr)

    @jax.jit
    def apply_canonicalize(self, ops):
        op_cell, op_coor, op_qr = ops
        assert not self.batched, "apply vmap"

        sp: SystemParams = self.apply_minkowski_reduction(op_cell)

        if op_qr is not None:
            sp: SystemParams = sp.apply_rotation(op_qr)

        sp: SystemParams = sp.apply_wrap(op_coor)

        return sp

    def min_distance(self, index_1, index_2):
        assert self.batched is False

        sp, _ = self.canonicalize()  # necessary if cell is skewed
        coor1 = sp.coordinates[index_1, :]
        coor2 = sp.coordinates[index_2, :]

        @partial(vmap, in_axes=(None, None, 0))
        @partial(vmap, in_axes=(None, 0, None))
        @partial(vmap, in_axes=(0, None, None))
        def dist(n0, n1, n2):
            return jnp.linalg.norm(
                coor2 - coor1 + n0 * sp.cell[0, :] + n1 * sp.cell[1, :] + n2 * sp.cell[2, :],
            )

        ind = jnp.array([-1, 0, 1])
        return jnp.min(dist(ind, ind, ind))

    def super_cell(self, n: int | list[int]) -> SystemParams:
        if self.batched:
            return vmap(Partial(SystemParams.super_cell, n=n))(self)

        if isinstance(n, int):
            n = [n, n, n]

        coor = []

        for a, b, c in itertools.product(range(n[0]), range(n[1]), range(n[2])):
            coor.append(self.coordinates + a * self.cell[0, :] + b * self.cell[1, :] + c * self.cell[2, :])

        return SystemParams(coordinates=jnp.concatenate(coor, axis=0), cell=jnp.array(n) @ self.cell)


class NeighbourList(PyTreeNode):
    atom_indices: Array
    r_cut: float = field(pytree_node=False)
    r_skin: float = field(pytree_node=False)

    op_cell: Array | None
    op_coor: Array | None
    op_center: Array | None

    sp_orig: SystemParams | None = field()
    ijk_indices: Array | None = field(default=None)
    nxyz: tuple[int] | None = field(pytree_node=False, default=None)
    z_array: tuple[int] | None = field(pytree_node=False, default=None)
    z_unique: tuple[int] | None = field(pytree_node=False, default=None)
    num_z_unique: tuple[int] | None = field(pytree_node=False, default=None)

    @jax.jit
    def canonicalized_sp(self, sp: SystemParams) -> SystemParams:
        app_can = SystemParams.apply_canonicalize

        if self.batched:
            assert sp.batched
            app_can = vmap(app_can)

        return app_can(sp, (self.op_cell, self.op_coor, None))

    @jax.jit
    def neighbour_pos(self, sp_orig):
        @partial(vmap, in_axes=(0, 0, None, None))
        def neighbour_translate(ijk, a, sp_centered_on_atom, cell):
            # a: index of atom
            out = sp_centered_on_atom[a]
            if ijk is not None:
                (i, j, k) = ijk
                out = out + i * cell[0, :] + j * cell[1, :] + k * cell[2, :]
            return out

        @partial(vmap, in_axes=(0, 0, 0, None, 0))
        def vmap_atoms(
            atom_center_coordinates: Array,
            ijk_indices: Array,
            atom_indices: Array,
            canon_sp: SystemParams,
            op_center: Array,
        ):
            # center on atom

            sp_centered_on_atom = SystemParams(canon_sp.coordinates - atom_center_coordinates, can_sp.cell)
            sp_centered_on_atom = SystemParams.apply_wrap(sp_centered_on_atom, op_center)

            # transform centered neighbours
            return neighbour_translate(
                ijk_indices,
                atom_indices,
                sp_centered_on_atom.coordinates,
                canon_sp.cell,
            )

        if sp_orig.batched:
            assert self.batched
            vmap_atoms = vmap(vmap_atoms)

        can_sp = self.canonicalized_sp(sp_orig)

        return vmap_atoms(
            can_sp.coordinates,
            self.ijk_indices,
            self.atom_indices,
            can_sp,
            self.op_center,
        )

    def apply_fun_neighbour(
        self,
        sp: SystemParams,
        func,
        r_cut=None,
        fill_value=0,
        reduce="full",  # or 'z' or 'none'
        split_z=False,  #
        exclude_self=False,
        chunk_size_neigbourgs=None,
        chunk_size_atoms=None,
        chunk_size_batch=None,
    ):
        if sp.batched:
            return chunk_map(
                vmap(
                    lambda nl, sp: NeighbourList.apply_fun_neighbour(
                        self=nl,
                        sp=sp,
                        func=func,
                        r_cut=r_cut,
                        fill_value=fill_value,
                        reduce=reduce,
                        split_z=split_z,
                        exclude_self=exclude_self,
                        chunk_size_neigbourgs=chunk_size_neigbourgs,
                        chunk_size_atoms=chunk_size_atoms,
                        chunk_size_batch=None,
                    ),
                ),
                chunk_size=chunk_size_batch,
            )(self=self, sp=sp)

        if r_cut is None:
            r_cut = self.r_cut

        pos = self.neighbour_pos(sp)
        ind = self.atom_indices
        r = jnp.linalg.norm(pos, axis=-1)

        bools = r**2 < r_cut**2

        if exclude_self:
            bools = jnp.logical_and(bools, r**2 != 0.0)

        bools = jnp.logical_and(bools, ind != -1)

        if func is None:
            return bools, None

        true_val = chunk_map(
            vmap(
                chunk_map(
                    vmap(func),
                    chunk_size=chunk_size_neigbourgs,
                ),
            ),
            chunk_size=chunk_size_atoms,
        )(pos, ind)

        false_val = jax.tree_map(
            lambda a: jnp.zeros_like(a) + fill_value,
            true_val,
        )

        def _get(bools):
            def _red(bools):
                val = vmap(
                    lambda b, x, y: jax.tree_map(
                        lambda t, f: vmap(jnp.where)(b, t, f),
                        x,
                        y,
                    ),
                )(bools, true_val, false_val)

                if reduce == "full":
                    return jax.tree_map(lambda x: jnp.sum(x, axis=(1)), (bools, val))
                elif reduce == "z":
                    return jax.tree_map(lambda x: jnp.sum(x, axis=(0, 1)), (bools, val))
                elif reduce == "none":
                    return bools, val
                else:
                    raise ValueError(
                        f"unknown value {reduce} for reduce argument of neighbourghfunction, try 'none','z' or 'full'",
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

            return _f(jnp.array(self.z_unique), jnp.array(self.z_array))

        if not split_z:
            return _get(bools)

        assert self.z_array is not None, "provide z_array to neighbourlist"

        @partial(vmap, in_axes=(0, None), out_axes=1)
        def sel(u1, at):
            @vmap
            def _b(at):
                bools_z1 = vmap(lambda x: x == u1)(at)
                return bools_z1

            return _get(jnp.logical_and(_b(at), bools))

        return sel(jnp.array(self.z_unique), jnp.array(self.z_array)[ind])

        # return _apply_fun_neighbour(sp, self)

    def apply_fun_neighbour_pair(
        self,
        sp: SystemParams,
        func_double,
        func_single=None,
        r_cut=None,
        fill_value=0,
        reduce="full",  # or 'z' or 'none'
        split_z=False,  #
        exclude_self=True,
        unique=True,
        chunk_size_neigbourgs=None,
        chunk_size_atoms=None,
        chunk_size_batch=None,
    ):
        """
        Args:
        ______
        func_single=lambda r_ij, atom_index_j: (1,),
        func_double=lambda r_ij, atom_index_j, data_j, r_ik, atom_index_k, data_k: (
            r_ij,
            atom_index_j,
            r_ik,
            atom_index_k,
        ),

        """
        if sp.batched:
            return chunk_map(
                vmap(
                    lambda nl, sp: NeighbourList.apply_fun_neighbour_pair(
                        self=nl,
                        sp=sp,
                        func_single=func_single,
                        func_double=func_double,
                        r_cut=r_cut,
                        fill_value=fill_value,
                        reduce=reduce,
                        split_z=split_z,
                        exclude_self=exclude_self,
                        unique=unique,
                    ),
                ),
                chunk_size=chunk_size_batch,
            )(self=self, sp=sp)

        pos = self.neighbour_pos(sp)
        ind = self.atom_indices

        bools, data_single = self.apply_fun_neighbour(
            sp=sp,
            func=func_single,
            r_cut=r_cut,
            reduce="none",
            fill_value=fill_value,
            exclude_self=exclude_self,
            chunk_size_neigbourgs=chunk_size_neigbourgs,
            chunk_size_atoms=chunk_size_atoms,
        )

        out_ijk = chunk_map(
            lambda x, y, z: vmap(
                lambda x1, y1, z1, x2, y2, z2: chunk_map(
                    vmap(
                        lambda vx1, vy1, vz1: chunk_map(
                            vmap(lambda vx2, vy2, vz2: func_double(vx2, vy2, vz2, vx1, vy1, vz1)),
                            chunk_size=chunk_size_neigbourgs,
                        )(x2, y2, z2),
                    ),
                    chunk_size=chunk_size_neigbourgs,
                )(x1, y1, z1),
            )(x, y, z, x, y, z),
            chunk_size=chunk_size_atoms,
        )(pos, ind, data_single)

        bools = vmap(
            lambda b: vmap(
                vmap(jnp.logical_and, in_axes=(0, None)),
                in_axes=(None, 0),
            )(b, b),
        )(bools)

        if unique:
            bools = vmap(lambda b: b.at[jnp.diag_indices_from(b)].set(False))(bools)

        # check if indices are not -1

        out_ijk_f = jax.tree_map(
            lambda o: jnp.full_like(o, fill_value),
            out_ijk,
        )

        def get(bools):
            def _red(bools):
                val = jax.tree_map(
                    lambda t, f: vmap(vmap(vmap(jnp.where)))(bools, t, f),
                    out_ijk,
                    out_ijk_f,
                )

                if reduce == "full":
                    return jax.tree_map(lambda x: jnp.sum(x, axis=(1, 2)), (bools, val))
                elif reduce == "z":
                    return jax.tree_map(
                        lambda x: jnp.sum(x, axis=(0, 1, 2)),
                        (bools, val),
                    )
                elif reduce == "none":
                    return bools, val
                else:
                    raise ValueError(
                        f"unknown value {reduce} for reduce argument of neighbourghfunction, try 'none','z' or 'full'",
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

            return _f(jnp.array(self.z_unique), jnp.array(self.z_array))

        if not split_z:
            return get(bools)

        assert self.z_array is not None, "provide z_array to neighbourlist"

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

        return sel(
            jnp.array(self.z_unique),
            jnp.array(self.z_unique),
            jnp.array(self.z_array)[self.atom_indices],
        )

    @property
    def batched(self):
        return len(self.atom_indices.shape) == 3

    @property
    def batch_dim(self):
        if self.batched:
            return self.shape[0]
        return 1

    @property
    def shape(self):
        return self.atom_indices.shape

    @property
    def num_neigh(self):
        return self.shape[-1]

    def __getitem__(self, slices):
        return NeighbourList(
            r_cut=self.r_cut,
            atom_indices=self.atom_indices[slices, :] if self.atom_indices is not None else None,
            ijk_indices=self.ijk_indices[slices, :] if self.ijk_indices is not None else None,
            op_cell=self.op_cell[slices, :, :] if self.op_cell is not None else None,
            op_coor=self.op_coor[slices, :, :] if self.op_coor is not None else None,
            op_center=self.op_center[slices, :] if self.op_center is not None else None,
            z_array=self.z_array,
            z_unique=self.z_unique,
            num_z_unique=self.num_z_unique,
            r_skin=self.r_skin,
            sp_orig=self.sp_orig[slices],
            nxyz=self.nxyz,
        )

    @jax.jit
    def needs_update(self, sp: SystemParams) -> bool:
        if self.sp_orig is None:
            return True

        max_displacement = jnp.max(
            jnp.linalg.norm(self.neighbour_pos(self.sp_orig) - self.neighbour_pos(sp), axis=-1),
        )

        return max_displacement > self.r_skin

    @jax.jit
    def update(self, sp: SystemParams) -> tuple[bool, NeighbourList]:
        a, _, _, b = sp._get_neighbour_list(
            r_cut=self.r_cut,
            r_skin=self.r_skin,
            z_array=self.z_array,
            z_unique=self.z_unique,
            num_z_unique=self.num_z_unique,
            num_neighs=self.num_neigh,
            nxyz=self.nxyz,
        )

        return a, b

    def nl_split_z(self, p):
        if self.batched:
            return vmap(NeighbourList.nl_split_z)(self, p)

        bool_masks = [jnp.array(self.z_array) == zu for zu in self.z_unique]

        arg_split = [jnp.argsort(~bm, kind="stable")[0:nzu] for bm, nzu in zip(bool_masks, self.num_z_unique)]
        p = [jax.tree_map(lambda pi: pi[a], tree=p) for a in arg_split]

        return jnp.array(bool_masks), arg_split, p

    def batch(self):
        if self.batched:
            return self

        return NeighbourList(
            r_cut=self.r_cut,
            atom_indices=jnp.expand_dims(self.atom_indices, axis=0),
            ijk_indices=jnp.expand_dims(self.ijk_indices, axis=0) if self.ijk_indices is not None else None,
            op_cell=jnp.expand_dims(self.op_cell, axis=0) if self.op_cell is not None else None,
            op_coor=jnp.expand_dims(self.op_coor, axis=0) if self.op_coor is not None else None,
            op_center=jnp.expand_dims(self.op_center, axis=0) if self.op_center is not None else None,
            z_array=self.z_array,
            z_unique=self.z_unique,
            num_z_unique=self.num_z_unique,
            r_skin=self.r_skin,
            sp_orig=self.sp_orig,
            nxyz=self.nxyz,
        )

    def __add__(self, other):
        assert isinstance(other, NeighbourList)

        if not self.batched:
            self = self.batch()

        if not other.batched:
            other = other.batch()

        assert self.r_cut == other.r_cut
        assert self.r_skin == other.r_skin
        if self.z_array is None:  # pragma: no cover
            assert other.z_array is None
        else:
            assert jnp.all(self.z_array == other.z_array)

        if self.z_unique is None:
            assert other.z_unique is None
        else:
            assert jnp.all(self.z_unique == other.z_unique)

        if self.atom_indices is not None:
            assert other.atom_indices is not None

        if self.nxyz is None:
            assert other.nxyz is None
            nxyz = None
        else:
            nxyz = [max(a, b) for a, b in zip(self.nxyz, other.nxyz)]

        if self.sp_orig is None:
            assert other.sp_orig is None

        if self.ijk_indices is None:
            assert other.ijk_indices is None

        if self.op_cell is None:
            assert other.op_cell is None

        if self.op_coor is None:
            assert other.op_coor is None

        if self.op_center is None:
            assert other.op_center is None

        if self.num_z_unique is None:
            assert other.num_z_unique is None
        else:
            assert jnp.all(self.num_z_unique == other.num_z_unique)

        op_cell = None
        op_coor = None
        op_center = None
        ijk_indices = None
        atom_indices = None

        if self.atom_indices is not None or self.ijk_indices is not None:  # depends on number of neighbours
            if self.atom_indices is not None:
                m = jnp.max(
                    jnp.array([self.atom_indices.shape[-1], other.atom_indices.shape[-1]]),
                )
            else:
                m = jnp.max(
                    jnp.array([self.ijk_indices.shape[-2], other.ijk_indices.shape[-2]]),
                )

            @vmap
            def _p(a):
                n = m - a.shape[-1]

                return jnp.pad(
                    array=a,
                    pad_width=((0, 0), (0, n)),
                    constant_values=-1,
                )

            if self.atom_indices is not None:
                atom_indices = jnp.vstack([_p(self.atom_indices), _p(other.atom_indices)])

            if self.ijk_indices is not None:
                _p = vmap(_p, in_axes=(-1), out_axes=(-1))
                ijk_indices = jnp.vstack([_p(self.ijk_indices), _p(other.ijk_indices)])

        if self.op_cell is not None:
            op_cell = jnp.vstack([self.op_cell, other.op_cell])
        if self.op_coor is not None:
            op_coor = jnp.vstack([self.op_coor, other.op_coor])

        if self.op_center is not None:
            op_center = jnp.vstack([self.op_center, other.op_center])

        return NeighbourList(
            r_cut=self.r_cut,
            r_skin=self.r_skin,
            atom_indices=atom_indices,
            z_array=self.z_array,
            z_unique=self.z_unique,
            nxyz=nxyz,
            sp_orig=self.sp_orig + other.sp_orig if self.sp_orig is not None else None,
            ijk_indices=ijk_indices,
            op_cell=op_cell,
            op_coor=op_coor,
            op_center=op_center,
            num_z_unique=self.num_z_unique,
        )

    @staticmethod
    def stack(*nls: NeighbourList) -> NeighbourList:
        return sum(nls[1:], nls[0])


class CV(PyTreeNode):
    cv: Array = field(pytree_node=True)
    mapped: bool = field(pytree_node=False, default=False)
    atomic: bool = field(pytree_node=False, default=False)
    _combine_dims: list | None = field(pytree_node=False, default=None)
    _stack_dims: list | None = field(pytree_node=False, default=None)

    @property
    def batched(self):
        if self.atomic:
            return len(self.cv.shape) == 3

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

        return self.replace(cv=self.cv + other)

    def __radd__(self, other) -> CV:
        return other + self

    def __sub__(self, other) -> CV:
        assert isinstance(other, Array)
        return self.replace(cv=self.cv - other)

    def __rsub__(self, other) -> CV:
        return other - self

    def __mul__(self, other) -> CV:
        assert isinstance(other, Array)
        return self.replace(cv=self.cv * other)

    def __rmul__(self, other) -> CV:
        return other * self

    def __matmul__(self, other) -> CV:
        assert isinstance(other, Array)
        return self.replace(cv=self.cv @ other)

    def __rmatmul__(self, other) -> CV:
        return other @ self

    def __div__(self, other) -> CV:
        assert isinstance(other, Array)
        return self.replace(cv=self.cv / other)

    def batch(self) -> CV:
        if self.batched:
            return self
        return CV(
            cv=jnp.array([self.cv]),
            mapped=self.mapped,
            atomic=self.atomic,
            _stack_dims=self._stack_dims,
            _combine_dims=self._combine_dims,
        )

    def __iter__(self):
        if not self.batched:
            yield self
            return

        for i in range(self.cv.shape[0]):
            yield self[i]
        return

    def __getitem__(self, idx):
        assert self.batched
        return CV(
            cv=self.cv[idx, :],
            mapped=self.mapped,
            atomic=self.atomic,
            _stack_dims=None,
            _combine_dims=self._combine_dims,
        )

    def unbatch(self) -> CV:
        if not self.batched:
            return self
        assert self.cv.shape[0] == 1
        return CV(
            cv=self.cv[0, :],
            mapped=self.mapped,
            atomic=self.atomic,
            _stack_dims=self._stack_dims,
            _combine_dims=self._combine_dims,
        )

    @staticmethod
    def stack(*cvs: CV) -> CV:
        """stacks a list of CVs into a single CV. The dimenisions are stored such that it can later be unstacked into separated CVs. The CVs are stacked over the batch dimension"""

        assert len(cvs) != 0
        atomic = cvs[0].atomic

        in_dims = None
        mapped = None

        cv_arr = []
        stack_dims = []

        for cv in cvs:
            assert atomic == cv.atomic

            # assert isinstance(cv, CV)
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
            atomic=atomic,
        )

    def unstack(self) -> list[CV]:
        i = 0

        out: list[CV] = []

        for j in self.stack_dims:
            if j == 0:
                # print("skipping empty stack")
                continue

            out += [
                CV(
                    cv=self.cv[i : i + j, :],
                    mapped=self.mapped,
                    _combine_dims=self._combine_dims,
                    atomic=self.atomic,
                ),
            ]
            i += j
        return out

    def split(self, flatten=False) -> list[CV]:
        """inverse operation of combine"""

        if self._combine_dims is None:
            return [CV(cv=self.cv, mapped=self.mapped, atomic=self.atomic)]

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
                    self.cv,
                    start_index=s,
                    slice_size=e,
                    axis=-1,
                ),
                _combine_dims=out_dim[i] if isinstance(out_dim[i], list) else None,
                _stack_dims=self._stack_dims,
                atomic=self.atomic,
            )
            for i, (s, e) in enumerate(zip(start, sz))
        ]

    @staticmethod
    def combine(*cvs: CV, flatten=False) -> CV:
        """merges a list of CVs into a single CV. The dimenisions are stored such that it can later be split into separated CVs. The CVs are combined over the last dimension"""

        out_cv: list[Array] = []
        out_dim: list[int] = []

        mapped = cvs[0].mapped
        batched = cvs[0].batched
        atomic = cvs[0].atomic
        bdim = cvs[0].batch_dim

        stack_dims = cvs[0].stack_dims

        assert len(cvs) != 0
        if len(cvs) == 1:
            return cvs[0]

        def _inner(
            cv: CV,
            batched,
            mapped,
            atomic,
            bdim,
            stack_dims,
        ) -> tuple[list[Array], list[int]]:
            assert mapped == cv.mapped

            if batched is None:
                batched = cv.batched
                if batched:
                    bdim = cv.batch_dim
            else:
                assert batched == cv.batched
                if batched:
                    assert bdim == cv.batch_dim

            if atomic is None:
                atomic = cv.atomic
            else:
                assert atomic == cv.atomic

            assert stack_dims == cv.stack_dims

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
                        a, b = a, b = _inner(ii, batched, mapped, atomic, bdim)

                    cvi += a
                    dimi += b

            else:
                cvi, dimi = simple(cv)

            return cvi, dimi

        for cv in cvs:
            a, b = _inner(cv, batched, mapped, atomic, bdim, stack_dims)
            out_cv += a
            out_dim += b

        # type: ignore
        return CV(
            cv=jnp.hstack(out_cv),
            mapped=mapped,
            _combine_dims=out_dim,
            _stack_dims=cvs[0]._stack_dims,
            atomic=atomic,
        )


class CvMetric(PyTreeNode):
    """class to keep track of topology of given CV. Identifies the periodicitie of CVs and maps to unit square with correct peridicities"""

    bounding_box: jax.Array
    periodicities: jax.Array

    @classmethod
    def create(
        self,
        periodicities=None,
        bounding_box=None,
        # map_meshgrids=None,
    ) -> None:
        if periodicities is None:
            assert bounding_box is not None

            periodicities = [False for _ in bounding_box]

        if isinstance(periodicities, list):
            periodicities = jnp.array(periodicities)

        assert periodicities.ndim == 1

        if bounding_box is None:
            assert periodicities is not None

            bounding_box = jnp.zeros((len(periodicities), 2))
            bounding_box = bounding_box.at[:, 1].set(1.0)
        else:
            if isinstance(bounding_box, list):
                bounding_box = jnp.array(bounding_box)

            if bounding_box.ndim == 1:
                bounding_box = jnp.reshape(bounding_box, (1, 2))

        return CvMetric(bounding_box=bounding_box, periodicities=periodicities)

    def norm(self, x1: CV, x2: CV, k=1.0):
        diff = self.difference(x1=x1, x2=x2) * k
        return jnp.linalg.norm(diff)

    @partial(jit, static_argnums=(2))
    def periodic_wrap(self, x: CV, min=False) -> CV:
        out = CV(cv=self.__periodic_wrap(self.map(x.cv), min=min), mapped=True)

        if x.mapped:
            return out
        return self.unmap(out)

    @jit
    def difference(self, x1: CV, x2: CV) -> Array:
        assert not x1.mapped
        assert not x2.mapped

        return self.min_cv(
            x2.cv - x1.cv,
        )

    def min_cv(self, cv: Array):
        mapped = self.map(cv, displace=False)
        wrapped = self.__periodic_wrap(mapped, min=True)

        return self.unmap(
            wrapped,
            displace=False,
        )

    @partial(jit, static_argnums=(2))
    def __periodic_wrap(self, xs: Array, min=False):
        """Translate cvs such over unit cell.

        min=True calculates distances, False translates one vector inside box
        """

        coor = jnp.mod(xs, 1)  # between 0 and 1
        if min:
            coor = jnp.where(coor > 0.5, coor - 1, coor)  # between [-0.5,0.5]

        return jnp.where(self.periodicities, coor, xs)

    @partial(jit, static_argnums=(2))
    def map(self, x: Array, displace=True) -> Array:
        """transform CVs to lie in unit square."""

        if displace:
            x -= self.bounding_box[:, 0]

        y = x / (self.bounding_box[:, 1] - self.bounding_box[:, 0])

        return y

    @partial(jit, static_argnums=(2))
    def unmap(self, x: Array, displace=True) -> Array:
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

    def grid(self, n, endpoints=None, margin=0.1):
        """forms regular grid in mapped space. If coordinate is periodic, last rows are ommited.

        Args:
            n: number of points in each dim
            map: boolean. True: work in mapped space (default), False: calculate grid in space without metric
            endpoints: if

        Returns:
            meshgrid and vector with distances between points

        """

        if endpoints is None:
            endpoints = ~self.periodicities
        elif isinstance(endpoints, bool):
            endpoints = jnp.full(self.periodicities.shape, endpoints)

        b = self.bounding_box

        if margin is not None:
            diff = (b[:, 1] - b[:, 0]) * margin
            b = b.at[:, 0].set(b[:, 0] - diff)
            b = b.at[:, 1].set(b[:, 1] + diff)

        assert not (jnp.abs(b[:, 1] - b[:, 0]) < 1e-12).any(), "give proper boundaries"
        grid = [jnp.linspace(row[0], row[1], n, endpoint=bool(endpoints[i])) for i, row in enumerate(b)]

        return grid

    @property
    def ndim(self):
        return len(self.periodicities)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, statedict: dict):
        self.__init__(**statedict)

    @staticmethod
    def bounds_from_cv(cv: CV, percentile=1.0, margin=None):
        if margin is None:
            margin = percentile / 100 * 2

        bounds = jnp.percentile(cv.cv, jnp.array([percentile, 100 - percentile]), axis=0).T

        bounds_margin = (bounds[:, 1] - bounds[:, 0]) * margin
        bounds = bounds.at[:, 0].set(bounds[:, 0] - bounds_margin)
        bounds = bounds.at[:, 1].set(bounds[:, 1] + bounds_margin)

        @vmap
        def get_mask(x):
            return jnp.logical_and(jnp.all(x > bounds[:, 0]), jnp.all(x < bounds[:, 1]))

        return bounds, get_mask(cv.cv)


######################################
#       CV tranformations            #
######################################


# U = TypeVar("U")  # Declare type variable "U"

# Callable = Callable[[Callable[[Any], U], Any], Callable[[Any], U]]


def jac_compose(jac1: CV | SystemParams, jac2: CV) -> CV:
    return jac2.replace(cv=jnp.einsum("ij,jnm->inm", jac2.cv, jac1.cv))

    # return jac2.replace(
    #     cv=jac1.replace(
    #         sp=jnp.einsum("ij,jnm->inm", jac2.cv,jac1.coordinates),
    #         cell=jnp.einsum("ij,jnm->inm", jac2.cv,jac1.cell) if jac2.cell is not None else None,
    #     )
    # )


class CvFunInput(PyTreeNode):
    __: KW_ONLY
    input: int = field(pytree_node=False)
    conditioners: tuple[int] | None = field(pytree_node=False, default=None)

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


class _CvFunBase:
    def calc(
        self,
        x: CV,
        nl: NeighbourList | None,
        reverse=False,
        log_det=False,
        jacobian=False,
    ) -> tuple[CV, Array | None]:
        if self.cv_input is not None:
            y, cond = self.cv_input.split(x)
        else:
            y, cond = x, None

        if jacobian:
            out, jac, log_det = self._jac(y, nl, reverse=reverse, conditioners=cond)

        elif log_det:
            out, log_det = self._log_Jf(y, nl, reverse=reverse, conditioners=cond)
            jac = None
        else:
            out = self._calc(y, nl, reverse=reverse, conditioners=cond)
            jac = None
            log_det = None

        if self.cv_input:
            out = self.cv_input.combine(x, out)

        return out, jac, log_det

    @abstractmethod
    def _calc(
        self,
        x: CV,
        nl: NeighbourList | None,
        reverse=False,
        conditioners: list[CV] | None = None,
    ) -> CV:
        pass

    def _log_Jf(
        self,
        x: CV,
        nl: NeighbourList | None,
        reverse=False,
        conditioners: list[CV] | None = None,
    ) -> tuple[CV, Array | None]:
        """naive automated implementation, overrride this"""

        def f(x):
            return self._calc(x, nl, reverse, conditioners)

        a = f(x)
        b = jacfwd(f)(x)
        log_det = jnp.log(jnp.abs(jnp.linalg.det(b.cv.cv)))

        return a, log_det

    def _jac(
        self,
        x: CV,
        nl: NeighbourList | None,
        reverse=False,
        conditioners: list[CV] | None = None,
    ) -> tuple[CV, Array | None]:
        def f(x):
            return self._calc(x, nl, reverse, conditioners)

        a = f(x)
        b = self.jacfun(f)(x).cv
        # print(b)

        return a, b, None

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, statedict: dict):
        # if "static_kwarg_names" in statedict:
        # statedict["static_kwargs"] = statedict.pop("kwargs")

        self.__init__(**statedict)


class CvFunBase(_CvFunBase, PyTreeNode):
    __: KW_ONLY
    cv_input: CvFunInput | None = None
    kwargs: dict = field(pytree_node=True, default_factory=dict)
    static_kwargs: dict = field(pytree_node=False, default_factory=dict)
    jacfun: Callable = field(pytree_node=False, default=jax.jacfwd)


class CvFun(CvFunBase, PyTreeNode):
    __: KW_ONLY
    forward: Callable[[CV, NeighbourList | None, CV | None], CV] | None = field(pytree_node=False, default=None)
    backward: Callable[[CV, NeighbourList | None, CV | None], CV] | None = field(pytree_node=False, default=None)
    conditioners = False

    def _calc(
        self,
        x: CV,
        nl: NeighbourList | None,
        reverse=False,
        conditioners: list[CV] | None = None,
    ) -> CV:
        if conditioners is None:
            c = None
        else:
            c = CV.combine(*conditioners)

        if reverse:
            assert self.backward is not None
            return jax.jit(Partial(self.backward, **self.static_kwargs))(x, nl, c, **self.kwargs)
        else:
            assert self.forward is not None
            return jax.jit(Partial(self.forward, **self.static_kwargs))(x, nl, c, **self.kwargs)

    # def __eq__(self, other):
    #     if not isinstance(other, CvFun):
    #         return False

    #     if not self.kwargs.keys() == self.kwargs.keys():
    #         return False

    #     if not self.forward == other.forward:
    #         return False

    #     if not self.backward == other.backward:
    #         return False

    #     if not self.cv_input == other.cv_input:
    #         return False

    #     if not self.conditioners == other.conditioners:
    #         return False

    #     for k in self.kwargs.keys():
    #         if self.kwargs[k] is None:
    #             if not other.kwargs[k] is None:
    #                 return False
    #         elif isinstance(self.kwargs[k], jax.Array):
    #             if not isinstance(other.kwargs[k], jax.Array):
    #                 return False
    #             if not (self.kwargs[k] == other.kwargs[k]).all():
    #                 return False
    #         else:
    #             if not self.kwargs[k] == other.kwargs[k]:
    #                 return False

    #     return True


class CvFunNn(nn.Module, _CvFunBase):
    """used to instantiate flax linen CvTrans"""

    cv_input: CvFunInput | None = None
    kwargs: dict = field(pytree_node=False, default_factory=dict)
    static_kwargs: dict = field(pytree_node=False, default_factory=dict)
    jacfun: Callable = field(pytree_node=False, default=jax.jacfwd)

    @abstractmethod
    def setup(self):
        pass

    def _calc(
        self,
        x: CV,
        nl: NeighbourList | None,
        reverse=False,
        conditioners: list[CV] | None = None,
    ) -> tuple[CV, CV | None, Array | None]:
        if reverse:
            return self.backward(x, nl, conditioners, **self.kwargs)
        else:
            return self.forward(x, nl, conditioners, **self.kwargs)

    @abstractmethod
    def forward(
        self,
        x: CV,
        nl: NeighbourList | None,
        conditioners: list[CV] | None = None,
    ) -> CV:
        pass

    @abstractmethod
    def backward(
        self,
        x: CV,
        nl: NeighbourList | None,
        conditioners: list[CV] | None = None,
    ) -> CV:
        pass


class CvFunDistrax(nn.Module, _CvFunBase):
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
    jacfun: Callable = dataclasses.field(default=jax.jacfwd)
    cv_input: CvFunInput | None = None

    @abstractmethod
    def setup(self):
        """setups self.bijector"""

    def _calc(
        self,
        x: CV,
        nl: NeighbourList | None,
        reverse=False,
        log_det=False,
        conditioners: list[CV] | None = None,
    ) -> tuple[CV, CV, Array | None]:
        if conditioners is not None:
            z = CV.combine(*conditioners, x).cv
        else:
            z = x.cv

        assert nl is None, "not implemented"

        if reverse:
            assert self.bijector is not None
            return CV(self.bijector.inverse(z), atomic=x.atomic)
        else:
            assert self.bijector is not None
            return CV(self.bijector.forward(z), atomic=x.atomic)

    def _log_Jf(
        self,
        x: CV,
        nl: NeighbourList | None,
        reverse=False,
        conditioners: list[CV] | None = None,
    ) -> tuple[CV, Array | None]:
        """naive implementation, overrride this"""
        assert nl is None, "not implemented"
        assert self.bijector is not None
        if conditioners is not None:
            z = CV.combine(*conditioners, x).cv
        else:
            z = x.cv

        f = self.bijector.inverse_and_log_det if reverse else self.bijector.forward_and_log_det
        z, jac = f(z)
        return CV(cv=z, atomic=x.atomic), jac


class _CombinedCvFun(_CvFunBase):
    __: KW_ONLY
    classes: tuple[tuple[_CvFunBase]]

    def calc(
        self,
        x: CV | list[SystemParams],
        nl: NeighbourList | None,
        reverse=False,
        log_det=False,
        jacobian=False,
    ) -> tuple[CV, CV | None, Array | None]:
        if isinstance(x, CV):
            assert x._combine_dims is not None, "combine dims not set. Use CV.combine(*cvs)"
            splitted = x.split()
        else:
            splitted = x

        out_x = []
        out_log_det = []
        out_jac = []

        for cl, xi in zip(self.classes, splitted):
            ordered = reversed(cl) if reverse else cl

            if log_det:
                log_det = jnp.array(0.0)
            else:
                log_det = None

            jac = None

            for tr in ordered:
                xi, jac_i, log_det_i = tr.calc(x=xi, nl=nl, reverse=reverse, log_det=log_det, jacobian=jacobian)

                if jacobian:
                    if jac is None:
                        jac = jac_i
                    else:
                        jac = jac_compose(jac, jac_i)

                if log_det:
                    assert log_det_i is not None
                    log_det += log_det_i

            out_jac.append(jac)
            out_x.append(xi)
            out_log_det.append(log_det)

        return (
            CV.combine(*out_x),
            CV.combine(*out_jac) if jacobian else None,
            jnp.sum(jnp.array(out_log_det), axis=0) if log_det else None,
        )

    def _log_Jf(
        self,
        x: CV,
        nl: NeighbourList | None,
        reverse=False,
        conditioners: list[CV] | None = None,
    ) -> tuple[CV, Array | None]:
        """naive automated implementation, overrride this"""

        raise NotImplementedError("untested")


class CombinedCvFun(_CombinedCvFun, PyTreeNode):
    __: KW_ONLY
    classes: tuple[tuple[CvFunBase]]


class CombinedCvFunNN(nn.Module, _CombinedCvFun):
    __: KW_ONLY
    classes: tuple[tuple[CvFunNn]]


def _from_array_function(
    x: CV,
    nl: NeighbourList | None,
    conditioners: list[CV] | None = None,
    array_func=None,
    atomic=False,
    **kwargs,
):
    assert conditioners is None, "implement this"

    if x.batched:
        out = vmap(array_func)(x.cv, nl, None, **kwargs)
    else:
        out = array_func(x.cv, nl, None, **kwargs)

    return x.replace(cv=out, atomic=atomic)


def _duplicate_trans(x: CV | SystemParams, nl: NeighbourList | None, _, n):
    if isinstance(x, SystemParams):
        return [x] * n

    return CV.combine(*[x] * n)


class _CvTrans:
    """f can either be a single CV tranformation or a list of transformations"""

    trans: tuple[CvFunBase]

    @property
    def _comb(self) -> type[_CombinedCvFun]:
        if isinstance(self, CvTrans):
            return CombinedCvFun

        if isinstance(self, CvTransNN):
            return CombinedCvFunNN

        raise

    @property
    def _cv_trans(self) -> type[_CvTrans]:
        return self.__class__

    @staticmethod
    def from_array_function(
        f: Callable[[Array, NeighbourList | None, None], Array],
        jacfun: Callable = None,
        atomic=False,
        **kwargs,
    ):
        return CvTrans.from_cv_function(f=_from_array_function, array_func=f, atomic=atomic, jacfun=jacfun, **kwargs)

    @staticmethod
    def from_cv_function(
        f: Callable[[CV, NeighbourList | None, CV | None], CV],
        jacfun: Callable = None,
        static_argnames=None,
        **kwargs,
    ) -> CvTrans:
        static_kwargs = {}

        if static_argnames is not None:
            for a in static_argnames:
                static_kwargs[a] = kwargs.pop(a)

        kw = dict(forward=f, kwargs=kwargs, static_kwargs=static_kwargs)

        if jacfun is not None:
            kw["jacfun"] = jacfun

        return CvTrans.from_cv_fun(proto=CvFun(**kw))

    @staticmethod
    def from_cv_fun(proto: _CvFunBase):
        if isinstance(proto, nn.Module):
            return CvTransNN(trans=(proto,))
        return CvTrans(trans=(proto,))

    def compute_cv_trans(
        self,
        x: CV | SystemParams,
        nl: NeighbourList | None = None,
        reverse=False,
        log_Jf=False,
        chunk_size=None,
        jacobian=False,
    ) -> tuple[CV, CV | None, Array | None]:
        """
        result is always batched
        arg: CV
        """
        if x.batched:
            return chunk_map(
                vmap(
                    Partial(
                        _CvTrans.compute_cv_trans,
                        self=self,
                        reverse=reverse,
                        log_Jf=log_Jf,
                        chunk_size=None,
                        jacobian=jacobian,
                    ),
                ),
                chunk_size=chunk_size,
            )(x=x, nl=nl)

        ordered = reversed(self.trans) if reverse else self.trans

        if log_Jf:
            log_det = jnp.array(0.0)
        else:
            log_det = None

        jac = None

        for tr in ordered:
            # print(f"{tr=} {x=}")

            x, jac_i, log_det_i = tr.calc(x=x, nl=nl, reverse=reverse, log_det=log_Jf, jacobian=jacobian)

            if jacobian:
                if jac is None:
                    jac = jac_i
                else:
                    jac = jac_compose(jac, jac_i)

            if log_Jf:
                assert log_det_i is not None
                log_det += log_det_i

        return x, jac, log_det

    def __mul__(self, other):
        assert isinstance(other, self._cv_trans), f"can only multiply by {self._cv_trans} object"
        return self._cv_trans(
            trans=(
                *self.trans,
                *other.trans,
            ),
        )

    def __add__(self, other: _CvTrans) -> _CvTrans:
        assert isinstance(other, self._cv_trans), f"can only add by {self._cv_trans} object"

        dt = CvTrans.from_cv_function(_duplicate_trans, static_argnames=["n"], n=2)

        return dt * self._cv_trans(
            trans=(self._comb(classes=(self.trans, other.trans)),),
        )

    @staticmethod
    def stack(*cv_trans: _CvTrans):
        n = len(cv_trans)

        _cv_trans = cv_trans[0]._cv_trans
        _cv_comb = cv_trans[0]._comb

        for i in cv_trans:
            assert isinstance(i, _cv_trans)

        dt = CvTrans.from_cv_function(_duplicate_trans, static_argnames=["n"], n=n)

        return dt * _cv_trans(
            trans=tuple([_cv_comb(classes=tuple([cvt.trans for cvt in cv_trans]))]),
        )

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, statedict: dict):
        self.__init__(**statedict)


class CvTrans(_CvTrans, PyTreeNode):
    __: KW_ONLY
    trans: tuple[CvFunBase]

    @partial(jit, static_argnames=("reverse", "log_Jf", "chunk_size", "jacobian"))
    def compute_cv_trans(
        self,
        x: CV | SystemParams,
        nl: NeighbourList | None = None,
        reverse=False,
        log_Jf=False,
        jacobian=False,
        chunk_size=None,
    ) -> tuple[CV, CV, Array | None]:
        return _CvTrans.compute_cv_trans(
            self=self,
            x=x,
            nl=nl,
            reverse=reverse,
            log_Jf=log_Jf,
            chunk_size=chunk_size,
            jacobian=jacobian,
        )


class CvTransNN(nn.Module, _CvTrans):
    trans: tuple[CvFunNn]

    def setup(self) -> None:
        for i, a in enumerate(self.trans):
            if not isinstance(a, nn.Module):
                self.trans[i] = CvFunNn(cv_input=a.cv_input)

    @nn.compact
    def compute_cv_trans(
        self,
        x: CV | SystemParams,
        nl: NeighbourList | None,
        reverse=False,
        log_Jf=False,
    ) -> tuple[CV, CV, Array | None]:
        return _CvTrans.compute_cv_trans(
            self=self,
            x=x,
            nl=nl,
            reverse=reverse,
            log_Jf=log_Jf,
        )

    @nn.nowrap
    def __mul__(self, other):
        return _CvTrans.__mul__(self, other)

    @nn.nowrap
    def __add__(self, other):
        return _CvTrans.__add__(self, other)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, statedict: dict):
        self.__init__(**statedict)


class NormalizingFlow(nn.Module):
    """normalizing flow. _ProtoCvTransNN are stored separately because they need to be initialized by this module in setup"""

    flow: CvTransNN

    def setup(self) -> None:
        if isinstance(self.flow, CvTrans):
            self.flow = CvTransNN(trans=self.flow.trans)

    def calc(self, x: CV, nl: NeighbourList | None, reverse: bool):
        a, _, b = self.flow.compute_cv_trans(x, nl, reverse=reverse, log_Jf=True)

        return a, b

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, statedict: dict):
        self.__init__(**statedict)


class CvFlow(PyTreeNode):
    func: CvTrans
    trans: CvTrans | None = None

    @staticmethod
    def from_function(
        f: Callable[[SystemParams, NeighbourList | None, CV | None], CV],
        static_argnames=None,
        **kwargs,
    ) -> CvFlow:
        return CvFlow(func=_CvTrans.from_cv_function(f=f, static_argnames=static_argnames, **kwargs))

    @partial(jax.jit, static_argnames=["chunk_size", "jacobian"])
    def compute_cv_flow(
        self,
        x: SystemParams,
        nl: NeighbourList | None = None,
        chunk_size: int | None = None,
        jacobian=False,
    ) -> tuple[CV, CV | None]:
        if x.batched:
            return chunk_map(
                vmap(
                    Partial(self.compute_cv_flow, jacobian=jacobian),
                    in_axes=0,
                ),
                chunk_size=chunk_size,
            )(x=x, nl=nl)

        out, out_jac, _ = self.func.compute_cv_trans(x, nl, jacobian=jacobian, chunk_size=chunk_size)
        # print(f"{out_jac}")

        if self.trans is not None:
            out, out_jac_i, _ = self.trans.compute_cv_trans(x=out, nl=nl, jacobian=jacobian)

            if jacobian:
                out_jac = jac_compose(out_jac, out_jac_i)

        return out, out_jac

    def __add__(self, other) -> CvFlow:
        assert isinstance(other, CvFlow)

        assert self.trans is None
        return CvFlow(func=self.func + other.func)

    def __mul__(self, other) -> CvFlow:
        assert isinstance(other, CvTrans), "can only multiply by CvTrans object"

        if self.trans is None:
            trans = other
        else:
            trans = self.trans * other

        return CvFlow(func=self.func, trans=trans)

    def save(self, file):
        filename = Path(file)

        if filename.suffix == ".json":
            with open(filename, "w") as f:
                f.writelines(jsonpickle.encode(self, indent=1, use_base85=True))
        else:
            with open(filename, "wb") as f:
                cloudpickle.dump(self, f)

    @staticmethod
    def load(file, **kwargs) -> CvFlow:
        filename = Path(file)

        if filename.suffix == ".json":
            with open(filename) as f:
                self = jsonpickle.decode(f.read(), context=Unpickler())
        else:
            with open(filename, "rb") as f:
                self = cloudpickle.load(f)

        for key in kwargs.keys():
            self.__setattr__(key, kwargs[key])

        return self

    def find_sp(
        self,
        x0: SystemParams,
        target: CV,
        target_nl: NeighbourList,
        nl0: NeighbourList | None = None,
        maxiter=10000,
        tol=1e-4,
        norm=lambda cv1, cv2, nl1, nl2: jnp.linalg.norm(cv1 - cv2),
        solver=jaxopt.GradientDescent,
    ) -> SystemParams:
        def loss(sp: SystemParams, nl: NeighbourList, norm):
            b, nl = jit(nl.update)(sp)
            cvi, _ = self.compute_cv_flow(sp, nl)
            nn = norm(cvi.cv, target.cv, nl, target_nl)

            return nn, (b, nl, nn)  # aux output

        _l = jit(partial(loss, norm=norm))

        slvr = solver(
            fun=_l,
            tol=tol,
            has_aux=True,
            maxiter=10,
        )
        state = slvr.init_state(x0, nl=nl0)
        r = jit(slvr.update)

        for _ in range(maxiter):
            x0, state = r(x0, state, nl=nl0)
            b, nl0, nn = state.aux

            print(
                f"step:{state.iter_num} norm {nn:.14f} err {state.error:.4f} update nl={not b}",
            )

            if not b:
                nl0 = x0.get_neighbour_list(
                    r_cut=nl0.r_cut,
                    r_skin=nl0.r_skin,
                    z_array=nl0.z_array,
                )

        sp0 = _l(x0, nl0)
        return sp0

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, statedict: dict):
        self.__init__(**statedict)


######################################
#       Collective variable          #
######################################


class CollectiveVariable(PyTreeNode):
    f: CvFlow
    metric: CvMetric
    jac: Callable = field(pytree_node=False, default=jax.jacrev)  # jacfwd is generally faster, but not always supported

    @partial(jax.jit, static_argnames=["chunk_size", "jacobian"])
    def compute_cv(
        self,
        sp: SystemParams,
        nl: NeighbourList | None = None,
        jacobian=False,
        chunk_size: int | None = None,
    ) -> tuple[CV, CV]:
        if sp.batched:
            if nl is not None:
                assert nl.batched

                return chunk_map(
                    vmap(
                        Partial(
                            self.compute_cv,
                            jacobian=jacobian,
                            chunk_size=None,
                        ),
                    ),
                    chunk_size=chunk_size,
                )(sp=sp, nl=nl)

            return chunk_map(
                vmap(
                    Partial(
                        self.compute_cv,
                        jacobian=jacobian,
                        chunk_size=None,
                        nl=None,
                    ),
                ),
                chunk_size=chunk_size,
            )(sp=sp)

        # cv, dcv = self.f.compute_cv_flow(sp, nl, jacobian=jacobian)
        def _f(sp):
            cv, _ = self.f.compute_cv_flow(sp, nl, jacobian=False)
            return cv

        cv = _f(sp)
        dcv = self.jac(_f)(sp) if jacobian else None
        return (cv, dcv)

    @property
    def n(self):
        return self.metric.ndim

    def save(self, file):
        filename = Path(file)

        if filename.suffix == ".json":
            with open(filename, "w") as f:
                f.writelines(jsonpickle.encode(self, indent=1, use_base85=True))
        else:
            with open(filename, "wb") as f:
                cloudpickle.dump(self, f)

    @staticmethod
    def load(file, **kwargs) -> CollectiveVariable:
        filename = Path(file)

        print("loading CV")

        if filename.suffix == ".json":
            with open(filename) as f:
                self = jsonpickle.decode(f.read(), context=Unpickler())
        else:
            with open(filename, "rb") as f:
                self = cloudpickle.load(f)

        for key in kwargs.keys():
            self.__setattr__(key, kwargs[key])

        return self

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, statedict: dict):
        self.__init__(**statedict)
