from __future__ import annotations

import dataclasses
import itertools
import tempfile
from abc import abstractmethod

# from IMLCV.base.CV import CV, CvTrans
from functools import partial
from importlib import import_module
from typing import TYPE_CHECKING, Callable

import jax
import jax.lax

# import numpy as np
import jax.numpy as jnp
import jax_dataclasses as jdc
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
    from IMLCV.base.MdEngine import StaticTrajectoryInfo

keras: KerasAPI = import_module("tensorflow.keras")


@jdc.pytree_dataclass
class SystemParams:
    coordinates: jnp.ndarray
    cell: jnp.ndarray | None = None

    def __post_init__(self):
        if isinstance(self.coordinates, np.ndarray):
            self.__dict__["coordinates"] = jnp.array(self.coordinates)
        if isinstance(self.cell, np.ndarray):
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
            cell=self.cell if self.cell is None else jnp.vstack([s.cell, o.cell]),
        )

    def __str__(self):
        str = f"coordinates shape: \n{self.coordinates.shape}"
        if self.cell is not None:
            str += f"\n cell [Angstrom]:\n{self.cell/angstrom }"

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

    def apply_fun_neighbourghs(
        self,
        r_cut,
        center_coordiantes=jnp.zeros(3),
        g=lambda r_ij, atom_index_j: 1,
        exclude_self=True,
    ):
        """

        the function g should be jax jttable, and takes as arguments the relative vector r_ij and its atom index
        """

        if exclude_self is not True:
            raise NotImplementedError

        @jit
        def apply_g(farg, ijk=None):
            if ijk is not None:
                i, j, k = ijk

            @jit
            def _f(atom_index_j, farg, ijk):
                c = self.coordinates[atom_index_j]
                if ijk is not None:
                    pos = c + jnp.array([i, j, k]) @ sp.cell
                else:
                    pos = c
                r_ij = pos - center_coordiantes
                norm = jnp.linalg.norm(r_ij)

                def true_fun():
                    out = g(r_ij, atom_index_j)
                    if len(farg) > 2:
                        return tuple(a + b for a, b in zip(farg, (1, *out)))
                    return farg[0] + 1, farg[1] + out

                return jax.lax.cond(
                    jnp.logical_and(norm < r_cut, norm != 0.0),
                    true_fun,
                    lambda: farg,
                )

            farg = jax.lax.fori_loop(
                0,
                self.coordinates.shape[0],
                lambda at_ind, farg: _f(atom_index_j=at_ind, farg=farg, ijk=ijk),
                init_val=farg,
            )

            if ijk is None:
                return farg
            else:
                return j, (i, farg)

        out = jax.eval_shape(g, jnp.zeros(3), 0)
        if not isinstance(out, jax.ShapeDtypeStruct):
            init_val = (0, *[jnp.zeros(shape=o.shape, dtype=o.dtype) for o in out])
        else:
            init_val = (0, jnp.zeros(shape=out.shape, dtype=out.dtype))

        if self.cell is None:
            return apply_g(init_val)

        else:
            # unit cell angles are in range [45,135] degrees
            sp = self.minkowski_reduce()
            _, r = jnp.linalg.qr(sp.cell.T)

            # orthogonal distance for number of blocks
            bounds = jnp.abs(r_cut / jnp.diag(r.T)) + 1
            # off diagonal added distance
            bounds = bounds + jnp.ceil(bounds @ jnp.abs(r.T - jnp.diag(jnp.diag(r.T))))
            bounds = jnp.array([jnp.floor(-bounds), jnp.ceil(bounds) + 1]).T

            # loop over combinations of unit cell vectors
            return jax.lax.fori_loop(
                lower=bounds[0, 0],
                upper=bounds[0, 1],
                body_fun=lambda i, farg: jax.lax.fori_loop(
                    lower=bounds[1, 0],
                    upper=bounds[1, 1],
                    body_fun=lambda j, i_farg: jax.lax.fori_loop(
                        lower=bounds[2, 0],
                        upper=bounds[2, 1],
                        body_fun=lambda k, j_i_farg: apply_g(
                            farg=j_i_farg[1][1], ijk=(j_i_farg[1][0], j_i_farg[0], k)
                        ),
                        init_val=(j, i_farg),
                    )[1],
                    init_val=(i, farg),
                )[1],
                init_val=init_val,
            )

    def apply_fun_neighbourgh_pairs(
        self,
        r_cut,
        center_coordiantes=jnp.zeros(3),
        g=lambda r_ij, r_ik, atom_index_j, atom_index_k: 1,
        exclude_self=True,
        sp2: SystemParams | None = None,
    ):
        """
        function that loops over all pairs of atoms within cutoff radius.

        if sp2 is none, all pairs with itself are made, otherwise all pairs with sp2
        """

        if sp2 is None:
            sp2 = self

        out = jax.eval_shape(g, jnp.zeros(3), jnp.zeros(3), 0, 0)
        if not isinstance(out, jax.ShapeDtypeStruct):
            init_val = (0, *[jnp.zeros(shape=o.shape, dtype=o.dtype) for o in out])
        else:
            init_val = (0, jnp.zeros(shape=out.shape, dtype=out.dtype))

        @jit
        def h(r_ij, r_ik, atom_index_j, atom_index_k):
            def true_fun():
                out = g(r_ij, r_ik, atom_index_j, atom_index_k)

                if len(init_val) > 2:
                    return (1, *out)
                return (1, out)

            return jax.lax.cond(
                jnp.linalg.norm(r_ij - r_ik) != 0.0,
                true_fun,
                lambda: init_val,
            )

        @jit
        def f(r_ij, atom_index_j):
            return sp2.apply_fun_neighbourghs(
                r_cut,
                center_coordiantes=center_coordiantes,
                g=lambda r_ik, atom_index_k: h(r_ij, r_ik, atom_index_j, atom_index_k),
                exclude_self=exclude_self,
            )

        i, j, k, val = self.apply_fun_neighbourghs(
            r_cut,
            center_coordiantes=center_coordiantes,
            g=f,
            exclude_self=exclude_self,
        )

        return k, val

    def minkowski_reduce(self) -> SystemParams:
        """base on code from ASE: https://wiki.fysik.dtu.dk/ase/_modules/ase/geometry/minkowski_reductsp.cellion.html#minkowski_reduce"""
        if self.cell is None:
            return self

        import itertools

        TOL = 1e-12
        # MAX_IT = 100000  # in practice this is not exceeded

        @partial(jit, static_argnums=(0,))
        def cycle_checker(d):
            assert d in [2, 3]
            max_cycle_length = {2: 60, 3: 3960}[d]
            return jnp.zeros((max_cycle_length, 3 * d), dtype=int)

        def add_site(visited, H):
            # flatten array for simplicity
            H = H.ravel()

            # check if site exists
            found = (visited == H).all(axis=1).any()

            # shift all visited sites down and place current site at the top
            visited = jnp.roll(visited, 1, axis=0)
            visited = visited.at[0].set(H)
            return visited, found

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

        def relevant_vectors_2D(u, v):
            cs = jnp.array(list(itertools.product([-1, 0, 1], repeat=2)))
            vs = cs @ jnp.array([u, v])
            indices = jnp.argsort(jnp.linalg.norm(vs, axis=1))[:7]
            return vs[indices], cs[indices]

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

            # for it in range(MAX_IT):

            # raise RuntimeError(f"Closest vector not found after {MAX_IT} iterations")

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

            return jax.lax.cond(
                is_minkowski_reduced(cell=cell),
                lambda: cell,
                lambda: reduction_full(cell)[1] @ cell,
            )

        if self.batched:
            cell = vmap(minkowski_reduce)(self.cell)
        else:
            cell = minkowski_reduce(self.cell)

        return SystemParams(coordinates=self.coordinates, cell=cell)


@jdc.pytree_dataclass
class CV:
    cv: jnp.ndarray
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
            mapped=mapped,
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

        out_cv: list[jnp.ndarray] = []
        out_dim: list[int] = []

        mapped = None
        batched = None
        bdim = None

        assert len(cvs) != 0
        if len(cvs) == 1:
            return cvs[0]

        def inner(cv: CV) -> tuple[list[jnp.ndarray], list[int]]:
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

                cvi: list[jnp.ndarray] = []
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


class Metric:
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
    def difference(self, x1: CV, x2: CV) -> jnp.ndarray:
        assert not x1.mapped
        assert not x2.mapped

        return self.min_cv(
            x2.cv - x1.cv,
        )

    def min_cv(self, cv: jnp.ndarray):

        mapped = self.__map(cv, displace=False)
        wrapped = self.__periodic_wrap(mapped, min=True)

        return self.__unmap(
            wrapped,
            displace=False,
        )

    @partial(jit, static_argnums=(0, 2))
    def __periodic_wrap(self, xs: jnp.ndarray, min=False):
        """Translate cvs such over unit cell.

        min=True calculates distances, False translates one vector inside box
        """

        coor = jnp.mod(xs, 1)  # between 0 and 1
        if min:
            coor = jnp.where(coor > 0.5, coor - 1, coor)  # between [-0.5,0.5]

        return jnp.where(self.periodicities, coor, xs)

    @partial(jit, static_argnums=(0, 2))
    def __map(self, x: jnp.ndarray, displace=True) -> jnp.ndarray:
        """transform CVs to lie in unit square."""

        if displace:
            x -= self.bounding_box[:, 0]

        y = x / (self.bounding_box[:, 1] - self.bounding_box[:, 0])

        return y

    @partial(jit, static_argnums=(0, 2))
    def __unmap(self, x: jnp.ndarray, displace=True) -> jnp.ndarray:
        """transform CVs to lie in unit square."""

        y = x * (self.bounding_box[:, 1] - self.bounding_box[:, 0])

        if displace:
            x += self.bounding_box[:, 0]

        return y

    def __add__(self, other):
        assert isinstance(self, Metric)
        if other is None:
            return self

        assert isinstance(other, Metric)

        periodicities = jnp.hstack((self.periodicities, other.periodicities))
        bounding_box = jnp.vstack((self.bounding_box, other.bounding_box))

        return Metric(
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


import dataclasses


@dataclasses.dataclass(kw_only=True)
class CvTrans:
    """f can either be a single CV tranformation or a list of transformations"""

    trans: tuple[CvLink]

    @dataclasses.dataclass(kw_only=True)
    class CVInput:
        inputs: list[int]
        relative: bool

        @staticmethod
        def combine(
            self: list[CvTrans.CVInput], other: list[CvTrans.CVInput]
        ) -> list[CvTrans.CVInput]:

            out = self.copy()

            offset = len(self)
            for o in other:
                if o.relative:
                    inp = [a + offset for a in o.inputs]
                else:
                    inp = o.inputs
                out.append(CvTrans.CVInput(inputs=inp, relative=o.relative))

            return out

    @dataclasses.dataclass(kw_only=True)
    class CvFun:
        """used to instantiate flax linen CvTrans"""

        f: Callable[[CV], CV] | None = None
        b: Callable[[CV], CV] | None = None
        cv_input: CvTrans.CVInput | None = None

        def to_cv_link(self):
            if self.cv_input is None:
                self.cv_input = CvTrans.CVInput(inputs=[0], relative=True)

            return CvTrans.CvLink(func=[self], split_indices=[self.cv_input])

        def __add__(self, other):
            assert isinstance(other, CvTrans.CvFun | CvTrans.CvFunNn)
            return self.to_cv_link() + other.to_cv_link()

    class CvFunNn(nn.Module):
        """used to instantiate flax linen CvTrans"""

        _: dataclasses.KW_ONLY
        cv_input: CvTrans.CVInput | None = None

        @abstractmethod
        def setup(self) -> None:
            pass

        @abstractmethod
        def f(self, x: CV) -> CV:
            pass

        def b(self, x: CV) -> CV | None:
            return None

        @nn.nowrap
        def to_cv_link(self):
            if self.cv_input is None:
                self.cv_input = CvTrans.CVInput(inputs=[0], relative=True)

            return CvTrans.CvLink(func=[self], split_indices=[self.cv_input])

        def __add__(self, other) -> CvTrans.CvLink:
            return other + self

    @dataclasses.dataclass(kw_only=True)
    class CvLink:

        _: dataclasses.KW_ONLY
        # funcs= list[ CvTrans.Cv]

        func: list[CvTrans.CvFun | CvTrans.CvFunNn] = dataclasses.field(
            default_factory=lambda: []
        )
        split_indices: list[CvTrans.CVInput]

        def _calc(self, x: CV, reversed=False):
            # todo: fix non used split indices as identity

            cvs = x.split(flatten=True)

            if self.split_indices is not None:

                args = [
                    CV.combine(*[cvs[j] for j in i.inputs]) for i in self.split_indices
                ]
            else:
                args = cvs

            out: list[CV] = []
            for xi, fi in zip(args, self.func):

                if reversed:
                    if fi.b is None:
                        raise
                    val = fi.b(xi)
                    if val is None:
                        raise
                    out.append(val)
                else:
                    if fi.f is None:
                        raise
                    val = fi.f(xi)
                    if val is None:
                        raise
                    out.append(val)

            return CV.combine(*out)

        def fw(self, x: CV):
            return self._calc(x, reversed=False)

        def bw(self, x: CV):
            return self._calc(x, reversed=True)

        def to_chain(self) -> CvTrans:
            return CvTrans(trans=(self,))

        def __add__(
            self, other: CvTrans.CvLink | CvTrans.CvFun | CvTrans.CvFunNn
        ) -> CvTrans.CvLink:

            if not isinstance(other, CvTrans.CvLink):
                other = CvTrans.CvLink(func=[other])

            return CvTrans.CvLink(
                func=[*self.func, *other.func],
                split_indices=CvTrans.CVInput.combine(
                    self.split_indices, other.split_indices
                ),
            )

        def __mul__(
            self, other: CvTrans | CvTrans.CvLink | CvTrans.CvFun | CvTrans.CvFunNn
        ) -> CvTrans:
            if not isinstance(other, CvTrans.CvLink | CvTrans):
                other = other.to_cv_link()

            if not isinstance(other, CvTrans):
                other = other.to_chain()

            return self.to_chain() * other

    @staticmethod
    def from_array_function(f: Callable[[Array], Array]):
        def f2(x: CV):
            if x.batched:
                out = vmap(f)(x.cv)
            else:
                out = f(x.cv)

            return CV(cv=out)

        return CvTrans.from_cv_function(f=f2)

    @staticmethod
    def from_cv_function(f: Callable[[CV], CV]) -> CvTrans:
        return CvTrans.from_cv_fun(proto=CvTrans.CvFun(f=f))

    @staticmethod
    def from_cv_fun(proto: CvFun | CvFunNn):
        return CvTrans.from_cv_link(proto.to_cv_link())

    @staticmethod
    def from_cv_link(cv_link: CvLink):
        return cv_link.to_chain()

    # @partial(jit, static_argnums=(0,))
    def compute_cv_trans(self, x: CV, reverse=False) -> CV:
        """
        result is always batched
        arg: CV
        """

        if reverse:
            for tr in reversed(self.trans):
                x = tr.bw(x)
        else:
            for tr in self.trans:
                x = tr.fw(x)

        return x

    def __mul__(self, other):
        assert isinstance(other, CvTrans), "can only multiply by CvTrans object"
        return CvTrans(trans=(*self.trans, *other.trans))

    def __add__(self, other):
        assert isinstance(other, CvTrans), "can only multiply by CvTrans object"
        return CvTrans(trans=tuple([s + o for s, o in zip(self.trans, other.trans)]))

    def __radd__(self, other):
        return self + other


class CvTransNN(nn.Module, CvTrans):

    trans: tuple[CvTrans.CvLink]

    def setup(self) -> None:
        """replaces all CvFunNn by initilized version"""

        for s, t, link in self.iter_nn_links():
            name = f"links_{s}_{t}"
            self.__setattr__(name, link)
            self.trans[s].func[t] = self.__getattr__(name)

    def iter_nn_links(self):
        for j, tr in enumerate(self.trans):
            for i, fun in enumerate(tr.func):
                if isinstance(fun, CvTrans.CvFunNn):
                    yield j, i, fun

    def compute_cv_trans(self, x: CV, reverse=False) -> CV:
        return super().compute_cv_trans(x, reverse)

    @nn.nowrap
    def __add__(self, other):
        assert isinstance(other, CvTrans | CvTransNN)
        return CvTransNN(trans=(*self.trans, *other.trans))

    @nn.nowrap
    def __mul__(self, other):
        assert isinstance(other, CvTrans | CvTransNN)
        return CvTransNN(trans=tuple([s + o for s, o in zip(self.trans, other.trans)]))


class NormalizingFlow(nn.Module):
    """normalizing flow. _ProtoCvTransNN are stored separately because they need to be initialized by this module in setup"""

    flow: CvTransNN | CvTransNN

    def setup(self) -> None:
        if isinstance(self.flow, CvTrans):
            self.nn_flow = CvTransNN(trans=self.flow.trans)
        else:
            self.nn_flow = self.flow

    def _calc(self, y, reverse: bool):
        def g(cv):
            return self.nn_flow.compute_cv_trans(cv, reverse=reverse)

        a = g(y)
        b = jacfwd(g)(y)

        return a, jnp.abs(jnp.linalg.det(b.cv.cv))

    def forward(self, x: CV) -> CV:
        return self._calc(x, reverse=False)

    def backward(self, x: CV) -> CV:
        return self._calc(x, reverse=True)


class CvFlow:
    def __init__(
        self,
        func: Callable[[SystemParams], CV],
        trans: CvTrans | None = None,
        batched=False,
    ) -> None:
        self.f0 = func
        self.f1 = trans
        self.batched = batched

    @staticmethod
    def from_function(f: Callable[[SystemParams], Array]) -> CvFlow:
        def f2(sp: SystemParams):

            return CV(cv=f(sp))

        return CvFlow(func=f2)

    @partial(jit, static_argnums=(0,))
    def compute_cv_flow(self, x: SystemParams) -> CV:

        out = vmap(self.f0)(x.batch())

        if self.f1 is not None:
            out = self.f1.compute_cv_trans(out)

        if not x.batched:
            out = out.unbatch()

        return out

    def __add__(self, other):
        assert isinstance(other, CvFlow)

        def f_add(x: SystemParams):
            cv1: CV = self.compute_cv_flow(x)
            cv2: CV = other.compute_cv_flow(x)

            return CV.combine(cv1, cv2)

        return CvFlow(func=f_add, batched=True)

    def __mul__(self, other):
        assert isinstance(other, CvTrans), "can only multiply by CvTrans object"

        if self.f1 is None:
            self.f1 = other
        else:
            self.f1 *= other

        return self


class CollectiveVariable:
    def __init__(self, f: CvFlow, metric: Metric, jac=jacfwd) -> None:
        "jac: kind of jacobian. Default is jacfwd (more efficient for tall matrices), but functions with custom jvp's only support jacrev"

        self.metric = metric
        self.f = f
        self.jac = jac

    @partial(jit, static_argnums=(0, 2))
    def compute_cv(self, sp: SystemParams, jacobian=False) -> tuple[CV, CV]:

        # if map:

        #     def cvf(x):
        #         return self.metric.__map(self.f.compute_cv_flow(x))

        # else:
        cvf = self.f.compute_cv_flow

        if sp.batched:
            dcv = vmap(self.jac(cvf))
        else:
            dcv = self.jac(cvf)

        cv = cvf(sp)
        dcv = dcv(sp) if jacobian else None

        assert sp.batched == cv.batched

        return (cv, dcv)

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
    def compute_cv_trans(self, cc: jnp.ndarray):
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
def Volume(sp: SystemParams):
    assert sp.cell is not None, "can only calculate volume if there is a unit cell"

    vol = jnp.abs(jnp.dot(sp.cell[0], jnp.cross(sp.cell[1], sp.cell[2])))
    return vol


def distance_descriptor(tic, permutation="none"):
    @CvFlow.from_function
    def h(x: SystemParams):

        coor = x.coordinates

        n = coor.shape[0]
        out = jnp.zeros((n * (n - 1) // 2,))

        for a, (i, j) in enumerate(itertools.combinations(range(n), 2)):
            out = out.at[a].set(jnp.linalg.norm(coor[i, :] - coor[j, :], 2))

        return out

    return h


def dihedral(numbers):
    """from https://stackoverflow.com/questions/20305272/dihedral-torsion-
    angle-from-four-points-in-cartesian- coordinates-in-python.

    args:
        numbers: list with index of 4 atoms that form dihedral
    """

    @CvFlow.from_function
    def f(sp: SystemParams):

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
    r_cut, sti: StaticTrajectoryInfo, z1: int | None = None, z2: int | None = None
):

    from IMLCV.base.tools.soap_kernel import p_i, p_inl_sb

    @CvFlow.from_function
    def f(sp: SystemParams):

        return p_i(
            sp=sp,
            sti=sti,
            p=p_inl_sb,
            r_cut=r_cut,
            z1=z1,
            z2=z2,
        )

    return f


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


@dataclasses.dataclass(kw_only=True)
class IdentityTrans(CvTrans.CvFun):
    id: Callable[[CV], CV] = lambda x: x

    f: Callable[[CV], CV] = id
    b: Callable[[CV], CV] = id


class RealNVP(CvTrans.CvFunNn):
    """use in combination with swaplink"""

    _: dataclasses.KW_ONLY
    features: int
    cv_input: CvTrans.CVInput

    def setup(self) -> None:
        self.s = Dense(features=self.features)
        self.t = Dense(features=self.features)

    def f(self, x: CV):
        x1, x2 = x.split()
        return CV(cv=x2.cv * self.s(x1.cv) + self.t(x1.cv))

    def b(self, z: CV):
        z1, z2 = z.split()
        return CV(cv=(z2.cv - self.t(z1.cv)) / self.s(z1.cv))


############################
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

        var_a = mf.init(k2, x, False, method=NormalizingFlow._calc)

        y, Jy = mf.apply(variables=var_a, method=NormalizingFlow.forward, x=x)
        x2, Jx = mf.apply(variables=var_a, method=NormalizingFlow.backward, x=y)

        print(f"dx{x.cv - x2.cv}  jac f(f^-1) {Jy*Jx} ")

        print(var_a)

    def get_chain():
        chain = (
            IdentityTrans()
            + RealNVP(
                features=features,
                cv_input=CvTrans.CVInput(inputs=[-1, 0], relative=True),
            )
        ).to_chain()
        return chain

    features = 5

    prng = jax.random.PRNGKey(seed=42)
    ##test 0

    key_1, key_2, prng = jax.random.split(prng, 3)
    x = CV(cv=jax.random.uniform(key=key_1, shape=(10,)), _combine_dims=[[5, 5]])

    test(NormalizingFlow(flow=get_chain()), x, key_2)

    ##test 1

    key_1, key_2, prng = jax.random.split(prng, 3)
    x = CV(cv=jax.random.uniform(key=key_1, shape=(10,)), _combine_dims=[[5, 5]])
    test(NormalizingFlow(flow=get_chain() * get_chain()), x, key_2)

    ##test 2

    key_1, key_2, prng = jax.random.split(prng, 3)
    x2 = CV(
        cv=jax.random.uniform(key=key_1, shape=(20,)), _combine_dims=[[5, 5], [5, 5]]
    )
    test(NormalizingFlow(flow=get_chain() + get_chain()), x2, key_2)


if __name__ == "__main__":
    test_cv_split_combine()

    test_nf()
