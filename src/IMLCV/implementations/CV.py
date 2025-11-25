from functools import partial
from itertools import combinations, permutations

import jax
import jax.numpy as jnp
from jax import Array
from jaxopt.linear_solve import solve_normal_cg

from IMLCV.base.CV import (
    CV,
    CollectiveVariable,
    CvMetric,
    CvTrans,
    NeighbourList,
    NeighbourListInfo,
    SystemParams,
)
from IMLCV.base.datastructures import vmap_decorator
from IMLCV.base.UnitsConstants import angstrom

######################################
#       CV transformations           #
######################################


def _identity_trans(x, nl, shmap, shmap_kwargs):
    return x


def _zero_cv_trans(x: CV, nl, shmap, shmap_kwargs):
    return x.replace(cv=jnp.array([0.0]), combine_dims=None)


def _zero_cv_flow(x: SystemParams, nl, shmap, shmap_kwargs):
    return CV(cv=jnp.array([0.0]))


identity_trans = CvTrans.from_cv_function(_identity_trans)
zero_trans = CvTrans.from_cv_function(_zero_cv_trans)
zero_flow = CvTrans.from_cv_function(_zero_cv_flow)


def _Volume(
    sp: SystemParams,
    _nl,
    shmap,
    shmap_kwargs,
):
    assert sp.cell is not None, "can only calculate volume if there is a unit cell"

    vol = jnp.abs(jnp.linalg.det(sp.cell))
    return CV(cv=jnp.array([vol]))


Volume = CvTrans.from_cv_function(_Volume)


def _real(
    x: CV,
    nl,
    shmap,
    shmap_kwargs,
):
    return x.replace(cv=jnp.angle(x.cv))


cv_trans_real = CvTrans.from_cv_function(_real)


def _lattice_invariants(
    sp: SystemParams,
    nl,
    shmap,
    shmap_kwargs,
):
    sp_red, _ = sp.minkowski_reduce()

    assert sp_red.cell is not None

    M = vmap_decorator(vmap_decorator(jnp.dot, in_axes=(0, None)), in_axes=(None, 0))(sp_red.cell, sp_red.cell)

    l = jnp.linalg.eigvalsh(M)

    return CV(cv=l)


LatticeInvariants = CvTrans.from_cv_function(_lattice_invariants)


def _lattice_invariants_2(
    sp: SystemParams,
    nl,
    shmap,
    shmap_kwargs,
):
    sp_red, _ = sp.minkowski_reduce()

    assert sp_red.cell is not None

    c = sp_red.cell @ sp_red.cell.T

    c0 = jnp.sqrt(c[0, 0])
    c1 = jnp.sqrt(c[1, 1])
    c2 = jnp.sqrt(c[2, 2])

    a01 = jnp.arccos(jnp.abs(c[0, 1] / (c0 * c1)))
    a12 = jnp.arccos(jnp.abs(c[1, 2] / (c1 * c2)))
    a02 = jnp.arccos(jnp.abs(c[0, 2] / (c0 * c2)))

    return CV(cv=jnp.array([c0, c1, c2, a01, a02, a12]))


LatticeInvariants2 = CvTrans.from_cv_function(_lattice_invariants_2)


def _distance(x: SystemParams, *_):
    x = x.canonicalize()[0]

    n = x.shape[-2]

    out = vmap_decorator(vmap_decorator(x.min_distance, in_axes=(0, None)), in_axes=(None, 0))(
        jnp.arange(n),
        jnp.arange(n),
    )

    return x.replace(cv=out[jnp.triu_indices_from(out, k=1)], _combine_dims=None)


def distance_descriptor():
    return CvTrans.from_cv_function(_distance)


def _position_index(sp: SystemParams, _nl, shmap, shmap_kwargs, idx):
    return CV(cv=sp.coordinates.reshape(-1)[idx])


def position_index(indices, sp):
    idx = jnp.ravel_multi_index(indices.T, sp.coordinates.shape)

    return CvTrans.from_cv_function(_position_index, idx=idx)


def _dihedral(sp: SystemParams, _nl, shmap, shmap_kwargs, numbers):
    coor = sp.coordinates
    p0 = coor[numbers[0]]
    p1 = coor[numbers[1]]
    p2 = coor[numbers[2]]
    p3 = coor[numbers[3]]

    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    b1, _ = _norm_safe(b1)

    v = b0 - jnp.dot(b0, b1) * b1
    w = b2 - jnp.dot(b2, b1) * b1

    v, _ = _norm_safe(v)
    w, _ = _norm_safe(w)

    x = jnp.dot(v, w)
    y = jnp.dot(jnp.cross(b1, v), w)

    return CV(cv=jnp.array([jnp.arctan2(y, x)]))


def dihedral(numbers: tuple[int, int, int, int] | Array):
    """from https://stackoverflow.com/questions/20305272/dihedral-torsion-
    angle-from-four-points-in-cartesian- coordinates-in-python.

    args:
        numbers: list with index of 4 atoms that form dihedral
    """

    return CvTrans.from_cv_function(_dihedral, static_argnames=["numbers"], numbers=numbers)


def _atomic_inv_variance_weighing(
    x: CV,
    nl: NeighbourList,
    shmap,
    shmap_kwargs,
    U_list: list[jax.Array],
    # l: list[jax.Array],
    z_indices: list[jax.Array],
    mu: list[jax.Array] | None = None,
):
    print(f"{x=}")

    assert x.atomic, "can only do atomic weighing on atomic CV"
    cv = x.cv

    print(f"{cv.shape=}")

    for i, Ui in enumerate(U_list):
        zi = z_indices[i]
        y = cv[zi, :]

        if mu is not None:
            y -= mu[i]

        y = y @ Ui

        cv = cv.at[zi].set(y)

    return x.replace(cv=cv)


def get_atomic_inv_variance_weighing(
    cv_1: list[CV],
    info: NeighbourListInfo,
    w: list[jax.Array] | None = None,
    remove_mean=True,
    tf_op=lambda x: x**2,
):
    assert cv_1[0].atomic, "can only do atomic weighing on atomic CV"

    cv_tot = jnp.concatenate([a.cv for a in cv_1], axis=0)
    arg_split, _, p_split = jax.vmap(info.nl_split_z)(cv_tot)
    arg_split = [jnp.argwhere(a)[:, 0] for a in arg_split[0]]
    print(f"{arg_split=}")

    U_list = []
    mu_list = []

    for i, a in enumerate(p_split):
        print(f"{i} {a.shape=}")

        x = a.reshape((-1, a.shape[-1]))

        mu = jnp.mean(x, axis=0)

        # y = tf_op(x - mu)
        # y -= jnp.mean(y, axis=0)

        y = x - mu

        print(f"{mu=}")

        l = jnp.var(y, axis=0)
        U = jnp.eye(x.shape[1])

        # cov = jnp.einsum("ni,nj->ij", y, y) / (x.shape[0] - 1)

        # l, U = jnp.linalg.eigh(cov)

        # necessary for to diagonalize distance
        l_inv = 1 / (l + 1e-8)

        # print(f"{l=}")

        l_inv /= jnp.sum(l_inv) ** 0.5
        # print(f"{l_inv=}")

        # l_inv /= jnp.sqrt(y.shape[0])

        # l_inv /= jnp.sqrt(jnp.sum(l_inv**2))

        U_list.append(U @ jnp.diag(l_inv))
        mu_list.append(mu)

    return CvTrans.from_cv_function(
        _atomic_inv_variance_weighing,
        U_list=U_list,
        z_indices=arg_split,
        mu=mu_list,
    )


def _sb_descriptor(
    sp: SystemParams,
    nl: NeighbourList,
    shmap,
    shmap_kwargs,
    r_cut,
    chunk_size_atoms,
    chunk_size_neigbourgs,
    reduce,
    reshape,
    n_max,
    l_max,
    bessel_fun="jax",
    mul_Z=False,
    Z_weights: jax.Array | None = None,
    normalize=False,
    reduce_Z=False,
):
    assert nl is not None, "provide neighbourlist for sb describport"

    from IMLCV.tools.soap_kernel import p_i, p_inl_sb

    f_single, f_double = p_inl_sb(
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        bessel_fun=bessel_fun,
    )

    a = p_i(
        sp=sp,
        nl=nl,
        f_single=f_single,
        f_double=f_double,
        r_cut=r_cut,
        chunk_size_atoms=chunk_size_atoms,
        chunk_size_neigbourgs=chunk_size_neigbourgs,
        shmap=shmap,
        shmap_kwargs=shmap_kwargs,
        mul_Z=mul_Z,
        merge_ZZ=True,
        reshape=True,
        Z_weights=Z_weights,
        normalize=normalize,
        reduce_Z=reduce_Z,
    )

    return CV(cv=a, atomic=True)


def sb_descriptor(
    r_cut,
    n_max: int,
    l_max: int,
    reduce=True,
    reshape=True,
    chunk_size_atoms=None,
    chunk_size_neigbourgs=None,
    bessel_fun="jax",
    mul_Z=True,
    Z_weights: jax.Array | None = None,
    normalize=True,
    reduce_Z=False,
) -> CvTrans:
    return CvTrans.from_cv_function(
        _sb_descriptor,
        static_argnames=[
            "r_cut",
            "chunk_size_atoms",
            "chunk_size_neigbourgs",
            "reduce",
            "reshape",
            "n_max",
            "l_max",
            "bessel_fun",
            "mul_Z",
            "normalize",
            "reduce_Z",
        ],
        r_cut=r_cut,
        chunk_size_atoms=chunk_size_atoms,
        chunk_size_neigbourgs=chunk_size_neigbourgs,
        reduce=reduce,
        reshape=reshape,
        n_max=n_max,
        l_max=l_max,
        bessel_fun=bessel_fun,
        mul_Z=mul_Z,
        Z_weights=Z_weights,
        normalize=normalize,
        reduce_Z=reduce_Z,
    )


def _soap_descriptor(
    sp: SystemParams,
    nl: NeighbourList,
    shmap,
    shmap_kwargs,
    r_cut,
    reduce,
    reshape,
    n_max,
    l_max,
    sigma_a,
    r_delta,
    num,
    basis,
    mul_Z=True,
):
    assert nl is not None, "provide neighbourlist for soap describport"

    from IMLCV.tools.soap_kernel import p_i, p_innl_soap

    f_single, f_double = p_innl_soap(
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        sigma_a=sigma_a,
        r_delta=r_delta,
        num=num,
        basis=basis,
        reduce=reduce,
    )

    a = p_i(
        sp=sp,
        nl=nl,
        f_single=f_single,
        f_double=f_double,
        r_cut=r_cut,
        shmap=shmap,
        shmap_kwargs=shmap_kwargs,
        mul_Z=mul_Z,
        merge_ZZ=True,
        reshape=True,
    )

    return CV(cv=a, atomic=True)


def soap_descriptor(
    r_cut,
    n_max: int,
    l_max: int,
    sigma_a: float,
    r_delta: float,
    reduce=True,
    reshape=True,
    num=50,
    basis="cos",
    mul_Z=True,
) -> CvTrans:
    return CvTrans.from_cv_function(
        _soap_descriptor,
        static_argnames=[
            "r_cut",
            "reduce",
            "reshape",
            "n_max",
            "l_max",
            "sigma_a",
            "r_delta",
            "num",
            "basis",
            "mul_Z",
        ],
        r_cut=r_cut,
        reduce=reduce,
        reshape=reshape,
        n_max=n_max,
        l_max=l_max,
        sigma_a=sigma_a,
        r_delta=r_delta,
        num=num,
        basis=basis,
        mul_Z=mul_Z,
    )


def NoneCV() -> CollectiveVariable:
    return CollectiveVariable(
        f=zero_flow,
        metric=CvMetric.create(
            periodicities=[False],
            bounding_box=[[-0.5, 0.5]],
        ),
    )


def _norm_safe(x, axis=None, ord=2):
    x_norm = jnp.sum(x**2)

    x_norm_safe = jnp.where(x_norm <= 1e-10, 1.0, x_norm)

    x_inv_norm = jnp.where(x_norm <= 1e-10, 0.0, 1 / jnp.sqrt(x_norm_safe))
    x_norm = jnp.where(x_norm <= 1e-10, 0.0, jnp.sqrt(x_norm_safe))

    return x * x_inv_norm, x_norm


def _quad(sp: SystemParams, _nl, shmap, shmap_kwargs, quads):
    @jax.vmap
    def get(p0, p1, p2, p3):
        b0 = -1.0 * (p1 - p0)
        b1 = p2 - p1
        b2 = p3 - p2

        b1, _ = _norm_safe(b1)

        v = b0 - jnp.dot(b0, b1) * b1
        w = b2 - jnp.dot(b2, b1) * b1

        v, _ = _norm_safe(v)
        w, _ = _norm_safe(w)

        cos = jnp.dot(v, w)
        sin = jnp.dot(jnp.cross(b1, v), w)

        return jnp.array([cos, sin])

    print(f"{quads.shape=}")

    vals = get(
        sp.coordinates[quads[:, 0]],
        sp.coordinates[quads[:, 1]],
        sp.coordinates[quads[:, 2]],
        sp.coordinates[quads[:, 3]],
    )

    print(f"quad {vals.shape=}")

    return CV(cv=vals.reshape(-1))


def quad(quads: Array):
    """from https://stackoverflow.com/questions/20305272/dihedral-torsion-
    angle-from-four-points-in-cartesian- coordinates-in-python.

    args:
        numbers: list with index of 4 atoms that form dihedral
    """

    print(f"{quads.shape=}")

    return CvTrans.from_cv_function(_quad, quads=quads)


def _trip(sp: SystemParams, _nl, shmap, shmap_kwargs, trips):
    @jax.vmap
    def get(p0, p1, p2):
        b0 = p0 - p1
        b1 = p2 - p1

        b0, _ = _norm_safe(b0)
        b1, _ = _norm_safe(b1)

        cos = jnp.dot(b0, b1)

        # _, sin = _norm_safe(jnp.cross(b0, b1))

        return jnp.array([cos])

    vals = get(
        sp.coordinates[trips[:, 0]],
        sp.coordinates[trips[:, 1]],
        sp.coordinates[trips[:, 2]],
    )

    print(f"trip {vals.shape=}")

    return CV(cv=vals.reshape(-1))


def trip(trips: Array):
    return CvTrans.from_cv_function(_trip, trips=trips)


def _pair(sp: SystemParams, _nl, shmap, shmap_kwargs, pairs):
    @jax.vmap
    def get(p0, p1):
        b0 = p0 - p1

        return _norm_safe(b0)[1]

    vals = get(
        sp.coordinates[pairs[:, 0]],
        sp.coordinates[pairs[:, 1]],
    )

    print(f"pair {vals.shape=}")

    return CV(cv=vals)


def pair(pairs: Array):
    return CvTrans.from_cv_function(_pair, pairs=pairs)


def _pip_pol(
    sp: SystemParams,
    _nl,
    shmap,
    shmap_kwargs,
    pairs,
    trips,
    quads,
    r_cut=3 * angstrom,
    r_delta=0.5 * angstrom,
):
    def f_cut(r: Array, r_cut, r_delta) -> Array:
        return jax.lax.cond(
            r > r_cut,
            lambda: 0.0,
            lambda: jax.lax.cond(
                r < r_cut - r_delta,
                lambda: 1.0,
                lambda: jnp.exp(-1 / (1 - ((r - (r_cut - r_delta)) / r_delta) ** 4) + 1),
            ),
        )

    def get_u(x1, x2):
        r = jnp.sum((x1 - x2) ** 2)

        r_safe = jnp.where(r == 0, 1.0, r)
        r = jnp.where(r == 0, 0.0, jnp.sqrt(r_safe))

        u = r / r_cut * f_cut(r, r_cut, r_delta)

        return u

    @jax.jit
    def generate_f2_terms(p):
        u = jnp.array([get_u(a, b) for a, b in combinations(p, 2)])

        return u

    @jax.jit
    def generate_f3_terms(p):
        u = jnp.array([get_u(a, b) for a, b in combinations(p, 2)])

        # f_{3,1}: Sum of all distances
        f_3_1 = jnp.sum(u)

        # f_{3,2}: Sum of squares of all distances
        f_3_2 = jnp.sum(u**2)

        # f_{3,3}: Pairwise products of all distances
        f_3_3 = jnp.sum(jnp.array([u[i] * u[j] for i, j in combinations(range(3), 2)]))

        return jnp.array([f_3_1, f_3_2, f_3_3])

    @jax.jit
    def generate_f4_terms(p):
        u = jnp.array([get_u(a, b) for a, b in combinations(p, 2)])
        # f_{4,1}: Sum of all distances
        f_4_1 = jnp.sum(u)

        # f_{4,2}: Sum of squares of all distances
        f_4_2 = jnp.sum(u**2)

        # f_{4,3}: Pairwise products of all distances
        f_4_3 = jnp.sum(jnp.array([u[i] * u[j] for i, j in combinations(range(6), 2)]))

        # f_{4,4}: Sum of cubes of all distances
        f_4_4 = jnp.sum(u**3)

        # f_{4,5}: Products of triples of distances
        f_4_5 = jnp.sum(jnp.array([u[i] * u[j] * u[k] for i, j, k in combinations(range(6), 3)]))

        # f_{4,6}: Sum of fourth powers of all distances
        f_4_6 = jnp.sum(u**4)

        # f_{4,7}: Pairwise products with one squared distance
        f_4_7 = jnp.sum(jnp.array([u[i] ** 2 * u[j] for i, j in permutations(range(6), 2)]))

        # f_{4,8}: Products of triples with one squared distance
        f_4_8 = jnp.sum(jnp.array([u[i] ** 2 * u[j] * u[k] for i, j, k in permutations(range(6), 3)]))

        # f_{4,9}: Products of triples with two squared distances
        f_4_9 = jnp.sum(jnp.array([u[i] ** 2 * u[j] ** 2 * u[k] for i, j, k in permutations(range(6), 3)]))

        return jnp.array([f_4_1, f_4_2, f_4_3, f_4_4, f_4_5, f_4_6, f_4_7, f_4_8, f_4_9])

    out = []

    if pairs is not None:
        vals_f2 = jax.vmap(generate_f2_terms)(sp.coordinates[pairs])
        out.append(vals_f2.flatten())

    if trips is not None:
        vals_f3 = jax.vmap(generate_f3_terms)(sp.coordinates[trips])
        out.append(vals_f3.flatten())

    if quads is not None:
        vals_f4 = jax.vmap(generate_f4_terms)(sp.coordinates[quads])
        out.append(vals_f4.flatten())

    if len(out) == 0:
        raise ValueError("provide at least pairs, trips or quads")

    vals = jnp.concatenate(out, axis=0)

    return CV(cv=vals)


def pip_pol(pairs: Array, trips: Array, quads: Array, r_cut=3 * angstrom, a=0.5 * angstrom):
    return CvTrans.from_cv_function(
        _pip_pol,
        pairs=pairs,
        trips=trips,
        quads=quads,
        r_cut=r_cut,
        r_delta=a,
    )


def _get_pair_pol(sp: SystemParams, nl: NeighbourList, shmap, shmap_kwargs, n_max=4, b: Array | None = None):
    def f(p, ind):
        def f_cut(r: Array, r_cut, r_delta) -> Array:
            return jax.lax.cond(
                r > r_cut,
                lambda: 0.0,
                lambda: jax.lax.cond(
                    r < r_cut - r_delta,
                    lambda: 1.0,
                    lambda: jnp.exp(-1 / (1 - ((r - (r_cut - r_delta)) / r_delta) ** 3) + 1),
                ),
            )

        r = jnp.sum(p**2)

        r_safe = jnp.where(r == 0, 1.0, r)
        r = jnp.where(r == 0, 0.0, jnp.sqrt(r_safe))

        p_norm = jnp.where(r == 0, jnp.zeros_like(p), p / r_safe)

        # print(f"new")

        return (p_norm, r, f_cut(r, nl.info.r_cut, nl.info.r_cut / 2))

    def f_double(p_j, ind_j, data_j, p_k, ind_k, data_k):
        # print(f"new 1")

        def to_sum_k_rec(n, k):
            if n == 1:
                yield (k,)
            else:
                for x in range(1, k):
                    for i in to_sum_k_rec(n - 1, k - x):
                        yield (x,) + i

        def to_sum_k(n, k_max):
            out = []

            for k in range(0, k_max + 1):
                out.append(jnp.array(list(to_sum_k_rec(n, n + k))))

            return jnp.vstack(out) - 1

        # print(f"{n_max=}")

        degs = to_sum_k(3, n_max)

        p_j_n, r_j, fc_j = data_j
        p_k_n, r_k, fc_k = data_k

        u_j = (r_j / nl.info.r_cut) ** (jnp.arange(n_max + 1)) * fc_j
        u_k = (r_k / nl.info.r_cut) ** (jnp.arange(n_max + 1)) * fc_k

        cos_theta = jnp.dot(p_j_n, p_k_n)

        ang_vec = cos_theta ** jnp.arange((n_max + 1))

        vals = jnp.vstack([u_j, u_k, ang_vec])

        # print(f"vals {vals.shape=} {degs.shape=} ")

        return jax.vmap(lambda i: vals[0, i[0]] * vals[1, i[1]] * vals[2, i[2]])(degs)

    # print(f"{n_max=} getting pair pol")

    _, d = nl.apply_fun_neighbour_pair(
        sp=sp,  # (n_atoms, 3)
        func_single=f,
        func_double=f_double,
        reduce="z",
    )

    # take out diag elements
    print(f"{d.shape=}")

    d = d[:, jnp.triu_indices(d.shape[1], k=0)[0], jnp.triu_indices(d.shape[1], k=0)[1], :]

    print(f"{d.shape=}")

    d = d.flatten()

    if b is not None:
        d = d[b]

    return CV(cv=d)


def get_pair_pol(n_max=3, sp: SystemParams | None = None, nl: NeighbourList | None = None):
    if sp is not None and nl is not None:
        d = _get_pair_pol(sp, nl, None, None, n_max)
        b = jnp.argwhere(d.cv != 0).reshape(-1)

        print(f"{d=}")

        print(f"pair_pol {d.cv.shape=} {b.shape=}")

    else:
        b = None

    return CvTrans.from_cv_function(_get_pair_pol, static_argnames=["n_max"], n_max=n_max, b=b)


# pair_pol = CvTrans.from_cv_function(_get_pair_pol)


def _pol_features(
    sp: SystemParams,
    _nl,
    shmap,
    shmap_kwargs,
    pairs,
    trips,
    quads,
    r_cut=3 * angstrom,
    r_delta=0.5 * angstrom,
    max_deg_n=2,
    max_deg_l=2,
):
    # print(f"pol_features called with {pairs=}, {trips=}, {quads=} {r_cut=}, {r_delta=}, {max_deg=}")

    def f_cut(r: Array, r_cut, r_delta) -> Array:
        return jax.lax.cond(
            r > r_cut,
            lambda: 0.0,
            lambda: jax.lax.cond(
                r < r_cut - r_delta,
                lambda: 1.0,
                lambda: jnp.exp(-1 / (1 - ((r - (r_cut - r_delta)) / r_delta) ** 4) + 1),
            ),
        )

    def get_u(x1, x2):
        r = jnp.sum((x1 - x2) ** 2)

        r_safe = jnp.where(r == 0, 1.0, r)
        r = jnp.where(r == 0, 0.0, jnp.sqrt(r_safe))
        r_inv = jnp.where(r == 0, 0.0, 1 / r_safe)

        return x1 - x2, r, r_inv, f_cut(r, r_cut, r_delta)

    @jax.jit
    def generate_f2_terms(p):
        _, r, _, fc = get_u(p[0], p[1])
        out = r ** jnp.arange(max_deg_n + 1) * fc

        # print(f"f2 terms: {out=}")
        return out

    @jax.jit
    def generate_f3_terms(p):
        x1, _, r1_inv, f1 = get_u(p[0], p[1])
        x2, _, r2_inv, f2 = get_u(p[2], p[1])

        cos_theta = jnp.dot(x1 * r1_inv, x2 * r2_inv)
        theta = jnp.arccos(jnp.clip(cos_theta, -1.0, 1.0))

        return theta ** jnp.arange(max_deg_l + 1)

    @jax.jit
    def generate_f4_terms(p):
        p0, p1, p2, p3 = p[0], p[1], p[2], p[3]

        b0 = -1.0 * (p1 - p0)
        b1 = p2 - p1
        b2 = p3 - p2

        b1, _ = _norm_safe(b1)

        v = b0 - jnp.dot(b0, b1) * b1
        w = b2 - jnp.dot(b2, b1) * b1

        v, _ = _norm_safe(v)
        w, _ = _norm_safe(w)

        cos = jnp.dot(v, w)
        sin = jnp.dot(jnp.cross(b1, v), w)

        theta = jnp.arctan2(sin, cos)

        # out = theta * jnp.arange(max_deg_l + 1)

        return theta * jnp.arange(max_deg_l + 1)

    out = []

    if pairs is not None:
        vals_f2 = jax.vmap(generate_f2_terms)(sp.coordinates[pairs])
        out.append(vals_f2.flatten())

    if trips is not None:
        vals_f3 = jax.vmap(generate_f3_terms)(sp.coordinates[trips])
        out.append(vals_f3.flatten())

    if quads is not None:
        vals_f4 = jax.vmap(generate_f4_terms)(sp.coordinates[quads])
        out.append(vals_f4.flatten())

    if len(out) == 0:
        raise ValueError("provide at least pairs, trips or quads")

    vals = jnp.concatenate(out, axis=0)

    return CV(cv=vals)


def pol_features(
    pairs: Array, trips: Array, quads: Array, r_cut=3 * angstrom, a=0.5 * angstrom, max_deg_n=2, max_deg_l=2
):
    periodic = []
    if pairs is not None:
        periodic.extend([False] * (max_deg_n + 1) * pairs.shape[0])
    if trips is not None:
        periodic.extend([False] * (max_deg_l + 1) * trips.shape[0])
    if quads is not None:
        periodic.extend([True] * (max_deg_l + 1) * quads.shape[0])

    return CvTrans.from_cv_function(
        _pol_features,
        static_argnames=["max_deg_l", "max_deg_n"],
        pairs=pairs,
        trips=trips,
        quads=quads,
        r_cut=r_cut,
        r_delta=a,
        max_deg_l=max_deg_l,
        max_deg_n=max_deg_n,
    ), jnp.array(periodic)


def _pair_features_2(
    sp: SystemParams,
    _nl,
    shmap,
    shmap_kwargs,
    pairs,
    neg_deg_n=2,
    pos_deg_n=2,
    r_cut: float = 3 * angstrom,
):
    def get_u(x1, x2):
        r = jnp.sum((x1 - x2) ** 2)
        r /= r_cut**2

        r_safe = jnp.where(r == 0, 1.0, r)
        r = jnp.where(r == 0, 0.0, jnp.sqrt(r_safe))
        r_inv = jnp.where(r == 0, 0.0, 1 / r_safe)

        return r, r_inv

    @jax.jit
    def generate_f2_terms(p):
        r, r_inv = get_u(p[0], p[1])

        out = jnp.hstack([r ** jnp.arange(1, pos_deg_n + 1), r_inv ** jnp.arange(1, neg_deg_n + 1)])

        return out

    vals = jax.vmap(generate_f2_terms)(sp.coordinates[pairs]).flatten()

    print(f"pair features 2 {vals.shape=}")

    return CV(cv=vals)


def pair_features_2(pairs: Array, neg_deg_n=2, pos_deg_n=2, r_cut: float = 3 * angstrom):
    return CvTrans.from_cv_function(
        _pair_features_2,
        static_argnames=["neg_deg_n", "pos_deg_n"],
        pairs=pairs,
        neg_deg_n=neg_deg_n,
        pos_deg_n=pos_deg_n,
        r_cut=r_cut,
    )


def _linear_layer_apply_rule(
    kwargs,
    static_kwargs,
):
    print(f"apply rule linear layer")

    spectral_norm = static_kwargs.get("spectral_norm", False)

    if spectral_norm:
        weights = kwargs["weights"]
        sigma = jnp.linalg.norm(weights, ord=2)

        kwargs["weights"] = weights / sigma

    return kwargs


def _linear_layer(
    x: CV,
    _nl,
    shmap,
    shmap_kwargs,
    weights: jnp.ndarray,
    biases: jnp.ndarray,
    spectral_norm=False,
):
    print(f"{weights=},{biases=}")

    # if spectral_norm:
    #     sigma = jnp.linalg.norm(weights, ord=2)

    #     weights = weights / sigma

    return x.replace(cv=weights @ x.cv + biases)


def linear_layer(
    weights: jnp.ndarray,
    biases: jnp.ndarray,
    spectral_norm: bool = False,
) -> CvTrans:
    return CvTrans.from_cv_function(
        _linear_layer,
        learnable_argnames=["weights", "biases"],
        static_argnames=["spectral_norm"],
        weights=weights,
        biases=biases,
        spectral_norm=spectral_norm,
        apply_rule=_linear_layer_apply_rule,
    )


activation_functions = {
    "relu": jax.nn.relu,
    "tanh": jnp.tanh,
    "sigmoid": jax.nn.sigmoid,
    "swish": jax.nn.swish,
    "gelu": jax.nn.gelu,
    "softplus": jax.nn.softplus,
    "selu": jax.nn.selu,
    "leaky_relu": jax.nn.leaky_relu,
    "elu": jax.nn.elu,
    "hard_sigmoid": jax.nn.hard_sigmoid,
    "hard_swish": jax.nn.hard_swish,
    "softsign": jax.nn.soft_sign,
}


def _activation_layer(
    x: CV,
    _nl,
    shmap,
    shmap_kwargs,
    activation: str,
):
    func = activation_functions.get(activation)

    if func is None:
        return x

    return x.replace(cv=func(x.cv))


def get_activation_trans(name: str):
    name = name.lower()
    assert name in activation_functions, (
        f"activation {name} not recognized. Available: {list(activation_functions.keys())}"
    )

    return CvTrans.from_cv_function(
        _activation_layer,
        static_argnames=["activation"],
        activation=name,
    )


######################################
#           CV trans                 #
######################################
def _rotate_2d(cv: CV, _nl: NeighbourList, shmap, shmap_kwargs, alpha):
    return cv.replace(
        cv=jnp.array(
            [[jnp.cos(alpha), jnp.sin(alpha)], [-jnp.sin(alpha), jnp.cos(alpha)]],
        )
        @ cv.cv
    )


def _mean(x: CV, nl, shmap, shmap_kwargs):
    return x.replace(cv=jnp.mean(x.cv, axis=0, keepdims=True))


def _func(x: CV, nl, shmap, shmap_kwargs, _f):
    return x.replace(cv=_f(x.cv))


def get_func_trans(f):
    return CvTrans.from_cv_function(_func, _f=f, static_argnames=["_f"])


mean_trans = CvTrans.from_cv_function(_mean)


def rotate_2d(alpha):
    return CvTrans.from_cv_function(_rotate_2d, alpha=alpha)


def _project_distances(cvs: CV, nl, shmap, shmap_kwargs, a: float):
    "projects the distances to a reaction coordinate"
    import jax.numpy as jnp

    from IMLCV.base.CV import CV

    assert cvs.dim == 2

    r1 = cvs.cv[0]
    r2 = cvs.cv[1]

    x = (r2**2 - r1**2) / (2 * a)
    y2 = r2**2 - (a / 2 + x) ** 2

    y2_safe = jnp.where(y2 <= 0, jnp.ones_like(y2), y2)
    y = jnp.where(y2 <= 0, 0.0, jnp.sqrt(y2_safe))

    return CV(cv=jnp.array([x / a + 0.5, y / a]))


def project_distances(a):
    return CvTrans.from_cv_function(_project_distances, a=a)


def _scale_cv_trans(x, nl, shmap, shmap_kwargs, upper, lower, mini, diff):
    return x.replace(cv=((x.cv - mini) / diff) * (upper - lower) + lower)


def scale_cv_trans(array: CV, lower: float = 0.0, upper: float = 1.0, periodic: jax.Array | None = None) -> CvTrans:
    "axis 0 is batch axis"
    maxi = jnp.nanmax(array.cv, axis=0)
    mini = jnp.nanmin(array.cv, axis=0)

    diff = maxi - mini
    diff = jnp.where(diff == 0, 1, diff)

    upper = jnp.full_like(maxi, upper)
    lower = jnp.full_like(mini, lower)

    if periodic is not None:
        mini = jnp.where(periodic, 0.0, mini)
        diff = jnp.where(periodic, 1.0, diff)

        upper = jnp.where(periodic, 1.0, upper)
        lower = jnp.where(periodic, 0.0, lower)

    print(f"{mini=}, {maxi=}, {diff=} {upper=} {lower=} ")

    return CvTrans.from_cv_function(_scale_cv_trans, upper=upper, lower=lower, mini=mini, diff=diff)


def _trunc_svd(x: CV, nl: NeighbourList | None, shmap, shmap_kwargs, m_atomic, v, cvi_shape):
    if m_atomic:
        out = jnp.einsum("ni,jni->j", x.cv, v)

    else:
        out = jnp.einsum("i,ji->j", x.cv, v)

    out = out * jnp.sqrt(cvi_shape)

    return x.replace(cv=out, atomic=False)


def trunc_svd(m: CV, range=Ellipsis) -> tuple[CV, CvTrans]:
    assert m.batched

    if m.atomic:
        cvi = m.cv.reshape((m.cv.shape[0], -1))
    else:
        cvi = m.cv

    cvi = cvi[range, :]

    cvi_shape = cvi.shape[0]
    m_atomic = m.atomic

    u, s, v = jnp.linalg.svd(cvi, full_matrices=False)

    include_mask = s > 10 * jnp.max(
        jnp.array([u.shape[-2], v.shape[-1]]),
    ) * jnp.finfo(
        u.dtype,
    ).eps * jnp.max(s)

    def _f(u, s, v, include_mask):
        u, s, v = u[:, include_mask], s[include_mask], v[include_mask, :]

        out = v

        return s, out

    s, v = _f(u, s, v, include_mask)

    if m.atomic:
        v = v.reshape((v.shape[0], m.cv.shape[1], m.cv.shape[2]))

    out = CvTrans.from_cv_function(
        _trunc_svd,
        static_argnames=["m_atomic", "cvi_shape"],
        m_atomic=m_atomic,
        v=v,
        cvi_shape=cvi_shape,
    )

    return out.compute_cv(m)[0], out


def _inv_sigma_weighing(
    x: CV,
    nl: NeighbourList,
    shmap,
    shmap_kwargs,
    mu: list[jax.Array] | None,
    sigma: list[jax.Array],
    norm: list[jax.Array],
    info: NeighbourListInfo | None = None,
):
    if nl is not None:
        _info = nl.info
    else:
        assert info is not None
        _info = info

    assert x.atomic

    _, argmask, cv_z = _info.nl_split_z(x)

    x_new = jnp.zeros_like(x.cv)

    for i, (idx_i, cv_i, sigma_i, n_i) in enumerate(zip(argmask, cv_z, sigma, norm)):
        sigma_safe = jnp.where(sigma_i == 0, 1.0, sigma_i)
        sigma_inv = jnp.where(sigma_i == 0, 0.0, 1 / sigma_safe)
        # var_inv = sigma_i**2

        # n = jnp.sum(simga_inv)

        # jax.debug.print("{}",n)

        y = cv_i.cv
        if mu is not None:
            mu_i = mu[i]
            y -= mu_i

        y *= sigma_inv**2 / jnp.mean(sigma_inv)

        x_new = x_new.at[idx_i, :].set(y)

    return x.replace(cv=x_new)


def get_inv_sigma_weighing(
    cv_0: list[CV],
    nli: NeighbourListInfo,
    remove_mean=True,
):
    _, _, cvs = jax.vmap(nli.nl_split_z)(CV.stack(*cv_0))

    if remove_mean:
        per_feature_mu = [jax.vmap(lambda x: jnp.mean(x.cv), in_axes=2)(cv_i) for cv_i in cvs]
    else:
        per_feature_mu = None

    per_feature_sigma = [jax.vmap(lambda x: jnp.std(x.cv), in_axes=2)(cv_i) for cv_i in cvs]

    n_sigma = [jnp.nansum(1 / a) for a in per_feature_sigma]

    print(f"{n_sigma=}")

    return CvTrans.from_cv_function(_inv_sigma_weighing, mu=per_feature_mu, sigma=per_feature_sigma, norm=n_sigma)


def sinkhorn_divergence_2(
    x1: CV,
    x2: CV,
    # w2: CV,
    nl1: NeighbourListInfo,
    nl2: NeighbourListInfo,
    # z_scale: jax.Array,
    alpha=1e-2,
    jacobian=False,
    lse=True,
    kernel_fun=lambda x, y: jnp.sum(x * y),
    verbose=False,
) -> CV:
    """caluculates the sinkhorn divergence between two CVs. If x2 is batched, the resulting divergences are stacked"""

    assert x1.atomic
    assert x2.atomic

    assert not x1.batched
    assert not x2.batched

    def solve_svd(matvec, b: Array) -> Array:
        from jax.numpy.linalg import lstsq
        from jaxopt._src.linear_solve import _materialize_array

        r_cond_svd = 1e-10
        print(f"solve_svd {  b= }")

        if len(b.shape) == 0:
            return b / _materialize_array(matvec, b.shape)

        A = _materialize_array(matvec, b.shape, b.dtype)

        b_shape = b.shape

        if len(b_shape) > 2:
            raise ValueError("b must be 1D or 2D")

        if len(b.shape) == 2:
            A = A.reshape(-1, b.shape[0] * b.shape[1])
            b = b.ravel()

        x, _, _, s = lstsq(A, b, rcond=r_cond_svd)

        # jax.debug.print("s {} {} {}", s, A, b)

        if len(b_shape) == 2:
            x = x.reshape(*b_shape[1])  # type: ignore

        return x

    def _core_sinkhorn(
        p1: jax.Array,
        p2: jax.Array,
        # w2: jax.Array,
        epsilon: float | None,
    ):
        print(f"{p1.shape=} {p2.shape=}")

        @partial(vmap_decorator, in_axes=(0, None))
        @partial(vmap_decorator, in_axes=(None, 0))
        def kernel_dist(p1: jax.Array, p2: jax.Array):
            return jnp.sum((p1 - p2) ** 2)

        c = kernel_dist(p1, p2)

        # jax.debug.print("c  {}", c)

        if epsilon is None:
            return jnp.diag(c), p1

        # jax.debug.print("c {} {}", c, c.shape)

        n = c.shape[0]
        m = c.shape[1]

        # if verbose:
        #     m = n

        a = jnp.ones((c.shape[0],)) / n
        b = jnp.ones((c.shape[1],)) / m

        # print(f"{n=} {m=} ")

        #

        from jaxopt import FixedPointIteration

        if lse:  # log space
            g_over_eps = jnp.log(b)
            c_over_eps = c / epsilon

            log_a = jnp.log(a)
            log_b = jnp.log(b)

            def _f0(c_over_epsilon, h_over_eps):
                u = h_over_eps - c_over_epsilon
                u_max = jax.lax.stop_gradient(jnp.max(u))

                return jnp.log(jnp.sum(jnp.exp(u - u_max))) + u_max

            def _T_lse(g_over_eps, c_over_eps, log_a, log_b):
                f_over_eps = log_a - vmap_decorator(_f0, in_axes=(0, None))(c_over_eps, g_over_eps)
                g_over_eps = log_b - vmap_decorator(_f0, in_axes=(1, None))(c_over_eps, f_over_eps)

                return g_over_eps, f_over_eps

            fpi = FixedPointIteration(
                fixed_point_fun=_T_lse,
                maxiter=1000,
                tol=1e-12,  # solve exactly
                implicit_diff_solve=partial(
                    solve_normal_cg,
                    # ridge=1e-10,
                    tol=1e-10,
                    maxiter=1000,
                ),
                has_aux=True,
            )

            out = fpi.run(g_over_eps, c_over_eps, log_a, log_b)

            g_over_eps = out.params
            f_over_eps = out.state.aux

            # print(f"{out=}")

            #

            @partial(vmap_decorator, in_axes=(None, 0, 1), out_axes=1)
            @partial(vmap_decorator, in_axes=(0, None, 0), out_axes=0)
            def _P(f_over_eps, g_over_eps, c_over_eps):
                return jnp.exp(f_over_eps - c_over_eps + g_over_eps)

            P = _P(f_over_eps, g_over_eps, c_over_eps)

        else:
            K = jnp.exp(-c / epsilon)

            # initialization
            v = jnp.ones((c.shape[1],))

            def _T(v, K, a, b):
                u = a / jnp.einsum("ij,j->i", K, v)
                v = b / jnp.einsum("ij,i->j", K, u)

                return v, u

            fpi = FixedPointIteration(
                fixed_point_fun=_T,
                maxiter=100,
                tol=1e-12,
                implicit_diff_solve=partial(solve_normal_cg, ridge=1e-8),
                has_aux=True,
            )

            out = fpi.run(v, K, a, b)

            v = out.params
            u = out.state.aux

            P = jnp.einsum("i,j,ij->ij", u, v, K)

        # if verbose:
        # jax.debug.print("f={} g={} ", f_over_eps, g_over_eps)
        # jax.debug.print("P={}", P)
        # jax.debug.print("sinkhorn iters {} error {}", out.state.iter_num, out.state.error)

        # print(f"{P.shape=} {p1.shape=} {c.shape=} ")

        p1_j = jnp.einsum("ij,i...->j...", P, p1)

        return jnp.einsum("ij,ij->j", P, c), p1_j
        # return jnp.einsum("ij,ij->j", P, c), p1_j

    def get_d_p12(p1_i: jax.Array, p2_i: jax.Array):
        print(f"new")

        # d11, _ = _core_sinkhorn(p1_i, p1_i, epsilon=alpha)
        # d22, _ = _core_sinkhorn(p2_i, p2_i, epsilon=alpha)
        d12_j, p1_j = _core_sinkhorn(p1_i, p2_i, epsilon=alpha)

        # return (d12 - 0.5 * (d11 + d22)) / p2_i.shape[0]
        return d12_j, p1_j

    # if jacobian:
    #     get_d_p12 = jax.value_and_grad(get_d_p12, argnums=1)  # type:ignore

    src_mask, _, p1_cv = nl1.nl_split_z(x1)
    tgt_mask, tgt_split, p2_cv = nl2.nl_split_z(x2)
    # _, _, w2_cv = nl2.nl_split_z(w2)

    p1_l = [a.cv.reshape(a.shape[0], -1) for a in p1_cv]
    p2_l = [a.cv.reshape(a.shape[0], -1) for a in p2_cv]
    # w2_l = [a.cv.reshape(a.shape[0], -1) for a in w2_cv]

    if jacobian:
        out = jnp.zeros((x2.shape[0], x2.shape[1] + 1))
    else:
        out = jnp.zeros((x2.shape[0], 1))

    for i, (p1_i, p2_i, out_i) in enumerate(zip(p1_l, p2_l, tgt_split)):
        if jacobian:
            d_j, p1_j = get_d_p12(p1_i, p2_i)

            out = out.at[out_i, 0].set(d_j)
            out = out.at[out_i, 1:].set(p1_j)
        else:
            d_j, _ = get_d_p12(p1_i, p2_i)
            out = out.at[out_i, 0].set(d_j)

    return x2.replace(
        cv=out,
        _combine_dims=(1, x2.shape[1]) if jacobian else None,
        atomic=True,
        _stack_dims=x1._stack_dims,
    )


def _sinkhorn_divergence_trans_2(
    cv: CV,
    nl: NeighbourList | None,
    shmap,
    shmap_kwargs,
    nl_i: NeighbourListInfo,
    p_i: CV,
    # w_i: CV,
    alpha_rematch,
    jacobian=False,
    verbose=False,
):
    assert nl is not None, "Neigbourlist required for rematch"

    if isinstance(nl_i, NeighbourList):
        print("converting nli to info")
        nl_i = nl_i.info

    def f(p_ii, cv):
        return sinkhorn_divergence_2(
            x1=cv,
            x2=p_ii,
            nl1=nl.info,
            nl2=nl_i,
            alpha=alpha_rematch,
            # w2=w_ii,
            jacobian=jacobian,
            verbose=verbose,
        )

    b = p_i.batched

    if b:
        n = p_i.shape[0]

        print(f"{p_i.shape=} {n=}")

        # p_i = CV.combine(*[a.unbatch() for a in p_i])

        p_i = p_i.replace(cv=jnp.vstack([a.unbatch().cv for a in p_i]))

        print(f"{p_i.shape=}")

        nl_i = NeighbourListInfo(
            r_cut=nl_i.r_cut,
            r_skin=nl_i.r_skin,
            z_array=nl_i.z_array * n,
            z_unique=nl_i.z_unique,
            num_z_unique=tuple(a * n for a in nl_i.num_z_unique),
        )

        # f = vmap_decorator(f, in_axes=(0, None))

    out = f(p_i, cv)

    if b:
        print(f"{out.shape=} {n=}")

        out = CV.combine(*[a.unbatch() for a in out.replace(cv=out.cv.reshape((n, -1, out.shape[1])))])

        print(f"{out.shape=}")

    return out


def get_sinkhorn_divergence_2(
    nli: NeighbourListInfo | NeighbourList,
    pi: CV,
    # wi: CV,
    alpha_rematch: float | None = 0.1,
    jacobian=True,
    verbose=False,
    # scale_z=False,
) -> CvTrans:
    """Get a function that computes the sinkhorn divergence between two point clouds. p_i and nli are the points to match against."""

    assert pi.atomic, "pi must be atomic"

    assert pi.batched, "not fully implemented"

    if isinstance(nli, NeighbourList):
        print("converting nli to info")
        nli = nli.info

    return CvTrans.from_cv_function(
        _sinkhorn_divergence_trans_2,
        static_argnames=[
            "jacobian",
            "verbose",
        ],
        nl_i=nli,
        p_i=pi,
        alpha_rematch=alpha_rematch,
        jacobian=jacobian,
        verbose=verbose,
        # z_scale=dz,
        # w_i=wi,
    )


def _un_atomize(
    x: CV,
    nl,
    shmap,
    shmap_kwargs,
):
    if not x.atomic:
        return x

    return x.replace(
        cv=jnp.reshape(x.cv, (-1,)),
        atomic=False,
    )


un_atomize = CvTrans.from_cv_function(_un_atomize)


def _append_trans(
    x: CV,
    nl,
    shmap,
    shmap_kwargs,
    v: Array,
):
    # print(f"x {x.cv.shape=}")

    return x.replace(cv=jnp.hstack([jnp.reshape(x.cv, (-1,)), v]))


def append_trans(v: Array):
    return CvTrans.from_cv_function(_append_trans, v=v)


def _stack_reduce(cv: CV, nl: NeighbourList | None, shmap, shmap_kwargs, op):
    cvs = cv.split()

    return CV(
        op(jnp.stack([cvi.cv for cvi in cvs]), axis=0),  # type:ignore
        _stack_dims=cvs[0]._stack_dims,
        _combine_dims=cvs[0]._combine_dims,
        atomic=cvs[0].atomic,
        mapped=cvs[0].mapped,
    )


def stack_reduce(op=jnp.mean):
    return CvTrans.from_cv_function(_stack_reduce, static_argnames=["op"], op=op)


def _affine_trans(x: CV, nl, shmap, shmap_kwargs, C):
    assert x.dim == 2

    u = (C[0] * x.cv[0] + C[1] * x.cv[1] + C[2]) / (C[6] * x.cv[0] + C[7] * x.cv[1] + 1)
    v = (C[3] * x.cv[0] + C[4] * x.cv[1] + C[5]) / (C[6] * x.cv[0] + C[7] * x.cv[1] + 1)
    return x.replace(cv=jnp.array([u, v]))


def affine_2d(old: Array, new: Array):
    """old: set of coordinates in the original space, new: set of coordinates in the new space"""

    # source:https://github.com/opencv/opencv/blob/11b020b9f9e111bddd40bffe3b1759aa02d966f0/modules/imgproc/src/imgwarp.cpp#L3001

    # /* Calculates coefficients of perspective transformation
    #  * which maps (xi,yi) to (ui,vi), (i=1,2,3,4):
    #  *
    #  *      c00*xi + c01*yi + c02
    #  * ui = ---------------------
    #  *      c20*xi + c21*yi + c22
    #  *
    #  *      c10*xi + c11*yi + c12
    #  * vi = ---------------------
    #  *      c20*xi + c21*yi + c22
    #  *
    #  * Coefficients are calculated by solving linear system:
    #  * / x0 y0  1  0  0  0 -x0*u0 -y0*u0 \ /c00\ /u0\
    #  * | x1 y1  1  0  0  0 -x1*u1 -y1*u1 | |c01| |u1|
    #  * | x2 y2  1  0  0  0 -x2*u2 -y2*u2 | |c02| |u2|
    #  * | x3 y3  1  0  0  0 -x3*u3 -y3*u3 |.|c10|=|u3|,
    #  * |  0  0  0 x0 y0  1 -x0*v0 -y0*v0 | |c11| |v0|
    #  * |  0  0  0 x1 y1  1 -x1*v1 -y1*v1 | |c12| |v1|
    #  * |  0  0  0 x2 y2  1 -x2*v2 -y2*v2 | |c20| |v2|
    #  * \  0  0  0 x3 y3  1 -x3*v3 -y3*v3 / \c21/ \v3/
    #  *
    #  * where:
    #  *   cij - matrix coefficients, c22 = 1
    #  */

    A = jnp.array(
        [
            [old[0, 0], old[0, 1], 1, 0, 0, 0, -old[0, 0] * new[0, 0], -old[0, 1] * new[0, 0]],
            [old[1, 0], old[1, 1], 1, 0, 0, 0, -old[1, 0] * new[1, 0], -old[1, 1] * new[1, 0]],
            [old[2, 0], old[2, 1], 1, 0, 0, 0, -old[2, 0] * new[2, 0], -old[2, 1] * new[2, 0]],
            [old[3, 0], old[3, 1], 1, 0, 0, 0, -old[3, 0] * new[3, 0], -old[3, 1] * new[3, 0]],
            [0, 0, 0, old[0, 0], old[0, 1], 1, -old[0, 0] * new[0, 1], -old[0, 1] * new[0, 1]],
            [0, 0, 0, old[1, 0], old[1, 1], 1, -old[1, 0] * new[1, 1], -old[1, 1] * new[1, 1]],
            [0, 0, 0, old[2, 0], old[2, 1], 1, -old[2, 0] * new[2, 1], -old[2, 1] * new[2, 1]],
            [0, 0, 0, old[3, 0], old[3, 1], 1, -old[3, 0] * new[3, 1], -old[3, 1] * new[3, 1]],
        ],
    )

    X = jnp.array([new[0, 0], new[1, 0], new[2, 0], new[3, 0], new[0, 1], new[1, 1], new[2, 1], new[3, 1]])

    C = jnp.linalg.solve(A, X)

    return CvTrans.from_cv_function(_affine_trans, C=C)


def _remove_mean(cv: CV, nl: NeighbourList | None, shmap, shmap_kwargs, mean):
    return cv - mean


def get_remove_mean_trans(c: CV, range=Ellipsis):
    assert c.batched
    mean = jnp.mean(c.cv[range, :], axis=0)

    trans = CvTrans.from_cv_function(_remove_mean, mean=mean)

    return trans.compute_cv(c)[0], trans


def _normalize(cv: CV, nl: NeighbourList | None, shmap, shmap_kwargs, mu, std_inv):
    x = cv.cv

    if mu is not None:
        x = mu + (x - mu) * std_inv
    else:
        x *= std_inv

    return cv.replace(cv=x)


def get_normalize_trans(c: CV, remove_mu=True, range=Ellipsis):
    assert c.batched

    if remove_mu:
        mu = jnp.mean(c.cv[range, :], axis=0)
    else:
        mu = None

    std = jnp.std(c.cv[range, :], axis=0)

    std_inv = jnp.where(std == 0, 0, 1 / std)

    trans = CvTrans.from_cv_function(_normalize, mu=mu, std_inv=std_inv)

    return trans.compute_cv(c)[0], trans


def _cv_slice(cv: CV, nl: NeighbourList, shmap, shmap_kwargs, indices):
    return cv.replace(cv=jnp.take(cv.cv, indices, axis=-1), _combine_dims=None)


def _cv_index(cv: CV, nl: NeighbourList, shmap, shmap_kwargs, indices):
    return cv.replace(
        cv=jnp.take(
            cv.cv,
            indices,
            unique_indices=True,
            indices_are_sorted=True,
        ),
        _combine_dims=None,
        atomic=False,
    )


@staticmethod
def _transform(
    cv,
    nl,
    shmap,
    shmap_kwargs,
    argmask: jax.Array | None = None,
    pi: jax.Array | None = None,
    add_1: bool = False,
    add_1_pre: bool = False,
    q: jax.Array | None = None,
    l: jax.Array | None = None,
) -> jax.Array:
    x = cv.cv

    # print(f"inside {x.shape=} {q=} {argmask=} ")

    if argmask is not None:
        x = x[argmask]

    if pi is not None:
        x = x - pi

    if q is not None:
        x = x @ q

    if l is not None:
        x = x * l

    return cv.replace(cv=x)


def get_non_constant_trans(
    c: list[CV],
    c_t: list[CV] | None = None,
    nl: NeighbourList | None = None,
    nl_t: NeighbourList | None = None,
    w: list[Array] | None = None,
    epsilon=1e-14,
    max_functions=None,
    tr: CvTrans | None = None,
):
    from IMLCV.base.rounds import Covariances

    cov = Covariances.create(
        cv_0=c,
        cv_1=c_t,
        nl=nl,
        nl_t=nl_t,
        w=w,
        symmetric=True,
        calc_pi=False,
        only_diag=True,
        trans_f=tr,
        trans_g=tr,
    )

    assert cov.C00 is not None

    cov = jnp.sqrt(cov.C00)

    idx = jnp.argsort(cov, descending=True)
    cov_sorted = cov[idx]

    pos = int(jnp.argwhere(cov_sorted > epsilon)[-1][0])

    if max_functions is not None:
        if pos > max_functions:
            pos = max_functions
    if pos == 0:
        raise ValueError("No features found")

    idx = idx[:pos]

    print(f"selected auto variances {cov[idx]}")

    trans = CvTrans.from_cv_function(_cv_slice, indices=idx)

    return trans


def get_feature_cov(
    c_0: list[CV],
    c_tau: list[CV],
    nl: list[NeighbourList] | NeighbourList | None = None,
    nl_tau: list[NeighbourList] | NeighbourList | None = None,
    w: list[jax.Array] | None = None,
    trans: CvTrans | None = None,
    epsilon=0.1,
    smallest_correlation=1e-12,
    max_functions=None,
    # T_scale=1,
) -> CvTrans:
    from IMLCV.base.rounds import Covariances

    print("computing feature covariances")

    cov = Covariances.create(
        cv_0=c_0,
        cv_1=c_tau,
        nl=nl,
        nl_t=nl_tau,
        w=w,
        symmetric=False,
        calc_pi=False,
        only_diag=True,
        # T_scale=T_scale,
        trans_f=trans,
        trans_g=trans,
    )

    print("computing feature covariances done")
    assert cov.C00 is not None
    assert cov.C01 is not None
    assert cov.C11 is not None

    cov_n = jnp.sqrt(cov.C00 * cov.C11)
    cov_01 = jnp.where(cov_n > smallest_correlation, cov.C01 / cov_n, 0)  # avoid division by zero

    idx = jnp.argsort(cov_01, descending=True)
    cov_sorted = cov_01[idx]

    pos = int(jnp.argwhere(cov_sorted > epsilon)[-1][0])

    if max_functions is not None:
        if pos > max_functions:
            pos = max_functions
    if pos == 0:
        raise ValueError("No features found")

    idx = idx[:pos]

    print(f"selected auto covariances {cov_01[idx]} {pos=}")

    trans = CvTrans.from_cv_function(_cv_slice, indices=idx)

    return trans


def whiten_trans(
    c_0: list[CV],
    c_tau: list[CV],
    nl: list[NeighbourList] | NeighbourList | None = None,
    nl_tau: list[NeighbourList] | NeighbourList | None = None,
    w: list[jax.Array] | None = None,
    trans: CvTrans | None = None,
    epsilon=0.1,
    # smallest_correlation=1e-12,
    max_functions=None,
    # T_scale=1,
) -> CvTrans:
    from IMLCV.base.rounds import Covariances

    print("computing feature covariances")

    cov = Covariances.create(
        cv_0=c_0,
        cv_1=c_tau,
        nl=nl,
        nl_t=nl_tau,
        w=w,
        symmetric=False,
        calc_pi=False,
        only_diag=False,
        # T_scale=T_scale,
        trans_f=trans,
        trans_g=trans,
    )

    W0 = cov.whiten_rho(choice="rho_00", epsilon=epsilon)

    print("computing feature covariances done")

    return CvTrans.from_cv_function(
        _transform,
        static_argnames=["argmask", "pi", "add_1", "add_1_pre", "q", "l"],
        argmask=None,
        pi=None,
        add_1=False,
        add_1_pre=False,
        q=W0.T,
        l=None,
    )

    # print("computing feature covariances done")
    # assert cov.C00 is not None
    # assert cov.C11 is not None

    # cov_n = jnp.sqrt(cov.C00 * cov.C11)
    # cov_01 = jnp.where(cov_n > smallest_correlation, cov.C01 / cov_n, 0)  # avoid division by zero

    # idx = jnp.argsort(cov_01, descending=True)
    # cov_sorted = cov_01[idx]

    # pos = int(jnp.argwhere(cov_sorted > epsilon)[-1][0])

    # if max_functions is not None:
    #     if pos > max_functions:
    #         pos = max_functions
    # if pos == 0:
    #     raise ValueError("No features found")

    # idx = idx[:pos]

    # print(f"selected auto covariances {cov_01[idx]} {pos=}")

    # trans = CvTrans.from_cv_function(_cv_slice, indices=idx)

    # return trans


def _eigh_rot(
    cv,
    nl,
    shmap,
    shmap_kwargs,
    argmask: jax.Array | None = None,
    pi: jax.Array | None = None,
    W: jax.Array | None = None,
) -> jax.Array:
    x = cv.cv

    # print(f"inside {x.shape=} {q=} {argmask=} ")

    if argmask is not None:
        x = x[argmask]

    if pi is not None:
        x = x - pi

    if W is not None:
        x = x @ W

    return cv.replace(cv=x, _combine_dims=None)


def eigh_rot(x: list[CV], w: list[Array] | None):
    from IMLCV.base.rounds import Covariances

    c = Covariances.create(
        cv_0=x,
        shrink=False,
        w=w,
    )

    l, U = jnp.linalg.eigh(c.rho_00)

    tr_rot = CvTrans.from_cv_function(
        _eigh_rot,
        W=U.T,
    )

    return tr_rot


def _matmul_trans(
    cv: CV,
    nl: NeighbourList | None,
    shmap,
    shmap_kwargs,
    M: jax.Array,
):
    return cv.replace(cv=cv.cv @ M, _combine_dims=None)


def _coordination_number(
    x: SystemParams, nl, shmap, shmap_kwargs, group_1: jax.Array, group_2: jax.Array, r: float, n: int = 6, m: int = 12
):
    @partial(jax.vmap, in_axes=(0, None))
    @partial(jax.vmap, in_axes=(None, 0))
    def get(idx1, idx2):
        r_min = x.min_distance(idx1, idx2)
        _r = r_min / r

        eps = _r - 1

        out = jnp.where(
            jnp.abs(eps) < 1e-3,
            n / m
            + (n) * (n - m) / (2 * m) * eps
            + n * (m**2 - 3 * m * (n - 1) + n * (2 * n - 3)) / (12 * m) * (eps**2),
            (1 - _r**n) / (1 - _r**m),
        )

        return out

    coordination = jnp.sum(get(group_1, group_2))

    return CV(cv=jnp.array([coordination]))


def _mix(x: CV, nl, shmap, shmap_kwargs, order: int = 2, include_inverse: bool = False):
    if include_inverse:
        d_inv = jnp.where(x.cv != 0, 1 / x.cv, 0)
        d = jnp.hstack([x.cv, d_inv, 1])
    else:
        d = jnp.hstack([x.cv, 1])

    d = jnp.ravel(d)
    n = d.shape[0]
    idx = jnp.arange(n)

    # Create an order-dimensional meshgrid of indices
    grids = jnp.meshgrid(*([idx] * order), indexing="ij")
    stacked = jnp.stack(grids, axis=-1)  # shape (..., order)

    # Keep only non-decreasing tuples (i1 <= i2 <= ...), i.e. unique combinations up to permutation

    n_masked = 1
    for i in range(order):
        n_masked *= (n - i) / (i + 1)

    n_masked += n  # diagonal
    print(f"{n_masked=}")

    comb_indices = jnp.argwhere(jnp.all(jnp.diff(stacked, axis=-1) >= 0, axis=-1), size=int(n_masked))
    print(f"{comb_indices.shape=}")

    # Values corresponding to each unique combination (product along each tuple)
    comb_values = jnp.prod(jnp.take(d, comb_indices), axis=1)

    return x.replace(cv=comb_values)


def mix_trans(order=2, include_inverse=False):
    return CvTrans.from_cv_function(
        _mix, static_argnames=["order", "include_inverse"], order=order, include_inverse=include_inverse
    )
