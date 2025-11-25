import jax
import jax.numpy as jnp

from IMLCV.base.CV import CV, NeighbourList, NeighbourListInfo, SystemParams
from IMLCV.base.datastructures import vmap_decorator

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


def _get_sp_rand(
    prng,
    n=15,
    r_cut=3,
) -> tuple[jax.Array, SystemParams, NeighbourList]:
    k1, k2, k3, prng = jax.random.split(prng, 4)

    r_side = 6 * (n / 5) ** (1 / 3)

    sp0, _ = SystemParams(
        coordinates=jax.random.uniform(k1, shape=(n, 3)) * r_side,
        cell=jnp.array(
            [
                [n, 0, 0],
                [0, n, 0],
                [0, 0, n],
            ],
        )
        * 1.0,
    ).canonicalize()

    # r_cut = 4 * angstrom
    z_array = jax.random.randint(k2, (n,), 0, 5)

    info = NeighbourListInfo.create(r_cut=r_cut, z_array=z_array)

    nl0 = sp0.get_neighbour_list(info=info)
    assert nl0 is not None

    return prng, sp0, nl0


def _permute_sp_rand(
    prng,
    sp0: SystemParams,
    nl0: NeighbourList,
    eps,
) -> tuple[jax.Array, SystemParams, NeighbourList]:
    k1, k2, k3, prng = jax.random.split(prng, 4)

    sp1 = SystemParams(
        coordinates=sp0.coordinates + jax.random.uniform(k1, shape=sp0.coordinates.shape) * eps,
        cell=sp0.cell + jax.random.normal(k3, (3, 3)) * eps,
    )

    nl1 = sp1.get_neighbour_list(info=nl0.info)
    assert nl1 is not None
    return prng, sp1, nl1


def _get_equival_sp(sp, rng) -> tuple[jax.Array, SystemParams]:
    # check rotational and translationa invariance
    from scipy.spatial.transform import Rotation as R

    key, rng = jax.random.split(rng)

    rot_mat = jnp.array(
        R.random(random_state=int(jax.random.randint(key, (), 0, 100))).as_matrix(),
    )
    pos2 = vmap_decorator(lambda a: rot_mat @ a, in_axes=0)(sp.coordinates) + jax.random.normal(key, (3,)) * 5
    cell_r = vmap_decorator(lambda a: rot_mat @ a, in_axes=0)(sp.cell)
    sp2 = SystemParams(coordinates=pos2, cell=cell_r)
    return rng, sp2


def test_neigh():
    rng = jax.random.PRNGKey(42)

    r_cut = 200
    n = 15

    rng, sp, nl = _get_sp_rand(prng=rng, n=n, r_cut=r_cut)

    def func(r_ij: jax.Array, index: jax.Array) -> tuple[jax.Array, jax.Array]:
        return (jnp.linalg.norm(r_ij), index)

    s1 = nl.apply_fun_neighbour(sp=sp, r_cut=r_cut, func_single=func)
    neigh_calc, (r_calc, index_calc) = s1

    # campare agains mathematical predictions for uniform fille sphere
    neigh_exp = n / jnp.abs(jnp.linalg.det(sp.cell)) * (4 / 3 * jnp.pi * r_cut**3)
    r_exp = 0.75 * r_cut
    print(
        f"err neigh density {jnp.abs((jnp.mean(neigh_calc) - neigh_exp) / neigh_exp)}",
    )
    print(
        f"err neigh density {jnp.abs((jnp.mean(r_calc / neigh_calc) - r_exp) / r_exp)}",
    )

    def _comp(s1, s2):
        neigh_calc1, (r_calc1, index_calc1) = s1
        neigh_calc2, (r_calc2, index_calc2) = s2

        assert (neigh_calc1 == neigh_calc2).all(), f"{neigh_calc1} != {neigh_calc2}"
        assert jnp.sqrt(jnp.mean((r_calc2 - r_calc1) ** 2 / r_calc2**2)) < 1e-7
        assert (index_calc1 - index_calc2 == 0).all()

    # test 2: campare equivalent sp
    rng, sp2 = _get_equival_sp(sp, rng)

    nl2 = sp2.get_neighbour_list(info=nl.info)
    assert nl2 is not None

    s2 = nl2.apply_fun_neighbour(sp=sp2, func_single=func, r_cut=r_cut)

    _comp(s1, s2)

    # test 3: with neighbourlist, split z

    s3 = nl.apply_fun_neighbour(sp=sp, r_cut=r_cut, func_single=func, split_z=True)
    s3_u = jax.tree.map(lambda x: jnp.sum(x, axis=1), s3)  # resum over z axis
    _comp(s1, s3_u)

    # method 6: with neighbourlist, no reduction

    s4 = nl.apply_fun_neighbour(
        sp=sp,
        r_cut=r_cut,
        func_single=func,
        reduce="none",
    )
    s4_u = jax.tree.map(lambda x: jnp.sum(x, axis=1), s4)  # resum over z axis
    _comp(s1, s4_u)

    # split to ge tz components

    # method 7: with neighbourlist, sort_z_self

    s5 = nl.apply_fun_neighbour(
        sp=sp,
        r_cut=r_cut,
        func_single=func,
        reduce="z",
    )

    # method 7: with neighbourlist, sort_z_self

    s6 = nl.apply_fun_neighbour(
        sp=sp,
        r_cut=r_cut,
        func_single=func,
        reduce="none",
    )

    _, _, s6_z = nl.nl_split_z(s6)
    a, bc = tuple(
        zip(*jax.tree.map(lambda x: jnp.sum(x, axis=(0, -1)), s6_z)),
    )  # sum over z and over neighbours
    b, c = tuple(zip(*bc))

    a = jnp.stack(a)
    b = jnp.stack(b)
    c = jnp.stack(c)

    s6_c = a, (b, c)

    _comp(s5, s6_c)
    # check nl update code

    key1, key2, key3, rng = jax.random.split(rng, 4)
    # sp4 = SystemParams(
    #     coordinates=sp.coordinates + 0.1 * jax.random.normal(key1, sp.coordinates.shape),
    #     cell=sp.cell + 0.1 * jax.random.normal(key2, sp.cell.shape),
    # )
    bool, nl4 = nl.update_nl(sp)


def test_neigh_pair():
    rng = jax.random.PRNGKey(42)

    r_cut = 15

    rng, sp, nl = _get_sp_rand(rng, 40, r_cut=r_cut)

    z_array = nl.info.z_array

    # neighbourghlist
    def func_double(r_ij, atom_index_j, data_j, r_ik, atom_index_k, data_k):
        return (
            jnp.linalg.norm(r_ij - r_ik),
            jnp.array(z_array)[atom_index_j],
            jnp.array(z_array)[atom_index_k],
        )

    info = NeighbourListInfo.create(r_cut=r_cut, z_array=z_array)

    nl = sp.get_neighbour_list(info=info)
    assert nl is not None

    s1 = nl.apply_fun_neighbour_pair(
        sp=sp,
        r_cut=r_cut,
        func_double=func_double,
        exclude_self=True,
        unique=True,
    )

    k, (pair_dist, index_j, index_k) = s1

    # test1: mahth val
    pair_dist_avg = pair_dist / k * 2  # factor 2 account for pairs
    pair_dist_exact = (
        36 / 35 * r_cut
    )  # https://math.stackexchange.com/questions/167932/mean-distance-between-2-points-in-a-ball

    print(
        f"err neigh density {jnp.abs((jnp.mean(pair_dist_avg) - pair_dist_exact) / pair_dist_exact)}",
    )

    #####

    def _comp(s1, s2):
        k1, (pair_dist1, index_j1, index_k1) = s1
        k2, (pair_dist2, index_j2, index_k2) = s2

        assert (k1 == k2).all()
        assert (index_j1 == index_j2).all()
        assert (index_k1 == index_k2).all()
        assert jnp.mean(((pair_dist1 - pair_dist2) / pair_dist1) ** 2) ** 0.5 < 1e-7

    # test 2: z split, resummations should yield orriginal result

    s2 = nl.apply_fun_neighbour_pair(
        sp=sp,
        r_cut=r_cut,
        func_double=func_double,
        exclude_self=True,
        unique=True,
        split_z=False,
    )

    # reduce to non split version
    s2_u = jax.tree.map(lambda x: jnp.sum(jnp.sum(x, axis=2), axis=1), s2)
    _comp(s1, s2_u)

    # same but without reduction

    k5_zz, (pair_dist5_zz, index_j5_zz, index_k5_zz) = nl.apply_fun_neighbour_pair(
        sp=sp,
        r_cut=r_cut,
        func_double=func_double,
        reduce="none",
        exclude_self=True,
        unique=True,
    )

    # test if j and k indices are correctly fragmented
    for j, zj in enumerate(jnp.unique(jnp.array(z_array))):
        for k, zk in enumerate(jnp.unique(jnp.array(z_array))):
            assert (index_j5_zz[:, j, k, :][k5_zz[:, j, k, :]] == zj).all()
            assert (index_k5_zz[:, j, k, :][k5_zz[:, j, k, :]] == zk).all()

    # same but with reduction per z

    k6_zz, (pair_dist6_zz, index_j6_zz, index_k6_zz) = nl.apply_fun_neighbour_pair(
        sp=sp,
        r_cut=r_cut,
        func_double=func_double,
        reduce="z",
        exclude_self=True,
        unique=True,
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
            ],
        ),
    ).minkowski_reduce()[0]

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
    coordinates2 = coordinates + jax.random.randint(k1, (3,), minval=-3, maxval=3) @ cell

    sp0 = SystemParams(cell=cell, coordinates=coordinates)
    sp1 = SystemParams(cell=cell2, coordinates=coordinates2)

    # test distance
    assert jnp.abs(sp0.min_distance(0, 1) - sp1.min_distance(0, 1)) < 1e-6

    # test minkowski reduction
    sp0, sp1 = sp0.minkowski_reduce()[0], sp1.minkowski_reduce()[0]

    assert sp0.cell is not None
    assert sp1.cell is not None

    for i in range(3):
        assert jnp.all(jnp.abs(sp0.cell[i, :] - sp1.cell[i, :]) < 1e-6) or jnp.all(
            jnp.abs(sp0.cell[i, :] + sp1.cell[i, :]) < 1e-6,
        )

    # test qr
    sp0, sp1 = sp0.rotate_cell()[0], sp1.rotate_cell()[0]

    assert sp0.cell is not None
    assert sp1.cell is not None

    assert jnp.all(jnp.abs(sp0.cell - sp1.cell) < 1e-6), f"{sp0.cell},{sp1.cell}"

    sp0, sp1 = sp0.wrap_positions()[0], sp1.wrap_positions()[0]

    assert jnp.all(jnp.abs(sp0.coordinates - sp1.coordinates) < 1e-6), f"{sp0.coordinates}, {sp1.coordinates}"


def test_sp_apply():
    prng = jax.random.PRNGKey(42)
    k1, k2, prng = jax.random.split(prng, 3)

    cell = jax.random.uniform(k1, (500, 3, 3))
    coordinates = jax.random.uniform(k2, (500, 2, 3))

    sp = SystemParams(cell=cell, coordinates=coordinates)

    sp_wrapped, op = sp.wrap_positions(min=True)
    sp_wrapped_2 = vmap_decorator(SystemParams.apply_wrap)(sp, op)
    assert jnp.linalg.norm(sp_wrapped.coordinates - sp_wrapped_2.coordinates) < 1e-10

    sp_rot, op = sp.rotate_cell()
    sp_rot_2 = vmap_decorator(SystemParams.apply_rotation)(sp, op)

    assert sp_rot.cell is not None
    assert sp_rot_2.cell is not None

    assert jnp.linalg.norm(sp_rot.cell - sp_rot_2.cell) < 1e-10
    assert jnp.linalg.norm(sp_rot.coordinates - sp_rot_2.coordinates) < 1e-10

    sp_min, op = sp.minkowski_reduce()
    sp_min_2 = vmap_decorator(SystemParams.apply_minkowski_reduction)(sp, op)

    assert sp_min.cell is not None
    assert sp_min_2.cell is not None

    assert jnp.linalg.norm(sp_min.cell - sp_min_2.cell) < 1e-10

    sp_can, op = sp.canonicalize(min=False, qr=True)
    sp_can_2 = vmap_decorator(SystemParams.apply_canonicalize)(sp, op)

    assert sp_can.cell is not None
    assert sp_can_2.cell is not None

    assert jnp.linalg.norm(sp_can.cell - sp_can_2.cell) < 1e-10
    assert jnp.linalg.norm(sp_can.coordinates - sp_can_2.coordinates) < 1e-10


if __name__ == "__main__":
    test_neigh()
    test_neigh_pair()
