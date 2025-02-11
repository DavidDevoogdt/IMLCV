from functools import partial

import jax.debug
import jax.dtypes
import jax.lax
import jax.numpy as jnp
import jax.numpy.linalg
import jax.random
import jax.scipy
import numpy as onp
import pytest
import scipy.special
from jax import grad, vmap

from IMLCV.base.CV import CollectiveVariable, NeighbourListInfo, SystemParams
from IMLCV.implementations.CV import get_sinkhorn_divergence_2, sb_descriptor, soap_descriptor
from IMLCV.tools.bessel_callback import iv, ive, ive_b, jv, kv, kve, spherical_jn, spherical_yn, yv
from IMLCV.tools.soap_kernel import p_inl_sb


def get_sps(
    r_side,
    r_cut,
    eps,
    n,
    with_cell=True,
):
    # sp
    rng = jax.random.PRNGKey(42)
    key, rng = jax.random.split(rng)
    pos = jax.random.uniform(key, (n, 3)) * r_side
    key, rng = jax.random.split(rng)
    if with_cell:
        cell = jnp.eye(3) * r_side + jax.random.normal(key, (3, 3)) * eps * r_side
    else:
        cell = None
    sp1 = SystemParams(coordinates=pos, cell=cell)

    # sp2
    from scipy.spatial.transform import Rotation as R

    key1, key2, key3, rng = jax.random.split(rng, 4)
    rot_mat = jnp.array(
        R.random(random_state=int(jax.random.randint(key1, (), 0, 100))).as_matrix(),
    )
    pos2 = vmap(lambda a: rot_mat @ a, in_axes=0)(sp1.coordinates) + jax.random.normal(key2, (3,)) * eps * r_side
    if with_cell:
        cell_r = vmap(lambda a: rot_mat @ a, in_axes=0)(sp1.cell)
    else:
        cell_r = None
    perm = jax.random.permutation(key3, n)

    sp2 = SystemParams(coordinates=pos2[perm], cell=cell_r)

    # raise "do permutation on p2"

    # sp3
    key, rng = jax.random.split(rng)
    pos = jax.random.uniform(key, (n, 3)) * n
    key, rng = jax.random.split(rng)
    if with_cell:
        cell = jnp.eye(3) * r_side + jax.random.normal(key, (3, 3)) * 0.5
    else:
        cell = None
    sp3 = SystemParams(coordinates=pos, cell=cell)

    key, rng = jax.random.split(rng)

    z_array = jax.random.randint(key, (n,), 0, 5)

    nl1 = sp1.get_neighbour_list(info=NeighbourListInfo.create(r_cut=r_cut, z_array=z_array))
    nl2 = sp2.get_neighbour_list(info=NeighbourListInfo.create(r_cut=r_cut, z_array=z_array[perm]))
    nl3 = sp3.get_neighbour_list(info=NeighbourListInfo.create(r_cut=r_cut, z_array=z_array))
    return sp1, sp2, sp3, nl1, nl2, nl3


@pytest.mark.parametrize(
    "cell, pp",
    [(True, "sb"), (False, "soap")],
)
def test_SOAP_SB_sinkhorn(cell, pp):
    n = 10
    r_side = 6 * (n / 5) ** (1 / 3)

    l_max = 2
    n_max = 2

    r_cut = 5
    eps = 0.2
    alpha = 1e-1

    sp1, sp2, sp3, nl1, nl2, nl3 = get_sps(r_side, r_cut, eps, n, with_cell=cell)

    if pp == "sb":
        desc = sb_descriptor(
            l_max=l_max,
            n_max=n_max,
            r_cut=r_cut,
        )
    elif pp == "soap":
        desc = soap_descriptor(
            l_max=l_max,
            n_max=n_max,
            r_cut=r_cut,
            sigma_a=1.0,
            r_delta=1.0,
            num=50,
        )

    p_ref, _ = desc.compute_cv_flow(sp1, nl1)
    nl_ref = nl1

    cv = CollectiveVariable(
        f=desc
        * get_sinkhorn_divergence_2(
            nli=nl_ref,
            pi=p_ref,
            alpha_rematch=alpha,
            output="both",
        ),
        metric=None,
        jac=jax.jacrev,
    )

    # n.b. each function jit compiles again bc shape nl constants/ shape are different.
    # with jax.disable_jit():
    x1, dx1 = cv.compute_cv(sp1, nl1, jacobian=True)
    x3, dx3 = cv.compute_cv(sp3, nl3, jacobian=True)
    x2, dx2 = cv.compute_cv(sp2, nl2, jacobian=True)

    assert jnp.allclose(x1.cv, x2.cv)

    print(f"{dx1.cv.coordinates=}")
    print(f"{dx2.cv.coordinates=}")
    print(f"{dx1.cv.coordinates-dx2.cv.coordinates=}")

    assert jnp.allclose(dx1.cv.coordinates, dx2.cv.coordinates, atol=1e-5)

    if cell:
        assert jnp.allclose(dx1.cv.cell, dx2.cv.cell, atol=1e-5)

    assert not jnp.allclose(x1.cv, x3.cv)


def test_SB_basis():
    n_max = 3
    l_max = 3
    r_cut = 1.0
    n = 10000

    r = jnp.zeros((n, 3))
    r = r.at[:, 0].set(jnp.linspace(1e-3, 1.0, n))

    a, _ = p_inl_sb(n_max, l_max, r_cut)
    # f = jax.vmap(Partial(a, atom_index_j=_))

    o_g = jax.vmap(lambda r: a(r, _))(r)

    @partial(jax.vmap, in_axes=(None, 2, 2))
    @partial(jax.vmap, in_axes=(None, None, 1))
    @partial(jax.vmap, in_axes=(None, 1, None))
    def integrate(r, g, h):
        r = jnp.linalg.norm(r, axis=1)
        n = r.shape[0]
        return jnp.sum(r**2 * g * h) / n

    o_i = integrate(r, o_g, o_g)
    o_i_2 = jax.vmap(lambda _: jnp.eye(o_g.shape[1]), in_axes=(2))(o_g)

    print(o_i)
    print(o_i_2)


def test_bessel():
    for func, name, sp, dsp, wz in zip(
        [jv, yv, iv, kv, spherical_jn, spherical_yn, ive, kve],
        ["jv", "yv", "iv", "kv", " spherical_jv", "spherical_yv", "ive", "ive_b" "kve"],
        [
            scipy.special.jv,
            scipy.special.yv,
            scipy.special.iv,
            scipy.special.kv,
            scipy.special.spherical_jn,
            scipy.special.spherical_yn,
            scipy.special.ive,
            scipy.special.ive,
            scipy.special.kve,
        ],
        [
            scipy.special.jvp,
            scipy.special.yvp,
            scipy.special.ivp,
            scipy.special.kvp,
            lambda x, y: scipy.special.spherical_jn(x, y, derivative=True),
            lambda x, y: scipy.special.spherical_yn(x, y, derivative=True),
            None,
            None,
            None,
        ],
        [1, 0, 1, 0, 0, 0, None, None],
    ):
        # try to vmap and jit
        _ = vmap(func, in_axes=(0, None))(jnp.array([2, 5]), 2)
        _ = vmap(func, in_axes=(None, 0))(2, jnp.array([2, 5]))
        _ = vmap(vmap(func, in_axes=(0, None)), in_axes=(None, 0))(
            jnp.array([2, 5]),
            jnp.array([2, 5]),
        )

        _ = vmap(
            vmap(vmap(func, in_axes=(0, None)), in_axes=(0, None)),
            in_axes=(None, 0),
        )(jnp.array([[2, 5], [2, 5]]), jnp.array([2, 5]))

        if wz == 1:
            x = jnp.linspace(0, 20, 1000)
        else:
            x = jnp.linspace(1e-5, 20, 1000)
        v = jnp.arange(0, 5)
        y = vmap(vmap(func, in_axes=(0, None)), in_axes=(None, 0))(v, x)
        y2 = jnp.array([sp(vi, onp.array(x)) for vi in v]).T
        assert jnp.allclose(y, y2, atol=1e-5)

        if dsp is not None:
            # with jax.disable_jit():
            dy = vmap(
                vmap(grad(func, argnums=1), in_axes=(0, None)),
                in_axes=(None, 0),
            )(v, x)

            dy2 = jnp.array([dsp(vi, onp.array(x)) for vi in v]).T

            mask = jnp.logical_or(jnp.isnan(dy2), jnp.isinf(dy2))

            assert jnp.allclose(dy[~mask], dy2[~mask], atol=1e-5, equal_nan=True)

    # sanity chekc to see wither ive and kve behave correctly

    def k1e(x):
        return kve(1, x)

    def k1e2(x):
        return kv(1, x) * jnp.exp(x)

    x = jnp.linspace(1, 5, 1000)
    assert jnp.linalg.norm(vmap(k1e)(x) - vmap(k1e2)(x)) < 1e-5
    assert jnp.linalg.norm(vmap(grad(k1e))(x) - vmap(grad(k1e2))(x)) < 1e-5

    def i1e(x):
        return ive(1, x)

    def i1eb(x):
        return ive_b(1, x)

    def i1e2(x):
        return iv(1, x) * jnp.exp(-jnp.abs(x))

    assert jnp.linalg.norm(vmap(i1e)(x) - vmap(i1e2)(x)) < 1e-5
    assert jnp.linalg.norm(vmap(i1eb)(x) - vmap(i1e2)(x)) < 1e-5
    assert jnp.linalg.norm(vmap(grad(i1e))(x) - vmap(grad(i1e2))(x)) < 1e-5


def test_bessel_2():
    for func, name, sp, dsp, wz in zip(
        [spherical_jn],
        ["spherical_jv"],
        [scipy.special.spherical_jn],
        [lambda x, y: scipy.special.spherical_jn(x, y, derivative=True)],
        [0],
    ):

        def func(v, x):
            return spherical_jn(v, x)[v]

        if wz == 1:
            x = jnp.linspace(0, 20, 1000)
        else:
            x = jnp.linspace(1e-5, 20, 1000)

        from matplotlib import pyplot as plt

        plt.figure()

        for v in range(0, 5):
            y = jax.vmap(func, in_axes=(None, 0))(v, x)
            y2 = sp(v, onp.array(x))
            assert jnp.allclose(y, y2, atol=1e-5)

            dy = vmap(grad(func, argnums=1), in_axes=(None, 0))(v, x)
            dy2 = dsp(v, onp.array(x))

            mask = jnp.logical_or(jnp.isnan(dy2), jnp.isinf(dy2))

            plt.plot(x, y, label=f"{v}")
            plt.plot(x, dy, label=f"{v} dy")

            assert jnp.allclose(dy[~mask], dy2[~mask], atol=1e-5, equal_nan=True), f"{dy} {dy2} "

        plt.show()
        print("done")


if __name__ == "__main__":
    # with jax.disable_jit():
    # test_SOAP_SB_sinkhorn(cell=True, pp="sb")
    # test_bessel_2()
    # test_SB_basis()

    test_bessel()

    # print(jax.vmap(spherical_jn, in_axes=(None, 0))(5, jnp.ones(5)))
