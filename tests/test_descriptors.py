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
from IMLCV.base.CV import CollectiveVariable
from IMLCV.base.CV import CV
from IMLCV.base.CV import NeighbourList
from IMLCV.base.CV import SystemParams
from IMLCV.implementations.CV import get_sinkhorn_divergence
from IMLCV.implementations.CV import sb_descriptor
from IMLCV.implementations.CV import soap_descriptor
from IMLCV.tools.bessel_callback import iv
from IMLCV.tools.bessel_callback import ive
from IMLCV.tools.bessel_callback import jv
from IMLCV.tools.bessel_callback import kv
from IMLCV.tools.bessel_callback import kve
from IMLCV.tools.bessel_callback import spherical_jn
from IMLCV.tools.bessel_callback import spherical_yn
from IMLCV.tools.bessel_callback import yv
from jax import grad
from jax import vmap


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

    nl1 = sp1.get_neighbour_list(r_cut=r_cut, z_array=z_array)
    nl2 = sp2.get_neighbour_list(r_cut=r_cut, z_array=z_array[perm])
    nl3 = sp3.get_neighbour_list(r_cut=r_cut, z_array=z_array)
    return sp1, sp2, sp3, nl1, nl2, nl3


@pytest.mark.parametrize(
    "cell,matching, pp",
    [(True, "rematch", "sb"), (False, "average", "soap")],
)
def test_SOAP_SB_sinkhorn(cell, matching, pp):
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
        * get_sinkhorn_divergence(
            nli=nl_ref,
            pi=p_ref,
            sort=matching,
            alpha_rematch=alpha,
            output="divergence",
        ),
        metric=None,
        jac=jax.jacrev,
    )

    # n.b. each function jit compiles again bc shape nl constants/ shape are different.
    # with jax.disable_jit():
    x1, dx1 = cv.compute_cv(sp1, nl1, jacobian=True)
    x3, dx3 = cv.compute_cv(sp3, nl3, jacobian=True)
    x2, dx2 = cv.compute_cv(sp2, nl2, jacobian=True)

    assert jnp.allclose(x1.cv, 0)
    assert jnp.allclose(x2.cv, 0)

    assert jnp.allclose(dx1.cv.coordinates, 0, atol=1e-5)
    assert jnp.allclose(dx2.cv.coordinates, 0, atol=1e-5)

    if cell:
        assert jnp.allclose(dx1.cv.cell, 0, atol=1e-5)
        assert jnp.allclose(dx2.cv.cell, 0, atol=1e-5)

    assert x3.cv > 1e-3
    assert 1e-3 < jnp.mean(jnp.linalg.norm(dx3.cv.coordinates, axis=1)) < 1
    if cell:
        assert 1e-3 < jnp.mean(jnp.linalg.norm(dx3.cv.cell, axis=1)) < 1


def test_bessel():
    for func, name, sp, dsp, wz in zip(
        [jv, yv, iv, kv, spherical_jn, spherical_yn, ive, kve],
        ["jv", "yv", "iv", "kv", " spherical_jv", "spherical_yv", "ive", "kve"],
        [
            scipy.special.jv,
            scipy.special.yv,
            scipy.special.iv,
            scipy.special.kv,
            scipy.special.spherical_jn,
            scipy.special.spherical_yn,
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

    def i1e2(x):
        return iv(1, x) * jnp.exp(-jnp.abs(x))

    assert jnp.linalg.norm(vmap(i1e)(x) - vmap(i1e2)(x)) < 1e-5
    assert jnp.linalg.norm(vmap(grad(i1e))(x) - vmap(grad(i1e2))(x)) < 1e-5


if __name__ == "__main__":
    # test_SOAP_SB_sinkhorn(cell=True, matching="average", pp="soap")
    test_bessel()
