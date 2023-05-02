import jax.debug
import jax.dtypes
import jax.lax
import jax.numpy as jnp
import jax.numpy.linalg
import jax.random
import jax.scipy
import pytest
from jax import jacrev, jit, vmap

import IMLCV
from IMLCV.base.CV import NeighbourList, SystemParams
from IMLCV.base.tools.soap_kernel import Kernel, p_i, p_inl_sb, p_innl_soap


import jax.lax
import jax.numpy as jnp
import numpy as onp
import scipy.special
from jax import custom_jvp, grad, jit, pure_callback, vmap,jacfwd
from jax.custom_batching import custom_vmap

from IMLCV.base.tools.bessel_callback import jv, yv, iv, kv, spherical_jn, spherical_yn, ive, kve

def get_sps(
    r_side,
    r_cut,
    eps,
    n,
    with_cell=True,
):
    ## sp
    rng = jax.random.PRNGKey(42)
    key, rng = jax.random.split(rng)
    pos = jax.random.uniform(key, (n, 3)) * r_side
    key, rng = jax.random.split(rng)
    if with_cell:
        cell = jnp.eye(3) * r_side + jax.random.normal(key, (3, 3)) * eps * r_side
    else:
        cell = None
    sp1 = SystemParams(coordinates=pos, cell=cell)

    ## sp2
    from scipy.spatial.transform import Rotation as R

    key1, key2, key3, rng = jax.random.split(rng, 4)
    rot_mat = jnp.array(
        R.random(random_state=int(jax.random.randint(key1, (), 0, 100))).as_matrix()
    )
    pos2 = (
        vmap(lambda a: rot_mat @ a, in_axes=0)(sp1.coordinates)
        + jax.random.normal(key2, (3,)) * eps * r_side
    )
    if with_cell:
        cell_r = vmap(lambda a: rot_mat @ a, in_axes=0)(sp1.cell)
    else:
        cell_r = None
    perm = jax.random.permutation(key3, n)

    sp2 = SystemParams(coordinates=pos2[perm], cell=cell_r)

    # raise "do permutation on p2"

    ## sp3
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


@pytest.mark.parametrize("cell", [False, True])
@pytest.mark.parametrize(
    "matching",
    [
        "average",  # -> does not result in local minmum based on forces
        "norm",
        "rematch",
        #
    ],
)
@pytest.mark.parametrize("pp", ["sb", "soap"])
def test_SOAP(cell, matching, pp):
    n = 10
    r_side = 6 * (n / 5) ** (1 / 3)

    l_max = 3
    n_max = 3

    r_cut = 5
    eps = 0.2
    alpha = 1e-2

    sp1, sp2, sp3, nl1, nl2, nl3 = get_sps(r_side, r_cut, eps, n, with_cell=cell)

    if pp == "sb":
        pi = p_inl_sb(
            l_max=l_max,
            n_max=n_max,
            r_cut=r_cut,
        )
    elif pp == "soap":
        pi = p_innl_soap(
            l_max=l_max,
            n_max=n_max,
            r_cut=r_cut,
            sigma_a=0.5,
            r_delta=1.0,
            num=50,
        )

    @jit
    def f(sp, nl):
        return p_i(
            sp=sp,
            nl=nl,
            p=pi,
            r_cut=r_cut,
        )

    p1 = f(sp1, nl1)

    @jit
    def k(sp: SystemParams, nl: NeighbourList):
        return Kernel(p1, f(sp, nl), nl1, nl, matching=matching, alpha=alpha)

    da = k(sp1, nl1).block_until_ready()

    from time import time_ns

    before = time_ns()
    dab = k(sp2, nl2).block_until_ready()
    after = time_ns()

    dac = k(sp3, nl3)

    assert jnp.abs(da - 1) < 1e-3
    assert jnp.abs(dab - 1) < 1e-3
    # assert jnp.abs(dac) < 0.8

    print(
        f"{matching=}\t{cell=} {pp=}\tl_max {l_max}\tn_max {l_max}\t<kernel(orig,rot)>=\t\t{  dab :>.4f}\t<kernel(orig, rand)>=\t\t{ dac:.4f}\tevalutation time [ms] { (after-before)/ 10.0**6 :>.2f}  "
    )

    @jit
    def jk(sp, nl):
        _jk = jacrev(k)(sp, nl)
        return jnp.mean(jnp.linalg.norm(_jk.coordinates)), jnp.mean(
            jnp.linalg.norm(_jk.coordinates) if _jk.cell is not None else 0.0
        )

    jac_da_coor, jac_da_cell = jk(sp1, nl1)

    before = time_ns()
    jac_dab_coor, jac_dab_cell = jk(sp2, nl2)
    after = time_ns()

    jac_dac_coor, jac_dac_cell = jk(sp3, nl3)

    assert jnp.abs(jac_da_coor) < 10
    assert jnp.abs(jac_dab_coor) < 10
    assert jnp.abs(jac_dac_coor) < 10

    if cell:
        assert jnp.abs(jac_da_cell) < 10
        assert jnp.abs(jac_dab_cell) < 10
        assert jnp.abs(jac_dac_cell) < 10

    print(
        f"\t\t\t\t\t\t\t\t\t\t||d kernel(orig,rot)/d sp)=\t{ jac_dab_coor  :.4f}\t||d kernel(orig,rand)/d sp)=\t{jac_dac_coor  :>.4f}\tevalutation time [ms] { (after-before)/ 10.0**6  :>.2f}   "
    )


def test_bessel():
  
    for func, name,sp,dsp,wz in zip(
        [jv, yv, iv, kv, spherical_jn, spherical_yn, ive, kve],
        ["jv", "yv", "iv", "kv", " spherical_jv", "spherical_yv", "ive", "kve"],
        [  scipy.special.jv, scipy.special.yv, scipy.special.iv, scipy.special.kv, scipy.special.spherical_jn, scipy.special.spherical_yn, scipy.special.ive, scipy.special.kve],
        [  scipy.special.jvp, scipy.special.yvp, scipy.special.ivp, scipy.special.kvp, lambda x,y: scipy.special.spherical_jn(x,y,derivative=True  ), lambda x,y: scipy.special.spherical_yn(x,y,derivative=True  ), None,None],
        [  1,0,1,0,0,0, None,None]
    ):
        # try to vmap and jit
        _ = vmap(func, in_axes=(0, None))(jnp.array([2, 5]), 2)
        _ = vmap(func, in_axes=(None, 0))(2, jnp.array([2, 5]))
        _ = vmap(vmap(func, in_axes=(0, None)), in_axes=(None, 0))(
            jnp.array([2, 5]), jnp.array([2, 5])
        )

        _ = vmap(
            vmap(vmap(func, in_axes=(0, None)), in_axes=(0, None)), in_axes=(None, 0)
        )(jnp.array([[2, 5], [2, 5]]), jnp.array([2, 5]))

        if wz==1:
            x = jnp.linspace(0, 20, 1000)
        else:
            x = jnp.linspace(1e-5, 20, 1000)
        v= jnp.arange(0, 5)
        y = vmap(vmap(func,in_axes=(0,None)),in_axes=(None,0))(v,x )
        y2 = jnp.array([sp( vi,  onp.array(x) ) for vi in v] ).T
        assert jnp.allclose( y,y2,atol=1e-5) 

        if dsp is not None:

            # with jax.disable_jit():
            dy =   vmap(vmap(  grad(func, argnums=1),in_axes=(0,None)),in_axes=(None,0)) (v,x )

            dy2 = jnp.array([dsp( vi,  onp.array(x)  ) for vi in v] ).T

            mask = jnp.logical_or(jnp.isnan(dy2) , jnp.isinf(dy2))

            assert jnp.allclose( dy[~mask],dy2[~mask],atol=1e-5,equal_nan=True )
        

    # sanity chekc to see wither ive and kve behave correctly

    k1e = lambda x: kve(1, x)
    k1e2 = lambda x: kv(1, x) * jnp.exp(x)

    x = jnp.linspace(1, 5, 1000)
    assert jnp.linalg.norm(vmap(k1e)(x) - vmap(k1e2)(x)) < 1e-5
    assert jnp.linalg.norm(vmap(grad(k1e))(x) - vmap(grad(k1e2))(x)) < 1e-5

    i1e = lambda x: ive(1, x)
    i1e2 = lambda x: iv(1, x) * jnp.exp(-jnp.abs(x))

    assert jnp.linalg.norm(vmap(i1e)(x) - vmap(i1e2)(x)) < 1e-5
    assert jnp.linalg.norm(vmap(grad(i1e))(x) - vmap(grad(i1e2))(x)) < 1e-5
