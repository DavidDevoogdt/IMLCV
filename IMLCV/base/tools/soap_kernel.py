from functools import partial
from typing import Callable

import jax.debug
import jax.dtypes
import jax.lax
import jax.numpy as jnp
import jax.numpy.linalg
import jax.random
import jax.scipy
import matplotlib.pyplot as plt
from jax import jacfwd, jit, lax, vmap
from scipy.optimize import root
from scipy.spatial.transform import Rotation as R
from scipy.special import jn_zeros, spherical_jn
from tensorflow_probability.substrates import jax as tfp

from IMLCV.base.tools.bessel_callback import spherical_jv


@jit
def comb(n, k):
    val = 1.0

    def f(n, k):
        return lax.fori_loop(
            lower=1,
            upper=k + 1,
            body_fun=lambda k, prod: prod * (n - k + 1) / k,
            init_val=val,
        )

    return lax.cond(
        n - k < k,
        lambda: f(n, n - k),
        lambda: f(n, k),
    )


def legendre(x, n):
    val = x * 0.0 + 1

    return lax.fori_loop(
        lower=1,
        upper=n + 1,
        body_fun=lambda k, sum: sum
        + comb(n, k) * comb(n + k, k) * ((x - 1.0) / 2.0) ** k,
        init_val=val,
    )


from jax.config import config

from IMLCV.base.CV import SystemParams
from IMLCV.base.MdEngine import StaticTrajectoryInfo

config.update("jax_debug_nans", True)


@partial(jit, static_argnums=(1, 2, 4, 5))
def p_i(
    sp: SystemParams,
    sti: StaticTrajectoryInfo | None,
    p: Callable[[jnp.ndarray], jnp.ndarray],
    r_cut,
    z1=None,
    z2=None,
):
    """
    positions: array with shape (n,3)
    f: callable that converts calculates contribution for single position

    calculates relative postions vectors for each position as central atom and applies the corresponding power function
    """
    # with jax.disable_jit():

    if sp.cell is not None:
        sp.minkowski_reduce()

    if z1 is not None:
        assert sti is not None
        assert sti.atomic_numbers is not None
        sp_z1 = sp[sti.atomic_numbers == z1]
    else:
        assert sti is not None
        assert sti.atomic_numbers is not None
        sp_z1 = sp

    if z2 is not None:
        sp_z2 = sp[sti.atomic_numbers == z2]
    else:
        sp_z2 = sp

    k, val = vmap(
        jit(
            lambda coor: sp_z1.apply_fun_neighbourgh_pairs(
                r_cut=r_cut,
                center_coordiantes=coor,
                g=p,
            )
        )
    )(sp_z2.coordinates)

    # normalize
    return jnp.einsum("i...,i->i...", val, 1 / vmap(jnp.linalg.norm)(val))


@partial(vmap, in_axes=(0, None, None), out_axes=0)
def lengendre_l(l, pj, pk):
    cos_theta = jnp.dot(pj, pk) / (jnp.linalg.norm(pj) * jnp.linalg.norm(pk))
    return legendre(cos_theta, l)


def p_innl_soap(l_max, n_max, r_cut, sigma_a, r_delta, num=50):
    # for explanation soap:
    # https://aip.scitation.org/doi/suppl/10.1063/1.5111045

    def phi(n, n_max, r, r_cut, sigma_a):
        return jnp.exp(-((r - r_cut * n / n_max) ** 2) / (2 * sigma_a**2))

    # @partial(vmap, in_axes=(None, None, 0), out_axes=2)
    @partial(vmap, in_axes=(None, 0, None), out_axes=1)
    @partial(vmap, in_axes=(0, None, None), out_axes=0)
    def I_prime_ml(n, l, r_ij):
        def f(r):
            # https://mathworld.wolfram.com/ModifiedSphericalBesselFunctionoftheFirstKind.html
            return (
                r ** (3 / 2)
                * jnp.sqrt(sigma_a**2 * jnp.pi / (2 * r_ij))
                * tfp.math.bessel_ive(l + 0.5, r * r_ij / sigma_a**2)
                * phi(n, n_max, r, r_cut, sigma_a)
                * jnp.exp(-((r - r_ij) ** 2) / (2 * sigma_a**2))
            )

        x = jnp.linspace(jax.dtypes.finfo(r_ij.dtype).eps, r_cut, num=num)
        y = vmap(f)(x)

        return jnp.trapz(y=y, x=x)

    def f_cut(r):
        return lax.cond(
            r > r_cut,
            lambda: 0.0,
            lambda: lax.cond(
                r < r_cut - r_delta,
                lambda: 1.0,
                lambda: 0.5 * (1 + jnp.cos(jnp.pi * (r - r_cut + r_delta) / r_delta)),
            ),
        )

    def S_nm(ind):
        def g(r):
            return (
                phi(ind[0], n_max, r, r_cut, sigma_a)
                * phi(ind[1], n_max, r, r_cut, sigma_a)
                * r**2
            )

        x = jnp.linspace(0, r_cut, num=num)
        y = g(x)

        return jnp.trapz(y=y, x=x)

    l_vec = jnp.arange(0, l_max + 1)
    n_vec = jnp.arange(0, n_max + 1)

    indices = jnp.array(jnp.meshgrid(n_vec, n_vec))

    S = jnp.apply_along_axis(S_nm, axis=0, arr=indices)

    L, V = jnp.linalg.eigh(S)
    L = L.at[L < 0].set(0)

    U = jnp.diag(jnp.sqrt(L)) @ V.T
    U_inv_nm = jnp.linalg.pinv(U)

    def _p_i_soap_2(p_ij, p_ik, atom_index_j, atom_index_k):
        r_ij = jnp.linalg.norm(p_ij)
        r_ik = jnp.linalg.norm(p_ik)

        a_nlj = U_inv_nm @ I_prime_ml(n_vec, l_vec, r_ij) * f_cut(r_ij)
        a_nlk = U_inv_nm @ I_prime_ml(n_vec, l_vec, r_ik) * f_cut(r_ik)
        b_ljk = lengendre_l(l_vec, p_ij, p_ik)

        return jnp.einsum(
            "l,al,bl,l->abl", 4 * jnp.pi * (2 * l_vec + 1), a_nlj, a_nlk, b_ljk
        )

    return _p_i_soap_2


def p_inl_sb(l_max, n_max, r_cut):
    # for explanation soap:
    # https://aip.scitation.org/doi/suppl/10.1063/1.5111045

    assert l_max <= n_max, "l_max should be smaller or equal to n_max"

    def spherical_jn_zeros(n, m):
        def fn(x):
            return spherical_jn(n, x)

        return jnp.array(
            [root(fn, x0).x[0] for x0 in (jn_zeros(n + 1, m) + jn_zeros(n, m)) / 2]
        )

    def show_spherical_jn_zeros(n, m, ngrid=100):
        """Graphical test for the above function"""

        zeros = spherical_jn_zeros(n, m)
        zeros_guess = (jn_zeros(n + 1, m) + jn_zeros(n, m)) / 2

        x = jnp.linspace(0, jnp.max(zeros), num=1000)
        y = spherical_jn(n, x)

        plt.plot(x, y)

        [plt.axvline(x0, color="r") for x0 in zeros]

        [plt.axvline(x0, color="b") for x0 in zeros_guess]
        plt.axhline(0, color="k")

    # uln is the (n + 1)th nonzero root of jl(r),
    u_ln = jnp.array([spherical_jn_zeros(n, l_max) for n in range(n_max + 1)])

    @partial(vmap, in_axes=(None, 0, None), out_axes=0)
    def f_nl(n, l, r):

        a_nl = u_ln[l, n + 1] / spherical_jv(l + 1, u_ln[l, n])
        b_nl = u_ln[l, n] / spherical_jv(l + 1, u_ln[l, n + 1])
        norm_nl = jnp.sqrt(2 / (u_ln[l, n] ** 2 + u_ln[l, n + 1]) / r_cut**3)

        return (
            a_nl * spherical_jv(l, r * u_ln[l, n] / r_cut)
            + b_nl * spherical_jv(l, r * u_ln[l, n + 1] / r_cut)
        ) * norm_nl

    l_vec = jnp.arange(l_max + 1)
    n_vec = jnp.arange(n_max + 1)

    @partial(vmap, in_axes=(0, None), out_axes=0)
    def e_nl(l, n):
        return (
            u_ln[l, n - 1] ** 2
            * u_ln[l, n + 1] ** 2
            / (
                (u_ln[l, n - 1] ** 2 + u_ln[l, n] ** 2)
                * (u_ln[l, n + 1] ** 2 + u_ln[l, n] ** 2)
            )
        )

    def g_nl(r):

        d_xl = 1
        g_xl = None

        for n in range(n_max):
            fnl = f_nl(n, l_vec, r)
            if g_xl is None:  # n=0
                g_xl = [fnl]
                continue
            enl = e_nl(l_vec, n)

            d_xl = 1 - enl / d_xl
            g_xl.append(1 / jnp.sqrt(d_xl) * (fnl + jnp.sqrt(enl / d_xl) * g_xl[-1]))

        return jnp.array(g_xl)

    def _p_i_sb_2(p_ij, p_ik, atom_index_j, atom_index_k):
        r_ij = jnp.linalg.norm(p_ij)
        r_ik = jnp.linalg.norm(p_ik)

        a_jnl = g_nl(r_ij)
        a_knl = g_nl(r_ik)
        b_ljk = lengendre_l(l_vec, p_ij, p_ik)

        @partial(vmap, in_axes=(None, 0, None), out_axes=1)
        @partial(vmap, in_axes=(0, None, None), out_axes=0)
        def a_nml_l(n, l, a):
            return lax.cond(
                l <= n,
                lambda: a[n - l, l],
                lambda: 0.0,
            )

        g_nml_l_j = a_nml_l(n_vec, l_vec, a_jnl)
        g_nml_l_k = a_nml_l(n_vec, l_vec, a_knl)

        out = jnp.einsum(
            "l,nl,nl,l -> nl",
            (2 * l_vec + 1) / (4 * jnp.pi),
            g_nml_l_j,
            g_nml_l_k,
            b_ljk,
        )

        return out

    return _p_i_sb_2


def Kernel(p1, p2, xi=2):
    """p1 and p2 should be already normalised"""
    return jnp.diag(jnp.tensordot(p1, jnp.moveaxis(p2, 0, -1), axes=p1.ndim - 1)) ** xi


if __name__ == "__main__":
    rng = jax.random.PRNGKey(42)
    key, rng = jax.random.split(rng)
    pos = jax.random.uniform(key, (22, 3)) * 5

    key, key2, rng = jax.random.split(rng, 3)
    pos2 = pos @ jnp.array(R.random().as_matrix())
    # pos2 = jax.random.permutation(key, pos2) #random order other atoms
    pos2 = pos2 + jax.random.uniform(key, (3,))

    pos3 = jax.random.uniform(key, (22, 3)) * 5

    l_max = 10
    n_max = 10

    r_cut = 3.7

    for pp in [
        p_inl_sb(
            l_max=l_max,
            n_max=n_max,
            r_cut=r_cut,
        ),
        p_innl_soap(
            l_max=l_max,
            n_max=n_max,
            r_cut=r_cut,
            sigma_a=0.5,
            r_delta=1.0,
            num=100,
        ),
    ]:

        def f(pos):
            return p_i(
                sp=SystemParams(coordinates=pos, cell=None), p=pp, r_cut=r_cut, sti=None
            )

        a = f(pos)

        from time import time_ns

        before = time_ns()
        b = f(pos2)
        after = time_ns()
        c = f(pos3)

        print(
            f"l_max {l_max}  n_max {l_max} <kernel(orig,rot)>={ jnp.mean( Kernel( a,b ) ) }, <kernel(orig, rand)>= { jnp.mean( Kernel( a,c ) ) }, evalutation time [ms] { (after-before)/ 10.0**6  }  "
        )

        # check the derivatives
        c = jacfwd(f)(pos)
        print(f"gradient shape:{c.shape} nans: {jnp.sum(  jnp.isnan(c ) )}")
