from functools import partial

import jax.debug
import jax.dtypes
import jax.lax
import jax.numpy as jnp
import jax.numpy.linalg
import jax.random
import jax.scipy
import jaxopt
import matplotlib.pyplot as plt
import scipy.special
from IMLCV.base.CV import NeighbourList
from IMLCV.base.CV import SystemParams
from IMLCV.tools.bessel_callback import ive
from IMLCV.tools.bessel_callback import spherical_jn
from jax import Array
from jax import jit
from jax import lax
from jax import vmap
from scipy.special import legendre as sp_legendre


# @partial(jit, static_argnums=(1,))
def legendre(x, n):
    c = jnp.array(sp_legendre(n).c, dtype=x.dtype)

    y = jnp.zeros_like(x)
    y, _ = lax.scan(lambda y, p: (y * x + p, None), y, c, length=n + 1)

    return y


# @partial(jit, static_argnums=(2, 3))
def p_i(
    sp: SystemParams,
    nl: NeighbourList,
    p,
    r_cut,
    chunk_size_neigbourgs=None,
    chunk_size_atoms=None,
    chunk_size_batch=None,
):
    if sp.batched:
        return vmap(p_i, in_axes=(0, 0, None, None))(sp, nl, p, r_cut)

    ps, pd = p

    _, val0 = nl.apply_fun_neighbour_pair(
        sp=sp,
        func_single=ps,
        func_double=pd,
        r_cut=r_cut,
        fill_value=0.0,
        reduce="full",
        unique=True,
        split_z=True,
        chunk_size_neigbourgs=chunk_size_neigbourgs,
        chunk_size_atoms=chunk_size_atoms,
        chunk_size_batch=chunk_size_batch,
    )

    return val0


# @partial(vmap, in_axes=(0, None, None), out_axes=0)
# @partial(jit, static_argnums=(0,))
def lengendre_l(l, pj, pk):
    # https://jax.readthedocs.io/en/latest/faq.html#gradients-contain-nan-where-using-where

    n2 = jnp.dot(pj, pj) * jnp.dot(pk, pk)

    n2s = jax.lax.cond(
        n2 > 0,
        lambda: n2,
        lambda: jnp.ones_like(n2),
    )

    cos_ang = jnp.dot(pj, pk) * jax.lax.cond(
        n2 > 0,
        lambda: 1 / n2s ** (0.5),
        lambda: jnp.zeros_like(n2),
    )

    return legendre(cos_ang, l)


def p_innl_soap(l_max, n_max, r_cut, sigma_a, r_delta, num=50):
    # for explanation soap:
    # https://aip.scitation.org/doi/suppl/10.1063/1.5111045

    @jit
    def phi(n, n_max, r, r_cut, sigma_a):
        return jnp.exp(-((r - r_cut * n / n_max) ** 2) / (2 * sigma_a**2))

    @partial(vmap, in_axes=(None, 0, None), out_axes=1)
    @partial(vmap, in_axes=(0, None, None), out_axes=0)
    @jit
    def I_prime_ml(n, l_vec, r_ij):
        def f(r):
            # https://mathworld.wolfram.com/ModifiedSphericalBesselFunctionoftheFirstKind.html
            def fi(r, r_ij):
                return (
                    r ** (3 / 2)
                    * jnp.sqrt(sigma_a**2 * jnp.pi / (2 * r_ij))
                    * phi(n, n_max, r, r_cut, sigma_a)
                    * jnp.exp(-((r - r_ij) ** 2) / (2 * sigma_a**2))
                    * ive(l_vec + 0.5, r * r_ij / sigma_a**2)
                )

            return jnp.where(
                r > 0,
                jnp.where(
                    r_ij > 0,
                    fi(jnp.where(r > 0, r, 1), jnp.where(r_ij > 0, r_ij, 1)),
                    0.0,
                ),
                0.0,
            )

        x = jnp.linspace(0, r_cut, num=num)
        y = vmap(f)(x)

        return jnp.apply_along_axis(lambda y: jnp.trapz(y=y, x=x), axis=0, arr=y)

    @jit
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
            return phi(ind[0], n_max, r, r_cut, sigma_a) * phi(ind[1], n_max, r, r_cut, sigma_a) * r**2

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

    l_list = list(range(l_max + 1))

    @jit
    def _l(p_ij, p_ik):
        return jnp.array([lengendre_l(l, p_ij, p_ik) for l in l_list])

    def a_nlj(r_ij):
        return U_inv_nm @ I_prime_ml(n_vec, l_vec, r_ij) * f_cut(r_ij)

    @jit
    def _p_i_soap_2_s(p_ij, atom_index_j):
        r_ij2 = jnp.dot(p_ij, p_ij)
        r_ij2 = jax.lax.cond(r_ij2 == 0, lambda: jnp.ones_like(r_ij2), lambda: r_ij2)

        shape = jax.eval_shape(a_nlj, r_ij2)

        a_jnl = jax.lax.cond(
            r_ij2 == 0,
            lambda: jnp.full(shape=shape.shape, fill_value=0.0, dtype=shape.dtype),
            lambda: a_nlj(jnp.sqrt(r_ij2)),
        )

        return a_jnl

    @jit
    def _p_i_soap_2_d(p_ij, atom_index_j, data_j, p_ik, atom_index_k, data_k):
        a_nlj = data_j
        a_nlk = data_k
        b_ljk = _l(p_ij, p_ik)

        return jnp.einsum(
            "l,al,bl,l->abl",
            4 * jnp.pi * (2 * l_vec + 1),
            a_nlj,
            a_nlk,
            b_ljk,
        )

    return _p_i_soap_2_s, _p_i_soap_2_d


def p_inl_sb(l_max, n_max, r_cut):
    # for explanation soap:
    # https://aip.scitation.org/doi/suppl/10.1063/1.5111045

    assert l_max <= n_max, "l_max should be smaller or equal to n_max"

    def spherical_jn_zeros(n, m):
        return vmap(
            lambda x: jaxopt.GradientDescent(
                lambda x: spherical_jn(n, x) ** 2,
                maxiter=1000,
            )
            .run(x)
            .params,
        )(
            jnp.array(
                (scipy.special.jn_zeros(n + 1, m) + scipy.special.jn_zeros(n, m)) / 2,
            ),
        )

    def show_spherical_jn_zeros(n, m, ngrid=100):
        """Graphical test for the above function"""

        zeros = spherical_jn_zeros(n, m)
        zeros_guess = (scipy.special.jn_zeros(n + 1, m) + scipy.special.jn_zeros(n, m)) / 2

        x = jnp.linspace(0, jnp.max(zeros), num=1000)
        y = spherical_jn(n, x)

        plt.plot(x, y)

        [plt.axvline(x0, color="r") for x0 in zeros]

        [plt.axvline(x0, color="b") for x0 in zeros_guess]
        plt.axhline(0, color="k")

    u_ln = jnp.array([spherical_jn_zeros(n, l_max + 2) for n in range(n_max + 2)]).T

    def e(x):
        l, n = x
        return (
            u_ln[l, n - 1] ** 2
            * u_ln[l, n + 1] ** 2
            / ((u_ln[l, n - 1] ** 2 + u_ln[l, n] ** 2) * (u_ln[l, n + 1] ** 2 + u_ln[l, n] ** 2))
        )

    e_nl = jnp.apply_along_axis(
        func1d=e,
        axis=0,
        arr=jnp.array(jnp.meshgrid(jnp.arange(l_max + 2), jnp.arange(n_max + 2))),
    )

    @partial(vmap, in_axes=(0, None, None))
    @partial(vmap, in_axes=(None, 0, None))
    @jit
    def f_nl(n, l, r):
        def f(r):
            return (
                u_ln[l, n + 1] / spherical_jn(l + 1, u_ln[l, n]) * spherical_jn(l, r * u_ln[l, n] / r_cut)
                - u_ln[l, n] / spherical_jn(l + 1, u_ln[l, n + 1]) * spherical_jn(l, r * u_ln[l, n + 1] / r_cut)
            ) * (2 / (u_ln[l, n] ** 2 + u_ln[l, n + 1]) / r_cut**3) ** (0.5)

        return f(r)

    l_list = list(range(l_max + 1))
    l_vec = jnp.array(l_list)
    n_vec = jnp.arange(n_max + 1)

    nm1_vec = jnp.arange(n_max)

    @jit
    def _l(p_ij, p_ik):
        return jnp.array([lengendre_l(l, p_ij, p_ik) for l in l_list])

    @jit
    def g_nl(r: Array):
        fnl = f_nl(n_vec, l_vec, r)

        def body(args, n):
            def inner(args):
                d_xlm, g_xlm = args

                d_xl = 1 - e_nl[n, l_vec] / d_xlm
                g_xl = 1 / jnp.sqrt(d_xl) * (fnl[n, :] + jnp.sqrt(e_nl[n, l_vec] / d_xlm) * g_xlm)

                return (d_xl, g_xl), g_xl

            def first(args):
                return (jnp.ones_like(fnl[0, :]), fnl[0, :]), fnl[0, :]

            return jax.lax.cond(n == 0, first, inner, args)

        state, out = lax.scan(
            f=body,
            init=(
                fnl[0, :] * 0 + 1,
                fnl[0, :],
            ),
            xs=nm1_vec,
        )

        return out

    @jit
    def _p_i_sb_2_s(p_ij, atom_index_j):
        r_ij_sq = jnp.dot(p_ij, p_ij)
        r_ij_sq_safe = jax.lax.cond(
            r_ij_sq == 0,
            lambda: jnp.ones_like(r_ij_sq),
            lambda: r_ij_sq,
        )

        shape = jax.eval_shape(g_nl, r_ij_sq)

        a_jnl = jax.lax.cond(
            r_ij_sq == 0,
            lambda: jnp.full(shape=shape.shape, fill_value=0.0, dtype=shape.dtype),
            lambda: g_nl(jnp.sqrt(r_ij_sq_safe)),
        )

        return a_jnl

    @jit
    def _p_i_sb_2_d(p_ij, atom_index_j, data_j, p_ik, atom_index_k, data_k):
        a_jnl = data_j
        a_knl = data_k

        b_ljk = _l(p_ij, p_ik)

        @partial(vmap, in_axes=(None, 0, None), out_axes=1)
        @partial(vmap, in_axes=(0, None, None), out_axes=0)
        def a_nml_l(n, l, a):
            return lax.cond(
                l <= n,
                lambda: a[n - l, l],
                lambda: jnp.zeros_like(a[0, 0]),
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

    return _p_i_sb_2_s, _p_i_sb_2_d
