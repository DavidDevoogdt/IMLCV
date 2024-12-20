from functools import partial

import jax.debug
import jax.dtypes
import jax.lax
import jax.numpy as jnp
import jax.numpy.linalg
import jax.random
import jax.scipy
import jaxopt
import scipy.special
from jax import Array, lax, vmap
from jax.tree_util import Partial
from scipy.special import legendre as sp_legendre

from IMLCV.base.CV import NeighbourList, SystemParams, padded_shard_map, shmap_kwargs
from IMLCV.tools.bessel_callback import ive, spherical_jn


@partial(jax.jit, static_argnums=(1,))
def legendre(x, n):
    c = jnp.array(sp_legendre(n).c, dtype=x.dtype)
    return jnp.sum(c * x ** (n - jnp.arange(n + 1)))


# @partial(jit, static_argnums=(2, 3))
def p_i(
    sp: SystemParams,
    nl: NeighbourList,
    p,
    r_cut,
    chunk_size_neigbourgs=None,
    chunk_size_atoms=None,
    chunk_size_batch=None,
    shmap=True,
    shmap_kwargs=shmap_kwargs(),
):
    if sp.batched:
        f = Partial(
            p_i,
            p=p,
            chunk_size_neigbourgs=chunk_size_neigbourgs,
            chunk_size_atoms=chunk_size_atoms,
            chunk_size_batch=chunk_size_batch,
            shmap=False,
        )

        if shmap:
            f = padded_shard_map(f, shmap_kwargs)
        return f(sp, nl)

    ps, pd = p

    _, val0 = nl.apply_fun_neighbour_pair(
        sp=sp,
        func_single=ps,
        func_double=pd,
        r_cut=r_cut,
        fill_value=0.0,
        reduce="z",
        unique=True,
        split_z=False,
        chunk_size_neigbourgs=chunk_size_neigbourgs,
        chunk_size_atoms=chunk_size_atoms,
        chunk_size_batch=chunk_size_batch,
        shmap=shmap,
        shmap_kwargs=shmap_kwargs,
    )

    return val0


# @partial(vmap, in_axes=(0, None, None), out_axes=0)
@partial(jax.jit, static_argnums=(0,))
def legendre_l(l, pj, pk):
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

    def phi(n, n_max, r, r_cut, sigma_a):
        return jnp.exp(-((r - r_cut * n / n_max) ** 2) / (2 * sigma_a**2))

    @partial(vmap, in_axes=(None, 0, None), out_axes=1)
    @partial(vmap, in_axes=(0, None, None), out_axes=0)
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

        return jnp.apply_along_axis(lambda y: jax.scipy.integrate.trapezoid(y=y, x=x), axis=0, arr=y)

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

        return jax.scipy.integrate.trapezoid(y=y, x=x)

    l_vec = jnp.arange(0, l_max + 1)
    n_vec = jnp.arange(0, n_max + 1)

    print("Check indexing of meshgrid for SOAP!")

    indices = jnp.array(jnp.meshgrid(n_vec, n_vec))

    S = jnp.apply_along_axis(S_nm, axis=0, arr=indices)

    L, V = jnp.linalg.eigh(S)
    L = jnp.where(L < 0, jnp.zeros_like(L), L)

    U = jnp.diag(jnp.sqrt(L)) @ V.T
    U_inv_nm = jnp.linalg.pinv(U)

    l_list = list(range(l_max + 1))

    def _l(p_ij, p_ik):
        return jnp.array([legendre_l(l, p_ij, p_ik) for l in l_list])

    def a_nlj(r_ij):
        return U_inv_nm @ I_prime_ml(n_vec, l_vec, r_ij) * f_cut(r_ij)

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

    # @jit
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


def p_inl_sb(l_max, n_max, r_cut, bessel_fun="jax"):
    # for explanation soap:
    # https://aip.scitation.org/doi/suppl/10.1063/1.5111045

    assert l_max <= n_max, "l_max should be smaller or equal to n_max"

    if bessel_fun == "jax":
        spherical_jn_2 = jax.vmap(jax.vmap(spherical_jn, in_axes=(None, 0), out_axes=1), in_axes=(None, 1), out_axes=2)
    elif bessel_fun == "scipy":
        from jax import custom_jvp

        from IMLCV.tools.bessel_callback import spherical_jn_b

        # these avoid recalculation of the same values
        @partial(custom_jvp, nondiff_argnums=(0,))
        @partial(jax.jit, static_argnums=0)
        def vec_spherical_jn_2(n, z):
            @partial(jax.vmap, in_axes=(0, None), out_axes=0)
            def _spherical_jn_2(n, z):
                return spherical_jn_b(n, z)

            return _spherical_jn_2(jnp.arange(n + 1), z)

        @vec_spherical_jn_2.defjvp
        def vec_spherical_jn_2_jvp(n, primals, tangents):
            (x,) = primals
            (x_dot,) = tangents

            ni = n

            if ni == 0:
                ni = 1

            y = vec_spherical_jn_2(ni, x)

            dy = jnp.zeros(ni + 1)
            dy = dy.at[0].set(-y[1])
            dy = dy.at[1:].set(y[:-1] - (jnp.arange(1, len(y)) + 1.0) * y[1:] / x)

            return y[: n + 1], dy[: n + 1] * x_dot

        @partial(jax.vmap, in_axes=(None, 1), out_axes=2)
        @partial(jax.vmap, in_axes=(None, 0), out_axes=1)
        def spherical_jn_2(n, z):
            return vec_spherical_jn_2(n, z)
    else:
        raise ValueError(f"{bessel_fun=} not supported")

    def spherical_jn_zeros(n, m):
        x0 = jnp.array((scipy.special.jn_zeros(n + 1, m) + scipy.special.jn_zeros(n, m)) / 2)

        return vmap(
            lambda x: jaxopt.GradientDescent(
                lambda x: spherical_jn(n, x)[n] ** 2,
                maxiter=1000,
                tol=1e-20,
            )
            .run(x)
            .params,
        )(x0)

    def show_spherical_jn_zeros(n, m, ngrid=100):
        """Graphical test for the above function"""

        zeros = spherical_jn_zeros(n, m)

        zeros_guess = (scipy.special.jn_zeros(n + 1, m) + scipy.special.jn_zeros(n, m)) / 2

        x = jnp.linspace(0, jnp.max(zeros), num=1000)
        y = jax.vmap(spherical_jn, in_axes=(None, 0), out_axes=1)(n, x)[n, :]

        import matplotlib.pyplot as plt

        plt.plot(x, y)

        [plt.axvline(x0, color="r") for x0 in zeros]

        [plt.axvline(x0, color="b") for x0 in zeros_guess]
        plt.axhline(0, color="k")

        # plt.savefig("bessel_zero.png")

    # show_spherical_jn_zeros(0, 5)

    l_vec = jnp.arange(0, l_max + 1)
    n_vec = jnp.arange(0, n_max + 1)

    # n+1th zero of spherical bessel function of the first kind
    u_ln = jnp.array([spherical_jn_zeros(int(n), l_max + 2) for n in range(n_max + 2)])

    # print(f"{spherical_jn_2(l_max + 1, u_ln)}")

    # # perform check
    # @partial(vmap, in_axes=(0, None))
    # @partial(vmap, in_axes=(None, 0))
    # def _check(n, l):
    #     return spherical_jn(l_max + 1, u_ln[l, n])[l]

    # print(f"{_check(n_vec, l_vec)=}")

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

    s_lln = spherical_jn_2(l_max + 1, u_ln)

    @partial(vmap, in_axes=(0, None, None))
    @partial(vmap, in_axes=(None, 0, None))
    def f_nl(n, l, sj):
        return (
            u_ln[l, n + 1] / s_lln[l + 1, l, n] * sj[l, l, n] - u_ln[l, n] / s_lln[l + 1, l, n + 1] * sj[l, l, n + 1]
        ) * (2 / (u_ln[l, n] ** 2 + u_ln[l, n + 1] ** 2) / r_cut**3) ** (0.5)

    l_list = list(range(l_max + 1))
    l_vec = jnp.array(l_list)
    n_vec = jnp.arange(n_max + 1)

    nm1_vec = jnp.arange(n_max)

    @jax.jit
    def _l(p_ij, p_ik):
        return jnp.array([legendre_l(l, p_ij, p_ik) for l in l_list])

    def g_nl(r: Array):
        sj = spherical_jn_2(l_max, r * u_ln / r_cut)
        fnl = f_nl(n_vec, l_vec, sj)

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
            unroll=True,
        )

        return out

    def _p_i_sb_2_s(p_ij, atom_index_j):
        r_ij_sq = jnp.dot(p_ij, p_ij)

        r_ij_sq_safe = jax.lax.cond(
            r_ij_sq == 0,
            lambda: jnp.ones_like(r_ij_sq),
            lambda: r_ij_sq,
        )

        shape = jax.eval_shape(g_nl, r_ij_sq)

        # a_jnl = jnp.full(shape=shape.shape, fill_value=0.0, dtype=shape.dtype)
        # # print("bb")
        a_jnl = jax.lax.cond(
            r_ij_sq == 0,
            lambda: jnp.full(shape=shape.shape, fill_value=0.0, dtype=shape.dtype),
            lambda: g_nl(jnp.sqrt(r_ij_sq_safe)),
        )

        return a_jnl

    def _p_i_sb_2_d(p_ij, atom_index_j, data_j, p_ik, atom_index_k, data_k):
        a_jnl = data_j
        a_knl = data_k

        # return a_jnl

        # shape = jax.eval_shape(_l, p_ij, p_ik)
        # b_ljk = jnp.full(shape=shape.shape, fill_value=0.0, dtype=shape.dtype)

        # print("aab")

        # jax.debug.print("a {}", data_j.shape)

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
