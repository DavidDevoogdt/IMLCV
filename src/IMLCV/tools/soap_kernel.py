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
from jax import Array, lax
from scipy.special import legendre as sp_legendre

from IMLCV.base.CV import NeighbourList, ShmapKwargs, SystemParams, padded_shard_map
from IMLCV.base.datastructures import Partial_decorator, jit_decorator, vmap_decorator
from IMLCV.external.quadrature import quad
from IMLCV.tools.bessel_callback import ie_n, spherical_jn


@partial(jax.jit, static_argnums=(1,))
def legendre(x, n):
    c = jnp.array(sp_legendre(n).c, dtype=x.dtype)
    return jnp.sum(c * x ** (n - jnp.arange(n + 1)))


# @partial( jit_decorator, static_argnums=(2, 3))
def p_i(
    sp: SystemParams,
    nl: NeighbourList,
    p,
    r_cut,
    chunk_size_neigbourgs=None,
    chunk_size_atoms=None,
    chunk_size_batch=None,
    shmap=False,
    shmap_kwargs=ShmapKwargs.create(),
):
    if sp.batched:
        f = Partial_decorator(
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


# @partial(vmap_decorator, in_axes=(0, None, None), out_axes=0)
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


def p_innl_soap(l_max, n_max, r_cut, sigma_a, r_delta, num=30, basis="gto"):
    # for explanation soap:
    # https://aip.scitation.org/doi/suppl/10.1063/1.5111045

    # n_max = float(n_max)

    def f_cut(r):
        return lax.cond(
            r > r_cut,
            lambda: 0.0,
            lambda: lax.cond(
                r < r_cut - r_delta,
                lambda: 1.0,
                lambda: 1.0 / (1 + jnp.exp((r - r_cut + r_delta / 2) / r_delta * 15)),
            ),
        )

    if basis == "gto":
        # https://lab-cosmo.github.io/librascal/SOAP.html
        # TODO: integrates analytically

        # print(f"ignoring {sigma_a=}")

        # sigma_a = sigma_b

        def _phi(n, n_max, r, r_cut, sigma_a, l, l_max):  # type: ignore
            # compress in the x direction
            r_scale = r_cut * (n + 1) / (n_max + 1)

            a = -jnp.log(1e-3) / r_scale**2

            # we want this to be 1e-3 at r_scale=1

            return r**l * jnp.exp(-a * r**2) * f_cut(r)

    elif basis == "loc_gaussians":

        def _phi(n, n_max, r, r_cut, sigma_a, l, l_max):
            return jnp.exp(-((r - r_cut * n / n_max) ** 2) / (2 * sigma_a**2)) * f_cut(r)
    elif basis == "cos":

        def _phi_0(n, n_max, r, r_cut, sigma_a, l, l_max):
            return (r / r_cut) ** l * (jnp.cos(r / r_cut * (n + 0.5) * jnp.pi)) * jnp.where(r > r_cut, 0, 1)

        # make derivatives zero at r_cut

        def _phi_1(n, n_max, r, r_cut, sigma_a, l, l_max):
            return _phi_0(n, n_max, r, r_cut, sigma_a, l, l_max) / (n + 1 / 2) + _phi_0(
                n + 1, n_max, r, r_cut, sigma_a, l, l_max
            ) / (n + 3 / 2)

        _phi = _phi_1

    else:
        raise ValueError(f"{basis=} not known")

    if sigma_a != 0:

        @partial(jit_decorator, static_argnums=(0, 1))
        def I_prime_ml(  # type: ignore
            n_max,
            l_max,
            r_ij,
            sigma_a,
            r_cut,
        ):
            def _f(
                r,
                r_ij,
                # sigma_a,
                # r_cut,
            ):
                # https://mathworld.wolfram.com/ModifiedSphericalBesselFunctionoftheFirstKind.html

                r_safe = jnp.where(r > 0, r, 1)

                # this is exp version
                ive_l = ie_n(l_max, r_safe * r_ij / sigma_a**2, half=True)  # exponential scaled, corrected in c

                phi_nl = vmap_decorator(
                    vmap_decorator(
                        _phi,
                        in_axes=(0, None, None, None, None, None, None),
                        out_axes=0,
                    ),
                    in_axes=(None, None, None, None, None, 0, None),
                    out_axes=1,
                )(
                    jnp.arange(n_max + 1, dtype=jnp.int64),
                    n_max,
                    r_safe,
                    r_cut,
                    sigma_a,
                    jnp.arange(l_max + 1, dtype=jnp.int64),
                    l_max,
                )

                c = (
                    r ** (3 / 2)
                    * jnp.sqrt(sigma_a**2 * jnp.pi / (2 * r_ij))
                    * jnp.exp(-((r_safe - r_ij) ** 2) / (2 * sigma_a**2))
                )

                @partial(vmap_decorator, in_axes=(0, 1), out_axes=1)
                @partial(vmap_decorator, in_axes=(None, 0), out_axes=0)
                def _mul(_l, _n):
                    out = _l * _n * c

                    return jnp.where(
                        r > 0,
                        out,
                        0.0,
                    )

                out = _mul(ive_l, phi_nl)

                return out

            I_prime = quad(_f, 1e-10, r_cut, n=num)(
                r_ij,
                # sigma_a,
                # r_cut,
            )

            return I_prime

    else:
        # print("using delta function")

        @partial(jit_decorator, static_argnums=(0, 1))
        def I_prime_ml(
            n_max,
            l_max,
            r_ij,
            sigma_a,
            r_cut,
        ):
            phi_nl = vmap_decorator(
                vmap_decorator(
                    _phi,
                    in_axes=(0, None, None, None, None, None, None),
                    out_axes=0,
                ),
                in_axes=(None, None, None, None, None, 0, None),
                out_axes=1,
            )(
                jnp.arange(n_max + 1, dtype=jnp.int64),
                n_max,
                r_ij,
                r_cut,
                sigma_a,
                jnp.arange(l_max + 1, dtype=jnp.int64),
                l_max,
            )

            return phi_nl  # * r_ij**2 this term is also not used in SB

    def S_nm(n, m):
        def g(r, sigma_a, r_cut, n_max):
            _phi_nl = vmap_decorator(_phi, in_axes=(None, None, None, None, None, 0, None))(
                n,
                n_max,
                r,
                r_cut,
                sigma_a,
                jnp.arange(l_max + 1, dtype=jnp.int64),
                l_max,
            )

            _phi_ml = vmap_decorator(_phi, in_axes=(None, None, None, None, None, 0, None))(
                m,
                n_max,
                r,
                r_cut,
                sigma_a,
                jnp.arange(l_max + 1, dtype=jnp.int64),
                l_max,
            )

            return _phi_nl * _phi_ml * r**2

        return quad(g, 1e-10, r_cut, n=101)(sigma_a, r_cut, n_max)

    S_nml = jnp.zeros((n_max + 1, n_max + 1, l_max + 1))

    for n in range(int(n_max) + 1):
        for m in range(int(n_max) + 1):
            S_nml = S_nml.at[n, m, :].set(S_nm(n, m))

    # jax.debug.print("S {}", S_nml)

    @partial(vmap_decorator, in_axes=(2), out_axes=(2))
    def _u_inv(S):
        U = jnp.linalg.cholesky(S)
        U_inv_nml = jnp.linalg.pinv(U)
        return U_inv_nml

    U_inv_nml = _u_inv(S_nml)

    # print("U_inv {}", U_inv_nml)
    # jax.debug.print("U inv {}", U_inv_nml)

    def _l(p_ij, p_ik):
        return jnp.array([legendre_l(l, p_ij, p_ik) for l in range(l_max + 1)])

    def a_nlj(r_ij, sigma_a, r_cut, U_inv_nml):
        I_nl = I_prime_ml(
            n_max,
            l_max,
            r_ij,
            sigma_a,
            r_cut,
        )

        @partial(vmap_decorator, in_axes=(2, 1), out_axes=1)
        def _g_nl(
            U_inv_l,
            I_nl,
        ):
            return U_inv_l @ I_nl

        return _g_nl(
            U_inv_nml,
            I_nl,
        )

    def _p_i_soap_2_s(p_ij, atom_index_j):
        r_ij2 = jnp.dot(p_ij, p_ij)
        r_ij_safe = jnp.where(r_ij2 <= 1e-15, jnp.ones_like(r_ij2), r_ij2)

        a_jnl_safe = a_nlj(jnp.sqrt(r_ij_safe), sigma_a, r_cut, U_inv_nml)

        a_jnl = jnp.where(
            r_ij2 <= 1e-15,
            jnp.full_like(a_jnl_safe, fill_value=0.0),
            a_jnl_safe,
        )

        return a_jnl

    # @jit
    def _p_i_soap_2_d(p_ij, atom_index_j, data_j, p_ik, atom_index_k, data_k):
        a_nl_j = data_j
        a_nl_k = data_k
        b_l_jk = _l(p_ij, p_ik)

        n_vec = jnp.arange(n_max + 1)
        l_vec = jnp.arange(l_max + 1)

        # this ensures that l<=n
        @partial(vmap_decorator, in_axes=(None, 0, None), out_axes=1)
        @partial(vmap_decorator, in_axes=(0, None, None), out_axes=0)
        def a_nml_l(n, l, a):
            return lax.cond(
                l <= n,
                lambda: a[n - l, l],
                lambda: jnp.zeros_like(a[0, 0]),
            )

        g_nml_l_j = a_nml_l(n_vec, l_vec, a_nl_j)
        g_nml_l_k = a_nml_l(n_vec, l_vec, a_nl_k)

        return jnp.einsum(
            "l,al,bl,l->abl",
            4 * jnp.pi * (2 * l_vec + 1),
            g_nml_l_j,
            g_nml_l_k,
            b_l_jk,
        )

    return _p_i_soap_2_s, _p_i_soap_2_d


def p_inl_sb(l_max, n_max, r_cut, bessel_fun="jax"):
    # for explanation soap:
    # https://aip.scitation.org/doi/suppl/10.1063/1.5111045

    assert l_max <= n_max, "l_max should be smaller or equal to n_max"

    if bessel_fun == "jax":
        spherical_jn_2 = vmap_decorator(
            vmap_decorator(spherical_jn, in_axes=(None, 0), out_axes=1),
            in_axes=(None, 1),
            out_axes=2,
        )
    elif bessel_fun == "scipy":
        from jax import custom_jvp

        from IMLCV.tools.bessel_callback import spherical_jn_b

        # these avoid recalculation of the same values
        @partial(custom_jvp, nondiff_argnums=(0,))
        @partial(jit_decorator, static_argnums=0)
        def vec_spherical_jn_2(n, z):
            @partial(vmap_decorator, in_axes=(0, None), out_axes=0)
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

        @partial(vmap_decorator, in_axes=(None, 1), out_axes=2)
        @partial(vmap_decorator, in_axes=(None, 0), out_axes=1)
        def spherical_jn_2(n, z):
            return vec_spherical_jn_2(n, z)
    else:
        raise ValueError(f"{bessel_fun=} not supported")

    def spherical_jn_zeros(n, m):
        x0 = jnp.array((scipy.special.jn_zeros(n + 1, m) + scipy.special.jn_zeros(n, m)) / 2)

        return vmap_decorator(
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
        y = vmap_decorator(spherical_jn, in_axes=(None, 0), out_axes=1)(n, x)[n, :]

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

    s_lln = spherical_jn_2(l_max + 1, u_ln)

    @partial(vmap_decorator, in_axes=(0, None, None))
    @partial(vmap_decorator, in_axes=(None, 0, None))
    def f_nl(n, l, sj):
        return (
            u_ln[l, n + 1] / s_lln[l + 1, l, n] * sj[l, l, n] - u_ln[l, n] / s_lln[l + 1, l, n + 1] * sj[l, l, n + 1]
        ) * (2 / (u_ln[l, n] ** 2 + u_ln[l, n + 1] ** 2) / r_cut**3) ** (0.5)

    l_list = list(range(l_max + 1))
    l_vec = jnp.array(l_list)
    n_vec = jnp.arange(n_max + 1)

    # nm1_vec = jnp.arange(n_max)

    # perform gramm shmidt

    def g(r):
        sj = spherical_jn_2(l_max, r * u_ln / r_cut)

        n_range = jnp.arange(n_max + 1)
        l_range = jnp.arange(l_max + 1)

        _f_nl = f_nl(n_range, l_range, sj)

        return jnp.einsum("nl,ml->nml", _f_nl, _f_nl) * r**2

    S_nml = quad(g, 1e-10, r_cut, n=50)()

    @partial(vmap_decorator, in_axes=(2), out_axes=(2))
    def _u_inv(S):
        U = jnp.linalg.cholesky(S)
        U_inv_nml = jnp.linalg.pinv(U)
        return U_inv_nml

    U_inv_nml = _u_inv(S_nml)

    # jax.debug.print("S={}, U_inv={}", S_nml, U_inv_nml)

    @jit_decorator
    def _l(p_ij, p_ik):
        return jnp.array([legendre_l(l, p_ij, p_ik) for l in l_list])

    def g_nl(r: Array):
        sj = spherical_jn_2(l_max, r * u_ln / r_cut)
        fnl = f_nl(n_vec, l_vec, sj)

        @partial(vmap_decorator, in_axes=(2, 1), out_axes=1)
        def _g_nl(
            U_inv_l,
            fnl,
        ):
            return U_inv_l @ fnl

        return _g_nl(
            U_inv_nml,
            fnl,
        )

        # def body(args, n):
        #     def inner(args):
        #         d_xlm, g_xlm = args

        #         d_xl = 1 - e_nl[n, l_vec] / d_xlm
        #         g_xl = (
        #             1
        #             / jnp.sqrt(d_xl)
        #             * (fnl[n, :] + jnp.sqrt(e_nl[n, l_vec] / d_xlm) * g_xlm)
        #         )

        #         return (d_xl, g_xl), g_xl

        #     def first(args):
        #         return (jnp.ones_like(fnl[0, :]), fnl[0, :]), fnl[0, :]

        #     return jax.lax.cond(n == 0, first, inner, args)

        # state, out = lax.scan(
        #     f=body,
        #     init=(
        #         fnl[0, :] * 0 + 1,
        #         fnl[0, :],
        #     ),
        #     xs=nm1_vec,
        #     # unroll=True,
        # )

        # return out

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

        # this ensures that l<=n
        @partial(vmap_decorator, in_axes=(None, 0, None), out_axes=1)
        @partial(vmap_decorator, in_axes=(0, None, None), out_axes=0)
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
