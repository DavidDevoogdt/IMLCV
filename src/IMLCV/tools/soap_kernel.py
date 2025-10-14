from functools import partial
from typing import Callable, ParamSpec, TypeVar

import jax.lax
import jax.numpy as jnp
import scipy.special
from jax import Array, lax
from scipy.special import legendre as sp_legendre

from IMLCV.base.CV import NeighbourList, ShmapKwargs, SystemParams, padded_shard_map
from IMLCV.base.datastructures import Partial_decorator, custom_jvp_decorator, jit_decorator, vmap_decorator
from IMLCV.external.quadrature import quad
from IMLCV.tools.bessel_callback import ie_n, spherical_jn

P = ParamSpec("P")
T1 = TypeVar("T1")
T2 = TypeVar("T2")


@partial(jit_decorator, static_argnums=(1,))
def legendre(x, n):
    c = jnp.array(sp_legendre(n).c, dtype=x.dtype)
    return jnp.sum(c * x ** (n - jnp.arange(n + 1)))


# @partial( jit_decorator, static_argnums=(2, 3))
def p_i(
    sp: SystemParams,
    nl: NeighbourList,
    f_single: Callable[[Array, Array], T1],
    f_double: Callable[[Array, Array, T1, Array, Array, T1], Array],
    r_cut: float,
    chunk_size_neigbourgs=None,
    chunk_size_atoms=None,
    chunk_size_batch=None,
    shmap=False,
    shmap_kwargs=ShmapKwargs.create(),
    merge_ZZ=True,
    reshape=True,
    mul_Z=True,
    normalize=True,
    Z_weights=None,
    reduce_Z=False,
):
    if sp.batched:
        _f: Callable[[SystemParams, NeighbourList], Array] = Partial_decorator(
            p_i,
            func_single=f_single,
            func_double=f_double,
            chunk_size_neigbourgs=chunk_size_neigbourgs,
            chunk_size_atoms=chunk_size_atoms,
            chunk_size_batch=chunk_size_batch,
            shmap=False,
            merge_ZZ=merge_ZZ,
            reshape=reshape,
        )

        if shmap:
            _f = padded_shard_map(_f, shmap_kwargs)

        return _f(sp, nl)

    # ps, pd = p

    _, out_nzzx = nl.apply_fun_neighbour_pair(
        sp=sp,
        func_single=f_single,
        func_double=f_double,
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

    if mul_Z and Z_weights is not None:

        @partial(vmap_decorator, in_axes=(None, None, 0, 2), out_axes=2)
        @partial(vmap_decorator, in_axes=(None, 0, None, 1), out_axes=1)
        @partial(vmap_decorator, in_axes=(0, None, None, 0), out_axes=0)
        def _mul_Z(Z1: Array, Z2: Array, Z3: Array, x: Array) -> Array:
            w1 = Z_weights[jnp.argwhere(jnp.array(nl.info.z_unique) == Z1, size=1)[0, 0]]
            w2 = Z_weights[jnp.argwhere(jnp.array(nl.info.z_unique) == Z2, size=1)[0, 0]]
            w3 = Z_weights[jnp.argwhere(jnp.array(nl.info.z_unique) == Z3, size=1)[0, 0]]

            return x * w1 * w2 * w3

        out_nzzx = _mul_Z(
            jnp.array(nl.info.z_array),
            jnp.array(nl.info.z_unique),
            jnp.array(nl.info.z_unique),
            out_nzzx,
        )

    if merge_ZZ:

        @partial(vmap_decorator, in_axes=(3), out_axes=2)
        @partial(vmap_decorator, in_axes=(0), out_axes=0)
        def _merge_ZZ(out_nzzx: Array) -> Array:
            x = jnp.tril_indices(out_nzzx.shape[0])
            out = jnp.where(x[0] == x[1], 1.0 * out_nzzx[x[0], x[1]], jnp.sqrt(2.0) * out_nzzx[x[0], x[1]])
            return out

        out = _merge_ZZ(out_nzzx)

    else:
        out = out_nzzx

    if reshape:
        out = jnp.reshape(out, (out.shape[0], -1))

    if normalize:
        # print(f"pre {out=}")

        @partial(vmap_decorator, in_axes=(0,))
        def n(x):
            n = jnp.sum(x**2)
            n = jnp.where(n > 1e-12, n, 1.0)

            return x / jnp.sqrt(n)

        out = n(out)

        # print(f"post {out=}")

    # if sum_Z:

    if reduce_Z:
        print(f"summing over Z {out.shape=}")

        @vmap_decorator
        def add(val, z):
            index = jnp.argwhere(jnp.array(nl.info.z_unique) == z, size=1)[0, 0]

            out = jnp.zeros((len(nl.info.z_unique), *val.shape))
            out = out.at[index].add(val)

            return out

        out = add(out, jnp.array(nl.info.z_array))
        out = jnp.sum(out, axis=0)

        print(f"post summing {out.shape=}")

    return out


# @partial(vmap_decorator, in_axes=(0, None, None), out_axes=0)
@partial(jit_decorator, static_argnums=(0,))
def legendre_l(l: int, pj: jax.Array, pk: jax.Array):
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


def p_innl_soap(
    l_max: int,
    n_max: int,
    r_cut: float,
    sigma_a: float,
    r_delta: float,
    num=100,
    basis="gto",
    reduce=True,
):
    # for explanation soap:
    # https://aip.scitation.org/doi/suppl/10.1063/1.5111045

    # n_max = float(n_max)

    # def f_cut(r: Array) -> Array:
    #     return lax.cond(
    #         r > r_cut,
    #         lambda: 0.0,
    #         lambda: lax.cond(
    #             r < r_cut - r_delta,
    #             lambda: 1.0,
    #             lambda: 1.0 / (1 + jnp.exp(((r - r_cut) + r_delta / 2) / r_delta * 15)),
    #         ),
    #     )

    def f_cut(r: Array) -> Array:
        return lax.cond(
            r > r_cut,
            lambda: 0.0,
            lambda: lax.cond(
                r < r_cut - r_delta,
                lambda: 1.0,
                lambda: jnp.exp(-1 / (1 - ((r - (r_cut - r_delta)) / r_delta) ** 4) + 1),
            ),
        )

    # if basis == "gto":
    #     # https://lab-cosmo.github.io/librascal/SOAP.html
    #     # TODO: integrates analytically

    #     # print(f"ignoring {sigma_a=}")

    #     # sigma_a = sigma_b

    #     def _phi(n: int | Array, n_max: int, r: Array, r_cut: float) -> Array:  # type: ignore
    #         # compress in the x direction
    #         r_scale = r_cut * (n + 1) / (n_max + 1)

    #         a = -jnp.log(1e-3) / r_scale**2

    #         # we want this to be 1e-3 at r_scale=1

    #         return jnp.exp(-a * r**2)

    # elif basis == "loc_gaussians":

    def _phi(n: int | Array, n_max: int, r: Array, r_cut: float) -> Array:
        return jnp.exp(-((2 * n_max * (r - n * (r_cut - r_delta) / (n_max)) / r_cut) ** 2) / (2))

    # elif basis == "cos":

    #     def _phi_0(n: int | Array, n_max: int, r: Array, r_cut: float) -> Array:
    #         return (r / r_cut) ** l * (jnp.cos(r / r_cut * (n + 0.5) * jnp.pi)) * jnp.where(r > r_cut, 0, 1)

    #     # make derivatives zero at r_cut

    #     def _phi_1(n: int | Array, n_max: int, r: Array, r_cut: float) -> Array:
    #         return _phi_0(n, n_max, r, r_cut) / (n + 1 / 2) + _phi_0(n + 1, n_max, r, r_cut) / (n + 3 / 2)

    #     _phi = _phi_1

    # else:
    #     raise ValueError(f"{basis=} not known")

    if sigma_a != 0:
        print(f"integrating")

        @partial(jit_decorator, static_argnums=(0, 1))
        def I_prime_ml(  # type: ignore
            n_max: int,
            l_max: int,
            r_ij: Array,
            sigma_a: float,
            r_cut: float,
        ):
            def _f(
                r: Array,
                r_ij: Array,
            ) -> Array:
                # https://mathworld.wolfram.com/ModifiedSphericalBesselFunctionoftheFirstKind.html

                r_safe = jnp.where(r > 1e-6, r, 1)

                ive_l: Array = ie_n(l_max, r_safe * r_ij / sigma_a**2, True, True)  # exponential scaled, corrected in c

                phi_n = vmap_decorator(
                    _phi,
                    in_axes=(0, None, None, None),
                    out_axes=0,
                )(jnp.arange(n_max + 1, dtype=jnp.int64), n_max, r_safe, r_cut)

                c = (
                    r ** (3 / 2)
                    * jnp.sqrt((sigma_a**2 * jnp.pi) / (2 * r_ij))
                    * jnp.exp(-((r_safe - r_ij) ** 2) / (2 * sigma_a**2))
                )

                @partial(vmap_decorator, in_axes=(0, None), out_axes=1)
                @partial(vmap_decorator, in_axes=(None, 0), out_axes=0)
                def _mul(_l: Array, _n: Array) -> Array:
                    out = _l * _n * c

                    return jnp.where(
                        r > 0,
                        out,
                        0.0,
                    )  # type: ignore

                out = _mul(ive_l, phi_n)

                return out

            I_prime = quad(_f, 1e-10, r_cut, n=num)(
                r_ij,
            ) / (sigma_a**3)

            return I_prime

    else:
        # print("using delta function")
        # limting case of previous func

        @partial(jit_decorator, static_argnums=(0, 1))
        def I_prime_ml(
            n_max: int,
            l_max: int,
            r_ij: Array,
            sigma_a: float,
            r_cut: float,
        ):
            phi_n = vmap_decorator(
                _phi,
                in_axes=(0, None, None, None),
                out_axes=0,
            )(jnp.arange(n_max + 1, dtype=jnp.int64), n_max, r_ij, r_cut)

            l = jnp.arange(l_max + 1, dtype=jnp.int64)

            # print(f"{phi_n.shape=} {l.shape=}")
            print(f" r**l")

            return jnp.outer(phi_n, r_ij**l)

    def S_nm(n: int, m: int):
        def g(r: jax.Array, sigma_a: float, r_cut: float, n_max: int):
            _phi_nl = _phi(n, n_max, r, r_cut)

            _phi_ml = _phi(m, n_max, r, r_cut)

            return _phi_nl * _phi_ml * r**2

        return quad(g, 1e-6, r_cut, n=101)(sigma_a, r_cut, n_max)

    _S_nm = jnp.zeros((n_max + 1, n_max + 1))

    for n in range(int(n_max) + 1):
        for m in range(int(n_max) + 1):
            _S_nm = _S_nm.at[n, m].set(S_nm(n, m))

    def _u_inv(S):
        U = jnp.linalg.cholesky(S)
        U_inv_nml = jnp.linalg.pinv(U)
        return U_inv_nml

    U_inv_nm = _u_inv(_S_nm)

    def _l(p_ij, p_ik):
        return jnp.array([legendre_l(l, p_ij, p_ik) for l in range(l_max + 1)])

    def a_nlj(r_ij, sigma_a, r_cut, U_inv_nm):
        I_nl = I_prime_ml(
            n_max,
            l_max,
            r_ij,
            sigma_a,
            r_cut,
        )

        return U_inv_nm @ I_nl * f_cut(r_ij)

    def _p_i_soap_2_s(p_ij: Array, atom_index_j: Array):
        r_ij2 = jnp.dot(p_ij, p_ij)
        r_ij_safe = jnp.where(r_ij2 <= 1e-6, jnp.ones_like(r_ij2), r_ij2)

        a_jnl_safe = a_nlj(jnp.sqrt(r_ij_safe), sigma_a, r_cut, U_inv_nm)

        a_jnl: Array = jnp.where(
            r_ij2 <= 1e-6,
            jnp.full_like(a_jnl_safe, fill_value=0.0),
            a_jnl_safe,
        )  # type:ignore

        return a_jnl

    # @jit
    def _p_i_soap_2_d(p_ij: Array, atom_index_j: Array, data_j: Array, p_ik: Array, atom_index_k: Array, data_k: Array):
        a_nl_j = data_j
        a_nl_k = data_k
        b_l_jk = _l(p_ij, p_ik)

        n_vec = jnp.arange(n_max + 1)
        l_vec = jnp.arange(l_max + 1)

        out_nnl = jnp.einsum(
            "l,al,bl,l->abl",
            4 * jnp.pi * (2 * l_vec + 1),
            a_nl_j,
            a_nl_k,
            b_l_jk,
        )

        # reduce the nn pair
        x = jnp.tril_indices(out_nnl.shape[0])
        out_nnl = jax.vmap(
            lambda y: jnp.where(
                x[0] == x[1],
                1.0 * y[x[0], x[1]],
                jnp.sqrt(2.0) * y[x[0], x[1]],
            ),
            in_axes=2,
            out_axes=1,
        )(out_nnl)

        # print(f"{out_nnl.shape=}")

        return out_nnl.reshape(-1)

    return _p_i_soap_2_s, _p_i_soap_2_d


def p_inl_sb(l_max: int, n_max: int, r_cut: float, bessel_fun="jax"):
    # for explanation soap:
    # https://aip.scitation.org/doi/suppl/10.1063/1.5111045

    # assert l_max == n_max, "l_max should be  equal to n_max"

    def f_cut(r: Array) -> Array:
        return jnp.where(r > r_cut, 0.0, 1.0)  # type:ignore

    if bessel_fun == "jax":

        def spherical_jn_2(n, z):
            @partial(vmap_decorator, in_axes=(None, 1), out_axes=2)
            @partial(vmap_decorator, in_axes=(None, 0), out_axes=1)
            def _spherical_jn(n, z):
                return spherical_jn(n, z, True)

            return _spherical_jn(n, z)

    elif bessel_fun == "scipy":
        # from jax import custom_jvp

        from IMLCV.tools.bessel_callback import spherical_jn_b

        # these avoid recalculation of the same values
        @partial(custom_jvp_decorator, nondiff_argnums=(0,))
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

    def spherical_jn_zeros(n: int, m: int):
        x0 = jnp.array((scipy.special.jn_zeros(n + 1, m) + scipy.special.jn_zeros(n, m)) / 2)

        import jaxopt

        def _opt(x):
            out = spherical_jn(n, x, True)

            return out[n] ** 2

        @vmap_decorator
        def _grad_desc(x: Array) -> Array:
            return (
                jaxopt.GradientDescent(
                    _opt,
                    maxiter=1000,
                    tol=1e-20,
                )
                .run(x)
                .params
            )

        return _grad_desc(x0)

    def show_spherical_jn_zeros(n, m, ngrid=100):
        """Graphical test for the above function"""

        zeros = spherical_jn_zeros(n, m)

        zeros_guess = (scipy.special.jn_zeros(n + 1, m) + scipy.special.jn_zeros(n, m)) / 2

        x = jnp.linspace(0, jnp.max(zeros), num=1000)
        y = vmap_decorator(spherical_jn, in_axes=(None, 0, None), out_axes=1)(n, x, True)[n, :]

        import matplotlib.pyplot as plt

        plt.plot(x, y)

        [plt.axvline(x0, color="r") for x0 in zeros]

        [plt.axvline(x0, color="b") for x0 in zeros_guess]
        plt.axhline(0, color="k")

    # show_spherical_jn_zeros(0, 5)

    l_vec = jnp.arange(0, l_max + 1)
    n_vec = jnp.arange(0, n_max + 1)

    # n+1th zero of spherical bessel function of the first kind
    u_ln = jnp.array([spherical_jn_zeros(int(n), l_max + 2) for n in range(n_max + 2)])

    s_lln = spherical_jn_2(l_max + 1, u_ln)

    @partial(vmap_decorator, in_axes=(0, None, None))
    @partial(vmap_decorator, in_axes=(None, 0, None))
    def f_nl(n: Array, l: Array, sj: Array) -> Array:
        return (
            u_ln[l, n + 1] / s_lln[l + 1, l, n] * sj[l, l, n] - u_ln[l, n] / s_lln[l + 1, l, n + 1] * sj[l, l, n + 1]
        ) * (2 / (u_ln[l, n] ** 2 + u_ln[l, n + 1] ** 2) / r_cut**3) ** (0.5)

    l_list = list(range(l_max + 1))
    l_vec = jnp.array(l_list)
    n_vec = jnp.arange(n_max + 1)

    # perform gramm shmidt

    def g(r: Array) -> Array:
        sj = spherical_jn_2(l_max, r * u_ln / r_cut)

        n_range = jnp.arange(n_max + 1)
        l_range = jnp.arange(l_max + 1)

        _f_nl = f_nl(n_range, l_range, sj)

        return jnp.einsum("nl,ml->nml", _f_nl, _f_nl) * r**2

    # print(f"integrating S")

    S_nml = quad(g, 1e-10, r_cut, n=100)()

    @partial(vmap_decorator, in_axes=(2), out_axes=(2))
    def _u_inv(S: Array):
        U = jnp.linalg.cholesky(S)
        U_inv_nml = jnp.linalg.pinv(U)
        return U_inv_nml

    U_inv_nml = _u_inv(S_nml)

    @jit_decorator
    def _l(p_ij: Array, p_ik: Array):
        return jnp.array([legendre_l(l, p_ij, p_ik) for l in l_list])

    def g_nl(r: Array):
        sj = spherical_jn_2(l_max, r * u_ln / r_cut)
        fnl = f_nl(n_vec, l_vec, sj)

        @partial(vmap_decorator, in_axes=(2, 1), out_axes=1)
        def _g_nl(
            U_inv_l: Array,
            fnl: Array,
        ):
            return U_inv_l @ fnl

        return _g_nl(
            U_inv_nml,
            fnl,
        )

    def _p_i_sb_2_s(p_ij: Array, atom_index_j: Array) -> Array:
        r_ij_sq = jnp.dot(p_ij, p_ij)

        r_ij_sq_safe = jnp.where(r_ij_sq == 0, jnp.full_like(r_ij_sq, 1e-6), r_ij_sq)
        r_ij = jnp.sqrt(r_ij_sq_safe)

        out = g_nl(r_ij) * f_cut(r_ij)
        a_jnl = jnp.where(r_ij_sq == 0, jnp.full_like(out, 0.0), out)

        return a_jnl

    def _p_i_sb_2_d(
        p_ij: Array,
        atom_index_j: Array,
        data_j: Array,
        p_ik: Array,
        atom_index_k: Array,
        data_k: Array,
    ):
        a_jnl = data_j
        a_knl = data_k

        b_ljk = _l(p_ij, p_ik)

        # this ensures that l<=n
        @partial(vmap_decorator, in_axes=(None, 0, None), out_axes=1)
        @partial(vmap_decorator, in_axes=(0, None, None), out_axes=0)
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

        out = out[jnp.tril_indices_from(out)]

        return out

    return _p_i_sb_2_s, _p_i_sb_2_d
