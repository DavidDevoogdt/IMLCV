from collections.abc import Callable
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
from jax import Array, jacfwd, jit, lax, vmap
from scipy.spatial.transform import Rotation as R

from IMLCV.base.CV import NeighbourList, SystemParams
from IMLCV.base.MdEngine import StaticTrajectoryInfo
from IMLCV.base.tools.bessel_callback import ive, spherical_jn

# todo: Optimizing many-body atomic descriptors for enhanced computational performance of
# machine learning based interatomic potentials


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


def p_i(
    sp: SystemParams,
    nl: NeighbourList,
    # sti: StaticTrajectoryInfo,
    p,
    r_cut,
    split_z=True,
):
    @NeighbourList.batch_sp_nl
    def _p_i(sp: SystemParams, nl: NeighbourList, p):
        p, ps, pd = p

        k0, val0 = nl.apply_fun_neighbour_pair(
            sp=sp,
            func_single=ps,
            func_double=pd,
            r_cut=r_cut,
            fill_value=0.0,
            reduce=jnp.sum,
            unique=True,
            split_z=True,
        )

        if split_z:
            val0 = jnp.einsum("abi...->iab...", val0)

        norms = vmap(jnp.linalg.norm)(val0)
        norms_inv = jnp.where(norms != 0, 1 / norms, 1.0)
        return jnp.einsum("i...,i->i...", val0, norms_inv)

    return _p_i(sp, nl, p)


@partial(vmap, in_axes=(0, None, None), out_axes=0)
def lengendre_l(l, pj, pk):
    # https://jax.readthedocs.io/en/latest/faq.html#gradients-contain-nan-where-using-where

    n = jnp.linalg.norm(pj) * jnp.linalg.norm(pk)

    return jnp.where(
        n > 0,
        legendre(jnp.dot(pj, pk) / jnp.where(n > 0, n, 1), l),
        0.0,
    )


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

    @jit
    def _p_i_soap_2(p_ij, p_ik, atom_index_j, atom_index_k):
        r_ij = jnp.linalg.norm(p_ij)
        r_ik = jnp.linalg.norm(p_ik)

        a_nlj = U_inv_nm @ I_prime_ml(n_vec, l_vec, r_ij) * f_cut(r_ij)
        a_nlk = U_inv_nm @ I_prime_ml(n_vec, l_vec, r_ik) * f_cut(r_ik)
        b_ljk = lengendre_l(l_vec, p_ij, p_ik)

        return jnp.einsum(
            "l,al,bl,l->abl", 4 * jnp.pi * (2 * l_vec + 1), a_nlj, a_nlk, b_ljk
        )

    @jit
    def _p_i_soap_2_s(p_ij, atom_index_j):
        r_ij = jnp.linalg.norm(p_ij)
        return U_inv_nm @ I_prime_ml(n_vec, l_vec, r_ij) * f_cut(r_ij)

    @jit
    def _p_i_soap_2_d(p_ij, atom_index_j, data_j, p_ik, atom_index_k, data_k):

        a_nlj = data_j
        a_nlk = data_k
        b_ljk = lengendre_l(l_vec, p_ij, p_ik)

        return jnp.einsum(
            "l,al,bl,l->abl", 4 * jnp.pi * (2 * l_vec + 1), a_nlj, a_nlk, b_ljk
        )

    return _p_i_soap_2, _p_i_soap_2_s, _p_i_soap_2_d


def p_inl_sb(l_max, n_max, r_cut):
    # for explanation soap:
    # https://aip.scitation.org/doi/suppl/10.1063/1.5111045

    assert l_max <= n_max, "l_max should be smaller or equal to n_max"

    def spherical_jn_zeros(n, m):

        return vmap(
            lambda x: jaxopt.GradientDescent(
                lambda x: spherical_jn(n, x) ** 2, maxiter=1000
            )
            .run(x)
            .params
        )(
            jnp.array(
                (scipy.special.jn_zeros(n + 1, m) + scipy.special.jn_zeros(n, m)) / 2
            )
        )

    def show_spherical_jn_zeros(n, m, ngrid=100):
        """Graphical test for the above function"""

        zeros = spherical_jn_zeros(n, m)
        zeros_guess = (
            scipy.special.jn_zeros(n + 1, m) + scipy.special.jn_zeros(n, m)
        ) / 2

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
            / (
                (u_ln[l, n - 1] ** 2 + u_ln[l, n] ** 2)
                * (u_ln[l, n + 1] ** 2 + u_ln[l, n] ** 2)
            )
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
        return (
            u_ln[l, n + 1]
            / spherical_jn(l + 1, u_ln[l, n])
            * spherical_jn(l, r * u_ln[l, n] / r_cut)
            - u_ln[l, n]
            / spherical_jn(l + 1, u_ln[l, n + 1])
            * spherical_jn(l, r * u_ln[l, n + 1] / r_cut)
        ) * (2 / (u_ln[l, n] ** 2 + u_ln[l, n + 1]) / r_cut**3) ** (0.5)

    l_list = list(range(l_max + 1))
    l_vec = jnp.array(l_list)
    n_vec = jnp.arange(n_max + 1)

    nm1_vec = jnp.arange(n_max)

    @jit
    def g_nl(r: Array):

        fnl = f_nl(n_vec, l_vec, r)

        def body(args, n):
            def inner(args):
                d_xlm, g_xlm = args

                d_xl = 1 - e_nl[n, l_vec] / d_xlm
                g_xl = (
                    1
                    / jnp.sqrt(d_xl)
                    * (fnl[n, :] + jnp.sqrt(e_nl[n, l_vec] / d_xlm) * g_xlm)
                )

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

    # return _p_i_sb_2

    @jit
    def _p_i_sb_2_s(p_ij, atom_index_j):
        r_ij = jnp.linalg.norm(p_ij)

        a_jnl = g_nl(r_ij)

        return a_jnl

    @jit
    def _p_i_sb_2_d(p_ij, atom_index_j, data_j, p_ik, atom_index_k, data_k):

        a_jnl = data_j
        a_knl = data_k
        b_ljk = lengendre_l(l_vec, p_ij, p_ik)

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

    return _p_i_sb_2, _p_i_sb_2_s, _p_i_sb_2_d


def Kernel(p1, p2, xi=2, matching="REMatch"):
    """p1 and p2 should be already normalised"""

    if matching == "average":

        def _f(a, b):
            return jnp.tensordot(
                jnp.mean(a, axis=0), jnp.mean(b, axis=0), axes=a.ndim - 1
            )

    elif matching == "REMatch":

        # adapted from https://singroup.github.io/dscribe/0.3.x/_modules/dscribe/kernels/rematchkernel.html

        alpha = 0.1
        threshold = 1e-6

        # @jit
        def _f(p1, p2):
            """
            Computes the REMatch similarity between two structures A and B.

            Args:
                localkernel(np.ndarray): NxM matrix of local similarities between
                    structures A and B, with N and M atoms respectively.
            Returns:
                float: REMatch similarity between the structures A and B.
            """

            localkernel = (
                jnp.tensordot(p1, jnp.moveaxis(p2, 0, -1), axes=p1.ndim - 1) ** xi
            )

            n, m = localkernel.shape
            K = jnp.exp(-(1 - localkernel) / alpha)

            def f(K):
                # initialisation
                u = jnp.ones((n,)) / n
                v = jnp.ones((m,)) / m

                en = jnp.ones((n,)) / float(n)
                em = jnp.ones((m,)) / float(m)

                def cond_fun(val):
                    _, _, err = val
                    return err > threshold

                def loop_body(val):
                    u_prev, v_prev, _ = val

                    v = jnp.divide(em, jnp.dot(K.T, u_prev))
                    u = jnp.divide(en, jnp.dot(K, v))

                    err = jnp.sum((u - u_prev) ** 2) / jnp.sum((u) ** 2) + jnp.sum(
                        (v - v_prev) ** 2
                    ) / jnp.sum((v) ** 2)

                    # jax.debug.print("{}", err)

                    return u, v, err

                u, v, err = jax.lax.while_loop(
                    cond_fun=cond_fun, body_fun=loop_body, init_val=(u, v, 1.0)
                )

                return u, v

            u, v = f(K)

            # using Tr(X.T Y) = Sum[ij](Xij * Yij)
            # P.T * C
            # P_ij = u_i * v_j * K_ij
            pity = jnp.multiply(jnp.multiply(K, u.reshape((-1, 1))), v)

            glosim = jnp.sum(jnp.multiply(pity, localkernel))

            return glosim

    elif matching == "best":
        raise
    else:
        raise ValueError("unknown matching procedure")

    def _g(a, b):
        return _f(a, b) / jnp.sqrt(_f(a, a) * _f(b, b))

    return _g(p1, p2)


if __name__ == "__main__":

    n = 10

    l_max = 5
    n_max = 5

    r_cut = 6

    ## sp
    rng = jax.random.PRNGKey(42)
    key, rng = jax.random.split(rng)
    pos = jax.random.uniform(key, (n, 3)) * n
    key, rng = jax.random.split(rng)
    cell = jnp.eye(3) * n + jax.random.normal(key, (3, 3)) * 0.5
    sp = SystemParams(coordinates=pos, cell=cell)

    ## sp2
    from scipy.spatial.transform import Rotation as R

    key1, key2, key3, rng = jax.random.split(rng, 4)
    rot_mat = jnp.array(
        R.random(random_state=int(jax.random.randint(key1, (), 0, 100))).as_matrix()
    )
    pos2 = (
        vmap(lambda a: rot_mat @ a, in_axes=0)(sp.coordinates)
        + jax.random.normal(key2, (3,)) * 5
    )
    cell_r = vmap(lambda a: rot_mat @ a, in_axes=0)(sp.cell)
    perm = jax.random.permutation(key3, n)

    sp2 = SystemParams(coordinates=pos2[perm], cell=cell_r)

    # raise "do permutation on p2"

    ## sp3
    key, rng = jax.random.split(rng)
    pos = jax.random.uniform(key, (n, 3)) * n
    key, rng = jax.random.split(rng)
    cell = jnp.eye(3) * n + jax.random.normal(key, (3, 3)) * 0.5
    sp3 = SystemParams(coordinates=pos, cell=cell)

    key, rng = jax.random.split(rng)

    z_array = jax.random.randint(key, (n,), 0, 5)

    sp1, nl1 = sp.get_neighbour_list(r_cut=r_cut, z_array=z_array)
    sp2, nl2 = sp2.get_neighbour_list(r_cut=r_cut, z_array=z_array[perm])
    sp3, nl3 = sp3.get_neighbour_list(r_cut=r_cut, z_array=z_array)

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

        @jit
        def f(sp, nl):

            return p_i(
                sp=sp,
                nl=nl,
                p=pp,
                r_cut=r_cut,
            )

        a = f(sp1, nl1)

        @jit
        def k(sp, nl, a):

            return Kernel(a, f(sp, nl))

        da = k(sp1, nl1, a)

        from time import time_ns

        before = time_ns()
        dab = k(sp2, nl2, a)
        after = time_ns()

        dac = k(sp3, nl3, a)

        print(
            f"REMatch l_max {l_max}  n_max {l_max} <kernel(orig,rot)>={  dab  }, <kernel(orig, rand)>= { dac }, evalutation time [ms] { (after-before)/ 10.0**6  }  "
        )

        jk = jit(jacfwd(k))

        jac_da = jk(sp1, nl1, a)

        before = time_ns()
        jac_dab = jk(sp2, nl2, a)
        after = time_ns()

        jac_dac = jk(sp3, nl3, a)

        print(
            f"REMatch l_max {l_max}  n_max {l_max}  || d kernel(orig,rot)  / d sp )={ jnp.linalg.norm(jac_dab.coordinates)  },  || d kernel(orig,rand)  / d sp )= { jnp.linalg.norm(jac_dac.coordinates)  }, evalutation time [ms] { (after-before)/ 10.0**6  }  "
        )
