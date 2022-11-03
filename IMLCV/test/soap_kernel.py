from functools import partial
from typing import Callable

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


@partial(jit, static_argnums=(0,))
def spherical_bessel_1st_kind_1(n, x):
    # http://www.fresco.org.uk/functions/barnett/APP23.pdf
    xgn, xdgn = jnp.sin(x), jnp.cos(x) - jnp.sin(x) / x

    def gn_plus_1(xgn, xdgn, n, x):
        xgn1 = n * xgn / x - xdgn
        xdgn1 = xgn - (n + 2) * xgn1 / x

        return xgn1, xdgn1

    for i in range(n):
        xgn, xdgn = gn_plus_1(xgn, xdgn, i, x)

    return xgn / x


# @partial( jit, static_argnums=(0,)  )
# def spherical_bessel_1st_kind_2(n,x):
#     f = lambda y :  jnp.sinc(y/jnp.pi)

#     def g(f,x):
#         return  grad( f  )(x) / x

#     for i in range(n):
#         f =  partial( g,f)

#     return (-x)**n * f(x)


# # file:///home/david/Downloads/Numerical_Calculation_of_Bessel_Functions.pdf
# @jit
# def bessek_Jn_integral(alpha, x,num=500):
#     # https://en.wikipedia.org/wiki/Bessel_function hansen-bessel formula
#     def f(tau):
#         return jnp.cos( alpha *tau  -x*jnp.sin(tau) )

#     def g(t):
#         return jnp.exp(  -x * jnp.sinh(t) - alpha*t )

#     x = jnp.linspace( 0,jnp.pi,num=num   )

#     x = jnp.linspace( 0,jnp.pi,num=num   )

#     return jnp.trapz(x=x,y=f(x))/ jnp.pi - jnp.sin(alpha * jnp.pi )/ jnp.pi * jnp.trapz(x=x,y=g(x))


# @jax.custom_jvp
@partial(jit, static_argnums=(2,))
def bessel_Jn(nu, z, num=500):
    # https://dlmf.nist.gov/10.9

    # re(nu)>-1/2

    def f(t):
        return jnp.sin(t) ** (2.0 * nu) * jnp.cos(z * jnp.cos(t))

    x = jnp.linspace(0, jnp.pi, num=num)
    y = (
        (0.5 * z) ** nu
        / (jnp.pi ** (0.5) * jnp.exp(jax.lax.lgamma(nu + 0.5)))
        * vmap(f)(x)
    )

    return jnp.trapz(x=x, y=y)


# @bessel_Jn.defjvp
# def bessel_Jn_jvp(primals, tangents, nondiff_argnums=(0, 2)):
#     # https://math.libretexts.org/Bookshelves/Differential_Equations/Book%3A_Partial_Differential_Equations_(Walet)/10%3A_Bessel_Functions_and_Two-Dimensional_Problems/10.05%3A_Properties_of_Bessel_functions
#     nu, z, num = primals
#     nu_dot, z_dot, num_dot = tangents
#     primal_out = bessel_Jn(nu, z, num)
#     # tangent_out = (bessej_Jn(nu - 1, z, num) - bessej_Jn(nu + 1, z, num)) * z_dot

#     if nu == 0:
#         tangent_out = -bessel_Jn(1, z, num) * z_dot
#     else:
#         tangent_out = (
#             0.5 * (bessel_Jn(nu - 1, z, num) - bessel_Jn(nu + 1, z, num)) * z_dot
#         )

#     return primal_out, tangent_out


# besssel zeros:
# https://www.cl.cam.ac.uk/~jrh13/papers/bessel.pdf


def spherical_bessel_jn(nu, z, num=500):
    return bessel_Jn(nu + 0.5, z, num=num) * jnp.sqrt(jnp.pi / (2 * z))


# @bessel_Yn.defjvp
# def bessel_Yn_jvp(primals, tangents, nondiff_argnums=(0, 2)):
#     # https://math.libretexts.org/Bookshelves/Differential_Equations/Book%3A_Partial_Differential_Equations_(Walet)/10%3A_Bessel_Functions_and_Two-Dimensional_Problems/10.05%3A_Properties_of_Bessel_functions
#     nu, z, num = primals
#     nu_dot, z_dot, num_dot = tangents
#     primal_out = bessel_Yn(nu, z, num)
#     # tangent_out = (bessej_Jn(nu - 1, z, num) - bessej_Jn(nu + 1, z, num)) * z_dot

#     if nu == 0:
#         tangent_out = -bessel_Yn(1, z, num) * z_dot
#     else:
#         tangent_out = (
#             0.5 * (bessel_Yn(nu - 1, z, num) - bessel_Yn(nu + 1, z, num)) * z_dot
#         )

#     return primal_out, tangent_out


# @jit
# def bessek_Yn_integral(n:int, x,num=500):
#     # https://en.wikipedia.org/wiki/Bessel_function hansen-bessel formula
#     def f(tau):
#         return jnp.sin( -n *tau  +x*jnp.sin(tau) )

#     x = jnp.linspace( 0,jnp.pi,num=num   )
#     return jnp.trapz(f,x=x  )  /jnp.pi


@jit
def bessek_K(v, z):
    return tfp.math.bessel_kve(v, z) * jnp.exp(-jnp.abs(z))


@jit
def bessel_I(v, z):
    return tfp.math.bessel_ive(v, z) * jnp.exp(jnp.abs(z))


@jit
def spherical_bessel_i(v, z):

    return lax.cond(
        z == 0.0,
        lambda: lax.cond(v == 0, lambda: 1.0, lambda: 0.0),
        lambda: jnp.sqrt(jnp.pi / (2 * z)) * bessel_I(v + 0.5, z),
    )


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


plotbessel = False

if plotbessel:

    x = jnp.linspace(0, 20, 1000)

    # for i in range(6):
    #     plt.plot(x, bessel_I(jnp.float32(i), x))

    # plt.ylim(0, 5)

    # fig = plt.figure()
    # x = jnp.linspace(0, 5, 100)

    # def di1(v, x):
    #     return vmap(grad(partial(bessel_I, jnp.float32(v))))(x)

    # def di2(v, x):
    #     return 0.5 * (
    #         bessel_I(jnp.float32(v) - 1.0, x) + bessel_I(jnp.float32(v) + 1.0, x)
    #     )

    # for i in range(6):
    #     plt.plot(x, di1(i, x) - di2(i, x))

    # for i in range(6):
    #     plt.plot(x, legendre(x, i))

    fig = plt.figure()
    for i in range(6):
        plt.plot(
            x, vmap(partial(spherical_bessel_jn, i))(x) - spherical_jn(i, jnp.array(x))
        )
    plt.ylim(-1, 1)

    # fig = plt.figure()
    # for i in range(6):
    #     plt.plot(x, vmap(partial(spherical_bessel_1st_kind_1, i))(x))
    # plt.ylim(-1, 1)

    # fig = plt.figure()
    # for i in range(6):
    #     plt.plot(x, vmap(grad(partial(spherical_bessel_jn, i)))(x))
    # plt.ylim(-1, 1)

    plt.show()


# @partial(jit, static_argnums=(4, 5))
def p_i(
    positions,
    p: Callable[[jnp.ndarray], jnp.ndarray],
):
    """
    positions: array with shape (n,3)
    f: callable that converts calculates contribution for single position
    """

    @vmap
    def _g_i(index):
        # swap index 0 and i, remove row 0 and substract the position of atom i
        pos_i = positions[index, :]

        pos_j = positions.copy()
        pos_j = pos_j.at[index, :].set(pos_j[0, :])
        pos_j = pos_j[1:, :]
        pos_ij = pos_j - pos_i

        pi = p(pos_ij)

        return pi / jnp.linalg.norm(pi)

    return _g_i(index=jnp.arange(positions.shape[0]))


@partial(vmap, in_axes=(None, None, 0), out_axes=2)
@partial(vmap, in_axes=(None, 0, None), out_axes=1)
@partial(vmap, in_axes=(0, None, None), out_axes=0)
def lengendre_l(l, pj, pk):
    cos_theta = jnp.dot(pj, pk) / (jnp.linalg.norm(pj) * jnp.linalg.norm(pk))
    return legendre(cos_theta, l)


def p_innl_soap(l_max, n_max, r_cut, sigma_a, r_delta, num=50):
    # for explanation soap:
    # https://aip.scitation.org/doi/suppl/10.1063/1.5111045

    def phi(n, n_max, r, r_cut, sigma_a):
        return jnp.exp(-((r - r_cut * n / n_max) ** 2) / (2 * sigma_a**2))

    @partial(vmap, in_axes=(None, None, 0), out_axes=2)
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

    # @jit
    @vmap
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

    def _p_i_soap(pos_ij):

        r_ij = jnp.linalg.norm(pos_ij, axis=1)  # relative vector to r_i

        A_nlj = U_inv_nm @ I_prime_ml(n_vec, l_vec, r_ij) * f_cut(r_ij)
        B_ljk = lengendre_l(l_vec, pos_ij, pos_ij)

        #  (n1 l),(n2 l), (l) -> (n1 n2 l)
        return jnp.einsum(
            "l,alj,blk,ljk->abl", 4 * jnp.pi * (2 * l_vec + 1), A_nlj, A_nlj, B_ljk
        )

    # @jit
    def p_innl(positions):
        return p_i(
            positions,
            _p_i_soap,
        )

    return p_innl


def Kernel(p1, p2, xi=2):
    return (
        jnp.diag(jnp.tensordot(p1, jnp.moveaxis(p2, 0, -1), axes=p1.ndim - 1))
        / (vmap(jnp.linalg.norm)(p1) * vmap(jnp.linalg.norm)(p2))
    ) ** xi


##########################
# taken from https://gist.github.com/timothydmorton/33ed23d99e2663df4004cb236f1b8ba5


# def spherical_jn_sensible_grid(n, m, ngrid=100):
#     """Returns a grid of x values that should contain the first m zeros, but not too many."""
#     return np.linspace(n, n + 1.0 * m * (np.pi * (np.log(n) + 1)), ngrid)


# show_spherical_jn_zeros(3, 30, ngrid=2000)

#################################
# https://aip-prod-cdn.literatumonline.com/journals/content/adv/2020/adv.2020.10.issue-1/1.5111045/20200109/suppl/supplementary_material.pdf?b92b4ad1b4f274c70877518712abb28b0b955e0e0198d7e04e65de17a89cd071222f0cd1d23af6ec7a5117a0f769ce93f348debdb60f0876d88df2202f6e0a0d245867d0a89d9f3b625c880aa1349d7ae571df49508fbdbd7292ca2c697561b25d1cbe2b44b4f3afc6b7af23624977642ab1829da6f42587656e4d16181287efd0614a3142481067a058c452e6c64e75bfe8


def p_inl_sb(l_max, n_max, r_cut, num=50):
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
    u_ln = jnp.array(
        [
            spherical_jn_zeros(
                n,
                l_max,
            )
            for n in range(n_max + 1)
        ]
    )

    @partial(vmap, in_axes=(None, 0, None), out_axes=0)
    # @partial(vmap, in_axes=(0, None, None), out_axes=0)
    def f_nl(n, l, r):

        a_nl = u_ln[l, n + 1] / spherical_bessel_jn(l + 1, u_ln[l, n], num=num)
        b_nl = u_ln[l, n] / spherical_bessel_jn(l + 1, u_ln[l, n + 1], num=num)
        norm_nl = jnp.sqrt(2 / (u_ln[l, n] ** 2 + u_ln[l, n + 1]) / r_cut**3)

        return (
            a_nl * spherical_bessel_jn(l, r * u_ln[l, n] / r_cut, num=num)
            + b_nl * spherical_bessel_jn(l, r * u_ln[l, n + 1] / r_cut, num=num)
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

    @vmap
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

    def _p_i_sb(pos_ij):

        r_ij = jnp.linalg.norm(pos_ij, axis=1)  # relative vector to r_i

        A_jnl = g_nl(r_ij)
        B_ljk = lengendre_l(l_vec, pos_ij, pos_ij)

        @partial(vmap, in_axes=(None, 0), out_axes=1)
        @partial(vmap, in_axes=(0, None), out_axes=0)
        def a_nml_l(n, l):
            return lax.cond(
                l <= n,
                lambda: A_jnl[:, n - l, l],
                lambda: jnp.zeros(A_jnl.shape[0]),
            )

        g_nml_l_j = a_nml_l(n_vec, l_vec)

        out = jnp.einsum(
            "l,nlj,nlk,ljk -> nl  ",
            (2 * l_vec + 1) / 4 * jnp.pi,
            g_nml_l_j,
            g_nml_l_j,
            B_ljk,
        )

        # return out[jnp.tril_indices(n=n_max + 1, m=l_max + 1)]
        return out

    @jit
    def p_innl(positions):
        return p_i(
            positions,
            _p_i_sb,
        )

    return p_innl


rng = jax.random.PRNGKey(42)
key, rng = jax.random.split(rng)
pos = jax.random.uniform(key, (30, 3)) * 5

key, key2, rng = jax.random.split(rng, 3)
pos2 = pos @ jnp.array(R.random().as_matrix())
# pos2 = jax.random.permutation(key, pos2) #random order other atoms
pos2 = pos2 + jax.random.uniform(key, (3,))


pos3 = jax.random.uniform(key, (30, 3)) * 5


l_max = 9


for pp in [
    p_inl_sb(
        l_max=l_max,
        n_max=l_max,
        r_cut=5.0,
        num=50,
    ),
    p_innl_soap(
        l_max=l_max,
        n_max=l_max,
        r_cut=5.0,
        sigma_a=1.0,
        r_delta=1.0,
        num=50,
    ),
]:

    a = pp(pos)

    from time import time_ns

    before = time_ns()
    b = pp(pos2)
    after = time_ns()
    c = pp(pos3)

    print(
        f"l_max {l_max}  n_max {l_max} <kernel(orig,rot)>={ jnp.mean( Kernel( a,b ) ) }, <kernel(orig, rand)>= { jnp.mean( Kernel( a,c ) ) }, evalutation time [ms] { (after-before)/ 10.0**6  }  "
    )

    # check the derivatives
    c = jacfwd(pp)(pos)
    print(f"gradient shape:{c.shape} nans: {jnp.sum(  jnp.isnan(c ) )}")
