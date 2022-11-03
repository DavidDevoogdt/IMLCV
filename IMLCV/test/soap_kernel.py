from functools import partial

import jax.numpy as jnp
import jax.numpy.linalg
import jax.random
import jax.scipy
import jax.dtypes
import matplotlib.pyplot as plt
from jax import jacfwd, jit, lax, vmap,grad
import jax.lax
from scipy.spatial.transform import Rotation as R
from tensorflow_probability.substrates import jax as tfp


# @partial( jit, static_argnums=(1,)  )
# def spherical_bessel_1st_kind_1(n,x):
#     #http://www.fresco.org.uk/functions/barnett/APP23.pdf
#     xgn,xdgn =  jnp.sin(x) ,   jnp.cos(x) - jnp.sin(x)/x 

#     def gn_plus_1(xgn,xdgn,n,x):    
#         xgn1 =  n * xgn/ x  - xdgn 
#         xdgn1 =  xgn - (n+2) * xgn1/x

#         return xgn1,xdgn1

#     for i in range(n):
#         xgn,xdgn =  gn_plus_1(xgn,xdgn,i,x)

#     return xgn/x

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





def bessek_Jn_integral(nu, z,num=500):
    # https://dlmf.nist.gov/10.9
    def f(t):
        return (1-t**2)**(nu-0.5) * jnp.cos( z*t  )  


    x = jnp.linspace( 0,1,num=num   ) 
    y = 2* (0.5*x)**nu / ( jnp.pi**(0.5)*  jnp.exp( jax.lax.lgamma( nu +0.5 ) )  )   *   vmap(f)(x)

    return jnp.trapz(x=x,y=  y )






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


plotbessel = True

if plotbessel:

    fig = plt.figure()
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


    for i in range(6):
        plt.plot(x, vmap(  partial(bessek_Jn_integral,i))(x))

    plt.ylim(-1, 1)

    plt.show()


# for explanation soap:
# https://aip-prod-cdn.literatumonline.com/journals/content/adv/2020/adv.2020.10.issue-1/1.5111045/20200109/suppl/supplementary_material.pdf?b92b4ad1b4f274c70877518712abb28b0b955e0e0198d7e04e65de17a89cd071222f0cd1d23af6ec7a5117a0f769ce93f348debdb60f0876d88df2202f6e0a0d245867d0a89d9f3b625c880aa1349d7ae571df49508fbdbd7292ca2c697561b25d1cbe2b44b4f3afc6b7af23624977642ab1829da6f42587656e4d16181287efd0614a3142481067a058c453ecc24e724a3d


def phi(n, n_max, r, r_cut, sigma_a):
    return jnp.exp(-((r - r_cut * n / n_max) ** 2) / (2 * sigma_a**2))


def I_prime_nl(n, l, sigma_a, r_j, r_cut, n_max, num=50):
    def f(r):
        # https://mathworld.wolfram.com/ModifiedSphericalBesselFunctionoftheFirstKind.html
        return (
            r
            * jnp.sqrt(sigma_a**2 * jnp.pi * r / (2 * r_j))
            * tfp.math.bessel_ive(l + 0.5, r * r_j / sigma_a**2)
            * phi(n, n_max, r, r_cut, sigma_a)
            * jnp.exp(-((r - r_j) ** 2) / (2 * sigma_a**2))
        )

    x = jnp.linspace(0, r_cut, num=num)
    y = vmap(f)(x)

    return jnp.trapz(y=y, x=x)


@jit
def f_cut(r, r_cut, r_delta):
    return lax.cond(
        r > r_cut,
        lambda: 0.0,
        lambda: lax.cond(
            r < r_cut - r_delta,
            lambda: 1.0,
            lambda: 0.5 * (1 + jnp.cos(jnp.pi * (r - r_cut + r_delta) / r_delta)),
        ),
    )


# @partial(jit, static_argnums=(0, 3))
def U(n_max, r_cut, sigma_a, num=2000):
    def f(ind):
        def g(r):
            return (
                phi(ind[0], n_max, r, r_cut, sigma_a)
                * phi(ind[1], n_max, r, r_cut, sigma_a)
                * r**2
            )

        x = jnp.linspace(0, r_cut, num=num)
        y = g(x)

        return jnp.trapz(y=y, x=x)

    indices = jnp.array(jnp.meshgrid(jnp.arange(n_max + 1), jnp.arange(n_max + 1)))

    S = jnp.apply_along_axis(f, axis=0, arr=indices)

    L, V = jnp.linalg.eigh(S)
    L = L.at[L < 0].set(0)

    U = jnp.diag(jnp.sqrt(L)) @ V.T

    # print(U.T @ U - S)

    # jax.dtypes.finfo(U.dtype).eps

    # U = jnp.linalg.cholesky(S)

    return U
    # return jnp.linalg.inv(U)


# @partial(jit, static_argnums=(4, 5))
def I_prime_l(l, sigma_a, r_j, r_cut, n_max, num=50):
    def f(n):
        return I_prime_nl(n, l, sigma_a, r_j, r_cut, n_max, num=num)

    x = jnp.arange(0, n_max + 1)
    return vmap(f)(x)


# @partial(jit, static_argnums=(1, 2, 6))
def p(positions, l_max, n_max, r_cut, sigma_a, r_delta, num=50):
    @partial(vmap, in_axes=(None, 0), out_axes=1)
    @partial(vmap, in_axes=(0, None), out_axes=0)
    def A(l, r_j):
        # return jnp.linalg.solve(
        #     a=U(n_max, r_cut, sigma_a, num=50),
        #     b=I_prime_l(l, sigma_a, r_j, r_cut, n_max, num=50),
        # ) * f_cut(r_j, r_cut=r_cut, r_delta=r_delta)

        return (
            jnp.linalg.pinv(U(n_max, r_cut, sigma_a, num=num))
            @ I_prime_l(l, sigma_a, r_j, r_cut, n_max, num=num)
            * f_cut(r_j, r_cut=r_cut, r_delta=r_delta)
        )

    @partial(vmap, in_axes=(None, None, 0), out_axes=2)
    @partial(vmap, in_axes=(None, 0, None), out_axes=1)
    @partial(vmap, in_axes=(0, None, None), out_axes=0)
    def B(l, pj, pk):
        cos_theta = jnp.dot(pj, pk) / (jnp.linalg.norm(pj) * jnp.linalg.norm(pk))
        return legendre(cos_theta, l)

    s = positions.shape[0]

    @vmap
    def p_nnl(index):
        # swap index 0 and i, remove row 0 and substract the position of atom i
        pos_i = positions[index, :]

        pos_j = positions.copy()
        pos_j = pos_j.at[index, :].set(pos_j[0, :])
        pos_j = pos_j[1:, :]
        pos_j = pos_j - pos_i

        r_j = jnp.linalg.norm(pos_j, axis=1)  # relative vector to r_i
        l_vec = jnp.arange(l_max + 1, dtype=jnp.int32)
        A_ljn = A(l_vec, r_j)
        B_ljk = B(l_vec, pos_j, pos_j)

        p = jnp.einsum(
            "l,ljn,lkm,ljk->nml", 4 * jnp.pi * (2 * l_vec + 1), A_ljn, A_ljn, B_ljk
        )

        return p

    i_vec = jnp.arange(s)

    return p_nnl(i_vec)


def K_soap(p1, p2, xi=2):
    return (
        jnp.einsum("ijkl,ijkl->i", a, b)
        / (vmap(jnp.linalg.norm)(a) * vmap(jnp.linalg.norm)(b))
    ) ** xi


rng = jax.random.PRNGKey(42)

U(9, 5.0, 1.0, 300)

# for l_max in range(14):
if True:
    l_max = 9

    key, rng = jax.random.split(rng)
    pos = jax.random.uniform(key, (30, 3)) * 5

    def pp(pos):
        return p(
            positions=pos,
            l_max=l_max,
            n_max=l_max,
            r_cut=5.0,
            sigma_a=1.0,
            r_delta=1.0,
            num=50,
        )

    # check for invariance under rotations, permutations and translations

    a = pp(pos)
    key, key2, rng = jax.random.split(rng, 3)

    pos2 = pos @ jnp.array(R.random().as_matrix())
    # pos2 = jax.random.permutation(key, pos2) #random order other atoms
    pos2 = pos2 + jax.random.uniform(key2, (3,))

    b = pp(pos2)

    print(f"l_max {l_max}  n_max {l_max} <K_soap> = {  K_soap( a,b ) } ")


# mask = abs(a - b) > 1e-6
# a[mask]-b[]

# check the derivatives
c = jacfwd(pp)(pos)
print(c.shape)
