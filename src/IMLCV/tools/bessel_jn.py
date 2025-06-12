# taken from https://colab.research.google.com/drive/1JBM1ZXFCR6DtKxeS5p9sS1MgdcYNXigj?usp=sharing#scrollTo=78aef876-0377-4d8f-b069-9da94d9111a6,
# see https://github.com/google/jax/pull/17038

import functools

import jax
import numpy as np
from jax import lax
from jax import numpy as jnp
from jax._src import dtypes
from jax._src.numpy.util import promote_dtypes_inexact

from IMLCV.base.datastructures import custom_jvp_decorator, jit_decorator

# polynomial coefficients for J0
PP0 = np.array(
    [
        7.96936729297347051624e-4,
        8.28352392107440799803e-2,
        1.23953371646414299388e0,
        5.44725003058768775090e0,
        8.74716500199817011941e0,
        5.30324038235394892183e0,
        9.99999999999999997821e-1,
    ]
)
PQ0 = np.array(
    [
        9.24408810558863637013e-4,
        8.56288474354474431428e-2,
        1.25352743901058953537e0,
        5.47097740330417105182e0,
        8.76190883237069594232e0,
        5.30605288235394617618e0,
        1.00000000000000000218e0,
    ]
)

QP0 = np.array(
    [
        -1.13663838898469149931e-2,
        -1.28252718670509318512e0,
        -1.95539544257735972385e1,
        -9.32060152123768231369e1,
        -1.77681167980488050595e2,
        -1.47077505154951170175e2,
        -5.14105326766599330220e1,
        -6.05014350600728481186e0,
    ]
)
QQ0 = np.array(
    [
        1.0,
        6.43178256118178023184e1,
        8.56430025976980587198e2,
        3.88240183605401609683e3,
        7.24046774195652478189e3,
        5.93072701187316984827e3,
        2.06209331660327847417e3,
        2.42005740240291393179e2,
    ]
)

YP0 = np.array(
    [
        1.55924367855235737965e4,
        -1.46639295903971606143e7,
        5.43526477051876500413e9,
        -9.82136065717911466409e11,
        8.75906394395366999549e13,
        -3.46628303384729719441e15,
        4.42733268572569800351e16,
        -1.84950800436986690637e16,
    ]
)
YQ0 = np.array(
    [
        1.04128353664259848412e3,
        6.26107330137134956842e5,
        2.68919633393814121987e8,
        8.64002487103935000337e10,
        2.02979612750105546709e13,
        3.17157752842975028269e15,
        2.50596256172653059228e17,
    ]
)

DR10 = 5.78318596294678452118e0
DR20 = 3.04712623436620863991e1

RP0 = np.array(
    [
        -4.79443220978201773821e9,
        1.95617491946556577543e12,
        -2.49248344360967716204e14,
        9.70862251047306323952e15,
    ]
)
RQ0 = np.array(
    [
        1.0,
        4.99563147152651017219e2,
        1.73785401676374683123e5,
        4.84409658339962045305e7,
        1.11855537045356834862e10,
        2.11277520115489217587e12,
        3.10518229857422583814e14,
        3.18121955943204943306e16,
        1.71086294081043136091e18,
    ]
)

# J1
RP1 = np.array(
    [
        -8.99971225705559398224e8,
        4.52228297998194034323e11,
        -7.27494245221818276015e13,
        3.68295732863852883286e15,
    ]
)
RQ1 = np.array(
    [
        1.0,
        6.20836478118054335476e2,
        2.56987256757748830383e5,
        8.35146791431949253037e7,
        2.21511595479792499675e10,
        4.74914122079991414898e12,
        7.84369607876235854894e14,
        8.95222336184627338078e16,
        5.32278620332680085395e18,
    ]
)

PP1 = np.array(
    [
        7.62125616208173112003e-4,
        7.31397056940917570436e-2,
        1.12719608129684925192e0,
        5.11207951146807644818e0,
        8.42404590141772420927e0,
        5.21451598682361504063e0,
        1.00000000000000000254e0,
    ]
)
PQ1 = np.array(
    [
        5.71323128072548699714e-4,
        6.88455908754495404082e-2,
        1.10514232634061696926e0,
        5.07386386128601488557e0,
        8.39985554327604159757e0,
        5.20982848682361821619e0,
        9.99999999999999997461e-1,
    ]
)

QP1 = np.array(
    [
        5.10862594750176621635e-2,
        4.98213872951233449420e0,
        7.58238284132545283818e1,
        3.66779609360150777800e2,
        7.10856304998926107277e2,
        5.97489612400613639965e2,
        2.11688757100572135698e2,
        2.52070205858023719784e1,
    ]
)
QQ1 = np.array(
    [
        1.0,
        7.42373277035675149943e1,
        1.05644886038262816351e3,
        4.98641058337653607651e3,
        9.56231892404756170795e3,
        7.99704160447350683650e3,
        2.82619278517639096600e3,
        3.36093607810698293419e2,
    ]
)

YP1 = np.array(
    [
        1.26320474790178026440e9,
        -6.47355876379160291031e11,
        1.14509511541823727583e14,
        -8.12770255501325109621e15,
        2.02439475713594898196e17,
        -7.78877196265950026825e17,
    ]
)
YQ1 = np.array(
    [
        5.94301592346128195359e2,
        2.35564092943068577943e5,
        7.34811944459721705660e7,
        1.87601316108706159478e10,
        3.88231277496238566008e12,
        6.20557727146953693363e14,
        6.87141087355300489866e16,
        3.97270608116560655612e18,
    ]
)

Z1 = 1.46819706421238932572e1
Z2 = 4.92184563216946036703e1
PIO4 = 0.78539816339744830962  # pi/4
THPIO4 = 2.35619449019234492885  # 3*pi/4
SQ2OPI = 0.79788456080286535588  # sqrt(2/pi)


def j1_small(x):
    """
    Implementation of J1 for x < 5
    """
    z = x * x
    w = jnp.polyval(RP1, z) / jnp.polyval(RQ1, z)
    w = w * x * (z - Z1) * (z - Z2)
    return w


def j1_large(x):
    """
    Implementation of J1 for x > 5
    """
    w = 5.0 / x
    z = w * w
    p = jnp.polyval(PP1, z) / jnp.polyval(PQ1, z)
    q = jnp.polyval(QP1, z) / jnp.polyval(QQ1, z)
    xn = x - THPIO4
    p = p * jnp.cos(xn) - w * q * jnp.sin(xn)
    return p * SQ2OPI / jnp.sqrt(x)


@jit_decorator
def j1(z):
    """
    Bessel function of the first kind of order one and a real argument
    - using the implementation from CEPHES, translated to Jax, to match scipy to machine precision.

    Reference:
    Cephes mathematical library.

    Args:
        x: The sampling point(s) at which the Bessel function of the first kind are
        computed.

    Returns:
        An array of shape `x.shape` containing the values of the Bessel function
    """

    z = jnp.asarray(z)
    (z,) = promote_dtypes_inexact(z)
    z_dtype = lax.dtype(z)

    if dtypes.issubdtype(z_dtype, complex):
        raise ValueError("complex input not supported.")

    return jnp.sign(z) * jnp.where(jnp.abs(z) < 5.0, j1_small(jnp.abs(z)), j1_large(jnp.abs(z)))


def j0_small(x):
    """
    Implementation of J0 for x < 5
    """
    z = jnp.square(x)
    p = (z - DR10) * (z - DR20)
    p = p * jnp.polyval(RP0, z) / jnp.polyval(RQ0, z)
    return jnp.where(x < 1e-5, 1 - z / 4.0, p)


def j0_large(x):
    """
    Implementation of J0 for x >= 5
    """

    w = 5.0 / x
    q = 25.0 / jnp.square(x)
    p = jnp.polyval(PP0, q) / jnp.polyval(PQ0, q)
    q = jnp.polyval(QP0, q) / jnp.polyval(QQ0, q)
    xn = x - PIO4
    p = p * jnp.cos(xn) - w * q * jnp.sin(xn)
    return p * SQ2OPI / jnp.sqrt(x)


@jit_decorator
def j0(z):
    """
    Bessel function of the first kind of order zero and a real argument
    - using the implementation from CEPHES, translated to Jax, to match scipy to machine precision.

    Reference:
    Cephes Mathematical Library.

    Args:
        z: The sampling point(s) at which the Bessel function of the first kind are
        computed.

    Returns:
        An array of shape `x.shape` containing the values of the Bessel function
    """
    z = jnp.asarray(z)
    (z,) = promote_dtypes_inexact(z)
    z_dtype = lax.dtype(z)

    if dtypes.issubdtype(z_dtype, complex):
        raise ValueError("complex input not supported.")

    return jnp.where(jnp.abs(z) < 5.0, j0_small(jnp.abs(z)), j0_large(jnp.abs(z)))


@functools.partial(jit_decorator, static_argnames=["v", "maxiter"])
def _besseljn_power_series(z, v: int, maxiter: int = 100):
    """
    Power series expansion of the Bessel function of the first kind at z << v
    """
    out = jnp.zeros_like(z)

    # Prevent numerical underflow when computing multiplier
    a = jnp.exp(v * jnp.log(z / 2) - jax.scipy.special.gammaln(v + 1))
    t2 = (z / 2) ** 2

    def body_func(i, carry):
        out, a = carry
        out += a
        a *= -t2 / ((v + i + 1) * (i + 1))
        return (out, a)

    return jax.lax.fori_loop(0, maxiter, body_func, (out, a))[0]


@functools.partial(jit_decorator, static_argnames=["v", "n_iter"])
def _besseljn_backward_recurrence(z, v: int, n_iter: int):
    """
    Implementation of Miller's backward recurrence algorithm for computing Bessel function of the first kind at x < v
    """
    f0 = jnp.zeros_like(z)
    f1 = jnp.full_like(z, 2 ** jnp.finfo(z.dtype).minexp)
    _f = jnp.zeros_like(z)
    bs = jnp.zeros_like(z)

    def body_fun(k, carry):
        """
        Body function for the for loop
        """
        # Carry niter to start sum from largest value
        f0, f1, bs, z, _f = carry
        f = 2.0 * (n_iter - k + 1.0) * f1 / z - f0

        # Compute the updates for the sum function
        bs = jnp.where(jnp.mod(n_iter - k, 2) == 0, bs + 2.0 * f, bs)
        _f = jnp.where(v == (n_iter - k), f, _f)

        f0 = f1
        f1 = f
        return (f0, f1, bs, z, _f)

    (f0, f1, bs, _, _f) = lax.fori_loop(jnp.zeros_like(n_iter), n_iter + 1, body_fun, (f0, f1, bs, z, _f))

    return _f / (bs - f1)


def besseljn_backward_recurrence_cutoff(v, z):
    """
    Helper function to determine when backward recurrence reaches overflow based on the dtype of the input
    argument
    """
    return v / jnp.exp(
        (jnp.log(jnp.finfo(z.dtype).max) - np.log(2 ** jnp.finfo(z.dtype).minexp))
        / jnp.floor(v + 2.0 * jnp.sqrt(40.0 * v))
    )


@functools.partial(jit_decorator, static_argnames=["v"])
def _besseljn_forward_recurrence(z, v: int):
    """
    This is a stripped down version of the proposed Bessel function
    """

    # use recurrence relation J_v+1(z) = 2 v/z J_v(z) - J_v-1(z) recursively
    def body(k, carry):
        jnm1, jn = carry
        jnplus = (2 * k) / z * jn - jnm1
        return (jn, jnplus)

    return jax.lax.fori_loop(1, v, body, (j0(z), j1(z)))[-1]


# Add custom jvp
@functools.partial(custom_jvp_decorator, nondiff_argnums=(0,))
@jnp.vectorize
@jit_decorator
def bessel_jn(v, z):
    """
    Bessel function uses a range of computational techniques to achieve accuracy in three
    regimes

    (1) when x << nu:
            Power series expansion is used to compute small values of x.

            Reference:
              [Equation 6.5.1]
              https://iate.oac.uncor.edu/~mario/materia/nr/numrec/f6-5.pdf

    (2) when x < nu
            Miller's algorithm is used compute values of the Bessel function for the
            first order which is stable for x < nu.

            References:
              [Miller's Algorithm; Equation 5.5.16]
              https://iate.oac.uncor.edu/~mario/materia/nr/numrec/f5-5.pdf
              W. Gautschi (1967) Computational aspects of three-term recurrence relations.
                SIAM Rev. 9 (1), pp. 24–82.
              F. W. J. Olver and D. J. Sookne (1972) Note on backward recurrence algorithms.
                Math. Comp. 26 (120), pp. 941–947.

    (3) when x >= nu
            Forward recurrence relation is used to compute values of the Bessel function for
            the first order, which is stable so long as x > nu.

            Reference:
              Shanjie Zhang and Jian-Ming Jin. Computation of special functions.
              Wiley-Interscience, 1996.


    """
    z = jnp.asarray(z)
    (z,) = promote_dtypes_inexact(z)

    if dtypes.issubdtype(lax.dtype(z), complex):
        raise ValueError("complex input not supported.")

    if not dtypes.issubdtype(lax.dtype(v), np.integer):
        raise ValueError("Order must be integer type.")

    # Compute absolute values of input x-values and orders
    zabs = jnp.abs(z)
    vabs = jnp.abs(v)

    # Used to compute the number of terms necessary for backwards recurrence to be accurate
    n_iter = jnp.maximum(vabs + jnp.sqrt(40.0 * vabs), 50).astype(int)

    # Get sign conventions for negative z and v inputs
    z_sign = jnp.where(z < 0, (-1) ** vabs, 1)
    v_sign = jnp.where(v < 0, (-1) ** vabs, 1)

    return (
        z_sign
        * v_sign
        * jnp.select(
            [
                vabs == 0,
                vabs == 1,
                zabs < besseljn_backward_recurrence_cutoff(vabs, zabs),
                zabs < vabs,
                zabs >= vabs,
            ],
            [
                j0(zabs),
                j1(zabs),
                _besseljn_power_series(zabs, v=vabs),
                _besseljn_backward_recurrence(zabs, v=vabs, n_iter=n_iter),
                _besseljn_forward_recurrence(zabs, vabs),
            ],
        )
    )


@bessel_jn.defjvp
def besseljn_jvp(v, primals, tangents):
    """ """
    (z,), (zdot,) = primals, tangents
    primal_out = bessel_jn(v, z)
    # tangent_out = v / z * primal_out - bessel_jn(v + 1, z)
    tangent_out = (bessel_jn(v - 1, z) - bessel_jn(v + 1, z)) / 2
    return primal_out, tangent_out * zdot


## edits
