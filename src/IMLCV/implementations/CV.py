import dataclasses
from functools import partial

import distrax
import jax
import jax.numpy as jnp
import numba
import numpy as np
from flax.linen.linear import Dense
from IMLCV.base.CV import CollectiveVariable
from IMLCV.base.CV import CV
from IMLCV.base.CV import CvFlow
from IMLCV.base.CV import CvFunDistrax
from IMLCV.base.CV import CvFunInput
from IMLCV.base.CV import CvFunNn
from IMLCV.base.CV import CvMetric
from IMLCV.base.CV import CvTrans
from IMLCV.base.CV import NeighbourList
from IMLCV.base.CV import SystemParams
from jax import Array
from jax import vmap

######################################
#       CV transformations           #
######################################


@CvFlow.from_function
def Volume(sp: SystemParams, _):
    assert sp.cell is not None, "can only calculate volume if there is a unit cell"

    vol = jnp.abs(jnp.dot(sp.cell[0], jnp.cross(sp.cell[1], sp.cell[2])))
    return jnp.array([vol])


def distance_descriptor():
    @CvFlow.from_function
    def h(x: SystemParams, _):
        x = x.canoncialize()[0]

        n = x.shape[-2]

        out = vmap(vmap(x.min_distance, in_axes=(0, None)), in_axes=(None, 0))(
            jnp.arange(n),
            jnp.arange(n),
        )

        return out[jnp.triu_indices_from(out, k=1)]

        # return out

    return h


def dihedral(numbers: list[int] | Array):
    """from https://stackoverflow.com/questions/20305272/dihedral-torsion-
    angle-from-four-points-in-cartesian- coordinates-in-python.

    args:
        numbers: list with index of 4 atoms that form dihedral
    """

    @CvFlow.from_function
    def f(sp: SystemParams, _):
        coor = sp.coordinates
        p0 = coor[numbers[0]]
        p1 = coor[numbers[1]]
        p2 = coor[numbers[2]]
        p3 = coor[numbers[3]]

        b0 = -1.0 * (p1 - p0)
        b1 = p2 - p1
        b2 = p3 - p2

        b1 /= jnp.linalg.norm(b1)

        v = b0 - jnp.dot(b0, b1) * b1
        w = b2 - jnp.dot(b2, b1) * b1

        x = jnp.dot(v, w)
        y = jnp.dot(jnp.cross(b1, v), w)
        return jnp.arctan2(y, x)

    return f


def sb_descriptor(
    r_cut,
    n_max: int,
    l_max: int,
    reduce=True,
    reshape=True,
) -> CvFlow:
    from IMLCV.tools.soap_kernel import p_i, p_inl_sb

    def _reduce_sb(a):
        def _tril(a):
            return a[jnp.tril_indices_from(a)]

        def _triu(a):
            return a[jnp.triu_indices_from(a)]

        a = vmap(
            vmap(
                vmap(_tril, in_axes=(0), out_axes=(0)),
                in_axes=(1),
                out_axes=(1),
            ),
            in_axes=(2),
            out_axes=(2),
        )(
            a,
        )  # eliminate l>n
        a = vmap(
            vmap(_triu, in_axes=(0), out_axes=(0)),
            in_axes=(3),
            out_axes=(2),
        )(
            a,
        )  # eliminate Z2>Z1
        return a

    p = p_inl_sb(r_cut=r_cut, n_max=n_max, l_max=l_max)

    def f(sp: SystemParams, nl: NeighbourList):
        assert nl is not None, "provide neighbourlist for sb describport"

        a = p_i(
            sp=sp,
            nl=nl,
            p=p,
            r_cut=r_cut,
        )

        if reduce:
            a = _reduce_sb(a)

        if reshape:
            a = jnp.reshape(a, (a.shape[0], -1))

        return a

    return CvFlow.from_function(f, atomic=True)  # type: ignore


def soap_descriptor(
    r_cut,
    n_max: int,
    l_max: int,
    sigma_a: float,
    r_delta: float,
    reduce=True,
    reshape=True,
    num=50,
) -> CvFlow:
    from IMLCV.tools.soap_kernel import p_i, p_innl_soap

    def _reduce_sb(a):
        def _triu(a):
            return a[jnp.triu_indices_from(a)]

        a = vmap(
            vmap(_triu, in_axes=(0), out_axes=(0)),
            in_axes=(3),
            out_axes=(2),
        )(
            a,
        )  # eliminate Z2>Z1
        return a

    p = p_innl_soap(r_cut=r_cut, n_max=n_max, l_max=l_max, sigma_a=sigma_a, r_delta=r_delta, num=num)

    def f(sp: SystemParams, nl: NeighbourList):
        assert nl is not None, "provide neighbourlist for soap describport"

        a = p_i(
            sp=sp,
            nl=nl,
            p=p,
            r_cut=r_cut,
        )

        if reduce:
            a = _reduce_sb(a)

        if reshape:
            a = jnp.reshape(a, (a.shape[0], -1))

        return a

    return CvFlow.from_function(f, atomic=True)  # type: ignore


def NoneCV() -> CollectiveVariable:
    return CollectiveVariable(
        f=CvFlow.from_function(lambda sp, nl: jnp.array([0.0])),
        metric=CvMetric(periodicities=[None]),
    )


######################################
#           CV trans                 #
######################################


def rotate_2d(alpha):
    @CvTrans.from_cv_function
    def f(cv: CV, *_):
        return (
            jnp.array(
                [[jnp.cos(alpha), jnp.sin(alpha)], [-jnp.sin(alpha), jnp.cos(alpha)]],
            )
            @ cv
        )

    return f


def project_distances(a):
    @CvTrans.from_cv_function
    def project_distances(cvs: CV, *_):
        "projects the distances to a reaction coordinate"
        import jax.numpy as jnp

        from IMLCV.base.CV import CV

        assert cvs.dim == 2

        r1 = cvs.cv[0]
        r2 = cvs.cv[1]

        x = (r2**2 - r1**2) / (2 * a)
        y2 = r2**2 - (a / 2 + x) ** 2

        y2_safe = jnp.where(y2 <= 0, jnp.ones_like(y2), y2)
        y = jnp.where(y2 <= 0, 0.0, y2_safe ** (1 / 2))

        return CV(cv=jnp.array([x / a + 0.5, y / a]))

    return project_distances


def scale_cv_trans(array: CV):
    "axis 0 is batch axis"
    maxi = jnp.max(array.cv, axis=0)
    mini = jnp.min(array.cv, axis=0)
    diff = (maxi - mini) / 2
    diff = jnp.where(diff == 0, 1, diff)

    # mask = jnp.abs(diff) > 1e-6

    @CvTrans.from_array_function
    def f0(x, *_):
        return (x - (mini + maxi) / 2) / diff

    return f0


def trunc_svd(m: CV) -> tuple[CV, CvTrans]:
    assert m.batched

    if m.atomic:
        cvi = m.cv.reshape((m.cv.shape[0], -1))
    else:
        cvi = m.cv

    u, s, v = jnp.linalg.svd(cvi, full_matrices=False)

    include_mask = s > 10 * jnp.max(
        jnp.array([u.shape[-2], v.shape[-1]]),
    ) * jnp.finfo(
        u.dtype,
    ).eps * jnp.max(s)

    def _f(u, s, v, include_mask):
        u, s, v = u[:, include_mask], s[include_mask], v[include_mask, :]
        return (u, s, v)

    _, _, v = _f(u, s, v, include_mask)

    if m.atomic:
        v = v.reshape((v.shape[0], m.cv.shape[1], m.cv.shape[2]))

    @CvTrans.from_cv_function
    def f(x: CV, nl: NeighbourList | None, _):
        if m.atomic:

            def _f(x, v):
                return jnp.einsum("ni,jni->j", x, v)

        else:

            def _f(x, v):
                return jnp.einsum("i,ji->j", x, v)

        return CV(cv=_f(x.cv, v), _stack_dims=x._stack_dims, _combine_dims=x._combine_dims, atomic=False)

    return f.compute_cv_trans(m)[0], f


def get_sinkhorn_divergence(
    nli: NeighbourList | None,
    pi: Array | CV | None,
    sort="rematch",
    alpha_rematch=0.1,
    output="tensor",
):
    if isinstance(pi, CV):
        assert pi.atomic, "pi must be atomic"
        pi = pi.cv

    def __norm(p):
        n1_sq = jnp.einsum("...,...->", p, p)
        n1_sq_safe = jnp.where(n1_sq <= 1e-16, 1, n1_sq)
        n1_i = jnp.where(n1_sq == 0, 0.0, 1 / jnp.sqrt(n1_sq_safe))

        return p * n1_i

    pi = __norm(pi)

    def sinkhorn_divergence(
        cv: CV,
        nl: NeighbourList | None,
        _,
        nli: NeighbourList | None,
        pi: Array | None,
        sort="rematch",
        alpha_rematch=0.1,
    ):
        assert nl is not None, "Neigbourlist required for rematch"

        if pi is None:
            pi = cv.cv
        if nli is None:
            nli = nl

        _, (P11, P12, P22) = NeighbourList.match_kernel(
            cv.cv,
            pi,
            nl,
            nli,
            matching=sort,
            alpha=alpha_rematch,
        )

        cvn = __norm(cv.cv)

        if output == "tensor":
            sig = "ij,i...,j...->j..."
        elif output == "vector":
            sig = "ij,i...,j...->j"
        elif output == "scalar":
            sig = "ij,i...,j...->"
        else:
            raise ValueError(f"output must be vector or scalar, not {output}")

        out = jnp.einsum(sig, P11, cvn, cvn) + jnp.einsum(sig, P22, pi, pi) - 2 * jnp.einsum(sig, P12, cvn, pi)

        return CV(
            cv=out,
            _stack_dims=cv._stack_dims,
            _combine_dims=cv._combine_dims,
            atomic=output != "scalar",
        )

    return CvTrans.from_cv_function(
        partial(sinkhorn_divergence, pi=pi, nli=nli, sort=sort, alpha_rematch=alpha_rematch),
    )


@CvTrans.from_cv_function
def un_atomize(x: CV, nl, _):
    if x.atomic:
        x = CV(
            cv=jnp.reshape(x.cv, (-1,)),
            atomic=False,
            _combine_dims=x._combine_dims,
            _stack_dims=x._stack_dims,
        )
    return x


def stack_reduce(op=jnp.mean):
    # take average over both alignments
    @CvTrans.from_cv_function
    def _stack_reduce(cv: CV, nl: NeighbourList | None, _):
        cvs = cv.split(cv.stack_dims)

        return CV(
            op(jnp.stack([cvi.cv for cvi in cvs]), axis=0),
            _stack_dims=cvs[0]._stack_dims,
            _combine_dims=cvs[0]._combine_dims,
            atomic=cvs[0].atomic,
            mapped=cvs[0].mapped,
        )

    return _stack_reduce


######################################
#           CV Fun                   #
######################################


class RealNVP(CvFunNn):
    """use in combination with swaplink"""

    _: dataclasses.KW_ONLY
    features: int
    cv_input: CvFunInput

    def setup(self) -> None:
        self.s = Dense(features=self.features)
        self.t = Dense(features=self.features)

    def forward(self, x: CV, nl: NeighbourList | None, conditioners: list[CV] | None = None):
        y = CV.combine(*conditioners).cv
        return CV(cv=x.cv * self.s(y) + self.t(y))

    def backward(self, z: CV, nl: NeighbourList | None, conditioners: list[CV] | None = None):
        y = CV.combine(*conditioners).cv
        return CV(cv=(z.cv - self.t(y)) / self.s(y))


class DistraxRealNVP(CvFunDistrax):
    _: dataclasses.KW_ONLY
    latent_dim: int

    def setup(self):
        """Creates the flow model."""

        try:
            from tensorflow_probability.substrates import jax as tfp
        except ImportError:
            raise ImportError("isntall tensorflow-probability")

        self.s = Dense(features=self.latent_dim)
        self.t = Dense(features=self.latent_dim)

        # Alternating binary mask.
        self.bijector = distrax.as_bijector(
            tfp.bijectors.RealNVP(
                fraction_masked=0.5,
                shift_and_log_scale_fn=self.shift_and_scale,
            ),
        )

    def shift_and_scale(self, x0, input_depth, **condition_kwargs):
        return self.s(x0), self.t(x0)


######################################
#           Test                     #
######################################


class MetricUMAP(CvMetric):
    def __init__(self, periodicities, bounding_box=None) -> None:
        super().__init__(periodicities=periodicities, bounding_box=bounding_box)

        # bb = np.array(self.bounding_box)
        per = np.array(self.periodicities)

        # @numba.njit
        # def map(y):

        #     return (y - bb[:, 0]) / (
        #         bb[:, 1] - bb[:, 0])

        @numba.njit
        def _periodic_wrap(xs, min=False):
            coor = np.mod(xs, 1)  # between 0 and 1
            if min:
                # between [-0.5,0.5]
                coor = np.where(coor > 0.5, coor - 1, coor)

            return np.where(per, coor, xs)

        @numba.njit
        def g(x, y):
            # r1 = map(x)
            # r2 = map(y)

            return _periodic_wrap(x - y, min=True)

        @numba.njit
        def val_and_grad(x, y):
            r = g(x, y)
            d = np.sqrt(np.sum(r**2))

            return d, r / (d + 1e-6)

        self.umap_f = val_and_grad


class hyperTorus(CvMetric):
    def __init__(self, n) -> None:
        periodicities = [True for _ in range(n)]
        boundaries = jnp.zeros((n, 2))
        boundaries = boundaries.at[:, 0].set(-jnp.pi)
        boundaries = boundaries.at[:, 1].set(jnp.pi)

        super().__init__(periodicities, boundaries)
