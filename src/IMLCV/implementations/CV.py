import dataclasses
from functools import partial

import distrax
import jax
import jax.numpy as jnp
import lineax as lx
import numba
import numpy as np
import ott
from flax.linen.linear import Dense
from IMLCV.base.CV import chunk_map
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
from jax import jit
from jax import vmap
from ott.geometry.pointcloud import PointCloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import implicit_differentiation
from ott.solvers.linear import lineax_implicit
from ott.solvers.linear import sinkhorn

######################################
#       CV transformations           #
######################################


def _identity_trans(x, nl, _):
    return x


identity_trans = CvFlow.from_function(_identity_trans)


def _Volume(sp: SystemParams, *_):
    assert sp.cell is not None, "can only calculate volume if there is a unit cell"

    vol = jnp.abs(jnp.linalg.det(sp.cell))
    return CV(cv=jnp.array([vol]))


Volume = CvFlow.from_function(_Volume)


def _distance(x: SystemParams, *_):
    x = x.canonicalize()[0]

    n = x.shape[-2]

    out = vmap(vmap(x.min_distance, in_axes=(0, None)), in_axes=(None, 0))(
        jnp.arange(n),
        jnp.arange(n),
    )

    return CV(cv=out[jnp.triu_indices_from(out, k=1)])


def distance_descriptor():
    return CvFlow.from_function(_distance)


def _dihedral(sp: SystemParams, _nl, _c, numbers):
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
    return CV(cv=jnp.arctan2(y, x))


def dihedral(numbers: list[int] | Array):
    """from https://stackoverflow.com/questions/20305272/dihedral-torsion-
    angle-from-four-points-in-cartesian- coordinates-in-python.

    args:
        numbers: list with index of 4 atoms that form dihedral
    """

    return CvFlow.from_function(_dihedral, numbers=numbers)


def _sb_descriptor(
    sp: SystemParams,
    nl: NeighbourList,
    _,
    r_cut,
    chunk_size_atoms,
    chunk_size_neigbourgs,
    reduce,
    reshape,
    n_max,
    l_max,
):
    assert nl is not None, "provide neighbourlist for sb describport"

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

    p = p_inl_sb(
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
    )

    a = p_i(
        sp=sp,
        nl=nl,
        p=p,
        r_cut=r_cut,
        chunk_size_atoms=chunk_size_atoms,
        chunk_size_neigbourgs=chunk_size_neigbourgs,
    )

    if reduce:
        a = _reduce_sb(a)

    if reshape:
        a = jnp.reshape(a, (a.shape[0], -1))

    return CV(cv=a, atomic=True)


def sb_descriptor(
    r_cut,
    n_max: int,
    l_max: int,
    reduce=True,
    reshape=True,
    chunk_size_atoms=None,
    chunk_size_neigbourgs=None,
) -> CvFlow:
    return CvFlow.from_function(
        _sb_descriptor,
        r_cut=r_cut,
        chunk_size_atoms=chunk_size_atoms,
        chunk_size_neigbourgs=chunk_size_neigbourgs,
        reduce=reduce,
        reshape=reshape,
        n_max=n_max,
        l_max=l_max,
    )


def _soap_descriptor(
    sp: SystemParams,
    nl: NeighbourList,
    _,
    r_cut,
    reduce,
    reshape,
    n_max,
    l_max,
    sigma_a,
    r_delta,
    num,
):
    assert nl is not None, "provide neighbourlist for soap describport"

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

    p = p_innl_soap(
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        sigma_a=sigma_a,
        r_delta=r_delta,
        num=num,
    )

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

    return CV(cv=a, atomic=True)


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
    return CvFlow.from_function(
        _soap_descriptor,
        r_cut=r_cut,
        reduce=reduce,
        reshape=reshape,
        n_max=n_max,
        l_max=l_max,
        sigma_a=sigma_a,
        r_delta=r_delta,
        num=num,
    )


def NoneCV() -> CollectiveVariable:
    return CollectiveVariable(
        f=CvFlow.from_function(lambda sp, nl, c: CV(cv=jnp.array([0.0]))),
        metric=CvMetric.create(periodicities=[None]),
    )


######################################
#           CV trans                 #
######################################
def _rotate_2d(cv: CV, _nl: NeighbourList, _, alpha):
    return (
        jnp.array(
            [[jnp.cos(alpha), jnp.sin(alpha)], [-jnp.sin(alpha), jnp.cos(alpha)]],
        )
        @ cv
    )


def rotate_2d(alpha):
    return CvTrans.from_cv_function(_rotate_2d, alpha=alpha)


def _project_distances(cvs: CV, nl, _, a):
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


def project_distances(a):
    return CvTrans.from_cv_function(_project_distances, a=a)


def _scale_cv_trans(x, nl, _, upper, lower, mini, diff):
    return x.replace(cv=((x.cv - mini) / diff) * (upper - lower) + lower)


def scale_cv_trans(array: CV, lower=0, upper=1):
    "axis 0 is batch axis"
    maxi = jnp.max(array.cv, axis=0)
    mini = jnp.min(array.cv, axis=0)

    diff = maxi - mini
    diff = jnp.where(diff == 0, 1, diff)

    return CvTrans.from_cv_function(_scale_cv_trans, upper=upper, lower=lower, mini=mini, diff=diff)


def _trunc_svd(x: CV, nl: NeighbourList | None, _, m_atomic, v, cvi_shape):
    if m_atomic:
        out = jnp.einsum("ni,jni->j", x.cv, v)

    else:
        out = jnp.einsum("i,ji->j", x.cv, v)

    out = out * jnp.sqrt(cvi_shape)

    return CV(
        cv=out,
        _stack_dims=x._stack_dims,
        _combine_dims=x._combine_dims,
        atomic=False,
    )


def trunc_svd(m: CV, range=Ellipsis) -> tuple[CV, CvTrans]:
    assert m.batched

    if m.atomic:
        cvi = m.cv.reshape((m.cv.shape[0], -1))
    else:
        cvi = m.cv

    cvi = cvi[range, :]

    cvi_shape = cvi.shape[0]
    m_atomic = m.atomic

    u, s, v = jnp.linalg.svd(cvi, full_matrices=False)

    include_mask = s > 10 * jnp.max(
        jnp.array([u.shape[-2], v.shape[-1]]),
    ) * jnp.finfo(
        u.dtype,
    ).eps * jnp.max(s)

    def _f(u, s, v, include_mask):
        u, s, v = u[:, include_mask], s[include_mask], v[include_mask, :]

        out = v

        return s, out

    s, v = _f(u, s, v, include_mask)

    if m.atomic:
        v = v.reshape((v.shape[0], m.cv.shape[1], m.cv.shape[2]))

    out = CvTrans.from_cv_function(_trunc_svd, m_atomic=m_atomic, v=v, cvi_shape=cvi_shape)

    return out.compute_cv_trans(m)[0], out


@partial(jit, static_argnames=["matching", "alpha", "normalize", "chunk_size"])
def sinkhorn_divergence(
    x1: CV,
    x2: CV,
    nl1: NeighbourList,
    nl2: NeighbourList,
    matching="REMatch",
    alpha=1e-2,
    normalize=True,
    chunk_size=None,
) -> tuple[CV, CV]:
    """caluculates the sinkhorn divergence between two CVs. If x2 is batched, the resulting divergences are stacked"""

    assert x1.atomic
    assert x2.atomic

    eps = 100 * jnp.finfo(x1.cv.dtype).eps

    def p_norm(x: CV):
        p = x.cv

        p_sq = jnp.einsum("i...,i...->i", p, p)
        p_sq_safe = jnp.where(p_sq <= eps, 1, p_sq)
        p_norm_inv = jnp.where(p_sq == 0, 0.0, 1 / jnp.sqrt(p_sq_safe))

        return CV(
            cv=jnp.einsum("i...,i->i...", p, p_norm_inv),
            _stack_dims=x._stack_dims,
            _combine_dims=x._combine_dims,
            atomic=x.atomic,
        )

    if normalize:
        x1_norm = p_norm
        x2_norm = p_norm

        if x1.batched:
            x1_norm = vmap(x1_norm)

        if x2.batched:
            x2_norm = vmap(x2_norm)

        x1 = x1_norm(x1)
        x2 = x2_norm(x2)

    def get_b_p(x1: CV, nl1: NeighbourList):
        return x1.cv, jnp.array(nl1.nl_split_z(())[0])

    @jax.jit
    def get_P(x1: CV, nl1: NeighbourList, x2: CV | None = None, nl2: NeighbourList | None = None):
        p1, b1 = get_b_p(x1, nl1)

        if x2 is None:
            p2, b2 = p1, b1
        else:
            p2, b2 = get_b_p(x2, nl2)

        if matching == "average":
            P12 = jnp.einsum("ni,nj->ij", b1, b2)

        elif matching == "norm":
            raise

        else:

            def __gs_mask(p1, p2, m1, m2):
                n = p1.shape[0]

                geom = PointCloud(
                    x=jnp.reshape(p1, (n, -1)),
                    y=jnp.reshape(p2, (n, -1)),
                    # scale_cost=False,  # leads to problems if x==y
                    epsilon=alpha,
                )

                geom = geom.mask(m1, m2)

                prob = linear_problem.LinearProblem(geom)

                # make svd class that accepts rtol and atol params
                class _SVD(lx.SVD):
                    def __init__(self, rtol, atol, rcond=None):
                        super().__init__(rcond=rcond)

                # precondition_fun=lambda x: x,
                # symmetric=True,

                solver = sinkhorn.Sinkhorn(
                    implicit_diff=implicit_differentiation.ImplicitDiff(
                        solver_kwargs={
                            "nonsym_solver": _SVD,
                        },
                    ),
                    use_danskin=True,
                )
                out = solver(prob)

                P_ij = out.matrix

                # Tr(  P.T * C )
                return P_ij

            P12 = jnp.einsum(
                "nij,ni,nj->ij",
                vmap(__gs_mask, in_axes=(None, None, 0, 0))(p1, p2, b1, b2),
                b1,
                b2,
            )

        return P12

    get_P1 = get_P
    get_P2 = get_P
    get_P12 = get_P

    if nl1.batched:
        get_P1 = chunk_map(
            vmap(get_P1, in_axes=(0, 0, None, None)),
            chunk_size=chunk_size,
        )
        get_P12 = chunk_map(
            vmap(get_P12, in_axes=(0, 0, None, None)),
            chunk_size=chunk_size,
        )

    if nl2.batched:
        get_P2 = chunk_map(
            vmap(get_P2, in_axes=(0, 0, None, None)),
            chunk_size=chunk_size,
        )
        get_P12 = chunk_map(
            vmap(get_P12, in_axes=(None, None, 0, 0)),
            chunk_size=chunk_size,
        )

    P11 = get_P1(x1, nl1, None, None)
    P22 = get_P2(x2, nl2, None, None)
    P12 = get_P12(x1, nl1, x2, nl2)

    def combine(x1, x2, P11, P12, P22):
        p1 = x1.cv
        p2 = x2.cv

        # todo: P_11 p1 p1 term not aligned -> nonseniscal?
        cv = CV(
            cv=(
                -2 * jnp.einsum("i...,j...,ij->j...", p1, p2, P12)
                + jnp.einsum("i...,j...,ij->j...", p1, p1, P11)
                + jnp.einsum("i...,j...,ij->j...", p2, p2, P22)
            ),
            _stack_dims=x1._stack_dims,
            _combine_dims=x1._combine_dims,
            atomic=x1.atomic,
        )

        # avoid non differentiable code
        p2_sq = jnp.einsum("i...,i...->i...", p2, p2)
        p2_safe = jnp.where(p2_sq <= eps, 1, p2)
        p2_inv = jnp.where(p2_sq <= eps, 0.0, 1 / p2_safe)

        cv_2 = CV(
            cv=(
                -2 * jnp.einsum("i...,ij->j...", p1, P12)
                + jnp.einsum("i...,j...,j...,ij", p1, p1, p2_inv, P11)
                + jnp.einsum("i...,ij->j...", p2, P22)
            ),
            _stack_dims=x1._stack_dims,
            _combine_dims=x1._combine_dims,
            atomic=x1.atomic,
        )

        cv_3 = CV(
            cv=(-2 * jnp.einsum("i...,ij->j...", p1, P12) + 2 * jnp.einsum("i...,ij->j...", p2, P22)),
            _stack_dims=x1._stack_dims,
            _combine_dims=x1._combine_dims,
            atomic=x1.atomic,
        )

        div = CV(
            cv=(
                jnp.mean(
                    -2 * jnp.einsum("i...,j...,ij->j", p1, p2, P12)
                    + jnp.einsum("i...,j...,ij->j", p1, p1, P11)
                    + jnp.einsum("i...,j...,ij->j", p2, p2, P22),
                )
            ),
            _stack_dims=x1._stack_dims,
            _combine_dims=x1._combine_dims,
            atomic=False,
        )

        return div, cv, cv_2, cv_3

    if nl1.batched:
        combine = vmap(combine, in_axes=(0, None, 0, 0, None))

    if nl2.batched:
        combine = vmap(combine, in_axes=(None, 0, None, 0, 0))

    return combine(x1, x2, P11, P12, P22)


def _sinkhorn_divergence_trans(
    cv: CV,
    nl: NeighbourList | None,
    _,
    nli,
    pi,
    sort,
    alpha_rematch,
    output,
    normalize,
):
    assert nl is not None, "Neigbourlist required for rematch"

    divergence, out, matched, ddiv = sinkhorn_divergence(
        x1=cv,
        x2=pi,
        nl1=nl,
        nl2=nli,
        matching=sort,
        alpha=alpha_rematch,
        normalize=normalize,
    )

    if output == "divergence":
        return divergence

    if output == "aligned_cv":
        return CV.combine(
            *[
                CV(
                    cv=cvi,
                    _stack_dims=out._stack_dims,
                    _combine_dims=out._combine_dims,
                    atomic=out.atomic,
                )
                for cvi in out.cv
            ],
        )

    if output == "matched":
        return CV.combine(
            *[
                CV(
                    cv=cvi,
                    _stack_dims=out._stack_dims,
                    _combine_dims=out._combine_dims,
                    atomic=matched.atomic,
                )
                for cvi in matched.cv
            ],
        )

    if output == "ddiv":
        return CV.combine(
            *[
                CV(
                    cv=cvi,
                    _stack_dims=ddiv._stack_dims,
                    _combine_dims=ddiv._combine_dims,
                    atomic=ddiv.atomic,
                )
                for cvi in ddiv.cv
            ],
        )

    raise ValueError(f"{output=}")


def get_sinkhorn_divergence(
    nli: NeighbourList | None,
    pi: CV,
    sort="rematch",
    alpha_rematch=0.1,
    output="divergence",
    normalize=True,
) -> CvTrans:
    """Get a function that computes the sinkhorn divergence between two point clouds. p_i and nli are the points to match against."""

    assert pi.atomic, "pi must be atomic"

    return CvTrans.from_cv_function(
        _sinkhorn_divergence_trans,
        nli=nli,
        pi=pi,
        sort=sort,
        alpha_rematch=alpha_rematch,
        output=output,
        normalize=normalize,
    )


@partial(jit, static_argnames=["alpha", "normalize", "sum_divergence"])
def sinkhorn_divergence_2(
    x1: CV,
    x2: CV,
    nl1: NeighbourList,
    nl2: NeighbourList,
    alpha=1e-2,
    normalize=True,
    sum_divergence=False,
) -> Array:
    """caluculates the sinkhorn divergence between two CVs. If x2 is batched, the resulting divergences are stacked"""

    assert x1.atomic
    assert x2.atomic
    assert not x1.batched
    assert not x2.batched

    eps = 100 * jnp.finfo(x1.cv.dtype).eps

    def p_norm(x: CV):
        p = x.cv

        p_sq = jnp.einsum("i...,i...->i", p, p)
        p_sq_safe = jnp.where(p_sq <= eps, 1, p_sq)
        p_norm_inv = jnp.where(p_sq == 0, 0.0, 1 / jnp.sqrt(p_sq_safe))

        return CV(
            cv=jnp.einsum("i...,i->i...", p, p_norm_inv),
            _stack_dims=x._stack_dims,
            _combine_dims=x._combine_dims,
            atomic=x.atomic,
        )

    if normalize:
        x1 = p_norm(x1)
        x2 = p_norm(x2)

    from ott.tools import sinkhorn_divergence

    b1, _, p1 = nl1.nl_split_z(x1)
    _, _, p2 = nl2.nl_split_z(x2)

    def get_divergence(p1, p2):
        n = p1.shape[0]

        class _SVD(lx.SVD):
            def __init__(self, rtol, atol, rcond=None):
                super().__init__(rcond=rcond)

        return sinkhorn_divergence.sinkhorn_divergence(
            PointCloud,
            x=jnp.reshape(p1, (n, -1)),
            y=jnp.reshape(p2, (n, -1)),
            epsilon=alpha,
            sinkhorn_kwargs={
                "implicit_diff": implicit_differentiation.ImplicitDiff(
                    solver_kwargs={
                        "nonsym_solver": _SVD,
                    },
                ),
                "use_danskin": True,
            },
        ).divergence

    divergences = jnp.array([get_divergence(p1_z.cv, p2_z.cv) for p1_z, p2_z in zip(p1, p2)])

    if sum_divergence:
        # weigh according to number of atoms
        w = jnp.sum(b1, axis=1)
        w /= jnp.sum(w)
        divergences = divergences * w

    return divergences


def _sinkhorn_divergence_trans_2(
    cv: CV,
    nl: NeighbourList | None,
    _,
    nli,
    pi,
    alpha_rematch,
    output,
    normalize,
    sum_divergence,
    ddiv_arg=0,
):
    assert nl is not None, "Neigbourlist required for rematch"

    def f(pii, cv, nlii, nl):
        return CV(
            cv=sinkhorn_divergence_2(
                x1=cv,
                x2=pii,
                nl1=nl,
                nl2=nlii,
                alpha=alpha_rematch,
                normalize=normalize,
                sum_divergence=sum_divergence,
            ),
            _stack_dims=cv._stack_dims,
        )

    if output == "ddiv":
        f = jax.jacrev(f, argnums=ddiv_arg)
        div = CV.combine(*[f(pii, cv, nlii, nl).cv for pii, nlii in zip(pi, nli)])
    else:
        div = CV.combine(*[f(pii, cv, nlii, nl) for pii, nlii in zip(pi, nli)])

    return div.replace(cv=jnp.reshape(div.cv, (-1)), atomic=False)


def get_sinkhorn_divergence_2(
    nli: NeighbourList | None,
    pi: CV,
    alpha_rematch=0.1,
    output="divergence",
    normalize=True,
    sum_divergence=False,
    ddiv_arg=0,
) -> CvTrans:
    """Get a function that computes the sinkhorn divergence between two point clouds. p_i and nli are the points to match against."""

    assert pi.atomic, "pi must be atomic"

    return CvTrans.from_cv_function(
        _sinkhorn_divergence_trans_2,
        nli=nli,
        pi=pi,
        alpha_rematch=alpha_rematch,
        output=output,
        normalize=normalize,
        sum_divergence=sum_divergence,
        ddiv_arg=ddiv_arg,
    )


def _divergence_from_aligned_cv(
    cv: CV,
    nl: NeighbourList | None,
    _,
):
    splitted = jnp.array([a.cv for a in cv.split()])
    div = jnp.einsum("k...->k", splitted) / splitted.shape[0]

    return CV(cv=div)


divergence_from_aligned_cv = CvTrans.from_cv_function(f=_divergence_from_aligned_cv)


def _divergence_weighed_aligned_cv(
    cv: CV,
    nl: NeighbourList | None,
    _,
    scaling,
):
    divergence, _ = divergence_from_aligned_cv.compute_cv_trans(cv, nl)

    def soft_min(x):
        x = x / scaling
        a = jnp.exp(-x)
        return a / jnp.sum(a)

    weights = soft_min(divergence.cv)
    cvs = cv.split()

    return CV.combine(*[a * b for a, b in zip(cvs, weights)])


def divergence_weighed_aligned_cv(scaling):
    return CvTrans.from_cv_function(_divergence_weighed_aligned_cv, scaling=scaling)


def _un_atomize(x: CV, nl, _):
    if not x.atomic:
        return x

    return x.replace(
        cv=jnp.reshape(x.cv, (-1,)),
        atomic=False,
    )


un_atomize = CvTrans.from_cv_function(_un_atomize)


def _stack_reduce(cv: CV, nl: NeighbourList | None, _, op):
    cvs = cv.split(cv.stack_dims)

    return CV(
        op(jnp.stack([cvi.cv for cvi in cvs]), axis=0),
        _stack_dims=cvs[0]._stack_dims,
        _combine_dims=cvs[0]._combine_dims,
        atomic=cvs[0].atomic,
        mapped=cvs[0].mapped,
    )


def stack_reduce(op=jnp.mean):
    return CvTrans.from_cv_function(_stack_reduce, op=op)


def _affine_trans(x: CV, nl, _, C):
    assert x.dim == 2

    u = (C[0] * x.cv[0] + C[1] * x.cv[1] + C[2]) / (C[6] * x.cv[0] + C[7] * x.cv[1] + 1)
    v = (C[3] * x.cv[0] + C[4] * x.cv[1] + C[5]) / (C[6] * x.cv[0] + C[7] * x.cv[1] + 1)
    return CV(
        cv=jnp.array([u, v]),
        mapped=x.mapped,
        _combine_dims=x._combine_dims,
        _stack_dims=x._stack_dims,
        atomic=x.atomic,
    )


def affine_2d(old: Array, new: Array):
    """old: set of coordinates in the original space, new: set of coordinates in the new space"""

    # source:https://github.com/opencv/opencv/blob/11b020b9f9e111bddd40bffe3b1759aa02d966f0/modules/imgproc/src/imgwarp.cpp#L3001

    # /* Calculates coefficients of perspective transformation
    #  * which maps (xi,yi) to (ui,vi), (i=1,2,3,4):
    #  *
    #  *      c00*xi + c01*yi + c02
    #  * ui = ---------------------
    #  *      c20*xi + c21*yi + c22
    #  *
    #  *      c10*xi + c11*yi + c12
    #  * vi = ---------------------
    #  *      c20*xi + c21*yi + c22
    #  *
    #  * Coefficients are calculated by solving linear system:
    #  * / x0 y0  1  0  0  0 -x0*u0 -y0*u0 \ /c00\ /u0\
    #  * | x1 y1  1  0  0  0 -x1*u1 -y1*u1 | |c01| |u1|
    #  * | x2 y2  1  0  0  0 -x2*u2 -y2*u2 | |c02| |u2|
    #  * | x3 y3  1  0  0  0 -x3*u3 -y3*u3 |.|c10|=|u3|,
    #  * |  0  0  0 x0 y0  1 -x0*v0 -y0*v0 | |c11| |v0|
    #  * |  0  0  0 x1 y1  1 -x1*v1 -y1*v1 | |c12| |v1|
    #  * |  0  0  0 x2 y2  1 -x2*v2 -y2*v2 | |c20| |v2|
    #  * \  0  0  0 x3 y3  1 -x3*v3 -y3*v3 / \c21/ \v3/
    #  *
    #  * where:
    #  *   cij - matrix coefficients, c22 = 1
    #  */

    A = jnp.array(
        [
            [old[0, 0], old[0, 1], 1, 0, 0, 0, -old[0, 0] * new[0, 0], -old[0, 1] * new[0, 0]],
            [old[1, 0], old[1, 1], 1, 0, 0, 0, -old[1, 0] * new[1, 0], -old[1, 1] * new[1, 0]],
            [old[2, 0], old[2, 1], 1, 0, 0, 0, -old[2, 0] * new[2, 0], -old[2, 1] * new[2, 0]],
            [old[3, 0], old[3, 1], 1, 0, 0, 0, -old[3, 0] * new[3, 0], -old[3, 1] * new[3, 0]],
            [0, 0, 0, old[0, 0], old[0, 1], 1, -old[0, 0] * new[0, 1], -old[0, 1] * new[0, 1]],
            [0, 0, 0, old[1, 0], old[1, 1], 1, -old[1, 0] * new[1, 1], -old[1, 1] * new[1, 1]],
            [0, 0, 0, old[2, 0], old[2, 1], 1, -old[2, 0] * new[2, 1], -old[2, 1] * new[2, 1]],
            [0, 0, 0, old[3, 0], old[3, 1], 1, -old[3, 0] * new[3, 1], -old[3, 1] * new[3, 1]],
        ],
    )

    X = jnp.array([new[0, 0], new[1, 0], new[2, 0], new[3, 0], new[0, 1], new[1, 1], new[2, 1], new[3, 1]])

    C = jnp.linalg.solve(A, X)

    return CvTrans.from_cv_function(_affine_trans, C=C)


def _remove_mean(cv: CV, nl: NeighbourList | None, _, mean):
    return cv - mean


def get_remove_mean_trans(c: CV, range=Ellipsis):
    assert c.batched
    mean = jnp.mean(c.cv[range, :], axis=0)

    trans = CvTrans.from_cv_function(_remove_mean, mean=mean)

    return trans.compute_cv_trans(c)[0], trans


def _normalize(cv: CV, nl: NeighbourList | None, _, std):
    return cv * (1 / std)


def get_normalize_trans(c: CV, range=Ellipsis):
    assert c.batched
    std = jnp.std(c.cv[range, :], axis=0)

    trans = CvTrans.from_cv_function(_normalize, std=std)

    return trans.compute_cv_trans(c)[0], trans


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

    def forward(
        self,
        x: CV,
        nl: NeighbourList | None,
        conditioners: list[CV] | None = None,
    ):
        y = CV.combine(*conditioners).cv
        return CV(cv=x.cv * self.s(y) + self.t(y))

    def backward(
        self,
        z: CV,
        nl: NeighbourList | None,
        conditioners: list[CV] | None = None,
    ):
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
