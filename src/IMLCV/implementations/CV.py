import dataclasses

import distrax
import jax
import jax.numpy as jnp
import lineax as lx
import numba
import numpy as np
from equinox import Partial
from flax.linen.linear import Dense
from IMLCV.base.CV import _CvTrans, CvFlow
from IMLCV.base.CV import padded_vmap
from IMLCV.base.CV import CollectiveVariable
from IMLCV.base.CV import CV
from IMLCV.base.CV import CvFunDistrax
from IMLCV.base.CV import CvFunInput
from IMLCV.base.CV import CvFunNn
from IMLCV.base.CV import CvMetric
from IMLCV.base.CV import CvTrans
from IMLCV.base.CV import NeighbourList, NeighbourListInfo
from IMLCV.base.CV import SystemParams

from jax import Array
from jax import vmap
from ott.geometry.pointcloud import PointCloud
from ott.geometry.geometry import Geometry
from ott.problems.linear import linear_problem
from ott.solvers.linear import implicit_differentiation
from ott.solvers.linear import sinkhorn, solve
from functools import partial

######################################
#       CV transformations           #
######################################


def _identity_trans(x, nl, _, shmap):
    return x


def _zero_cv(x, nl, _, shmap):
    return CV(cv=jnp.array([0.0]))


identity_trans = _CvTrans.from_cv_function(_identity_trans)
zero_trans = _CvTrans.from_cv_function(_zero_cv)
zero_flow = CvFlow.from_function(_zero_cv)


def _Volume(sp: SystemParams, _nl, _c, shmap):
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


def _dihedral(sp: SystemParams, _nl, _c, shmap, numbers):
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


def dihedral(numbers: tuple[int] | Array):
    """from https://stackoverflow.com/questions/20305272/dihedral-torsion-
    angle-from-four-points-in-cartesian- coordinates-in-python.

    args:
        numbers: list with index of 4 atoms that form dihedral
    """

    return CvFlow.from_function(_dihedral, static_argnames=["numbers"], numbers=numbers)


def _sb_descriptor(
    sp: SystemParams,
    nl: NeighbourList,
    _,
    shmap,
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
        shmap=shmap,
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
        static_argnames=[
            "r_cut",
            "chunk_size_atoms",
            "chunk_size_neigbourgs",
            "reduce",
            "reshape",
            "n_max",
            "l_max",
        ],
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
    shmap,
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
        shmap=shmap,
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
        static_argnames=[
            "r_cut",
            "reduce",
            "reshape",
            "n_max",
            "l_max",
            "sigma_a",
            "r_delta",
            "num",
        ],
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
        f=zero_flow,
        metric=CvMetric.create(periodicities=[None]),
    )


######################################
#           CV trans                 #
######################################
def _rotate_2d(cv: CV, _nl: NeighbourList, _, shmap, alpha):
    return (
        jnp.array(
            [[jnp.cos(alpha), jnp.sin(alpha)], [-jnp.sin(alpha), jnp.cos(alpha)]],
        )
        @ cv
    )


def rotate_2d(alpha):
    return CvTrans.from_cv_function(_rotate_2d, alpha=alpha)


def _project_distances(cvs: CV, nl, _, shmap, a):
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


def _scale_cv_trans(x, nl, _, shmap, upper, lower, mini, diff):
    return x.replace(cv=((x.cv - mini) / diff) * (upper - lower) + lower)


def scale_cv_trans(array: CV, lower=0, upper=1):
    "axis 0 is batch axis"
    maxi = jnp.nanmax(array.cv, axis=0)
    mini = jnp.nanmin(array.cv, axis=0)

    diff = maxi - mini
    diff = jnp.where(diff == 0, 1, diff)

    return CvTrans.from_cv_function(_scale_cv_trans, upper=upper, lower=lower, mini=mini, diff=diff)


def _trunc_svd(x: CV, nl: NeighbourList | None, _, shmap, m_atomic, v, cvi_shape):
    if m_atomic:
        out = jnp.einsum("ni,jni->j", x.cv, v)

    else:
        out = jnp.einsum("i,ji->j", x.cv, v)

    out = out * jnp.sqrt(cvi_shape)

    return x.replace(cv=out, atomic=False)


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

    out = CvTrans.from_cv_function(
        _trunc_svd,
        static_argnames=["m_atomic", "cvi_shape"],
        m_atomic=m_atomic,
        v=v,
        cvi_shape=cvi_shape,
    )

    return out.compute_cv_trans(m)[0], out


# @partial(jit, static_argnames=["matching", "alpha", "normalize", "chunk_size"])
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
    def get_P(
        x1: CV,
        nl1: NeighbourList,
        x2: CV | None = None,
        nl2: NeighbourList | None = None,
    ):
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
        # raise
        get_P1 = padded_vmap(get_P1, in_axes=(0, 0, None, None), chunk_size=chunk_size)
        get_P12 = padded_vmap(get_P12, in_axes=(0, 0, None, None), chunk_size=chunk_size)

    if nl2.batched:
        get_P2 = padded_vmap(get_P2, in_axes=(0, 0, None, None), chunk_size=chunk_size)
        get_P12 = padded_vmap(get_P12, in_axes=(None, None, 0, 0), chunk_size=chunk_size)

    P11 = get_P1(x1, nl1, None, None)
    P22 = get_P2(x2, nl2, None, None)
    P12 = get_P12(x1, nl1, x2, nl2)

    def combine(x1, x2, P11, P12, P22):
        p1 = x1.cv
        p2 = x2.cv

        # todo: P_11 p1 p1 term not aligned -> nonseniscal?
        cv = x1.replace(
            cv=(
                -2 * jnp.einsum("i...,j...,ij->j...", p1, p2, P12)
                + jnp.einsum("i...,j...,ij->j...", p1, p1, P11)
                + jnp.einsum("i...,j...,ij->j...", p2, p2, P22)
            ),
        )

        # avoid non differentiable code
        p2_sq = jnp.einsum("i...,i...->i...", p2, p2)
        p2_safe = jnp.where(p2_sq <= eps, 1, p2)
        p2_inv = jnp.where(p2_sq <= eps, 0.0, 1 / p2_safe)

        cv_2 = x1.replace(
            cv=(
                -2 * jnp.einsum("i...,ij->j...", p1, P12)
                + jnp.einsum("i...,j...,j...,ij", p1, p1, p2_inv, P11)
                + jnp.einsum("i...,ij->j...", p2, P22)
            ),
        )

        cv_3 = x1.replace(
            cv=(-2 * jnp.einsum("i...,ij->j...", p1, P12) + 2 * jnp.einsum("i...,ij->j...", p2, P22)),
        )

        div = x1.replace(
            cv=(
                jnp.mean(
                    -2 * jnp.einsum("i...,j...,ij->j", p1, p2, P12)
                    + jnp.einsum("i...,j...,ij->j", p1, p1, P11)
                    + jnp.einsum("i...,j...,ij->j", p2, p2, P22),
                )
            ),
            atomic=False,
        )

        return div, cv, cv_2, cv_3

    if nl1.batched:
        combine = vmap(combine, in_axes=(0, None, 0, 0, None))

    if nl2.batched:
        combine = vmap(combine, in_axes=(None, 0, None, 0, 0))

    return combine(x1, x2, P11, P12, P22)


# @partial(jax.jit,static_argnames=["sort","alpha_rematch","output","normalize"])
def _sinkhorn_divergence_trans(
    cv: CV,
    nl: NeighbourList | None,
    _,
    shmap,
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
        static_argnames=["sort", "alpha_rematch", "output", "normalize"],
        nli=nli,
        pi=pi,
        sort=sort,
        alpha_rematch=alpha_rematch,
        output=output,
        normalize=normalize,
    )


# @partial(jit, static_argnames=["alpha", "normalize", "sum_divergence", "ridge", "sinkhorn_iterations"])
def sinkhorn_divergence_2(
    x1: CV,
    x2: CV,
    nl1: NeighbourListInfo,
    nl2: NeighbourListInfo,
    alpha=1e-2,
    normalize=True,
    sum_divergence=False,
    ridge=0,
    sinkhorn_iterations=None,
    use_dankskin=False,
    r_cond_svd=1e-10,
    method=None,
    op=None,
    scale=None,
    push_div=False,
    nan_zero=True,
) -> Array:
    """caluculates the sinkhorn divergence between two CVs. If x2 is batched, the resulting divergences are stacked"""

    assert x1.atomic
    assert x2.atomic

    assert not x1.batched
    assert not x2.batched

    eps = 10 * jnp.finfo(x1.cv.dtype).eps

    def p_norm(x: CV):
        p = x.cv

        p_sq = jnp.einsum("i...,i...->i", p, p)
        p_sq_safe = jnp.where(p_sq <= eps, 1, p_sq)
        p_norm_inv = jnp.where(p_sq == 0, 0.0, 1 / jnp.sqrt(p_sq_safe))

        return x.replace(cv=jnp.einsum("i...,i->i...", p, p_norm_inv))

    if scale is None:
        if normalize:
            x1 = p_norm(x1)
            x2 = p_norm(x2)

    else:
        x1 = x1.replace(cv=x1.cv * scale)
        x2 = x2.replace(cv=x2.cv * scale)

    src_mask, _, p1 = nl1.nl_split_z(x1)
    tgt_mask, _, p2 = nl2.nl_split_z(x2)

    def solve_lineax_svd(
        lin,
        b: Array,
        lin_t=None,
        symmetric: bool = False,
        ridge_identity: float = 0.0,
        ridge_kernel: float = 0.0,
        r_cond_svd: float = 1e-10,
        method="jvp",
    ) -> Array:
        def lin_reg(x, symmetric, lin, lin_t, ridge_kernel, ridge_identity):
            op = lin if symmetric else lambda x: lin_t(lin(x))

            if ridge_kernel == 0 and ridge_identity == 0:
                return op(x)

            return op(x) + ridge_kernel * jnp.sum(x) + ridge_identity * x

        fun = Partial(
            lin_reg,
            symmetric=symmetric,
            lin=lin,
            lin_t=lin_t,
            ridge_kernel=ridge_kernel,
            ridge_identity=ridge_identity,
        )

        fun, aux_args = jax.closure_convert(fun, b)

        def _solve(fun, r_cond_svd, aux_args, b):
            input_structure = jax.eval_shape(lambda: b)

            solver = lx.SVD(rcond=r_cond_svd)

            def _f(x):
                return fun(x, *aux_args)

            fn_operator = lx.FunctionLinearOperator(
                _f,
                input_structure,
                tags=[lx.positive_semidefinite_tag, lx.symmetric_tag],
            )

            return lx.linear_solve(fn_operator, b, solver).value

        # methods to differentiate through the solver
        # They're not needed as the the derivatives are taken as jacrev(solve) jacfwd(jacrev(solve)) below, but are needed for 3rd order derivatives
        if method == "jvp":
            _solve = jax.custom_jvp(_solve, nondiff_argnums=(0, 1))

            @_solve.defjvp
            def _solve_jvp(fun, r_cond_svd, primals, tangents):
                (aux_args, b) = primals
                (aux_args_dot, b_dot) = tangents

                y = _solve(fun, r_cond_svd, aux_args, b)

                solver = lx.SVD(rcond=r_cond_svd)

                def _f(x, _):
                    return fun(x, *aux_args)

                jac_operator = lx.JacobianLinearOperator(_f, y)

                dy = lx.linear_solve(jac_operator, b_dot, solver).value

                return y, dy

        elif method == "vjp":
            _solve = jax.custom_vjp(_solve, nondiff_argnums=(0, 1))

            def _solve_fwd(fun, r_cond_svd, aux_args, b):
                y = _solve(fun, r_cond_svd, aux_args, b)
                return y, (aux_args, y)

            def _solve_rev(fun, r_cond_svd, args, b_dot):
                aux_args, y = args
                solver = lx.SVD(rcond=r_cond_svd)

                def _f(x, _):
                    return fun(x, *aux_args)

                jac_operator = lx.JacobianLinearOperator(_f, y)
                y_dot = lx.linear_solve(jac_operator, b_dot, solver).value

                return (y_dot, None)

            _solve.defvjp(_solve_fwd, _solve_rev)

        return _solve(fun, r_cond_svd, aux_args, b)

    sinkhorn_kwargs = {
        "threshold": 1e-4,
        "implicit_diff": implicit_differentiation.ImplicitDiff(
            solver=solve_lineax_svd,  # solve_jax_cg, solve_lineax_svd,
            solver_kwargs={
                "ridge_identity": ridge,
                "ridge_kernel": ridge,
                "r_cond_svd": r_cond_svd,
                "method": method,
            },
        ),
        "use_danskin": use_dankskin,
    }

    if sinkhorn_iterations is not None:
        sinkhorn_kwargs["min_iterations"] = sinkhorn_iterations
        sinkhorn_kwargs["max_iterations"] = sinkhorn_iterations

    def get_divergence(p1, p2):
        cost_xy = PointCloud(
            x=jnp.reshape(p1, (p1.shape[0], -1)),
            y=jnp.reshape(p2, (p2.shape[0], -1)),
            epsilon=alpha,
        ).cost_matrix

        cost_xx = PointCloud(
            x=jnp.reshape(p1, (p1.shape[0], -1)),
            y=jnp.reshape(p1, (p1.shape[0], -1)),
            epsilon=alpha,
        ).cost_matrix

        cost_yy = PointCloud(
            x=jnp.reshape(p2, (p2.shape[0], -1)),
            y=jnp.reshape(p2, (p2.shape[0], -1)),
            epsilon=alpha,
        ).cost_matrix

        num_a, num_b = p1.shape[0], p2.shape[0]

        a = jnp.ones(num_a) / num_a
        b = jnp.ones(num_b) / num_b

        # raw sinkhorn optimal transport cost
        def _f(cost, a, b):
            geom = Geometry(cost, epsilon=alpha)
            return solve(geom, a, b, **sinkhorn_kwargs).reg_ot_cost

        @partial(jax.custom_jvp, nondiff_argnums=(0,))
        def f(_f, cost, a, b):
            return _f(cost, a, b)

        @partial(jax.custom_jvp, nondiff_argnums=(0,))
        def df(_f, cost, a, b):
            return jax.jacrev(_f, argnums=0)(cost, a, b)

        def ddf(_f, cost, a, b):
            # jacfwd(jacrev(_f)) needed to avoid predefined  number of sinkhorn iterations
            return jax.hessian(_f)(cost, a, b)

        @f.defjvp
        def f_jvp(_f, primals, tangents):
            (cost, a, b) = primals
            (cost_dot, _, _) = tangents

            y = f(_f, cost, a, b)
            dy = df(_f, cost, a, b)

            return y, jnp.einsum("ij,ij->", dy, cost_dot)

        @df.defjvp
        def df_jvp(f, primals, tangents):
            (cost, a, b) = primals
            (cost_dot, _, _) = tangents

            dy = df(_f, cost, a, b)
            ddy = ddf(_f, cost, a, b)

            return dy, jnp.einsum("ijkl,kl->ij", ddy, cost_dot)

        if op == "jacrev":
            fun = partial(df, _f)

        else:
            fun = partial(f, _f)

        out_xy = fun(cost_xy, a, b)
        out_xx = fun(cost_xx, a, a)
        out_yy = fun(cost_yy, b, b)

        div = out_xy - 0.5 * (out_xx + out_yy) + 0.5 * alpha * (jnp.sum(a) - jnp.sum(b)) ** 2

        return div

    if push_div:
        get_divergence = jax.jacrev(get_divergence, argnums=(1))

        divergences = []

        for p1_i, p2_i in zip(p1, p2):
            div_z = get_divergence(p1_i.cv, p2_i.cv)

            cv_z = p1_i.replace(cv=jnp.reshape(div_z, (-1,)), atomic=False)

            divergences.append(cv_z)
        comb = CV.combine(*divergences)

        return comb.replace(_stack_dims=x1._stack_dims)

    # this is not vmappable because it has different lengths. The src_mask and tgt_mask in ott jax geometry do not produce the desired results
    divergences = jnp.array([get_divergence(p1_i.cv, p2_i.cv) for p1_i, p2_i in zip(p1, p2)])

    if sum_divergence:
        raise NotImplementedError
        # weigh according to number of atoms
        w = jnp.sum(src_mask, axis=1)
        w /= jnp.sum(w)
        divergences = jnp.dot(divergences, w)

    return divergences


def _sinkhorn_divergence_trans_2(
    cv: CV,
    nl: NeighbourList | None,
    _,
    shmap,
    nli: NeighbourList | NeighbourListInfo,
    pi: CV,
    alpha_rematch,
    output,
    normalize,
    sum_divergence,
    ddiv_arg=0,
    ridge=0,
    sinkhorn_iterations=None,
    r_cond_svd=1e-10,
    method=None,
    op=None,
    scale=None,
    push_div=False,
):
    assert nl is not None, "Neigbourlist required for rematch"

    if isinstance(nli, NeighbourList):
        print("converting nli to info")
        nli = nli.info

    def f(pii, cv):
        return sinkhorn_divergence_2(
            x1=cv,
            x2=pii,
            nl1=nl.info,
            nl2=nli,
            alpha=alpha_rematch,
            normalize=normalize,
            sum_divergence=sum_divergence,
            ridge=ridge,
            sinkhorn_iterations=sinkhorn_iterations,
            use_dankskin=False,
            r_cond_svd=r_cond_svd,
            method=method,
            op=op,
            scale=scale,
            push_div=push_div,
        )

    def get_div(pi, cv):
        if pi.batched:
            cv_arr = vmap(f, in_axes=(0, None))(pi, cv)
        else:
            cv_arr = f(pi, cv)

        if pi.batched:
            _combine_dims = [cv_arr.shape[1]] * cv_arr.shape[0]
            cv_arr = jnp.hstack(cv_arr)

        else:
            _combine_dims = None

        _stack_dims = cv._stack_dims

        return CV(
            cv=cv_arr,
            _stack_dims=_stack_dims,
            _combine_dims=_combine_dims,
            atomic=False,
        )

    if push_div:
        if pi.batched:
            f = vmap(f, in_axes=(0, None))

            raise NotImplementedError("TODO: unstack and combine along batched axis")

        return f(pi, cv)

    if output == "ddiv":
        div = jax.jacrev(get_div, argnums=ddiv_arg)(pi, cv).cv

    elif output == "both":
        assert not sum_divergence, "not implemented"

        # taken from https://github.com/google/jax/pull/762
        def value_and_jacrev(g, x):
            y, pullback = jax.vjp(g, x)
            basis = y.replace(cv=jnp.eye(y.cv.size, dtype=y.cv.dtype))
            (jac,) = jax.vmap(pullback)(basis)
            return y, jac

        if ddiv_arg == 0:
            g = Partial(get_div, cv=cv)
            x = pi
        else:
            g = Partial(get_div, pi=pi)
            x = cv

        div, jac = value_and_jacrev(g, x)

        # d_size = int(div.shape[0])
        x_shape = int(x.cv.size / x.shape[0])
        n_z = x.shape[0]

        jac = jac.replace(
            cv=jac.cv.reshape(-1),
            _combine_dims=[[[x_shape] * n_z] * a for a in div._combine_dims] if div._combine_dims is not None else None,
            atomic=div.atomic,
            _stack_dims=div._stack_dims,
        )  # flatten according to ddiv combine dims

        div = CV.combine(div, jac)

    else:
        div = get_div(pi, cv)

    return div.replace(cv=jnp.reshape(div.cv, (-1)), atomic=False)


def get_sinkhorn_divergence_2(
    nli: NeighbourListInfo,
    pi: CV,
    alpha_rematch=0.1,
    output="divergence",
    normalize=False,
    sum_divergence=False,
    ddiv_arg=0,
    ridge=0,
    sinkhorn_iterations=None,
    merge=False,
    r_cond_svd=1e-10,
    method=None,
    op=None,
    scale=None,
    push_div=False,
) -> CvTrans:
    """Get a function that computes the sinkhorn divergence between two point clouds. p_i and nli are the points to match against."""

    assert pi.atomic, "pi must be atomic"

    if isinstance(nli, NeighbourList):
        print("converting nli to info")
        nli = nli.info

    if push_div:
        assert output == "jacrev"
        output = "divergence"

    return CvTrans.from_cv_function(
        _sinkhorn_divergence_trans_2,
        jacfun=jax.jacrev,
        static_argnames=[
            "alpha_rematch",
            "output",
            "normalize",
            "sum_divergence",
            "ddiv_arg",
            "ridge",
            "sinkhorn_iterations",
            "r_cond_svd",
            "method",
            "op",
            "push_div",
        ],
        nli=nli,
        pi=pi,
        alpha_rematch=alpha_rematch,
        output=output,
        normalize=normalize,
        sum_divergence=sum_divergence,
        ddiv_arg=ddiv_arg,
        ridge=ridge,
        sinkhorn_iterations=sinkhorn_iterations,
        r_cond_svd=r_cond_svd,
        method=method,
        op=op,
        scale=scale,
        push_div=push_div,
    )


def _divergence_from_aligned_cv(
    cv: CV,
    nl: NeighbourList | None,
    _,
    shmap,
):
    splitted = jnp.array([a.cv for a in cv.split()])
    div = jnp.einsum("k...->k", splitted) / splitted.shape[0]

    return CV(cv=div)


divergence_from_aligned_cv = CvTrans.from_cv_function(f=_divergence_from_aligned_cv)


def _divergence_weighed_aligned_cv(
    cv: CV,
    nl: NeighbourList | None,
    _,
    shmap,
    scaling,
):
    divergence, _, _ = divergence_from_aligned_cv.compute_cv_trans(cv, nl)

    def soft_min(x):
        x = x / scaling
        a = jnp.exp(-x)
        return a / jnp.sum(a)

    weights = soft_min(divergence.cv)
    cvs = cv.split()

    return CV.combine(*[a * b for a, b in zip(cvs, weights)])


def divergence_weighed_aligned_cv(scaling):
    return CvTrans.from_cv_function(_divergence_weighed_aligned_cv, scaling=scaling)


def _weighted_sinkhorn_divergence_2(
    cv: CV,
    nl: NeighbourList | None,
    _,
    shmap,
    scaling,
    mean=None,
    append_weights=True,
    sum_x=True,
):
    div, ddiv = cv.split()

    def soft_min(x):
        if sum_x:
            x = jnp.sum(x, axis=-1)

        a = jnp.exp(-x)
        return a / jnp.sum(a)

    if mean is not None:
        div = div - mean
    weights = soft_min(jnp.array([d.cv for d in (div * scaling).split()]))

    if sum_x:
        out = CV.combine(*[a * b for a, b in zip(ddiv.split(), weights)])
    else:
        out = CV.combine(
            *[CV.combine(*[ai * bi for ai, bi in zip(a.split(), b)]) for a, b in zip(ddiv.split(), weights)],
        )

    if append_weights:
        out = CV.combine(out, div.replace(cv=weights.reshape(-1)))

    return out


def weighted_sinkhorn_divergence_2(
    scaling,
    append_weights,
    mean=None,
    sum_x=False,
):
    return CvTrans.from_cv_function(
        _weighted_sinkhorn_divergence_2,
        static_argnames=["append_weights", "sum_x"],
        scaling=scaling,
        mean=mean,
        append_weights=append_weights,
        sum_x=sum_x,
    )


def _un_atomize(
    x: CV,
    nl,
    _,
    shmap,
):
    if not x.atomic:
        return x

    return x.replace(
        cv=jnp.reshape(x.cv, (-1,)),
        atomic=False,
    )


un_atomize = CvTrans.from_cv_function(_un_atomize)


def _stack_reduce(cv: CV, nl: NeighbourList | None, _, shmap, op):
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


def _affine_trans(x: CV, nl, _, shmap, C):
    assert x.dim == 2

    u = (C[0] * x.cv[0] + C[1] * x.cv[1] + C[2]) / (C[6] * x.cv[0] + C[7] * x.cv[1] + 1)
    v = (C[3] * x.cv[0] + C[4] * x.cv[1] + C[5]) / (C[6] * x.cv[0] + C[7] * x.cv[1] + 1)
    return x.replace(cv=jnp.array([u, v]))


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


def _remove_mean(cv: CV, nl: NeighbourList | None, _, shmap, mean):
    return cv - mean


def get_remove_mean_trans(c: CV, range=Ellipsis):
    assert c.batched
    mean = jnp.mean(c.cv[range, :], axis=0)

    trans = CvTrans.from_cv_function(_remove_mean, mean=mean)

    return trans.compute_cv_trans(c)[0], trans


def _normalize(cv: CV, nl: NeighbourList | None, _, shmap, std):
    return cv * (1 / std)


def get_normalize_trans(c: CV, range=Ellipsis):
    assert c.batched
    std = jnp.std(c.cv[range, :], axis=0)

    trans = CvTrans.from_cv_function(_normalize, std=std)

    return trans.compute_cv_trans(c)[0], trans


def _cv_slice(cv: CV, nl: NeighbourList, _, shmap, indices):
    return cv.replace(cv=jnp.take(cv.cv, indices, axis=-1), _combine_dims=None)


def get_non_constant_trans(
    c: list[CV], c_t: list[CV] | None = None, w: list[Array] | None = None, epsilon=1e-14, max_functions=None
):
    from IMLCV.base.rounds import Covariances

    cov = Covariances.create(
        cv_0=c,
        cv_1=c_t,
        w=w,
        symmetric=True,
        calc_pi=True,
        only_diag=True,
    )

    cov = jnp.sqrt(cov.C00)

    idx = jnp.argsort(cov, descending=True)
    cov_sorted = cov[idx]

    pos = int(jnp.argwhere(cov_sorted > epsilon)[-1][0])

    if max_functions is not None:
        if pos > max_functions:
            pos = max_functions
    if pos == 0:
        raise ValueError("No features found")

    idx = idx[:pos]

    print(f"selected auto variances {cov[idx]}")

    trans = CvTrans.from_cv_function(_cv_slice, indices=idx)

    return trans


def get_feature_cov(
    c_0: list[CV],
    c_tau: list[CV],
    w: list[jnp.array] | None = None,
    epsilon=1e-14,
    max_functions=None,
    T_scale=1,
) -> tuple[CV, CV, CvTrans]:
    from IMLCV.base.rounds import Covariances

    print("computing feature covariances")

    cov = Covariances.create(
        c_0,
        c_tau,
        w=w,
        symmetric=True,
        calc_pi=True,
        only_diag=True,
        T_scale=T_scale,
    )

    print("computing feature covariances done")

    cov_n = jnp.sqrt(cov.C00 * cov.C11)
    cov_01 = jnp.where(cov_n > epsilon, cov.C01 / cov_n, 0)

    idx = jnp.argsort(cov_01, descending=True)
    cov_sorted = cov_01[idx]

    pos = int(jnp.argwhere(cov_sorted > epsilon)[-1][0])

    if max_functions is not None:
        if pos > max_functions:
            pos = max_functions
    if pos == 0:
        raise ValueError("No features found")

    idx = idx[:pos]

    print(f"selected auto covariances {cov_01[idx]}")

    trans = CvTrans.from_cv_function(_cv_slice, indices=idx)

    return trans


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
        shmap=True,
    ):
        y = CV.combine(*conditioners).cv
        return CV(cv=x.cv * self.s(y) + self.t(y))

    def backward(
        self,
        z: CV,
        nl: NeighbourList | None,
        conditioners: list[CV] | None = None,
        shmap=True,
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
