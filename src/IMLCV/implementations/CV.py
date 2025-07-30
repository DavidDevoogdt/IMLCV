from functools import partial

import jax
import jax.numpy as jnp
from jax import Array
from jaxopt.linear_solve import solve_normal_cg

from IMLCV.base.CV import (
    CV,
    CollectiveVariable,
    CvMetric,
    CvTrans,
    NeighbourList,
    NeighbourListInfo,
    SystemParams,
)
from IMLCV.base.datastructures import vmap_decorator

######################################
#       CV transformations           #
######################################


def _identity_trans(x, nl, shmap, shmap_kwargs):
    return x


def _zero_cv_trans(x: CV, nl, shmap, shmap_kwargs):
    return x.replace(cv=jnp.array([0.0]), combine_dims=None)


def _zero_cv_flow(x: SystemParams, nl, shmap, shmap_kwargs):
    return CV(cv=jnp.array([0.0]))


identity_trans = CvTrans.from_cv_function(_identity_trans)
zero_trans = CvTrans.from_cv_function(_zero_cv_trans)
zero_flow = CvTrans.from_cv_function(_zero_cv_flow)


def _Volume(
    sp: SystemParams,
    _nl,
    shmap,
    shmap_kwargs,
):
    assert sp.cell is not None, "can only calculate volume if there is a unit cell"

    vol = jnp.abs(jnp.linalg.det(sp.cell))
    return CV(cv=jnp.array([vol]))


Volume = CvTrans.from_cv_function(_Volume)


def _lattice_invariants(
    sp: SystemParams,
    nl,
    shmap,
    shmap_kwargs,
):
    sp_red, _ = sp.minkowski_reduce()

    assert sp_red.cell is not None

    M = vmap_decorator(vmap_decorator(jnp.dot, in_axes=(0, None)), in_axes=(None, 0))(sp_red.cell, sp_red.cell)

    l = jnp.linalg.eigvalsh(M)

    return CV(cv=l)


LatticeInvariants = CvTrans.from_cv_function(_lattice_invariants)


def _lattice_invariants_2(
    sp: SystemParams,
    nl,
    shmap,
    shmap_kwargs,
):
    sp_red, _ = sp.minkowski_reduce()

    assert sp_red.cell is not None

    c = sp_red.cell @ sp_red.cell.T

    c0 = jnp.sqrt(c[0, 0])
    c1 = jnp.sqrt(c[1, 1])
    c2 = jnp.sqrt(c[2, 2])

    a01 = jnp.arccos(jnp.abs(c[0, 1] / (c0 * c1)))
    a12 = jnp.arccos(jnp.abs(c[1, 2] / (c1 * c2)))
    a02 = jnp.arccos(jnp.abs(c[0, 2] / (c0 * c2)))

    return CV(cv=jnp.array([c0, c1, c2, a01, a02, a12]))


LatticeInvariants2 = CvTrans.from_cv_function(_lattice_invariants_2)


def _distance(x: SystemParams, *_):
    x = x.canonicalize()[0]

    n = x.shape[-2]

    out = vmap_decorator(vmap_decorator(x.min_distance, in_axes=(0, None)), in_axes=(None, 0))(
        jnp.arange(n),
        jnp.arange(n),
    )

    return x.replace(cv=out[jnp.triu_indices_from(out, k=1)], _combine_dims=None)


def distance_descriptor():
    return CvTrans.from_cv_function(_distance)


def _position_index(sp: SystemParams, _nl, shmap, shmap_kwargs, idx):
    return CV(cv=sp.coordinates.reshape(-1)[idx])


def position_index(indices, sp):
    idx = jnp.ravel_multi_index(indices.T, sp.coordinates.shape)

    return CvTrans.from_cv_function(_position_index, idx=idx)


def _dihedral(sp: SystemParams, _nl, shmap, shmap_kwargs, numbers):
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

    # out = CV(cv=jnp.arctan2(y, x))

    # print(f"{out}")

    return CV(cv=jnp.array([jnp.arctan2(y, x)]))


def dihedral(numbers: tuple[int, int, int, int] | Array):
    """from https://stackoverflow.com/questions/20305272/dihedral-torsion-
    angle-from-four-points-in-cartesian- coordinates-in-python.

    args:
        numbers: list with index of 4 atoms that form dihedral
    """

    return CvTrans.from_cv_function(_dihedral, static_argnames=["numbers"], numbers=numbers)


def _sb_descriptor(
    sp: SystemParams,
    nl: NeighbourList,
    shmap,
    shmap_kwargs,
    r_cut,
    chunk_size_atoms,
    chunk_size_neigbourgs,
    reduce,
    reshape,
    n_max,
    l_max,
    bessel_fun="jax",
    mul_Z=False,
):
    assert nl is not None, "provide neighbourlist for sb describport"

    from IMLCV.tools.soap_kernel import p_i, p_inl_sb

    f_single, f_double = p_inl_sb(
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        bessel_fun=bessel_fun,
    )

    a = p_i(
        sp=sp,
        nl=nl,
        f_single=f_single,
        f_double=f_double,
        r_cut=r_cut,
        chunk_size_atoms=chunk_size_atoms,
        chunk_size_neigbourgs=chunk_size_neigbourgs,
        shmap=shmap,
        shmap_kwargs=shmap_kwargs,
        mul_Z=mul_Z,
        merge_ZZ=True,
        reshape=True,
    )

    return CV(cv=a, atomic=True)


def sb_descriptor(
    r_cut,
    n_max: int,
    l_max: int,
    reduce=True,
    reshape=True,
    chunk_size_atoms=None,
    chunk_size_neigbourgs=None,
    bessel_fun="jax",
    mul_Z=True,
) -> CvTrans:
    return CvTrans.from_cv_function(
        _sb_descriptor,
        static_argnames=[
            "r_cut",
            "chunk_size_atoms",
            "chunk_size_neigbourgs",
            "reduce",
            "reshape",
            "n_max",
            "l_max",
            "bessel_fun",
            "mul_Z",
        ],
        r_cut=r_cut,
        chunk_size_atoms=chunk_size_atoms,
        chunk_size_neigbourgs=chunk_size_neigbourgs,
        reduce=reduce,
        reshape=reshape,
        n_max=n_max,
        l_max=l_max,
        bessel_fun=bessel_fun,
        mul_Z=mul_Z,
    )


def _soap_descriptor(
    sp: SystemParams,
    nl: NeighbourList,
    shmap,
    shmap_kwargs,
    r_cut,
    reduce,
    reshape,
    n_max,
    l_max,
    sigma_a,
    r_delta,
    num,
    basis,
    mul_Z=True,
):
    assert nl is not None, "provide neighbourlist for soap describport"

    from IMLCV.tools.soap_kernel import p_i, p_innl_soap

    f_single, f_double = p_innl_soap(
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        sigma_a=sigma_a,
        r_delta=r_delta,
        num=num,
        basis=basis,
        reduce=reduce,
    )

    a = p_i(
        sp=sp,
        nl=nl,
        f_single=f_single,
        f_double=f_double,
        r_cut=r_cut,
        shmap=shmap,
        shmap_kwargs=shmap_kwargs,
        mul_Z=mul_Z,
        merge_ZZ=True,
        reshape=True,
    )

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
    basis="cos",
    mul_Z=True,
) -> CvTrans:
    return CvTrans.from_cv_function(
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
            "basis",
            "mul_Z",
        ],
        r_cut=r_cut,
        reduce=reduce,
        reshape=reshape,
        n_max=n_max,
        l_max=l_max,
        sigma_a=sigma_a,
        r_delta=r_delta,
        num=num,
        basis=basis,
        mul_Z=mul_Z,
    )


def NoneCV() -> CollectiveVariable:
    return CollectiveVariable(
        f=zero_flow,
        metric=CvMetric.create(
            periodicities=[False],
            bounding_box=[[-0.5, 0.5]],
        ),
    )


######################################
#           CV trans                 #
######################################
def _rotate_2d(cv: CV, _nl: NeighbourList, shmap, shmap_kwargs, alpha):
    return (
        jnp.array(
            [[jnp.cos(alpha), jnp.sin(alpha)], [-jnp.sin(alpha), jnp.cos(alpha)]],
        )
        @ cv
    )


def rotate_2d(alpha):
    return CvTrans.from_cv_function(_rotate_2d, alpha=alpha)


def _project_distances(cvs: CV, nl, shmap, shmap_kwargs, a: float):
    "projects the distances to a reaction coordinate"
    import jax.numpy as jnp

    from IMLCV.base.CV import CV

    assert cvs.dim == 2

    r1 = cvs.cv[0]
    r2 = cvs.cv[1]

    x = (r2**2 - r1**2) / (2 * a)
    y2 = r2**2 - (a / 2 + x) ** 2

    y2_safe = jnp.where(y2 <= 0, jnp.ones_like(y2), y2)
    y = jnp.where(y2 <= 0, 0.0, jnp.sqrt(y2_safe))

    return CV(cv=jnp.array([x / a + 0.5, y / a]))


def project_distances(a):
    return CvTrans.from_cv_function(_project_distances, a=a)


def _scale_cv_trans(x, nl, shmap, shmap_kwargs, upper, lower, mini, diff):
    return x.replace(cv=((x.cv - mini) / diff) * (upper - lower) + lower)


def scale_cv_trans(array: CV, lower: float = 0.0, upper: float = 1.0):
    "axis 0 is batch axis"
    maxi = jnp.nanmax(array.cv, axis=0)
    mini = jnp.nanmin(array.cv, axis=0)

    diff = maxi - mini
    diff = jnp.where(diff == 0, 1, diff)

    return CvTrans.from_cv_function(_scale_cv_trans, upper=upper, lower=lower, mini=mini, diff=diff)


def _trunc_svd(x: CV, nl: NeighbourList | None, shmap, shmap_kwargs, m_atomic, v, cvi_shape):
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

    return out.compute_cv(m)[0], out


def _inv_sigma_weighing(
    x: CV,
    nl: NeighbourList,
    shmap,
    shmap_kwargs,
    mu: list[jax.Array] | None,
    sigma: list[jax.Array],
    norm: list[jax.Array],
    info: NeighbourListInfo | None = None,
):
    if nl is not None:
        _info = nl.info
    else:
        assert info is not None
        _info = info

    assert x.atomic

    _, argmask, cv_z = _info.nl_split_z(x)

    x_new = jnp.zeros_like(x.cv)

    for i, (idx_i, cv_i, sigma_i, n_i) in enumerate(zip(argmask, cv_z, sigma, norm)):
        sigma_safe = jnp.where(sigma_i == 0, 1.0, sigma_i)
        sigma_inv = jnp.where(sigma_i == 0, 0.0, 1 / sigma_safe)
        # var_inv = sigma_i**2

        # n = jnp.sum(simga_inv)

        # jax.debug.print("{}",n)

        y = cv_i.cv
        if mu is not None:
            mu_i = mu[i]
            y -= mu_i

        y *= sigma_inv**2 / jnp.mean(sigma_inv)

        x_new = x_new.at[idx_i, :].set(y)

    return x.replace(cv=x_new)


def get_inv_sigma_weighing(
    cv_0: list[CV],
    nli: NeighbourListInfo,
    remove_mean=True,
):
    _, _, cvs = jax.vmap(nli.nl_split_z)(CV.stack(*cv_0))

    if remove_mean:
        per_feature_mu = [jax.vmap(lambda x: jnp.mean(x.cv), in_axes=2)(cv_i) for cv_i in cvs]
    else:
        per_feature_mu = None

    per_feature_sigma = [jax.vmap(lambda x: jnp.std(x.cv), in_axes=2)(cv_i) for cv_i in cvs]

    n_sigma = [jnp.nansum(1 / a) for a in per_feature_sigma]

    print(f"{n_sigma=}")

    return CvTrans.from_cv_function(_inv_sigma_weighing, mu=per_feature_mu, sigma=per_feature_sigma, norm=n_sigma)


def kernel_dist(p1: jax.Array, p2: jax.Array, xi=2.0):
    # print(f"new dist sum")

    # def log_safe(x: jax.Array):
    #     x = jnp.abs(x)
    #     x = jnp.where(x < 1e-10, 1e-10, x)

    #     return jnp.log(x)

    n_1 = jnp.sum(p1 * p1)
    n_1_safe = jnp.where(n_1 == 0, 1.0, n_1)
    p1 = jnp.where(n_1 == 0, 0.0, p1 / jnp.sqrt(n_1_safe))

    n_2 = jnp.sum(p2 * p2)
    n_2_safe = jnp.where(n_2 == 0, 1.0, n_2)
    p2 = jnp.where(n_2 == 0, 0.0, p2 / jnp.sqrt(n_2_safe))

    return jnp.sum(jnp.abs(p1 - p2) ** xi)

    # return -xi * log_safe(jnp.sum(p1*p2))


def sinkhorn_divergence_2(
    x1: CV,
    x2: CV,
    nl1: NeighbourListInfo,
    nl2: NeighbourListInfo,
    z_scale: jax.Array,
    alpha=1e-2,
    jacobian=False,
    lse=True,
    exp_factor: jax.Array | None = None,
    mass_weight=True,
    dist_fun=kernel_dist,
    scale_std=True,
    xi=2.0,
) -> CV:
    """caluculates the sinkhorn divergence between two CVs. If x2 is batched, the resulting divergences are stacked"""

    assert x1.atomic
    assert x2.atomic

    assert not x1.batched
    assert not x2.batched

    def solve_svd(matvec, b: Array) -> Array:
        from jax.numpy.linalg import lstsq
        from jaxopt._src.linear_solve import _materialize_array

        r_cond_svd = 1e-10
        print(f"solve_svd {  b= }")

        if len(b.shape) == 0:
            return b / _materialize_array(matvec, b.shape)

        A = _materialize_array(matvec, b.shape, b.dtype)

        b_shape = b.shape

        if len(b_shape) > 2:
            raise ValueError("b must be 1D or 2D")

        if len(b.shape) == 2:
            A = A.reshape(-1, b.shape[0] * b.shape[1])
            b = b.ravel()

        x, _, _, s = lstsq(A, b, rcond=r_cond_svd)

        # jax.debug.print("s {} {} {}", s, A, b)

        if len(b_shape) == 2:
            x = x.reshape(*b_shape[1])  # type: ignore

        return x

    def _core_sinkhorn(
        p1,
        p2,
        epsilon,
        scale,
    ):
        c = vmap_decorator(
            vmap_decorator(
                partial(dist_fun, xi=xi),
                in_axes=(None, 0),
            ),
            in_axes=(0, None),
        )(p1, p2)

        c /= scale

        if epsilon is None:
            return jnp.sum(jnp.diag(c)), p1

        # jax.debug.print("c {} {}", c, c.shape)

        n = c.shape[0]
        m = c.shape[1]

        a = jnp.ones((c.shape[0],)) / n
        b = jnp.ones((c.shape[1],)) / m

        #

        from jaxopt import FixedPointIteration

        if lse:  # log space
            g_over_eps = jnp.log(b)
            c_over_eps = c / epsilon

            def _f0(c_over_epsilon, h_over_eps):
                u = h_over_eps - c_over_epsilon
                u_max = jnp.max(u)

                return jnp.log(jnp.sum(jnp.exp(u - u_max))) + u_max

            def _T_lse(g_over_eps, c_over_eps):
                f_over_eps = -jnp.log(n) - vmap_decorator(_f0, in_axes=(0, None))(c_over_eps, g_over_eps)
                g_over_eps = -jnp.log(m) - vmap_decorator(_f0, in_axes=(1, None))(c_over_eps, f_over_eps)

                return g_over_eps, f_over_eps

            fpi = FixedPointIteration(
                fixed_point_fun=_T_lse,
                maxiter=1000,
                tol=1e-12,  # solve exactly
                implicit_diff_solve=partial(
                    solve_normal_cg,
                    ridge=1e-10,
                    tol=1e-12,
                    maxiter=1000,
                ),
                has_aux=True,
            )

            out = fpi.run(g_over_eps, c_over_eps)

            g_over_eps = out.params
            f_over_eps = out.state.aux

            @partial(vmap_decorator, in_axes=(None, 0, 1), out_axes=1)
            @partial(vmap_decorator, in_axes=(0, None, 0), out_axes=0)
            def _P(f_over_eps, g_over_eps, c_over_eps):
                return jnp.exp(f_over_eps - c_over_eps + g_over_eps)

            P = _P(f_over_eps, g_over_eps, c_over_eps)

        else:
            K = jnp.exp(-c / epsilon)

            # initialization
            v = jnp.ones((c.shape[1],))

            def _T(v, K, a, b):
                u = a / jnp.einsum("ij,j->i", K, v)
                v = b / jnp.einsum("ij,i->j", K, u)

                return v, u

            fpi = FixedPointIteration(
                fixed_point_fun=_T,
                maxiter=500,
                tol=1e-12,
                implicit_diff_solve=partial(solve_normal_cg, ridge=1e-8),
                has_aux=True,
            )

            out = fpi.run(v, K, a, b)

            v = out.params
            u = out.state.aux

            P = jnp.einsum("i,j,ij->ij", u, v, K)

        p1_j = jnp.einsum("ij,i...->j...", P, p1_i)

        return jnp.einsum("ij,ij", P, c), p1_j

    # @partial(jax.value_and_grad, argnums=1)
    def get_d_p12(p1_i: jax.Array, p2_i: jax.Array, scale: jax.Array):
        d11, _ = _core_sinkhorn(p1_i, p1_i, epsilon=alpha, scale=scale)
        d12, _ = _core_sinkhorn(p1_i, p2_i, epsilon=alpha, scale=scale)
        d22, _ = _core_sinkhorn(p2_i, p2_i, epsilon=alpha, scale=scale)

        return -(0.5 * d11 - d12 + 0.5 * d22) / p1_i.shape[0]

    if jacobian:
        get_d_p12 = jax.value_and_grad(get_d_p12, argnums=1)  # type:ignore

    # def get_aligined_x1(x1: CV, nl1: NeighbourListInfo, nl2: NeighbourListInfo):
    src_mask, _, p1_cv = nl1.nl_split_z(x1)
    tgt_mask, tgt_split, p2_cv = nl2.nl_split_z(x2)

    p1 = [a.cv.reshape(a.shape[0], -1) for a in p1_cv]
    p2 = [a.cv.reshape(a.shape[0], -1) for a in p2_cv]

    if jacobian:
        out = jnp.zeros((x2.shape[0], x2.shape[1] + 1))
    else:
        out = jnp.zeros((x2.shape[0], 1))

    # solve problem per atom kind
    for i, (p1_i, p2_i, out_i, zi) in enumerate(zip(p1, p2, tgt_split, z_scale)):
        # ef = exp_factor[i] if exp_factor is not None else None

        if jacobian:
            d_j, p1_j = get_d_p12(p1_i, p2_i, scale=zi)

            out = out.at[out_i, 0].set(d_j)
            out = out.at[out_i, 1:].set(p1_j)
        else:
            d_j = get_d_p12(p1_i, p2_i, scale=zi)
            out = out.at[out_i, 0].set(d_j)

    print(f"{out.shape=}")

    return x2.replace(
        cv=out,
        _combine_dims=(1, x2.shape[1]) if jacobian else None,
        atomic=True,
        _stack_dims=x1._stack_dims,
    )


def _sinkhorn_divergence_trans_2(
    cv: CV,
    nl: NeighbourList | None,
    shmap,
    shmap_kwargs,
    nli: NeighbourListInfo,
    pi: CV,
    alpha_rematch,
    z_scale: jax.Array,
    exp_factor: jax.Array | None = None,
    # normalize,
    jacobian=False,
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
            z_scale=z_scale,
            # exp_factor=exp_factor,
            # normalize=normalize,
            jacobian=jacobian,
        )

    if pi.batched:
        f = vmap_decorator(f, in_axes=(0, None))

    out = f(pi, cv)

    print(f"pre {out=}")

    if pi.batched:
        # unstacked = [a.unbatch() for a in out]

        # # vmap over z
        # @partial(jax.vmap, in_axes=(1, 1), out_axes=(1))
        # def softmax(d_cv: CV, p: CV):
        #     d = d_cv.cv

        #     w = jnp.exp(-d) / jnp.sum(jnp.exp(-d))

        #     return p.replace(cv=jnp.hstack([w * d, w * p.cv]), _combine_dims=(1, p.shape[1]))

        # d, p = out.split()

        # out = softmax(d, p)

        # print(f"{out=}")

        out = CV.combine(*[a.unbatch() for a in out])

        print(f"post {out=}")

    return out


def get_sinkhorn_divergence_2(
    nli: NeighbourListInfo | NeighbourList,
    pi: CV,
    alpha_rematch: float | None = 0.1,
    jacobian=True,
    scale_z=False,
) -> CvTrans:
    """Get a function that computes the sinkhorn divergence between two point clouds. p_i and nli are the points to match against."""

    assert pi.atomic, "pi must be atomic"

    assert pi.batched, "not fully implemented"

    if isinstance(nli, NeighbourList):
        print("converting nli to info")
        nli = nli.info

    # computes the average distance to other atoms

    if scale_z:

        @partial(jax.vmap, in_axes=(0, None, None))
        @partial(jax.vmap, in_axes=(None, 0, None))
        def get_d(cv_0: CV, cv_1: CV, nli: NeighbourListInfo):
            _, _, cv_0_split = nli.nl_split_z(cv_0)
            _, _, cv_1_split = nli.nl_split_z(cv_1)

            d = []

            for x, y in zip(cv_0_split, cv_1_split):
                c = vmap_decorator(
                    vmap_decorator(
                        partial(kernel_dist, xi=2.0),
                        in_axes=(None, 0),
                    ),
                    in_axes=(0, None),
                )(x.cv, y.cv)

                jnp.tril(c)

                d.append(jnp.mean(jnp.tril(c)))

            return jnp.array(d)

        dz = jax.vmap(jnp.mean, in_axes=(2))(get_d(pi, pi, nli))

        print(f"{dz=} {dz.shape=}")
    else:
        assert nli.num_z_unique is not None

        print(f"{nli.num_z_unique=}")
        dz = jnp.ones((len(nli.num_z_unique),))

    return CvTrans.from_cv_function(
        _sinkhorn_divergence_trans_2,
        static_argnames=[
            "jacobian",
        ],
        nli=nli,
        pi=pi,
        alpha_rematch=alpha_rematch,
        jacobian=jacobian,
        z_scale=dz,
    )


def _un_atomize(
    x: CV,
    nl,
    shmap,
    shmap_kwargs,
):
    if not x.atomic:
        return x

    return x.replace(
        cv=jnp.reshape(x.cv, (-1,)),
        atomic=False,
    )


un_atomize = CvTrans.from_cv_function(_un_atomize)


def _append_trans(
    x: CV,
    nl,
    shmap,
    shmap_kwargs,
    v: Array,
):
    # print(f"x {x.cv.shape=}")

    return x.replace(cv=jnp.hstack([jnp.reshape(x.cv, (-1,)), v]))


def append_trans(v: Array):
    return CvTrans.from_cv_function(_append_trans, v=v)


def _stack_reduce(cv: CV, nl: NeighbourList | None, shmap, shmap_kwargs, op):
    cvs = cv.split()

    return CV(
        op(jnp.stack([cvi.cv for cvi in cvs]), axis=0),  # type:ignore
        _stack_dims=cvs[0]._stack_dims,
        _combine_dims=cvs[0]._combine_dims,
        atomic=cvs[0].atomic,
        mapped=cvs[0].mapped,
    )


def stack_reduce(op=jnp.mean):
    return CvTrans.from_cv_function(_stack_reduce, static_argnames=["op"], op=op)


def _affine_trans(x: CV, nl, shmap, shmap_kwargs, C):
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


def _remove_mean(cv: CV, nl: NeighbourList | None, shmap, shmap_kwargs, mean):
    return cv - mean


def get_remove_mean_trans(c: CV, range=Ellipsis):
    assert c.batched
    mean = jnp.mean(c.cv[range, :], axis=0)

    trans = CvTrans.from_cv_function(_remove_mean, mean=mean)

    return trans.compute_cv(c)[0], trans


def _normalize(cv: CV, nl: NeighbourList | None, shmap, shmap_kwargs, mu, std_inv):
    x = cv.cv

    if mu is not None:
        x = mu + (x - mu) * std_inv
    else:
        x *= std_inv

    return cv.replace(cv=x)


def get_normalize_trans(c: CV, remove_mu=True, range=Ellipsis):
    assert c.batched

    if remove_mu:
        mu = jnp.mean(c.cv[range, :], axis=0)
    else:
        mu = None

    std = jnp.std(c.cv[range, :], axis=0)

    std_inv = jnp.where(std == 0, 0, 1 / std)

    trans = CvTrans.from_cv_function(_normalize, mu=mu, std_inv=std_inv)

    return trans.compute_cv(c)[0], trans


def _cv_slice(cv: CV, nl: NeighbourList, shmap, shmap_kwargs, indices):
    return cv.replace(cv=jnp.take(cv.cv, indices, axis=-1), _combine_dims=None)


def _cv_index(cv: CV, nl: NeighbourList, shmap, shmap_kwargs, indices):
    return cv.replace(
        cv=jnp.take(
            cv.cv,
            indices,
            unique_indices=True,
            indices_are_sorted=True,
        ),
        _combine_dims=None,
        atomic=False,
    )


def get_non_constant_trans(
    c: list[CV],
    c_t: list[CV] | None = None,
    nl: NeighbourList | None = None,
    nl_t: NeighbourList | None = None,
    w: list[Array] | None = None,
    epsilon=1e-14,
    max_functions=None,
    tr: CvTrans | None = None,
):
    from IMLCV.base.rounds import Covariances

    cov = Covariances.create(
        cv_0=c,
        cv_1=c_t,
        nl=nl,
        nl_t=nl_t,
        w=w,
        symmetric=True,
        calc_pi=False,
        only_diag=True,
        trans_f=tr,
        trans_g=tr,
    )

    assert cov.C00 is not None

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
    nl: list[NeighbourList] | NeighbourList | None = None,
    nl_tau: list[NeighbourList] | NeighbourList | None = None,
    w: list[jax.Array] | None = None,
    trans: CvTrans | None = None,
    epsilon=0.1,
    smallest_correlation=1e-12,
    max_functions=None,
    # T_scale=1,
) -> CvTrans:
    from IMLCV.base.rounds import Covariances

    print("computing feature covariances")

    cov = Covariances.create(
        cv_0=c_0,
        cv_1=c_tau,
        nl=nl,
        nl_t=nl_tau,
        w=w,
        symmetric=False,
        calc_pi=False,
        only_diag=True,
        # T_scale=T_scale,
        trans_f=trans,
        trans_g=trans,
    )

    print("computing feature covariances done")
    assert cov.C00 is not None
    assert cov.C11 is not None

    cov_n = jnp.sqrt(cov.C00 * cov.C11)
    cov_01 = jnp.where(cov_n > smallest_correlation, cov.C01 / cov_n, 0)  # avoid division by zero

    idx = jnp.argsort(cov_01, descending=True)
    cov_sorted = cov_01[idx]

    pos = int(jnp.argwhere(cov_sorted > epsilon)[-1][0])

    if max_functions is not None:
        if pos > max_functions:
            pos = max_functions
    if pos == 0:
        raise ValueError("No features found")

    idx = idx[:pos]

    print(f"selected auto covariances {cov_01[idx]} {pos=}")

    trans = CvTrans.from_cv_function(_cv_slice, indices=idx)

    return trans


######################################
#           CV Fun                   #
######################################


# class RealNVP(CvFunNn):
#     _: dataclasses.KW_ONLY
#     features: int
#     cv_input: CvFunInput

#     def setup(self) -> None:
#         self.s = Dense(features=self.features)
#         self.t = Dense(features=self.features)

#     def forward(
#         self,
#         x: CV,
#         nl: NeighbourList | None,
#         conditioners: list[CV] | None = None,
#         shmap=False,
#     ):
#         y = CV.combine(*conditioners).cv
#         return CV(cv=x.cv * self.s(y) + self.t(y))

#     def backward(
#         self,
#         z: CV,
#         nl: NeighbourList | None,
#         conditioners: list[CV] | None = None,
#         shmap=False,
#     ):
#         y = CV.combine(*conditioners).cv
#         return CV(cv=(z.cv - self.t(y)) / self.s(y))


# class DistraxRealNVP(CvFunDistrax):
#     _: dataclasses.KW_ONLY
#     latent_dim: int

#     def setup(self):
#         """Creates the flow model."""

#         try:
#             from tensorflow_probability.substrates import jax as tfp
#         except ImportError:
#             raise ImportError("isntall tensorflow-probability")

#         self.s = Dense(features=self.latent_dim)
#         self.t = Dense(features=self.latent_dim)

#         # Alternating binary mask.
#         self.bijector = distrax.as_bijector(
#             tfp.bijectors.RealNVP(
#                 fraction_masked=0.5,
#                 shift_and_log_scale_fn=self.shift_and_scale,
#             ),
#         )

#     def shift_and_scale(self, x0, input_depth, **condition_kwargs):
#         return self.s(x0), self.t(x0)


######################################
#           Test                     #
######################################


# class MetricUMAP(CvMetric):
#     def __init__(self, periodicities, bounding_box=None) -> None:
#         m = CvMetric.create(periodicities=periodicities, bounding_box=bounding_box)

#         # bb = np.array(self.bounding_box)
#         per = np.array(self.periodicities)

#         import numba

#         # @numba.njit
#         # def map(y):

#         #     return (y - bb[:, 0]) / (
#         #         bb[:, 1] - bb[:, 0])

#         @numba.njit
#         def _periodic_wrap(xs, min=False):
#             coor = np.mod(xs, 1)  # between 0 and 1
#             if min:
#                 # between [-0.5,0.5]
#                 coor = np.where(coor > 0.5, coor - 1, coor)

#             return np.where(per, coor, xs)

#         @numba.njit
#         def g(x, y):
#             # r1 = map(x)
#             # r2 = map(y)

#             return _periodic_wrap(x - y, min=True)

#         @numba.njit
#         def val_and_grad(x, y):
#             r = g(x, y)
#             d = np.sqrt(np.sum(r**2))

#             return d, r / (d + 1e-6)

#         self.umap_f = val_and_grad


# class hyperTorus(CvMetric):
#     def __init__(self, n) -> None:
#         periodicities = [True for _ in range(n)]
#         boundaries = jnp.zeros((n, 2))
#         boundaries = boundaries.at[:, 0].set(-jnp.pi)
#         boundaries = boundaries.at[:, 1].set(jnp.pi)

#         super().__init__(periodicities, boundaries)
