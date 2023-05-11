import dataclasses

import distrax
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
from IMLCV.base.rounds import Rounds
from IMLCV.configs.bash_app_python import bash_app_python
from jax import Array
from jax import jit
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
    references: SystemParams | None = None,
    references_nl: NeighbourList | None = None,
    reduce=True,
    reshape=False,
):
    from IMLCV.tools.soap_kernel import p_i, p_inl_sb

    # @jit #jit makes it not pickable
    def f(sp: SystemParams, nl: NeighbourList):
        assert nl is not None, "provide neighbourlist for sb describport"

        def _reduce(a):
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

        a = p_i(
            sp=sp,
            nl=nl,
            p=p_inl_sb(r_cut=r_cut, n_max=n_max, l_max=l_max),
            r_cut=r_cut,
        )

        if reduce:
            a = _reduce(a)

        if reshape:
            a = jnp.reshape(a, (-1,))

        return a

    if references is not None:
        raise NotImplementedError("adapt for divergences")

        # assert references_nl is not None

        # refs = (f)(references, references_nl)

        # # @NeighbourList.vmap_x_nl
        # # @partial(vmap, in_axes=(0, 0, None, None))
        # def _f(refs, references_nl, val, nl):
        #     return NeighbourList.match_kernel(val, refs, nl, references_nl)

        # if references.batched:
        #     _f = vmap(f, in_axes=(0, 0, None, None))

        # _f = partial(_f, refs, references_nl)

        # # @jit
        # def sb_descriptor_distance2(sp: SystemParams, nl: NeighbourList):
        #     assert nl is not None, "provide neighbourlist for sb describport"

        #     val = f(sp=sp, nl=nl)
        #     com = _f(val, nl)

        #     y2 = 1 - com
        #     y2_safe = jnp.where(y2 <= 0, jnp.ones_like(y2), y2)
        #     y = jnp.where(y2 <= 0, 0.0, jnp.sqrt(y2_safe))

        #     return jnp.ravel(y)

        # return CvFlow.from_function(sb_descriptor_distance2)  # type: ignore

    else:
        return CvFlow.from_function(f)  # type: ignore


def NoneCV() -> CollectiveVariable:
    return CollectiveVariable(
        f=CvFlow.from_function(lambda sp, nl: jnp.array([0.0])),
        metric=CvMetric(periodicities=[None]),
    )


def get_lda_cv(
    num_kfda,
    folder,
    descriptor: CvFlow,
    kernel=False,
    harmonic=True,
    kernel_type="rematch",
    sort="rematch",
    shrinkage=0,
    alpha_rematch=1e-2,
    execution_folder=None,
    stdout=None,
    stderr=None,
    max_iterations=50,
    **nl_kwargs,
) -> CollectiveVariable:
    try:
        import pymanopt
    except ImportError:
        raise ImportError(
            "pymanopt not installed, please install it to use LDA collective variable",
        )

    @bash_app_python(executors=["model", "reference"])
    def compute_lda_cv(
        num_kfda,
        folder,
        descriptor: CvFlow,
        kernel=False,
        harmonic=True,
        kernel_type="rematch",
        sort="rematch",
        shrinkage=0,
        alpha_rematch=1e-3,
        max_iterations=50,
        **nl_kwargs,
    ):
        assert folder.exists()
        pre_round = Rounds(folder=folder, new_folder=False)

        assert pre_round.round != -1, f"LDA {folder=} doesn not contain simulations"

        phase_sps = []
        phase_nls = []
        phase_cvs = []
        norm_data = []

        from IMLCV.base.CV import NeighbourList

        # @vmap
        def __norm(p):
            n1_sq = jnp.einsum("...,...->", p, p)
            n1_sq_safe = jnp.where(n1_sq <= 1e-16, 1, n1_sq)
            n1_i = jnp.where(n1_sq == 0, 0.0, 1 / jnp.sqrt(n1_sq_safe))

            return p * n1_i

        if kernel:
            raise NotImplementedError("kernel not implemented for lda")

        if sort == "rematch" or sort == "l2" or sort == "average":
            # @NeighbourList.vmap_x_nl

            def _norm(cvi, nli, pi):
                _, (P11, P12, P22) = NeighbourList.match_kernel(
                    cvi,
                    pi,
                    nli,
                    nli,
                    matching=sort,
                    alpha=alpha_rematch,
                )

                cvi = __norm(cvi)

                return (
                    0.5 * jnp.einsum("ij,i...,j...->j...", P11, cvi, cvi)
                    + 0.5 * jnp.einsum("ij,i...,j...->j...", P22, pi, pi)
                    - jnp.einsum("ij,i...,j...->j...", P12, cvi, pi)
                )

        else:
            raise NotImplementedError("todo: implement other sorts")

        for ri, ti in pre_round.iter():
            traj_info = ti.ti
            spi = traj_info.sp
            # cvi = traj_info.CV
            nli = spi.get_neighbour_list(**nl_kwargs)

            # import jax.tree_util
            # nli_flat, nli_unflatten = jax.tree_util.tree_flatten(nli)
            # spi_flat, spi_unflatten = jax.tree_util.tree_flatten(spi)
            # with serial_loop("l", 100):
            from netket.jax import vmap_chunked

            # def _f(sp, nl):
            #     return vmap(descriptor.compute_cv_flow)(sp, nl)
            # cvi = xmap(
            #     lambda sp, nl: _f(sp, nl),
            #     in_axes=[{0: "left"}, {0: "left"}],
            #     out_axes=["left", ...],
            #     axis_resources={"left": SerialLoop(100)},
            # )(spi, nli)

            cvi = vmap_chunked(
                descriptor.compute_cv_flow,
                in_axes=(0, 0),
                chunk_size=500,
            )(spi, nli)

            if sort == "l2" or sort == "average":
                if len(norm_data) == 0:
                    norm_data.append(None)

            elif sort == "rematch":
                pi = __norm(jnp.average(cvi.cv, axis=0))
                norm_data.append(pi)

            phase_sps.append(spi)
            phase_nls.append(nli)
            phase_cvs.append(cvi)

        lj = jnp.array([x.shape[0] for x in phase_sps])
        lj_min = jnp.min(lj)
        lj = jnp.full_like(lj, lj_min)

        sp = None
        nl = None

        for s, n, c in zip(phase_sps, phase_nls, phase_cvs):
            if sp is None:
                sp = s
            else:
                sp = sp + s

            if nl is None:
                nl = n
            else:
                nl = nl + n

        cv = jnp.vstack([pc.cv[0:lj_min, :] for pc in phase_cvs])

        alphas = []
        vs = []
        scale_factors = []

        for nd in norm_data:
            normed_cvs = vmap(_norm, in_axes=(0, 0, None))(cv, nl, nd)

            if kernel:
                raise NotImplementedError("todo: select subset to sample kernels from")

                # @jit
                # @partial(vmap, in_axes=(0, None, 0, None))
                # @partial(vmap, in_axes=(None, 0, None, 0))
                # def kernel_matrix(p0, p1, nl0, nl1):
                #     return NeighbourList.match_kernel(
                #         p0,
                #         p1,
                #         nl0,
                #         nl1,
                #         matching=kernel_type,
                #     )[0]

                # mat = []
                # for i, (cva, nla) in enumerate(zip(phase_cvs, phase_nls)):
                #     mat_i = []
                #     for j, (cvb, nlb) in enumerate(zip(phase_cvs, phase_nls)):
                #         mat_i.append(kernel_matrix(cva, cvb, nla, nlb))
                #     mat.append(jnp.array(mat_i))
                # lj = jnp.array([x.shape[0] for x in phase_cvs])

                # Kj_nm = jnp.concatenate(mat, axis=1)
                # Sj_n = jnp.mean(Kj_nm, axis=2)
                # S_n = vmap(lambda x: jnp.sum(x * lj), in_axes=1)(Sj_n) / sum(lj)
                # Sb = jnp.einsum(
                #     "ijk,i->jk",
                #     vmap(lambda x: jnp.outer(x - S_n, x - S_n))(Sj_n),
                #     lj,
                # )
                # Sw = jnp.sum(
                #     vmap(
                #         lambda Kj, lj: Kj @ (jnp.eye(Kj.shape[1]) - jnp.ones((Kj.shape[1], Kj.shape[1])) / lj) @ Kj.T,
                #     )(Kj_nm, lj),
                #     axis=0,
                # )

                # if not harmonic:
                #     # regularize
                #     Sw = Sw + 1e-3 * jnp.eye(Sw.shape[0])

                # manifold = pymanopt.manifolds.stiefel.Stiefel(n=Sb.shape[0], p=num_kfda)

                # @pymanopt.function.jax(manifold)
                # def cost(x):
                #     if harmonic:
                #         return jnp.trace(x.T @ Sw @ x) / jnp.trace(x.T @ Sb @ x)
                #     else:
                #         return -jnp.trace(x.T @ Sb @ x) / (jnp.trace(x.T @ Sw @ x) + 1e-3)

                # optimizer = pymanopt.optimizers.TrustRegions(max_iterations=50)
                # problem = pymanopt.Problem(manifold, cost)
                # result = optimizer.run(problem)

                # alpha = result.point

                # assert jnp.allclose(alpha.T @ alpha, jnp.eye(num_kfda))

                # # make cv
                # x = phase_sps[0] + phase_sps[1]
                # x_nl = x.get_neighbour_list(**nl_kwargs)
                # x_cv = descriptor.compute_cv_flow(x, x_nl)

                # scale_factor = jnp.mean(alpha.T @ Kj_nm, axis=2).T

                # @CvFlow
                # def klda_cv(sp: SystemParams, nl: NeighbourList):
                #     # @NeighbourList.vmap_x_nl
                #     @partial(vmap, in_axes=(0, 0, None, None))
                #     def __f(cv1, nl1, cv2, nl2):
                #         return NeighbourList.match_kernel(cv1.cv, cv2.cv, nl1, nl2)[0]

                #     cv = descriptor.compute_cv_flow(sp, nl)

                #     lda = alpha.T @ __f(x_cv, x_nl, cv, nl)

                #     return CV(
                #         cv=(lda - scale_factor[:, 0]) / (scale_factor[:, 1] - scale_factor[:, 0]),
                #     )

                # # check if close to 1
                # # cvs0 = klda_cv.compute_cv_flow(phase_sps[0], phase_nls[0])
                # # cvs1 = klda_cv.compute_cv_flow(phase_sps[1], phase_nls[1])
                # cv = CollectiveVariable(
                #     f=klda_cv,
                #     metric=CvMetric(
                #         periodicities=[False] * num_kfda,
                #         bounding_box=jnp.repeat(
                #             jnp.array([[-0.1, 1.1]]),
                #             num_kfda,
                #             axis=0,
                #         ),
                #     ),
                # )

            def trunc_svd(m):
                u, s, v = jnp.linalg.svd(m, full_matrices=False)

                include_mask = s > 10 * jnp.max(
                    jnp.array([u.shape[-2], v.shape[-1]]),
                ) * jnp.finfo(
                    normed_cvs.dtype,
                ).eps * jnp.max(s)

                def _f(u, s, v, include_mask, m):
                    u, s, v = u[:, include_mask], s[include_mask], v[include_mask, :]
                    return u, s, v, m @ v.T, include_mask

                if s.ndim == 1:
                    return _f(u, s, v, include_mask, m)
                if s.ndim == 2:
                    include_mask = jnp.sum(~include_mask, axis=0) == 0

                    return vmap(_f, in_axes=(0, 0, 0, None, 0))(
                        u,
                        s,
                        v,
                        include_mask,
                        m,
                    )

                raise ValueError("s.ndim must be 1 or 2")

            u, s, v, normed_cvs, include_mask = trunc_svd(
                vmap(lambda x: jnp.reshape(x, (-1)))(normed_cvs),
            )

            # step 1
            normed_cvs = jnp.reshape(normed_cvs, (len(lj), lj_min, -1))

            mu_i = vmap(lambda x, y: jnp.sum(x, axis=0) / y, in_axes=(0, 0))(
                normed_cvs,
                lj,
            )
            mu = jnp.sum(
                vmap(lambda x, y: x * y, in_axes=(0, 0))(mu_i, lj),
                axis=0,
            ) / jnp.sum(lj)

            u_w, s_w, v_w, _, _ = trunc_svd(
                vmap(lambda cvs, mu_i: cvs - mu_i)(normed_cvs, mu_i),
            )
            u_b, s_b, v_b, _, _ = trunc_svd(
                vmap(lambda lj, mu_i: jnp.sqrt(lj) * (mu_i - mu))(lj, mu_i),
            )

            manifold = pymanopt.manifolds.stiefel.Stiefel(n=mu.shape[0], p=num_kfda)

            # pip install pymanopt@git+https://github.com/pymanopt/pymanopt.git
            @pymanopt.function.jax(manifold)
            @jit
            def cost(x):
                a = jnp.einsum("ab, ija, ij  ,ijc, cb ", x, v_w, s_w**2, v_w, x)
                b = jnp.einsum("ab, ja, j, jc, cb ", x, v_b, s_b**2, v_b, x)

                if harmonic:
                    out = ((1 - shrinkage) * a + shrinkage) / ((1 - shrinkage) * b + shrinkage)
                else:
                    out = -((1 - shrinkage) * b + shrinkage) / ((1 - shrinkage) * a + shrinkage)

                return out

            optimizer = pymanopt.optimizers.TrustRegions(max_iterations=50)
            # optimizer = pymanopt.optimizers.ConjugateGradient(max_iterations=3000)

            problem = pymanopt.Problem(manifold, cost)
            result = optimizer.run(problem)

            alpha = result.point

            scale_factor = vmap(lambda x: alpha.T @ x)(mu_i)

            scale_factors.append(scale_factor)
            alphas.append(alpha)
            vs.append(v)

        # return norm_data, alphas, vs, scale_factors, _norm

        @CvFlow.from_function
        def _lda_cv(sp: SystemParams, nl: NeighbourList):
            cv = descriptor.compute_cv_flow(sp, nl)

            def cvs(nd, p, alpha, v, scale_factor):
                p = _norm(cv.cv, nl, nd)
                cv_unscaled = alpha.T @ (jnp.reshape(p, (-1,)) @ v.T)
                cv_scaled = (cv_unscaled - scale_factor[0, :]) / (scale_factor[1, :] - scale_factor[0, :])
                return cv_scaled

            return jnp.mean(
                jnp.array(
                    [
                        cvs(nd, cv, alpha, v, scale_factor)
                        for nd, alpha, v, scale_factor in zip(
                            norm_data,
                            alphas,
                            vs,
                            scale_factors,
                        )
                    ],
                ),
                axis=0,
            )

        # cv0 = _lda_cv.compute_cv_flow(phase_sps[0], phase_nls[0])
        # cv1 = _lda_cv.compute_cv_flow(phase_sps[1], phase_nls[1])

        # print(
        #     f"cv0 mean {jnp.mean(cv0.cv)} std {jnp.std(cv0.cv)}\n cv1 mean {jnp.mean(cv1.cv)} std {jnp.std(cv1.cv)}"
        # )

        return _lda_cv

    _lda_cv = compute_lda_cv(
        num_kfda=num_kfda,
        folder=folder,
        descriptor=descriptor,
        kernel=kernel,
        harmonic=harmonic,
        kernel_type=kernel_type,
        sort=sort,
        shrinkage=shrinkage,
        alpha_rematch=alpha_rematch,
        execution_folder=execution_folder,
        stdout=stdout,
        stderr=stderr,
        max_iterations=max_iterations,
        **nl_kwargs,
    ).result()

    cv0 = CollectiveVariable(
        f=_lda_cv,
        metric=CvMetric(
            periodicities=[False] * num_kfda,
            bounding_box=jnp.repeat(jnp.array([[-0.1, 1.1]]), num_kfda, axis=0),
        ),
    )

    return cv0


######################################
#           CV trans                 #
######################################


# class MeshGrid(CvTrans):
#     def __init__(self, meshgrid) -> None:
#         self.map_meshgrids = meshgrid
#         super().__init__(f)

#     def _f(self, x: CV):
#         #  if self.map_meshgrids is not None:
#         y = x.cv

#         y = y * (jnp.array(self.map_meshgrids[0].shape) - 1)
#         y = jnp.array(
#             [jsp.ndimage.map_coordinates(wp, y, order=1) for wp in self.map_meshgrids]
#         )


def rotate_2d(alpha):
    @CvTrans.from_cv_function
    def f(cv: CV, _):
        return (
            jnp.array(
                [[jnp.cos(alpha), jnp.sin(alpha)], [-jnp.sin(alpha), jnp.cos(alpha)]],
            )
            @ cv
        )

    return f


def project_distances(a):
    @CvTrans.from_cv_function
    def project_distances(cvs: CV, _):
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
    def f0(x, _):
        return (x - (mini + maxi) / 2) / diff

    return f0


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

    def forward(self, x: CV, *cond: CV):
        y = CV.combine(*cond).cv
        return CV(cv=x.cv * self.s(y) + self.t(y))

    def backward(self, z: CV, *cond: CV):
        y = CV.combine(*cond).cv
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
