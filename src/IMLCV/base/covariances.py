from __future__ import annotations

from typing import Callable, TypeVar, cast

import jax
import jax.numpy as jnp
from jax import Array

from IMLCV.base.CV import (
    CvTrans,
)
from IMLCV.base.dataobjects import CV, NeighbourList, ShmapKwargs, SystemParams, macro_chunk_map_fun
from IMLCV.base.decoratros import MyPyTreeNode, field

X = TypeVar("X", "CV", "SystemParams", "NeighbourList")
X2 = TypeVar("X2", "CV", "SystemParams", "NeighbourList")


class Covariances(MyPyTreeNode):
    rho_00: jax.Array
    rho_01: jax.Array | None
    rho_10: jax.Array | None
    rho_11: jax.Array | None

    rho_gen: jax.Array | None = None

    pi_s_0: jax.Array | None
    pi_s_1: jax.Array | None
    sigma_0: jax.Array
    sigma_1: jax.Array | None

    d_rho_00: jax.Array | None = None

    bc: list[jax.Array] | None = None  # used in BC estimator
    shrinkage_method: str = field(default="bidiag", pytree_node=False)

    time_series: bool = True

    # n: int | None = None

    W_0: jax.Array | None = None
    W_1: jax.Array | None = None

    only_diag: bool = field(default=False, pytree_node=False)
    trans_f: CvTrans | CvTrans | None = None
    trans_g: CvTrans | CvTrans | None = None
    # T_scale: float = 1.0
    symmetric: bool = field(default=False, pytree_node=False)
    pi_argmask: Array | None = None

    w2: float | None = None
    w3: float | None = None

    @staticmethod
    def create(
        cv_0: list[X],
        cv_1: list[X] | None = None,
        nl: list[NeighbourList] | NeighbourList | None = None,
        nl_t: list[NeighbourList] | NeighbourList | None = None,
        w: list[Array] | None = None,
        w_t: list[Array] | None = None,
        dynamic_weights: list[Array] | None = None,
        calc_pi=True,
        macro_chunk=1000,
        chunk_size=None,
        only_diag=False,
        trans_f: CvTrans | CvTrans | None = None,
        trans_g: CvTrans | CvTrans | None = None,
        # T_scale=1,
        symmetric=False,
        calc_C00=True,
        calc_C01=True,
        calc_C10=True,
        calc_C11=True,
        shmap_kwargs=ShmapKwargs.create(),
        verbose=True,
        shrink=True,
        # BC=False,
        shrinkage_method="bidiag",
        pi_argmask: Array | None = None,
        get_diff=False,
        bessel_correct=True,
        generator=False,
        eps_shrink=1e-3,
    ) -> Covariances:
        time_series = cv_1 is not None

        w_dyn = [jnp.exp((jnp.log(a) + jnp.log(b)) / 2) for a, b in zip(w, w_t if w_t is not None else w)]
        # w = w_pair
        # w_t = w_pair

        # assert not only_diag

        # if generator:
        #     generator = True
        #     print("generator not implemented, setting to False")

        if not time_series:
            if get_diff:
                print("cannot get differences without time series, setting to False")
                get_diff = False

            # assert trans_f is not None
            # assert trans_g is not None

        # print(f"{nl=}")

        if w is None:
            w = [jnp.ones((cvi.shape[0],)) for cvi in cv_0]

        n = jnp.sum(jnp.array([jnp.sum(wi) for wi in w]))

        w = [wi / n for wi in w]
        if time_series:
            if w_t is None:
                w_t = [jnp.ones((cvi.shape[0],)) for cvi in cv_1]

            n = jnp.sum(jnp.array([jnp.sum(wi) for wi in w_t]))
            w_t = [wi / n for wi in w_t]

        if dynamic_weights is None:
            dynamic_weights = [jnp.ones((cvi.shape[0],)) for cvi in cv_0]

        # print(f"{dynamic_weights=}")

        print(f"shrinking {shrinkage_method =} ")

        BC = shrinkage_method == "BC"

        print(f"{nl=}")

        def cov_pi(
            carry: tuple[
                jax.Array | None,
                jax.Array | None,
                jax.Array | None,
                jax.Array | None,
                jax.Array | None,
                jax.Array | None,
                jax.Array | None,
                jax.Array | None,
                jax.Array | None,
                jax.Array | None,
                jax.Array | None,
                jax.Array | None,
                jax.Array | None,
                list[jax.Array] | None,
                jax.Array | None,
            ],
            cv_0: CV,
            cv_1: CV | None,
            w: jax.Array | None,
            w_t: jax.Array | None = None,
            diff_w: jax.Array | None = None,
        ):
            assert w is not None

            # if diff_w is not None:
            #     w *= diff_w
            #     w_t *= diff_w

            (
                rho_00_prev,
                rho_01_prev,
                rho_10_prev,
                rho_11_prev,
                rho_gen_prev,
                pi_0_prev,
                pi_1_prev,
                sigma_0_prev,
                sigma_1_prev,
                w_prev,
                w_prev_t,
                w_dyn_prev,
                bc,
                d_rho_prev,
            ) = carry

            if (trans_f is not None) and generator and (trans_g is not None):
                assert not only_diag, "only_diag not implemented for generator with transformations"

                def phi(x):
                    # print(f"phi: {x=} ")

                    cv, _ = trans_f.compute_cv(x, nl, shmap=False)
                    return cv

                shape = cv_0.shape[1:]
                n_shape = len(shape)

                # shape (n_batch, n_cv,n_atom ,3)
                jac = jax.vmap(jax.jacrev(phi))(cv_0)

                print(f"{jac=}")
                jac = jac.cv.coordinates
                # inv_sqr_masses = 1 / jnp.sqrt(nl.info.masses)

                dL = jnp.einsum("nair,nbir,n,i->ab", jac, jac, w, 1 / nl.info.masses)

                # print(f"{dL=} { sigma_0_inv=}")

                if sigma_0_prev is not None:
                    sigma_0_inv = jnp.where(sigma_0_prev == 0, 1, 1 / sigma_0_prev)

                    dL = jnp.einsum("lk,l,k->lk", dL, sigma_0_inv, sigma_0_inv)

                cv_0 = jax.vmap(phi)(cv_0)

                if cv_1 is not None:
                    print(f"{cv_1.shape=}")
                    cv_1 = jax.vmap(phi)(cv_1)

                print(f"{cv_0.shape=}")

            assert not cv_0.atomic

            x_0 = cv_0.cv

            if cv_1 is not None:
                x_1 = cv_1.cv
            else:
                x_1 = None

            if sigma_1_prev is not None:
                assert x_1 is not None
                sigma_1_inv = jnp.where(sigma_1_prev == 0, 1, 1 / sigma_1_prev)
                x_1 = jnp.einsum("ni,i->ni", x_1, sigma_1_inv)
            else:
                sigma_1_inv = None

            if sigma_0_prev is not None:
                sigma_0_inv = jnp.where(sigma_0_prev == 0, 1, 1 / sigma_0_prev)

                x_0 = jnp.einsum("ni,i->ni", x_0, sigma_0_inv)

            if get_diff:
                if (sigma_0_prev is not None) and (sigma_1_prev is not None):
                    dx = (x_0 * sigma_0_prev - x_1 * sigma_1_prev) * jnp.sqrt(sigma_0_inv * sigma_1_inv)

                else:
                    dx = x_0 - x_1

            def get_pi(_x, _w, _pi_prev, _w_prev, calc_pi):
                _dw = jnp.sum(_w)

                if _w_prev is None:
                    _w_tot = _dw
                else:
                    _w_tot = _w_prev + _dw

                if not calc_pi:
                    return None, None, _w_tot, _dw

                _pi_new_dw = jnp.einsum("ni,n->i", _x, _w)

                if _pi_prev is None:
                    _pi_tot = _pi_new_dw / _dw

                    _dpi = None
                else:
                    _pi_tot = (_pi_prev * _w_prev + _pi_new_dw) / (_w_tot)

                    _dpi = _pi_tot - _pi_prev

                if pi_argmask is not None:
                    # print(f"{pi_argmask=}")

                    _pi_tot = _pi_tot.at[pi_argmask].set(0.0)
                    if _dpi is not None:
                        _dpi = _dpi.at[pi_argmask].set(0.0)

                    # print(f"{_pi_tot}")

                return _pi_tot, _dpi, _w_tot, _dw

            pi_0_new, dpi_0, w_tot, dw = get_pi(x_0, w, pi_0_prev, w_prev, calc_pi)

            # jax.debug.print("dw_o {}", dw_0)

            if time_series:
                pi_1_new, dpi_1, w_tot_t, dw_t = get_pi(x_1, w_t, pi_1_prev, w_prev_t, calc_pi)

                #     # # diff_w = jnp.sqrt(w * w_t)  # * w_dw

                # w0_dyn = w  # / diff_w
                # w1_dyn = w_t  # / diff_w
                w_dyn = w * diff_w

                _, _, w_dyn_tot, dw_dyn = get_pi(None, w_dyn, None, w_dyn_prev, False)
                # _, _, diff_tot_1, diff_dw_1 = get_pi(None, w1_dyn, None, diff_w_prev_1, False)

            def c(_x, _y, _w, _c_prev, _pi_x, _pi_y, _d_pi_x, _d_pi_y, _w_prev, _dw, dx=False):
                if _d_pi_x is not None and not dx:
                    assert _d_pi_y is not None

                    if only_diag:
                        _c_prev += jnp.conj(_d_pi_x) * _d_pi_y
                    else:
                        _c_prev += jnp.outer(jnp.conj(_d_pi_x), _d_pi_y)

                if calc_pi and not dx:
                    assert _pi_x is not None
                    assert _pi_y is not None

                    u = _x - _pi_x
                    v = _y - _pi_y
                else:
                    u = _x
                    v = _y

                if only_diag:
                    _c_new_dw = jnp.einsum("ni,ni,n->i", jnp.conj(u), v, _w)  # in case of complex cv
                else:
                    _c_new_dw = jnp.einsum("ni,nj,n->ij", jnp.conj(u), v, _w)  # in case of complex cv

                if _c_prev is None:
                    return _c_new_dw / _dw

                return (_w_prev * _c_prev + _c_new_dw) / (_w_prev + _dw)

            if calc_C00:
                rho_00 = c(x_0, x_0, w, rho_00_prev, pi_0_new, pi_0_new, dpi_0, dpi_0, w_prev, dw)

            rho_01, rho_10, rho_11 = None, None, None

            if time_series:
                assert w is not None

                if calc_C01:
                    # Optimal Data-Driven Estimation of Generalized Markov State Models for Non-Equilibrium Dynamics
                    # sampled under wt
                    rho_01 = c(x_0, x_1, w_dyn, rho_01_prev, pi_0_new, pi_1_new, dpi_0, dpi_1, w_dyn_prev, dw_dyn)
                    # rho_01 = c(x_0, x_1, w, rho_01_prev, pi_0_new, pi_1_new, dpi_0, dpi_1, w_prev, dw)

                if calc_C10:
                    # rho_10 = c(x_1, x_0, w, rho_10_prev, pi_1_new, pi_0_new, dpi_1, dpi_0, w_prev, dw)
                    # rho_10 = c(x_1, x_0, w, rho_10_prev, pi_1_new, pi_0_new, dpi_1, dpi_0, w_prev, dw)
                    rho_10 = rho_01.T

                if calc_C11:
                    rho_11 = c(x_1, x_1, w_t, rho_11_prev, pi_1_new, pi_1_new, dpi_1, dpi_1, w_prev_t, dw_t)

            if generator:
                if rho_gen_prev is not None:
                    rho_gen = (rho_gen_prev * w_prev + dL) / (w_prev + dw)
                else:
                    rho_gen = dL / dw
            else:
                rho_gen = None

            if get_diff:
                # d_rho = c(dx, dx, w, d_rho_prev, None, None, None, None, w_prev, dw, dx=True)
                d_rho = c(dx, dx, w, d_rho_prev, None, None, None, None, w_prev, dw, dx=True)
            else:
                d_rho = None

            # convert everything to new sigma

            diag_rho_00 = jnp.diag(rho_00) if not only_diag else rho_00

            d_sigma_0 = jnp.sqrt(jnp.where(diag_rho_00 <= 0, 0.0, diag_rho_00))
            f0 = jnp.where(d_sigma_0 == 0, 1.0, 1 / d_sigma_0)

            pi_0_new = jnp.einsum("i,i->i", pi_0_new, f0) if calc_pi else None

            def get_rho(rho, f_left, f_right):
                if only_diag:
                    return jnp.einsum("i,i,i->i", rho, jnp.conj(f_left), f_right)
                return jnp.einsum("ij,i,j->ij", rho, jnp.conj(f_left), f_right)

            rho_00 = get_rho(rho_00, f0, f0)

            if generator:
                rho_gen = get_rho(rho_gen, f0, f0)

            if sigma_0_prev is None:
                sigma_0 = d_sigma_0
            else:
                sigma_0 = jnp.where(sigma_0_prev == 0, d_sigma_0, sigma_0_prev * d_sigma_0)

            if time_series:
                diag_rho_11 = jnp.diag(rho_11) if not only_diag else rho_11

                d_sigma_1 = jnp.sqrt(jnp.where(diag_rho_11 <= 0, 0.0, diag_rho_11))  # type: ignore
                f1 = jnp.where(d_sigma_1 <= 0, 1.0, 1 / d_sigma_1)

                pi_1_new = jnp.einsum("i,i->i", pi_1_new, f1) if calc_pi else None

                if rho_01 is not None:
                    rho_01 = get_rho(rho_01, f0, f1)
                if rho_10 is not None:
                    rho_10 = get_rho(rho_10, f1, f0)

                if rho_11 is not None:
                    rho_11 = get_rho(rho_11, f1, f1)

                if sigma_1_prev is None:
                    sigma_1 = d_sigma_1
                else:
                    sigma_1 = jnp.where(sigma_1_prev == 0, d_sigma_1, sigma_1_prev * d_sigma_1)

            else:
                sigma_1 = None
                rho_01 = None
                rho_10 = None
                rho_11 = None

            if get_diff:
                d_rho = get_rho(d_rho, f0, f0)

            if BC:
                assert not only_diag, "BC not implemented for only_diag"

                def _bc(_x, _y, _w, _bc_prev, _w_prev, _dw):
                    _bc_new_dw = jnp.einsum("ni,nj,ni,nj,n->ij", _x, _x, _y, _y, _w)

                    if _bc_prev is None:
                        return _bc_new_dw / _dw

                    return (_w_prev * _bc_prev + _bc_new_dw) / (_w_prev + _dw)

                assert calc_pi is False, "bias correction with pi not implemented"

                bc_00 = _bc(x_0, x_0, w, bc[0] if bc is not None else None, w_prev, dw)
                bc_01 = _bc(x_0, x_1, w, bc[1] if bc is not None else None, w_prev, dw)

                bc_00 = jnp.einsum("ij,i,j->ij", bc_00, f0**2, f0**2)
                bc_01 = jnp.einsum("ij,i,j->ij", bc_01, f0**2, f1**2)

                bc = [bc_00, bc_01]

            return (
                rho_00,
                rho_01,
                rho_10,
                rho_11,
                rho_gen,
                pi_0_new,
                pi_1_new if time_series else None,
                sigma_0,
                sigma_1 if time_series else None,
                w_tot,
                w_tot_t if time_series else None,
                w_dyn_tot if time_series else None,
                bc,
                d_rho,
            )

        if trans_f is not None:

            def f_func(x: X, nl: NeighbourList | None) -> CV:
                # print(f"{x=} {nl=} {generator=}")

                if not generator:
                    x, _ = trans_f.compute_cv(x, nl, chunk_size=chunk_size, shmap=False)
                # else:
                #     print(f"{x=}")
                return x

        else:
            assert isinstance(cv_0[0], CV)

            def f_func(x: X, nl: NeighbourList | None) -> CV:
                assert isinstance(x, CV)
                return x

        if trans_g is not None:

            def g_func(x: X, nl: NeighbourList | None) -> CV:
                if not generator:
                    x, _ = trans_g.compute_cv(x, nl, chunk_size=chunk_size, shmap=False)

                return x

        else:

            def g_func(x: X, nl: NeighbourList | None) -> CV:
                assert isinstance(x, CV)
                return x

        g_func = cast(Callable[[CV | SystemParams, NeighbourList | None], CV], g_func)

        chunk_func_init_args = cast(
            tuple[
                jax.Array | None,
                jax.Array | None,
                jax.Array | None,
                jax.Array | None,
                jax.Array | None,
                jax.Array | None,
                jax.Array | None,
                jax.Array | None,
                jax.Array | None,
                jax.Array | None,
                jax.Array | None,
                jax.Array | None,
                list[jax.Array] | None,
                jax.Array | None,
            ],
            (None, None, None, None, None, None, None, None, None, None, None, None, None, None),
        )

        # print(f"{cv_1=}")

        # with jax.debug_nans():
        out = macro_chunk_map_fun(
            # f=lambda x, nl: x,
            # ft=lambda x, nl: x,
            f=f_func,
            ft=g_func,
            y=cv_0,
            y_t=cv_1,
            nl=nl,
            nl_t=nl,
            macro_chunk=macro_chunk,
            verbose=verbose,
            chunk_func=cov_pi,
            chunk_func_init_args=chunk_func_init_args,
            w=w,
            w_t=w_t,
            d_w=dynamic_weights,
            jit_f=True,
        )

        (
            rho_00,
            rho_01,
            rho_10,
            rho_11,
            rho_gen,
            pi_s_0,
            pi_s_1,
            sigma_0,
            sigma_1,
            _w,
            _wt,
            _wdyn,
            bc,
            d_rho,
        ) = out

        print(f"{_w=} {_wt=} {_wdyn=}")

        print(f"{rho_gen=}")

        # # divide time lagged covariances by total weight
        # if time_series:
        #     print(f"correcting")

        #     assert _dw_0 is not None
        #     rho_01 /= _dw_0
        #     rho_10 /= _dw_1

        if BC:
            # print(f"{bc=}")

            bc_00 = bc[0] - rho_00**2
            bc_01 = bc[1] - rho_00**2

            # print(f"{bc_00=} {rho_00=}")

            bc = [bc_00, bc_01]

        # print(f"test {rho_00=}")
        assert rho_00 is not None
        assert sigma_0 is not None

        # bessel correction: https://univ-lyon1.hal.science/hal-04693522/file/shrinkage_eusipco_2024_reviewed.pdf

        w_stack = jnp.hstack(w)

        w2 = jnp.sum(w_stack**2) / (jnp.sum(w_stack) ** 2)
        w3 = jnp.sum(w_stack**3) / (jnp.sum(w_stack) ** 3)

        print(f"{w2=} {w3=}")

        # if bessel_correct:
        eps = w2

        print(f"bessel correction {eps=} {1/eps=}  {w_stack.shape[0]=} ")

        # C -> (1-eps) C
        # sigma -> sqrt(1-eps) sigma
        # rho: stays the same
        # pi: stays the same

        gamma = 1 / jnp.sqrt(1 - eps)

        if sigma_0 is not None:
            sigma_0 *= gamma

        if sigma_1 is not None:
            sigma_1 *= gamma

        if pi_s_0 is not None:
            pi_s_0 /= gamma

        if pi_s_1 is not None:
            pi_s_1 /= gamma

        # rho_00 = jnp.real(rho_00)

        cov = Covariances(
            rho_00=rho_00,
            rho_01=rho_01,
            rho_10=rho_10,
            rho_11=rho_11,
            rho_gen=rho_gen,
            pi_s_0=pi_s_0,
            pi_s_1=pi_s_1,
            only_diag=only_diag,
            trans_f=trans_f,
            trans_g=trans_g,
            # T_scale=T_scale,
            sigma_0=sigma_0,
            sigma_1=sigma_1,
            # n=sum([a.shape[0] for a in cv_0]),
            bc=bc,
            shrinkage_method=shrinkage_method,
            d_rho_00=d_rho,
            w2=w2,
            w3=w3,
            time_series=time_series,
            symmetric=not time_series,
        )

        # print(f"pre {jnp.isnan(cov.C00).any()}")

        # print(f"shrinking {cov.C00 is None =} {cov.C11 is None=}")

        # print(f"debug nans v2")
        # with jax.debug_nans():
        if symmetric:
            cov = cov.symmetrize()

        # print(f"shrinking {cov.C00 is None =} {cov.C11 is None=}")

        # print(f"sym {jnp.isnan(cov.C00).any()}")0

        if shrink:
            cov = cov.shrink(eps_shrink=eps_shrink)

        # print(f"shrunk {jnp.isnan(cov.C00).any()}")

        return cov

    @property
    def generator(self):
        return self.rho_gen is not None

    @property
    def pi_0(self):
        if self.pi_s_0 is None:
            return None
        return jnp.einsum("i,i->i", self.pi_s_0, self.sigma_0)

    @property
    def pi_1(self):
        if self.pi_s_1 is None:
            return None
        return jnp.einsum("i,i->i", self.pi_s_1, self.sigma_1)

    @property
    def C00(self):
        if self.only_diag:
            return jnp.einsum("i,i->i", self.rho_00, self.sigma_0**2)

        return jnp.einsum("ij,i,j->ij", self.rho_00, self.sigma_0, self.sigma_0)

    @property
    def C_gen(self):
        if self.only_diag:
            return jnp.einsum("i,i->i", self.rho_gen, self.sigma_0**2)

        return jnp.einsum("ij,i,j->ij", self.rho_gen, self.sigma_0, self.sigma_0)

    @property
    def d_C00(self):
        if self.only_diag:
            return jnp.einsum("i,i->i", self.d_rho_00, self.sigma_0**2)

        return jnp.einsum("ij,i,j->ij", self.d_rho_00, self.sigma_0, self.sigma_0)

    @property
    def C01(self):
        if self.rho_01 is None:
            return None
        assert self.sigma_1 is not None

        if self.only_diag:
            return jnp.einsum("i,i->i", self.rho_01, self.sigma_0 * self.sigma_1)

        return jnp.einsum("ij,i,j->ij", self.rho_01, self.sigma_0, self.sigma_1)

    @property
    def C10(self):
        if self.rho_10 is None:
            return None
        assert self.sigma_1 is not None

        if self.only_diag:
            return jnp.einsum("i,i->i", self.rho_10, self.sigma_1 * self.sigma_0)

        return jnp.einsum("ij,i,j->ij", self.rho_10, self.sigma_1, self.sigma_0)

    @property
    def C11(self):
        if self.rho_11 is None:
            return None
        assert self.sigma_1 is not None

        if self.only_diag:
            return jnp.einsum("i,i->i", self.rho_11, self.sigma_1**2)

        return jnp.einsum("ij,i,j->ij", self.rho_11, self.sigma_1, self.sigma_1)

    @property
    def sigma_0_inv(self):
        return jnp.where(self.sigma_0 == 0, 0, 1 / self.sigma_0)

    @property
    def sigma_1_inv(self):
        if self.sigma_1 is None:
            return None
        return jnp.where(self.sigma_1 == 0, 0, 1 / self.sigma_1)

    # def glasso_whiten(self, choice: str):
    #     from gglasso.problem import glasso_problem
    #     import numpy as np

    #     if choice == "rho_00":
    #         rho = self.rho_00
    #         sigma_inv = self.sigma_0_inv
    #         C = self.C00
    #     elif choice == "rho_11":
    #         rho = self.rho_11
    #         sigma_inv = self.sigma_1_inv
    #         C = self.C11
    #     else:
    #         raise ValueError(f"choice {choice} not known")

    #     assert rho is not None

    #     print(f"glasso on {choice} {rho.shape=}")

    #     P = glasso_problem(
    #         S=rho.__array__(),
    #         N=self.n_eff,
    #         reg_params={"lambda1": 1e-3},
    #         latent=False,
    #         do_scaling=False,
    #     )

    #     P.solve()

    #     # lambda1_range = np.logspace(0, -3, 10)
    #     # modelselect_params = {"lambda1_range": lambda1_range}

    #     # P.model_selection(modelselect_params=modelselect_params, method="eBIC", gamma=0.1)

    #     sol = P.solution.precision_
    #     sol_l, sol_U = jnp.linalg.eigh(sol)
    #     inv_cov_glasso = sol_U @ jnp.diag(sol_l ** (0.5)) @ sol_U.T

    #     W = jnp.einsum(
    #         "ij,j->ij",
    #         inv_cov_glasso,
    #         sigma_inv,
    #     )

    #     print(f"{jnp.linalg.norm(W @ C @ W.T - jnp.eye(W.shape[0]))=}")

    #     return W

    def whiten_C(
        self,
        choice: str,
        epsilon: float = 1e-4,
        max_features: int | None = None,
        verbose=False,
        cholesky=False,
        apply_mask: bool = True,
        tikhonov=1e-6,
        return_eigh=False,
    ) -> Array:
        # returns W such that W C W.T = I and hence w.T W = C^-1

        # https://arxiv.org/pdf/1512.00809

        if choice == "rho_00":
            C = self.C00
            # sigma = self.sigma_0
        elif choice == "rho_11":
            C = self.C11
            # sigma = self.sigma_1
        else:
            raise ValueError(f"choice {choice} not known")

        assert C is not None

        if self.only_diag:
            return 1 / jnp.sqrt(C)

        C = C

        print(f"{jnp.linalg.norm(C - jnp.conj(C.T))=}")

        print(f"{epsilon=}")

        # if cholesky:
        #     print(f"cholesky")

        #     import scipy

        #     # this is pivoted cholesky

        #     if C.dtype == jnp.complex_:
        #         cho = scipy.linalg.lapack.zpstrf
        #         print("complex cholesky")
        #     else:
        #         cho = scipy.linalg.lapack.dpstrf

        #     X, P, r, info = cho(C, tol=epsilon**2, lower=True)
        #     X = jnp.array(X)

        #     # print(f"{X=}")

        #     pi = jnp.eye(P.shape[0])[:, P - 1][:, :r]
        #     X = X.at[jnp.triu_indices(X.shape[0], 1)].set(0)  # set upper half to zero
        #     X = X[:r, :][:, :r]

        #     err = jnp.linalg.norm(pi.T @ C @ pi - X @ jnp.conj(X).T)

        #     print(f" rank reduced chol {err=} rank {r} ")

        #     # P_out = P[:r]

        #     X_inv = jax.scipy.linalg.solve_triangular(
        #         X,
        #         jnp.eye(*X.shape),  # type: ignore
        #         lower=True,
        #     )

        #     W = X_inv @ pi.T

        # else:
        # else:
        theta, G = jnp.linalg.eigh(C)

        theta += tikhonov**2

        theta_inv = 1 / jnp.sqrt(theta)

        W = jnp.einsum(
            "i,ji->ij",
            # V_inv,
            theta_inv,
            G,
        )

        if apply_mask:
            if max_features is not None:
                if W.shape[0] > max_features:
                    print(f"whiten: reducing dim to {max_features=}")
                    W = W[:max_features, :]

        print(f"{jnp.linalg.norm(W @ C @ jnp.conj(W).T - jnp.eye(W.shape[0]))=}")

        if return_eigh:
            return W, theta

        return W

    def mask(
        self,
        eps_pre: float | None,
        max_features: int | None = 2000,
        auto_cov_threshold: float | None = None,
    ):
        argmask = jnp.arange(self.rho_00.shape[0])
        if eps_pre is not None:
            if self.sigma_1 is not None:
                b = jnp.logical_and(self.sigma_0 > eps_pre, self.sigma_1 > eps_pre)
            else:
                b = self.sigma_0 > eps_pre
            argmask = argmask[b]

        # print(f"{argmask.shape=}")

        if self.rho_01 is not None:
            d = jnp.diag(self.rho_01) if not self.only_diag else self.rho_01
        else:
            assert self.rho_gen is not None
            d = jnp.exp(-jnp.diag(self.rho_gen)) if not self.only_diag else jnp.exp(-self.rho_gen)

        argsort = jnp.argsort(d[argmask], descending=True)
        argmask = argmask[argsort]

        # print(f"{argmask.shape=}")

        if auto_cov_threshold is not None:
            b = d[argmask] > auto_cov_threshold
            argmask = argmask[b]

        # print(f"{argmask.shape=}")

        if max_features is not None:
            if argmask.shape[0] > max_features:
                argmask = argmask[:max_features]

        def mask_rho(rho: jax.Array | None) -> jax.Array | None:
            if rho is None:
                return None
            if self.only_diag:
                return rho[argmask]
            return rho[argmask, :][:, argmask]

        # self.rho_00 = self.rho_00[argmask, :][:, argmask]
        self.rho_00 = mask_rho(self.rho_00)

        if self.rho_11 is not None:
            # self.rho_11 = self.rho_11[argmask, :][:, argmask]
            self.rho_11 = mask_rho(self.rho_11)
        if self.rho_01 is not None:
            # self.rho_01 = self.rho_01[argmask, :][:, argmask]
            self.rho_01 = mask_rho(self.rho_01)
        if self.rho_10 is not None:
            # self.rho_10 = self.rho_10[argmask, :][:, argmask]
            self.rho_10 = mask_rho(self.rho_10)

        if self.pi_s_0 is not None:
            self.pi_s_0 = self.pi_s_0[argmask]
        if self.pi_s_1 is not None:
            self.pi_s_1 = self.pi_s_1[argmask]

        self.sigma_0 = self.sigma_0[argmask]
        if self.sigma_1 is not None:
            self.sigma_1 = self.sigma_1[argmask]

        if self.d_rho_00 is not None:
            # self.d_rho_00 = self.d_rho_00[argmask, :][:, argmask]
            self.d_rho_00 = mask_rho(self.d_rho_00)

        if self.rho_gen is not None:
            # self.rho_gen = self.rho_gen[argmask, :][:, argmask]
            self.rho_gen = mask_rho(self.rho_gen)

        return argmask

    def decompose(
        self,
        out_dim: int | None = None,
        sparse=True,
        eps: float = 1e-2,
        out_eps: float | None = None,
        glasso_whiten=False,
        verbose=True,
        apply_mask: bool = True,
        # generator=True,
        # shrink=False,
    ):
        W_0 = self.whiten_C("rho_00", epsilon=eps, apply_mask=apply_mask)

        print(f"{W_0.shape=}")

        if not self.symmetric:
            W_1 = self.whiten_C("rho_11", epsilon=eps, apply_mask=apply_mask)
        else:
            W_1 = W_0

        if self.only_diag:
            if self.generator:
                T_tilde = jnp.einsum("i,i,i->i", self.C_gen, W_0, jnp.conj(W_0))
            else:
                assert self.rho_01 is not None
                T_tilde = jnp.einsum("i,i,i->i", self.C01, W_1, jnp.conj(W_0))

        else:
            if self.generator:
                # T_tilde = W_1 @ (self.C10 - self.C00) @ W_0.T
                # T_tilde = (W_0 @ (self.C01 - self.C00) @ W_1.T).T

                T_tilde = (W_0 @ (self.C_gen) @ jnp.conj(W_0).T).T

            else:
                assert self.rho_01 is not None
                T_tilde = (W_0 @ self.C01 @ jnp.conj(W_1).T).T

        print(f"{T_tilde.shape=}")

        if out_dim is None:
            out_dim = 10

        if out_dim == -1:
            out_dim = T_tilde.shape[0]

        if out_dim < 20:
            out_dim = 20

        n_modes = out_dim

        k = min(n_modes, T_tilde.shape[0] - 1)
        if not self.only_diag:
            k = min(n_modes, T_tilde.shape[1] - 1)

        if self.only_diag:
            s = T_tilde
            idx = jnp.argsort(s, descending=True)[:n_modes]

            # W_0 = jnp.diag(W_0)[idx, :]
            # W_1 = jnp.diag(W_1)[idx, :]
            # s = s[idx]

            return idx, s

        if n_modes + 1 < T_tilde.shape[0] / 5 and sparse:
            from jax.experimental.sparse.linalg import lobpcg_standard
            from jax.random import PRNGKey, uniform

            x0 = uniform(PRNGKey(0), (T_tilde.shape[0], k))

            print(f"using lobpcg with {n_modes} modes ")

            if self.symmetric:
                # matrix should be psd

                s, U, n_iter = lobpcg_standard(
                    T_tilde.T,
                    x0,
                    m=200,
                )

                print(f"{n_iter=} {s=}")

                # VT = U.T
                V = U

            else:
                l, V, n_iter = lobpcg_standard(
                    T_tilde @ T_tilde.T,
                    x0,
                    m=200,
                )

                # VT = V.T

                print(n_iter)

                s = l ** (1 / 2)
                s_inv: jax.Array = jnp.where(s > 1e-12, 1 / s, 0)  # type: ignore
                U = T_tilde.T @ jnp.conj(VT).T @ jnp.diag(s_inv)

        else:
            if self.symmetric and W_0.shape[0] == W_1.shape[0]:
                print("using eigh")
                s, U = jax.numpy.linalg.eigh(
                    T_tilde.T,
                )
                # VT = U.T
                V = U
            else:
                print("using svd")
                U, s, VT = jax.numpy.linalg.svd(T_tilde.T)
                V = jnp.conj(VT).T

        if self.generator:
            print(f"generator, shiften spectrum {s=}")
            s = jnp.exp(-s)

        idx = jnp.argsort(s, descending=True)
        U = U[:, idx]
        s = s[idx]
        V = V[:, idx]

        W_0 = U.T @ W_0
        W_1 = V.T @ W_1

        if out_eps is not None:
            m = jnp.abs(1 - s) < out_eps

            U = U[:, m]
            VT = VT[m, :]
            s = s[m]

            print(f"{jnp.sum(m)=}")

        # W_0 = jnp.einsum("ij,j->ij", W_0, self.sigma_0_inv)
        # W_1 = jnp.einsum("ij,j->ij", W_1, self.sigma_1_inv if self.sigma_1_inv is not None else self.sigma_0_inv)

        return W_0, W_1, s

    @property
    def p(self):
        return self.rho_00.shape[0]

    @property
    def n_eff(self):
        return 1.0 / self.w2

    def shrink(self, shrinkage=None, eps_shrink: float = 1e-3):
        if self.only_diag:
            print("skip shrinkage for only_diag")
            return self

        if shrinkage is None:
            shrinkage = self.shrinkage_method
        # https://arxiv.org/pdf/1602.08776.pdf appendix b

        print(f"shrinking with {shrinkage}")

        n = self.n_eff
        assert n is not None, "C00 not provided"

        if shrinkage == "glasso":
            # https://arxiv.org/pdf/2403.02979v1#page=4.31

            from IMLCV.tools.pcglasso import pcglasso

            S = jnp.block([[self.C00, self.C01], [self.C10, self.C11]])

            d_opt, R, W = pcglasso(
                S,
                lmbda=1e-1,
                tol_newton=1e-5,
                tol_glasso=1e-5,
                tol_tot=1e-5,
                max_iter=100,
                max_iter_glasso=100,
                alpha=0.0,
                verbose=3,
                par=False,
            )

            print(f"{d_opt=} {R=} {W=}")

            raise

        # Todo https://papers.nips.cc/paper_files/paper/2014/file/11459f04a46a9e348cdeee6986fcf5f2-Paper.pdf
        # todo: paper Covariance shrinkage for autocorrelated data
        # assert shrinkage in ["RBLW", "OAS", "bidiag", "BC"]

        if self.time_series:
            p = self.C00.shape[0]

            C = jnp.block([[self.C00, self.C01], [self.C10, self.C11]])

            if shrinkage == "bidiag":
                T = jnp.block(
                    [
                        [jnp.diag(jnp.diag(self.C00)), jnp.diag(jnp.diag(self.C01))],
                        [jnp.diag(jnp.diag(self.C10)), jnp.diag(jnp.diag(self.C11))],
                    ]
                )

            elif shrinkage == "diag":
                T = jnp.block(
                    [
                        [jnp.diag(jnp.diag(self.C00)), self.C01 * 0],
                        [self.C10 * 0, jnp.diag(jnp.diag(self.C11))],
                    ]
                )
            elif shrinkage == "constant":
                T = jnp.block(
                    [
                        [jnp.mean(jnp.diag(self.C00)) * jnp.eye(self.C00.shape[0]), self.C01 * 0],
                        [self.C10 * 0, jnp.mean(jnp.diag(self.C11)) * jnp.eye(self.C11.shape[0])],
                    ]
                )
            elif shrinkage == "biconstant":
                T = jnp.block(
                    [
                        [
                            jnp.mean(jnp.diag(self.C00)) * jnp.eye(self.C00.shape[0]),
                            jnp.mean(jnp.diag(self.C01)) * jnp.eye(self.C01.shape[0]),
                        ],
                        [
                            jnp.mean(jnp.diag(self.C10)) * jnp.eye(self.C10.shape[0]),
                            jnp.mean(jnp.diag(self.C11)) * jnp.eye(self.C11.shape[0]),
                        ],
                    ]
                )
            else:
                raise NotImplementedError

        else:
            assert self.C00 is not None

            C = self.C00

            p = C.shape[0]
            if shrinkage == "bidiag":
                T = jnp.diag(jnp.diag(C))
            elif shrinkage == "diag":
                T = jnp.diag(jnp.diag(C))
            elif shrinkage == "constant":
                T = jnp.mean(jnp.diag(C)) * jnp.eye(C.shape[0])
            else:
                raise NotImplementedError

        assert self.w2 is not None
        assert self.w3 is not None

        gamma = 1.0 / (1.0 - self.w2)
        nu = 1 - self.w2 - 2 * self.w3
        eta = self.w2 + self.w2**2 - 2 * self.w3
        eps = self.w2

        C = C / gamma  # with bessel correction, C = gamma S

        # Shrinkage MMSE estimators of covariances beyond the zero-mean and stationary variance assumptions

        def get_C(lambd):
            return (1 - lambd) * C + lambd * T

        def get_lambd(C):
            tr_CS = jnp.trace(C @ C.T) - jnp.sum(jnp.diag(C) * jnp.diag(C))
            tr_CS_diag = jnp.trace(C) * jnp.trace(C) - jnp.sum(jnp.diag(C) * jnp.diag(C))

            return ((gamma * nu + eps - 1) * tr_CS + gamma * eta * tr_CS_diag) / (
                (gamma * nu) * tr_CS + gamma * eta * tr_CS_diag
            )

        lambd = 0

        for i in range(10):
            C_est = get_C(lambd)

            lambd_new = get_lambd(C_est)

            if jnp.abs(lambd - lambd_new) < 1e-6:
                break

            lambd = lambd_new

            print(f"{i}: {lambd=}")

        print(f"{i}: {lambd=}, finished")

        if lambd > 1:
            lambd = 1

        # figure out new C matrix

        def get_rho(i, j):
            X = C_est
            if i == 0:
                X = X[:p, :]
                sigma_0_inv = self.sigma_0_inv
            else:
                X = X[p:, :]
                sigma_0_inv = self.sigma_1_inv

            if j == 0:
                X = X[:, :p]
                sigma_1_inv = self.sigma_0_inv
            else:
                X = X[:, p:]
                sigma_1_inv = self.sigma_1_inv

            # print(f"{X.shape=} {sigma_0_inv.shape=}")

            return jnp.einsum("i,ij,j->ij", sigma_0_inv, X, sigma_1_inv)

        if not self.time_series:
            return Covariances(
                rho_00=get_rho(0, 0),
                pi_s_0=self.pi_s_0,
                pi_s_1=None,
                sigma_0=self.sigma_0,
                sigma_1=None,
                only_diag=self.only_diag,
                rho_01=None,
                rho_11=None,
                rho_10=None,
                rho_gen=self.rho_gen,
                trans_f=self.trans_f,
                trans_g=self.trans_g,
                symmetric=self.symmetric,
                shrinkage_method=shrinkage,
                d_rho_00=self.d_rho_00,
                w2=self.w2,
                w3=self.w3,
            )

        return Covariances(
            rho_00=get_rho(0, 0),
            rho_01=get_rho(0, 1),
            rho_11=get_rho(1, 1),
            pi_s_0=self.pi_s_0,
            pi_s_1=self.pi_s_1,
            rho_10=get_rho(1, 0),
            rho_gen=self.rho_gen,
            sigma_0=self.sigma_0,
            sigma_1=self.sigma_1,
            only_diag=self.only_diag,
            trans_f=self.trans_f,
            trans_g=self.trans_g,
            symmetric=self.symmetric,
            # n=self.n,
            shrinkage_method=shrinkage,
            d_rho_00=self.d_rho_00,
            w2=self.w2,
            w3=self.w3,
        )

    def symmetrize(self):
        if not self.time_series:
            return self

        _rho_00 = self.rho_00
        _rho_01 = self.rho_01
        _rho_10 = self.rho_10
        _rho_11 = self.rho_11
        _pi_s_0 = self.pi_s_0
        _pi_s_1 = self.pi_s_1

        _d_rho_00 = self.d_rho_00

        # make sigma same

        _sigma = jnp.sqrt(self.sigma_0 * self.sigma_1)

        d_sigma_0 = jnp.where(_sigma != 0, self.sigma_0 / _sigma, 0)
        d_sigma_1 = jnp.where(_sigma != 0, self.sigma_1 / _sigma, 0)

        # print(f"{d_sigma_0=}")

        def get_rho(rho, f1, f2) -> jax.Array:
            if self.only_diag:
                return jnp.einsum("i,i->i", rho, jnp.conj(f1) * f2)
            return jnp.einsum("ij,i,j->ij", rho, jnp.conj(f1), f2)

        _rho_00 = get_rho(_rho_00, d_sigma_0, d_sigma_0)
        _rho_01 = get_rho(_rho_01, d_sigma_0, d_sigma_1)
        _rho_10 = get_rho(_rho_10, d_sigma_1, d_sigma_0)
        _rho_11 = get_rho(_rho_11, d_sigma_1, d_sigma_1)

        if self.d_rho_00 is not None:
            # _d_rho_00 = jnp.einsum("ij,i,j->ij", _d_rho_00, d_sigma_0, d_sigma_0)
            _d_rho_00 = get_rho(_d_rho_00, d_sigma_0, d_sigma_0)

        # make pi same

        calc_pi = self.pi_s_0 is not None

        def get_outer(pi1, pi2):
            if self.only_diag:
                return jnp.einsum("i,i->i", pi1, pi2)
            return jnp.einsum("i,j->ij", pi1, pi2)

        if calc_pi:
            assert self.pi_s_1 is not None
            _pi_s_0 = jnp.einsum("i,i->i", _pi_s_0, d_sigma_0)
            _pi_s_1 = jnp.einsum("i,i->i", _pi_s_1, d_sigma_1)

            _pi_s = 0.5 * (_pi_s_0 + _pi_s_1)

            _rho_00 = _rho_00 + get_outer(_pi_s_0, _pi_s_0) - get_outer(_pi_s, _pi_s)
            _rho_11 = _rho_11 + get_outer(_pi_s_1, _pi_s_1) - get_outer(_pi_s, _pi_s)
            _rho_01 = _rho_01 + get_outer(_pi_s_0, _pi_s_1) - get_outer(_pi_s, _pi_s)
            _rho_10 = _rho_10 + get_outer(_pi_s_1, _pi_s_0) - get_outer(_pi_s, _pi_s)

        sym_rho_00 = (1 / 2) * (_rho_00 + _rho_11)
        sym_rho_01 = (1 / 2) * (_rho_01 + _rho_10.T)

        d_sigma = jnp.sqrt(jnp.diag(sym_rho_00)) if not self.only_diag else jnp.sqrt(sym_rho_00)
        d_sigma_inv = jnp.where(d_sigma != 0, 1 / d_sigma, 0)

        # sym_rho_00 = jnp.einsum("ij,i,j->ij", sym_rho_00, d_sigma_inv, d_sigma_inv)
        # sym_rho_01 = jnp.einsum("ij,i,j->ij", sym_rho_01, d_sigma_inv, d_sigma_inv)
        sym_rho_00 = get_rho(sym_rho_00, d_sigma_inv, d_sigma_inv)
        sym_rho_01 = get_rho(sym_rho_01, d_sigma_inv, d_sigma_inv)

        if calc_pi:
            _pi_s = jnp.einsum("i,i->i", _pi_s, d_sigma)
        else:
            _pi_s = None

        _sigma *= d_sigma

        return Covariances(
            rho_00=sym_rho_00,
            rho_11=sym_rho_00,
            rho_01=sym_rho_01,
            rho_10=sym_rho_01,
            rho_gen=self.rho_gen,
            pi_s_0=_pi_s,
            pi_s_1=_pi_s,
            sigma_0=_sigma,
            sigma_1=_sigma,
            only_diag=self.only_diag,
            symmetric=True,
            trans_f=self.trans_f,
            trans_g=self.trans_g,
            # n=self.n,
            shrinkage_method=self.shrinkage_method,
            bc=self.bc,
            d_rho_00=_d_rho_00,
            w2=self.w2,
            w3=self.w3,
        )

    @staticmethod
    def add(cov1: Covariances, cov2: Covariances, frac=0.5):
        w1 = frac
        w2 = 1 - frac

        def get_rho(rho, f1, f2, only_diag) -> jax.Array:
            if only_diag:
                return jnp.einsum("i,i->i", rho, jnp.conj(f1) * f2)
            return jnp.einsum("ij,i,j->ij", rho, jnp.conj(f1), f2)

        def get_outer(pi1, pi2, only_diag):
            if only_diag:
                return jnp.einsum("i,i->i", pi1, pi2)
            return jnp.einsum("i,j->ij", pi1, pi2)

        # 1. Bring everything to absolute scale (scale by sigma)
        # pi_abs = pi * sigma
        pi1_0 = cov1.pi_s_0 * cov1.sigma_0 if cov1.pi_s_0 is not None else 0
        pi1_1 = cov1.pi_s_1 * cov1.sigma_1 if cov1.pi_s_1 is not None else 0
        pi2_0 = cov2.pi_s_0 * cov2.sigma_0 if cov2.pi_s_0 is not None else 0
        pi2_1 = cov2.pi_s_1 * cov2.sigma_1 if cov2.pi_s_1 is not None else 0

        # 2. Compute the new weighted mean (scaled)
        new_pi_abs_0 = w1 * pi1_0 + w2 * pi2_0
        new_pi_abs_1 = w1 * pi1_1 + w2 * pi2_1

        # 3. Combine Second Moments
        # Formula: M2 = w1*(rho1_scaled + pi1*pi1') + w2*(rho2_scaled + pi2*pi2')
        def combine_component(r1, s1_a, s1_b, p1_a, p1_b, r2, s2_a, s2_b, p2_a, p2_b):
            # Scale rho to absolute covariance terms
            term1 = get_rho(r1, s1_a, s1_b, cov1.only_diag) + get_outer(p1_a, p1_b, cov1.only_diag)
            term2 = get_rho(r2, s2_a, s2_b, cov1.only_diag) + get_outer(p2_a, p2_b, cov1.only_diag)
            return w1 * term1 + w2 * term2

        m2_00 = combine_component(
            cov1.rho_00, cov1.sigma_0, cov1.sigma_0, pi1_0, pi1_0, cov2.rho_00, cov2.sigma_0, cov2.sigma_0, pi2_0, pi2_0
        )
        m2_11 = combine_component(
            cov1.rho_11, cov1.sigma_1, cov1.sigma_1, pi1_1, pi1_1, cov2.rho_11, cov2.sigma_1, cov2.sigma_1, pi2_1, pi2_1
        )
        m2_01 = combine_component(
            cov1.rho_01, cov1.sigma_0, cov1.sigma_1, pi1_0, pi1_1, cov2.rho_01, cov2.sigma_0, cov2.sigma_1, pi2_0, pi2_1
        )
        m2_10 = combine_component(
            cov1.rho_10, cov1.sigma_1, cov1.sigma_0, pi1_1, pi1_0, cov2.rho_10, cov2.sigma_1, cov2.sigma_0, pi2_1, pi2_0
        )

        # 4. Subtract the new mean to get centered covariance
        res_00 = m2_00 - get_outer(new_pi_abs_0, new_pi_abs_0, cov1.only_diag)
        res_11 = m2_11 - get_outer(new_pi_abs_1, new_pi_abs_1, cov1.only_diag)
        res_01 = m2_01 - get_outer(new_pi_abs_0, new_pi_abs_1, cov1.only_diag)
        res_10 = m2_10 - get_outer(new_pi_abs_1, new_pi_abs_0, cov1.only_diag)

        # 5. Extract new sigma and normalize back to rho
        new_sigma_0 = jnp.sqrt(jnp.diag(res_00)) if not cov1.only_diag else jnp.sqrt(res_00)
        new_sigma_1 = jnp.sqrt(jnp.diag(res_11)) if not cov1.only_diag else jnp.sqrt(res_11)

        # Avoid division by zero
        inv_sigma_0 = jnp.where(new_sigma_0 != 0, 1.0 / new_sigma_0, 0.0)
        inv_sigma_1 = jnp.where(new_sigma_1 != 0, 1.0 / new_sigma_1, 0.0)

        new_rho_00 = get_rho(res_00, inv_sigma_0, inv_sigma_0, cov1.only_diag)
        new_rho_11 = get_rho(res_11, inv_sigma_1, inv_sigma_1, cov1.only_diag)
        new_rho_01 = get_rho(res_01, inv_sigma_0, inv_sigma_1, cov1.only_diag)
        new_rho_10 = get_rho(res_10, inv_sigma_1, inv_sigma_0, cov1.only_diag)

        # 6. Final mean normalization (scaled back)
        final_pi_0 = new_pi_abs_0 * inv_sigma_0 if cov1.pi_s_0 is not None else None
        final_pi_1 = new_pi_abs_1 * inv_sigma_1 if cov1.pi_s_1 is not None else None

        # Handle derivative mixing if applicable
        new_d_rho_00 = None
        if cov1.d_rho_00 is not None and cov2.d_rho_00 is not None:
            # Basic linear mix for derivatives
            new_d_rho_00 = w1 * cov1.d_rho_00 + w2 * cov2.d_rho_00

        return Covariances(
            rho_00=new_rho_00,
            rho_11=new_rho_11,
            rho_01=new_rho_01,
            rho_10=new_rho_10,
            pi_s_0=final_pi_0,
            pi_s_1=final_pi_1,
            sigma_0=new_sigma_0,
            sigma_1=new_sigma_1,
            rho_gen=cov1.rho_gen,
            only_diag=cov1.only_diag,
            symmetric=cov1.symmetric and cov2.symmetric,
            trans_f=cov1.trans_f,
            trans_g=cov1.trans_g,
            shrinkage_method=cov1.shrinkage_method,
            bc=None,
            d_rho_00=new_d_rho_00,
            w2=(w1 * cov1.w2 + w2 * cov2.w2) if cov1.w2 is not None and cov2.w2 is not None else None,
            w3=(w1 * cov1.w3 + w2 * cov2.w3) if cov1.w3 is not None and cov2.w3 is not None else None,
        )
