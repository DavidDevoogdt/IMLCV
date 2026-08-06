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
from IMLCV.base.UnitsConstants import amu, angstrom, nanosecond

X = TypeVar("X", "CV", "SystemParams", "NeighbourList")
X2 = TypeVar("X2", "CV", "SystemParams", "NeighbourList")


def safe_inv(x, power: int = 1, fill: float = 0.0, cond=None):
    """Reciprocal with NaN-safe GRADIENTS at x == 0 (double-where trick).

    jnp.where(x == 0, fill, 1/x) is forward-safe but its backward pass
    computes d(1/x)/dx = -inf at x == 0 and multiplies it by the zero
    cotangent routed by where: 0 * inf = NaN.  Sanitizing the denominator
    BEFORE the reciprocal means the inf is never created, so gradients stay
    finite (and are exactly 0 for the masked entries)."""
    if cond is None:
        cond = x != 0
    x_safe = jnp.where(cond, x, 1.0)
    return jnp.where(cond, 1.0 / x_safe**power, fill)


class BendParams(MyPyTreeNode):
    n_probe: int = field(default=4, pytree_node=False)
    ridge: float = 1e-4
    hinge_p: int = 2
    mode: str = field(default="full", pytree_node=False)  # "full" or "projected"
    reduce: str = field(default="hinge", pytree_node=False)  # "trace", "frobenius", "hinge"
    R0: float = 1.0  # reference radius of curvature for hinge loss, in units of gradient length
    W: float = 1.0  # barstat weight, amounts to (Nf+d)k_B T tau_B^2 * ( tau_thermo/tau_baro) (last term is converserion to mass weighted units without thermostat constant)
    beta: float = 5.0  # softness of hinge loss


class CovarianceData(MyPyTreeNode):
    rho_00: jax.Array | None = None
    rho_01: jax.Array | None = None
    rho_10: jax.Array | None = None
    rho_11: jax.Array | None = None

    rho_gen: jax.Array | None = None
    rho_gen_sq: jax.Array | None = None
    rho_00_sq: jax.Array | None = None
    rho_bend: jax.Array | None = None

    pi_s_0: jax.Array | None = None
    pi_s_1: jax.Array | None = None

    sigma_0: jax.Array | None = None
    sigma_1: jax.Array | None = None

    w0: jax.Array | None = None
    w1: jax.Array | None = None

    key: jax.Array = field(default_factory=lambda: jax.random.PRNGKey(0))

    bp: BendParams = field(default_factory=BendParams)

    reduction_cv: CV | None = None


class Covariances(MyPyTreeNode):
    rho_00: jax.Array
    rho_01: jax.Array | None
    rho_10: jax.Array | None
    rho_11: jax.Array | None

    rho_gen: jax.Array | None = None
    rho_bend: jax.Array | None = None

    rho_gen_sq: jax.Array | None = None
    rho_00_sq: jax.Array | None = None

    pi_s_0: jax.Array | None
    pi_s_1: jax.Array | None
    pi_s_01: jax.Array | None = None
    sigma_0: jax.Array
    sigma_1: jax.Array | None

    bc: list[jax.Array] | None = None  # used in BC estimator
    shrinkage_method: str = field(default="bidiag", pytree_node=False)

    time_series: bool = True

    W_0: jax.Array | None = None
    W_1: jax.Array | None = None

    only_diag: bool = field(default=False, pytree_node=False)
    trans_f: CvTrans | CvTrans | None = None
    trans_g: CvTrans | CvTrans | None = None
    symmetric: bool = field(default=False, pytree_node=False)
    pi_argmask: Array | None = None

    w2: float | None = None
    w3: float | None = None

    scale_deriv: float | None = None

    reduction_cv: CV | None = None

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
        symmetric=False,
        calc_C00=True,
        calc_C01=True,
        calc_C10=True,
        calc_C11=True,
        shmap_kwargs=ShmapKwargs.create(),
        verbose=True,
        shrink=True,
        shrinkage_method="bidiag",
        pi_argmask: Array | None = None,
        get_diff=False,
        bessel_correct=True,
        generator=False,
        eps_shrink=1e-3,
        get_hess=False,
        scale_deriv=1.0 * jnp.sqrt(amu) * angstrom,
        curvature=True,
        sigma_0: Array | None = None,
        key: jax.Array = jax.random.PRNGKey(0),
        bp: BendParams = BendParams(),
        gen_squared=False,
        cov_squared=False,
        reduction_fun: CvTrans | None = None,
    ) -> Covariances:
        time_series = cv_1 is not None

        if not time_series:
            if get_diff:
                print("cannot get differences without time series, setting to False")
                get_diff = False

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

        if shrink:
            print(f"shrinking {shrinkage_method =} ")

        BC = shrinkage_method == "BC"

        print(f"{nl=}")

        def cov_pi(
            carry: CovarianceData,
            cv_0: CV,
            cv_1: CV | None,
            w: jax.Array | None,
            w_t: jax.Array | None = None,
            diff_w: jax.Array | None = None,
        ):
            assert w is not None

            rho_00_prev = carry.rho_00
            rho_01_prev = carry.rho_01
            rho_10_prev = carry.rho_10
            rho_11_prev = carry.rho_11
            rho_gen_prev = carry.rho_gen
            rho_bend = carry.rho_bend
            rho_gen_sq_prev = carry.rho_gen_sq

            reduction_cv_prev = carry.reduction_cv

            rho_00_sq_prev = carry.rho_00_sq

            pi_0_prev = carry.pi_s_0
            pi_1_prev = carry.pi_s_1

            sigma_0_prev = carry.sigma_0
            sigma_1_prev = carry.sigma_1

            w_prev = carry.w0
            w_prev_t = carry.w1

            key = carry.key

            bp = carry.bp

            if sigma_0_prev is not None:
                sigma_0_inv = safe_inv(sigma_0_prev, fill=1.0)

            if sigma_1_prev is not None:
                sigma_1_inv = safe_inv(sigma_1_prev, fill=1.0)

            if (trans_f is not None) and ((trans_g is not None) or (not time_series)) and (generator or get_hess):
                if nl is not None:
                    _, nl_0 = nl.update_nl(cv_0)
                else:
                    nl_0 = None

                spm: SystemParams = cv_0
                has_cell = spm.cell is not None
                tau_thermo_over_tau_baro = 0.2  # tau_T / tau_p

                # phi is differentiated, so it must stay REAL-valued: whiten with
                # sigma pushed back to the real representation (a folded pair
                # shares one sigma, see unfold_scale)
                if sigma_0_prev is not None:
                    s0_inv_real = sigma_0_inv

                def phi(x, nl_x):
                    cv, _ = trans_f.compute_cv(x, nl_x, shmap=False)
                    if sigma_0_prev is not None:
                        cv = cv.replace(cv=cv.cv * s0_inv_real)
                    return cv

                masses = nl_0.info.masses
                sqrt_inv_mass = jnp.where(masses == 0, 0.0, scale_deriv / jnp.sqrt(masses))

                jacobian = jax.vmap(jax.jacrev(phi))(spm, nl_0).cv
                jac_r = jacobian.coordinates  # (n_batch, n_cv, n_atom, 3)
                jac_r_w = jnp.einsum("nair, i -> nair", jac_r, sqrt_inv_mass)  # M^{-1/2} J_r

                if has_cell:
                    coords = spm.coordinates  # (n_batch, n_atom, 3)
                    cell = spm.cell  # (n_batch, 3, 3), rows = lattice vectors
                    s = jnp.einsum("nir, nrk -> nik", coords, jnp.linalg.inv(cell))  # fractional coords  x h^{-1}

                    # cell vectors as pseudo-atoms; affine gradient  dphi/dh|_s = dphi/dh|_x + sum_a s_a dphi/dr_a
                    jac_h = jacobian.cell + jnp.einsum("nik, nair -> nakr", s, jac_r)  # (n_batch, n_cv, 3, 3)

                    # scalar cell mass  W = (tau_T/tau_p) Mtot / 12   (uniform-density fractional gyration)
                    W = 1 / tau_thermo_over_tau_baro * masses.sum() / 12.0
                    sqrt_inv_cell = scale_deriv / jnp.sqrt(W)
                    jac_h_w = sqrt_inv_cell * jac_h
                else:
                    s = jnp.zeros_like(spm.coordinates)
                    jac_h_w = jnp.zeros((jac_r_w.shape[0], jac_r_w.shape[1], 3, 3))
                    sqrt_inv_cell = 0.0

                # --- metric ---
                if only_diag:
                    G = jnp.einsum("nair, nair -> na", jac_r_w, jac_r_w)
                    if has_cell:
                        G = G + jnp.einsum("nakr, nakr -> na", jac_h_w, jac_h_w)
                else:
                    G = jnp.einsum("nair, nbir -> nab", jac_r_w, jac_r_w)
                    if has_cell:
                        G_cell = jnp.einsum("nakr, nbkr -> nab", jac_h_w, jac_h_w)
                        G = G + G_cell

                if only_diag:
                    dL = jnp.einsum("na, n -> a", G, w)
                else:
                    dL = jnp.einsum("nab, n -> ab", G, w)

                if gen_squared:
                    # |G|^2 elementwise; real by construction, complex or not
                    if only_diag:
                        dL_sq = jnp.real(jnp.einsum("na,na, n -> a", jnp.conj(G), G, w))
                    else:
                        dL_sq = jnp.real(jnp.einsum("nab,nab, n -> ab", jnp.conj(G), G, w))

                if get_hess:

                    def hvp(x, v, nl_x):
                        # forward-over-reverse, seed-independent linearization done once
                        def phi_cv(y):
                            return phi(y, nl_x).cv  # (n_cv,)

                        def jvp_dir(y):
                            _, Jv = jax.jvp(phi_cv, (y,), (v,))  # (n_cv,), linear in v
                            return Jv

                        # jacrev of (x -> J v) gives d/dx (J v) = (H^{(c)} v)_c stacked  -> (n_cv, *x)
                        return jax.jacrev(jvp_dir)(x)

                    def frame_curvature(
                        jac_r_frame,
                        jac_h_frame,
                        s_frame,
                        sp_x,
                        nl_x,
                        key,
                    ):
                        """Per-frame curvature penalty, computed in the REAL
                        representation (the honest local geometry); the fold to
                        complex happens on the frame-averaged form, which is
                        legitimate because it is linear.
                        mode   :    "full"      -> full-Hessian Frobenius Gram via Hutchinson (all directions)
                                    "projected" -> Hessian restricted to the CV-gradient subspace
                        reduce :    "trace"     -> sum_i lambda_i      (curvature^2,  cheap, no eigh)
                                    "frobenius" -> sum_i lambda_i^2    (curvature^4,  cheap, no eigh)
                                    "hinge"     -> sum_i relu(bend_i - 1)^p   (thresholded radii)
                        lambda_i = eig(G^{-1} C) are the reparam-invariant local curvatures (units 1/length^2).
                        """
                        n_cv = jac_r_frame.shape[0]

                        # ---- metric  G_ab = <grad phi_a, grad phi_b>  (mass + cell weighted) ----
                        G = jnp.einsum("air, bir -> ab", jac_r_frame, jac_r_frame)
                        if has_cell:
                            G = G + jnp.einsum("akr, bkr -> ab", jac_h_frame, jac_h_frame)

                        if sigma_0_prev is not None:
                            eye = jnp.diag(jnp.real(s0_inv_real) ** 2)
                        else:
                            eye = jnp.eye(n_cv)

                        G_safe = G + bp.ridge * eye

                        L = jnp.linalg.cholesky(G_safe)  # G_safe = L L^T

                        # ---- Hessian-vector product of all CVs along an (atoms, cell) direction ----
                        def hvp_dir(v_r, v_h):
                            if has_cell:
                                dr = jnp.einsum("ik, kr -> ir", s_frame, v_h)
                                u = sp_x.replace(coordinates=v_r + dr, cell=v_h)
                            else:
                                u = sp_x.replace(coordinates=v_r, cell=None)
                            hv = hvp(sp_x, u, nl_x)
                            g_r = hv.coordinates * sqrt_inv_mass[None, :, None]
                            g_h = (
                                sqrt_inv_cell * (hv.cell + jnp.einsum("ik, cir -> ckr", s_frame, hv.coordinates))
                                if has_cell
                                else jnp.zeros((n_cv, 3, 3))
                            )
                            return g_r, g_h

                        # ---- curvature Gram  C_cd  (PSD, (2,0) over output CVs) ----
                        if bp.mode == "full":

                            def probe(kp):
                                kr, kh = jax.random.split(kp)
                                z_r = jax.random.rademacher(kr, sp_x.coordinates.shape, dtype=G.dtype)
                                v_r = z_r * sqrt_inv_mass[:, None]  # M^{-1/2} z
                                v_h = (
                                    sqrt_inv_cell * jax.random.rademacher(kh, (3, 3), dtype=G.dtype)
                                    if has_cell
                                    else jnp.zeros((3, 3))
                                )
                                return hvp_dir(v_r, v_h)

                            g_r, g_h = jax.vmap(probe)(jax.random.split(key, bp.n_probe))
                            C = jnp.einsum("pcir, pdir -> cd", g_r, g_r)
                            if has_cell:
                                C = C + jnp.einsum("pckr, pdkr -> cd", g_h, g_h)
                            C = C / bp.n_probe
                        elif bp.mode == "projected":

                            def beta(b):
                                v_r = jac_r_frame[b] * sqrt_inv_mass[:, None]  # M^{-1} grad_b
                                v_h = sqrt_inv_cell * jac_h_frame[b] if has_cell else jnp.zeros((3, 3))
                                return hvp_dir(v_r, v_h)

                            g_r, g_h = jax.vmap(beta)(jnp.arange(n_cv))
                            Htil = jnp.einsum("air, bcir -> abc", jac_r_frame, g_r)  # <grad_a, Hess_c grad_b>
                            if has_cell:
                                Htil = Htil + jnp.einsum("akr, bckr -> abc", jac_h_frame, g_h)
                            Htil = 0.5 * (Htil + jnp.swapaxes(Htil, 0, 1))  # symmetric in (a,b)
                            Mc = jnp.linalg.solve(G_safe, Htil.reshape(n_cv, -1)).reshape(n_cv, n_cv, n_cv)
                            C = jnp.einsum("abc, bad -> cd", Mc, Mc)  # tr(G^{-1} H~^c G^{-1} H~^d)
                        else:
                            raise ValueError(f"Unknown bend mode {bp.mode}")

                        Y = jax.scipy.linalg.solve_triangular(L, C, lower=True)  # L^{-1} C
                        S = jax.scipy.linalg.solve_triangular(L, Y.T, lower=True).T  # L^{-1} C L^{-T}
                        S = 0.5 * (S + S.T)  # symmetric PSD, eig = lambda_i

                        if bp.reduce == "trace":
                            out = S
                        elif bp.reduce == "frobenius":
                            out = S @ S
                        elif bp.reduce == "hinge":
                            lam, Vhat = jnp.linalg.eigh(S)
                            Vhat = jax.lax.stop_gradient(Vhat)  # directions = relevance gate, no grad
                            h = (
                                jax.nn.softplus(bp.beta * (lam * bp.R0**2 - 1.0)) / bp.beta
                            ) ** bp.hinge_p  # grad via lam
                            P = jax.lax.stop_gradient(L @ Vhat)
                            out = (P * h[None, :]) @ P.T  # = L Vhat diag(h) Vhat^T L^T  ->  covariant B
                        else:
                            raise ValueError(f"Unknown bend reduction {bp.reduce}")

                        return G, out

                    key, subkey = jax.random.split(key)

                    G_batch, H_batch = jax.vmap(frame_curvature, in_axes=(0, 0, 0, 0, 0, None))(
                        jac_r_w, jac_h_w, s, spm, nl_0, subkey
                    )

                    dH = jnp.einsum("nij, n ->ij ", H_batch, w)

                # Standard tracking variables return cleanly un-warped
                cv_0 = jax.vmap(phi)(spm, nl_0)

                if cv_1 is not None:
                    spm1 = cv_1
                    nl_1 = nl.update_nl(cv_1)[1]
                    cv_1 = jax.vmap(phi)(spm1, nl_1)

                x_0 = cv_0.cv
                x_1 = cv_1.cv if cv_1 is not None else None

            else:
                assert not cv_0.atomic

                x_0 = cv_0.cv

                if cv_1 is not None:
                    x_1 = cv_1.cv
                else:
                    x_1 = None

                if sigma_1_prev is not None:
                    assert x_1 is not None
                    x_1 = jnp.einsum("ni,i->ni", x_1, sigma_1_inv)

                if sigma_0_prev is not None:
                    x_0 = jnp.einsum("ni,i->ni", x_0, sigma_0_inv)

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
                    _pi_tot = _pi_tot.at[pi_argmask].set(0.0)
                    if _dpi is not None:
                        _dpi = _dpi.at[pi_argmask].set(0.0)

                return _pi_tot, _dpi, _w_tot, _dw

            pi_0_new, dpi_0, w_tot, dw = get_pi(x_0, w, pi_0_prev, w_prev, calc_pi)

            if time_series:
                pi_1_new, dpi_1, w_tot_t, dw_t = get_pi(x_1, w_t, pi_1_prev, w_prev_t, calc_pi)

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
                    # Optimal Data-Driven Estimation of Generalized Markov State Models
                    # for Non-Equilibrium Dynamics, sampled under w_dyn
                    rho_01 = c(
                        x_0,
                        x_1,
                        w,
                        rho_01_prev,
                        pi_0_new,
                        pi_1_new,
                        dpi_0,
                        dpi_1,
                        w_prev,
                        dw,
                    )

                if calc_C10:
                    # rho_10[i,j] = E[conj(x1_i) x0_j] = conj(rho_01[j,i]);
                    # plain .T would be wrong once the features are complex
                    rho_10 = jnp.conj(rho_01) if only_diag else jnp.conj(rho_01).T

                if calc_C11:
                    rho_11 = c(x_1, x_1, w_t, rho_11_prev, pi_1_new, pi_1_new, dpi_1, dpi_1, w_prev_t, dw_t)

            if generator:
                if rho_gen_prev is not None:
                    rho_gen = (rho_gen_prev * w_prev + dL) / (w_prev + dw)
                else:
                    rho_gen = dL / dw
            else:
                rho_gen = None

            if gen_squared:
                if rho_gen_sq_prev is not None:
                    rho_gen_sq = (rho_gen_sq_prev * w_prev + dL_sq) / (w_prev + dw)
                else:
                    rho_gen_sq = dL_sq / dw
            else:
                rho_gen_sq = rho_gen_sq_prev

            if reduction_fun is not None:
                if reduction_cv_prev is not None:
                    cv_red, _ = reduction_fun.compute_cv(cv_0, nl=None, shmap=False)

                    dra = jnp.einsum("n...,n->...", cv_red.cv, w)  # E[phi_red]

                    reduction_cv = (reduction_cv_prev * w_prev + dra) / (w_prev + dw)

                    reduction_cv = cv_red.replace(cv=reduction_cv)

                else:
                    cv_red, _ = reduction_fun.compute_cv(cv_0, nl=None, shmap=False)

                    print(f"{cv_red=}")

                    dra = jnp.einsum("n...,n->...", cv_red.cv, w)  # E[phi_red]

                    reduction_cv = dra / dw

                    reduction_cv = cv_red.replace(cv=reduction_cv)
            else:
                reduction_cv = reduction_cv_prev

            if get_hess:
                if rho_bend is not None:
                    rho_bend = (rho_bend * w_prev + dH) / (w_prev + dw)
                else:
                    rho_bend = dH / dw

            if cov_squared:
                x0_sq = jnp.real(x_0 * jnp.conj(x_0))  # |x|^2, = x^2 for real cv

                if only_diag:
                    drho_00_sq = jnp.einsum("ni,ni,n->i", x0_sq, x0_sq, w)  # E[|x_i|^4]
                else:
                    drho_00_sq = jnp.einsum("ni,nj,n->ij", x0_sq, x0_sq, w)  # E[|x_i|^2 |x_j|^2]

                if rho_00_sq_prev is not None:
                    rho_00_sq = (rho_00_sq_prev * w_prev + drho_00_sq) / (w_prev + dw)

                else:
                    rho_00_sq = drho_00_sq / dw

            else:
                rho_00_sq = rho_00_sq_prev

            # hermitian diagonals are real; the cast keeps the comparison below
            # (and hence sigma) real-valued for complex features
            diag_rho_00 = jnp.real(jnp.diag(rho_00) if not only_diag else rho_00)

            d_sigma_0 = jnp.sqrt(jnp.where(diag_rho_00 <= 0, 0.0, diag_rho_00))
            f0 = safe_inv(d_sigma_0, fill=1.0)

            pi_0_new = jnp.einsum("i,i->i", pi_0_new, f0) if calc_pi else None

            def get_rho(rho, f_left, f_right):
                if only_diag:
                    return jnp.einsum("i,i,i->i", rho, jnp.conj(f_left), f_right)
                return jnp.einsum("ij,i,j->ij", rho, jnp.conj(f_left), f_right)

            rho_00 = get_rho(rho_00, f0, f0)

            if generator:
                rho_gen = get_rho(rho_gen, f0, f0)

            if gen_squared:
                rho_gen_sq = get_rho(rho_gen_sq, f0**2, f0**2)

            if cov_squared:
                # E[x^4] carries sigma^2 per index of the pair matrix, E[x^3] carries sigma^3
                rho_00_sq = get_rho(rho_00_sq, f0**2, f0**2)

            # rho_bend doesn't scale with sigma (curvature form already whitened internally)

            if sigma_0_prev is None:
                sigma_0 = d_sigma_0
            else:
                sigma_0 = jnp.where(sigma_0_prev == 0, d_sigma_0, sigma_0_prev * d_sigma_0)

            if time_series:
                diag_rho_11 = jnp.real(jnp.diag(rho_11) if not only_diag else rho_11)

                d_sigma_1 = jnp.sqrt(jnp.where(diag_rho_11 <= 0, 0.0, diag_rho_11))  # type: ignore
                f1 = safe_inv(d_sigma_1, fill=1.0, cond=d_sigma_1 > 0)

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

            return CovarianceData(
                rho_00=rho_00,
                rho_01=rho_01,
                rho_10=rho_10,
                rho_11=rho_11,
                rho_gen=rho_gen,
                rho_gen_sq=rho_gen_sq,
                rho_00_sq=rho_00_sq,
                pi_s_0=pi_0_new,
                pi_s_1=pi_1_new if time_series else None,
                sigma_0=sigma_0,
                sigma_1=sigma_1 if time_series else None,
                w0=w_tot,
                w1=w_tot_t if time_series else None,
                key=key,
                rho_bend=rho_bend,
                bp=bp,
                reduction_cv=reduction_cv,
            )

        if trans_f is not None:

            def f_func(x: X, nl: NeighbourList | None) -> CV:
                if not generator:
                    x, _ = trans_f.compute_cv(x, nl, chunk_size=chunk_size, shmap=False)

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
                return x

        g_func = cast(Callable[[CV | SystemParams, NeighbourList | None], CV], g_func)

        chunk_func_init_args = CovarianceData(
            sigma_0=sigma_0,
            sigma_1=sigma_0,
            key=key,
            bp=bp,
        )

        out = macro_chunk_map_fun(
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

        rho_00 = out.rho_00
        rho_01 = out.rho_01
        rho_10 = out.rho_10
        rho_11 = out.rho_11
        rho_gen = out.rho_gen
        rho_bend = out.rho_bend

        rho_gen_sq = out.rho_gen_sq
        rho_00_sq = out.rho_00_sq

        reduction_cv = out.reduction_cv

        pi_s_0 = out.pi_s_0
        pi_s_1 = out.pi_s_1

        sigma_0 = out.sigma_0
        sigma_1 = out.sigma_1

        _w = out.w0
        _wt = out.w1

        assert rho_00 is not None
        assert sigma_0 is not None

        # bessel correction: https://univ-lyon1.hal.science/hal-04693522/file/shrinkage_eusipco_2024_reviewed.pdf

        w_stack = jnp.hstack(w)

        w2 = jnp.sum(w_stack**2) / (jnp.sum(w_stack) ** 2)
        w3 = jnp.sum(w_stack**3) / (jnp.sum(w_stack) ** 3)

        eps = w2

        if bessel_correct:
            print(f"bessel correction {eps=} {1/eps=}  {w_stack.shape[0]=} ")

        # C -> (1-eps) C  is absorbed by inflating sigma:
        #   sigma -> gamma sigma  with gamma = 1/sqrt(1-eps)
        #   rho   : stays the same (it IS the corrected covariance in sigma units)
        # Everything that is a plain weighted MEAN (not a centered second moment)
        # is already an unbiased plug-in estimate and must be preserved at raw
        # scale, i.e. deflated by the sigma-power it carries:
        #   pi_s      (sigma^1)            /= gamma
        #   rho_gen   (sigma^2 pair)       /= gamma^2   E[G_ab] is a mean of a
        #                                               frame quantity, no
        #                                               centering/Bessel applies
        #   rho_00_sq (sigma^2 per index)  /= gamma^4
        #   rho_gen_sq (sigma^2 per index) /= gamma^4
        #   m3_0      (sigma^3)            /= gamma^3
        # side effect on rho_gen: the generator quadratic form is now expressed
        # over the Bessel-CORRECTED variance, so decompose() rates pick up the
        # small-sample correction consistently.

        gamma = 1 / jnp.sqrt(1 - eps) if bessel_correct else 1.0

        if sigma_0 is not None:
            sigma_0 *= gamma

        if sigma_1 is not None:
            sigma_1 *= gamma

        if pi_s_0 is not None:
            pi_s_0 /= gamma

        if pi_s_1 is not None:
            pi_s_1 /= gamma

        if rho_gen is not None:
            rho_gen /= gamma**2

        if rho_00_sq is not None:
            rho_00_sq /= gamma**4

        if rho_gen_sq is not None:
            rho_gen_sq /= gamma**4

        cov = Covariances(
            rho_00=rho_00,
            rho_01=rho_01,
            rho_10=rho_10,
            rho_11=rho_11,
            rho_gen=rho_gen,
            rho_gen_sq=rho_gen_sq,
            rho_00_sq=rho_00_sq,
            pi_s_0=pi_s_0,
            pi_s_1=pi_s_1,
            only_diag=only_diag,
            trans_f=trans_f,
            trans_g=trans_g,
            sigma_0=sigma_0,
            sigma_1=sigma_1,
            shrinkage_method=shrinkage_method,
            w2=w2,
            w3=w3,
            time_series=time_series,
            symmetric=not time_series,
            rho_bend=rho_bend,
            scale_deriv=scale_deriv,
            reduction_cv=reduction_cv,
        )

        if symmetric:
            cov = cov.symmetrize()

        if shrink:
            cov = cov.shrink(eps_shrink=eps_shrink)

        return cov

    # ------------------------------------------------------------------ #
    #  basic properties                                                  #
    # ------------------------------------------------------------------ #

    @property
    def generator(self):
        return self.rho_gen is not None

    @property
    def complex_features(self):
        """True if the stored statistics live in a complex representation
        (either via create(..., complexify=...) or halve_even_odd)."""
        return bool(jnp.iscomplexobj(self.rho_00))

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
    def C_gen_sq(self):
        if self.rho_gen_sq is None:
            return None

        if self.only_diag:
            return jnp.einsum("i,i->i", self.rho_gen_sq, self.sigma_0**4)

        return jnp.einsum("ij,i,j->ij", self.rho_gen_sq, self.sigma_0**2, self.sigma_0**2)

    @property
    def C_gen_var(self):
        """Elementwise variance of the metric across frames:
            Var[G_ab] = E[|G_ab|^2] - |E[G_ab]|^2   (raw scale).

        Note this is a variance over FRAMES of the Gram entries -- not a
        centering in x; the Dirichlet form itself is correctly uncentered.
        Relies on rho_gen and rho_gen_sq being in consistent (raw-preserving)
        units, which the gamma deflation in create() guarantees.  For complex
        features it is the variance of the complex Gram,
        Var[G_zz,cd] = E[|G_zz,cd|^2] - |E[G_zz,cd]|^2 -- exact when complexify
        was used, and available after halve_even_odd only if the cross-block
        accumulator was tracked (create(..., gen_sq_cross=True))."""
        if self.rho_gen_sq is None or self.rho_gen is None:
            return None

        g = self.C_gen
        return self.C_gen_sq - jnp.real(g * jnp.conj(g))

    @property
    def C_00_sq(self):
        """Raw (uncentered) quartic cross moments E[|x_i|^2 |x_j|^2]."""
        if self.rho_00_sq is None:
            return None

        if self.only_diag:
            return jnp.einsum("i,i->i", self.rho_00_sq, self.sigma_0**4)

        return jnp.einsum("ij,i,j->ij", self.rho_00_sq, self.sigma_0**2, self.sigma_0**2)

    @property
    def C_bend(self):
        if self.rho_bend is None:
            return None

        if self.only_diag:
            return jnp.einsum("i,i->i", self.rho_bend, self.sigma_0 ** (-2))

        return jnp.einsum("ij,i,j->ij", self.rho_bend, self.sigma_0, self.sigma_0)

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
        return safe_inv(self.sigma_0)

    @property
    def sigma_1_inv(self):
        if self.sigma_1 is None:
            return None
        return safe_inv(self.sigma_1)

    # ------------------------------------------------------------------ #
    #  centered fourth-moment statistics                                 #
    # ------------------------------------------------------------------ #

    @property
    def _diag_rho_00(self):
        return self.rho_00 if self.only_diag else jnp.diag(self.rho_00).real

    @property
    def _diag_rho_00_sq(self):
        if self.rho_00_sq is None:
            return None
        return self.rho_00_sq if self.only_diag else jnp.diag(self.rho_00_sq).real

    @property
    def m2_central(self):
        """Per-feature CENTRAL second moment (variance) in sigma_0 units.

        The accumulator centers online (c() subtracts the running pi), so when
        calc_pi was on this is simply diag(rho_00) -- and equals 1 right after
        create() by construction, since sigma absorbs it.  It becomes
        informative after shrink / symmetrize / halve_even_odd, where rho_00 is
        modified while sigma is not re-absorbed.  If pi was never tracked,
        rho_00 is uncentered and this returns the raw second moment.
        Multiply by sigma_0**2 for raw scale (= diag(C00))."""
        return self._diag_rho_00

    @property
    def m4_central(self):
        """Per-feature CENTRAL fourth moment mu_4 in sigma_0 units.

        Assembled from the accumulated raw moments via
            mu_4 = m4 - 4 mu m3 + 6 mu^2 m2 - 3 mu^4        (real features)
        with everything expressed relative to sigma_0.  Exact when pi was
        tracked (calc_pi=True) and features are real; if pi is unavailable
        the raw moment is returned (i.e. mu is taken to be 0).

        NOTE: for complex features this real centering identity does not hold
        (and after halve_even_odd m3_0 is gone entirely), so prefer mod_dev
        there -- it needs only raw moments.
        """
        m4u = self._diag_rho_00_sq
        if m4u is None:
            return None

        return m4u

        p = jnp.real(self.pi_s_0)
        v = self._diag_rho_00  # central variance in sigma units
        m2u = v + p**2  # raw second moment
        m3u = jnp.real(self.m3_0)

        return m4u - 4.0 * p * m3u + 6.0 * p**2 * m2u - 3.0 * p**4

    # ------------------------------------------------------------------ #
    #  whitening / masking / decomposition                               #
    # ------------------------------------------------------------------ #

    def whiten_rho(
        self,
        choice: str,
        epsilon: float = 1e-4,
        max_features: int | None = None,
        verbose=False,
        cholesky=True,
        apply_mask: bool = True,
        tikhonov=1e-3,
        return_eigh=False,
    ) -> tuple[Array, Array, Array]:
        # returns W such that W C W^H = I and hence W^H W = C^-1
        # https://arxiv.org/pdf/1512.00809

        if choice == "rho_00":
            C = self.rho_00
        elif choice == "rho_11":
            C = self.rho_11
        else:
            raise ValueError(f"choice {choice} not known")

        assert C is not None

        if self.only_diag:
            return 1 / jnp.sqrt(C)

        C = C + tikhonov * jnp.eye(C.shape[0], dtype=C.dtype)

        if cholesky:
            L = jnp.linalg.cholesky(C)  # works for complex hermitian PSD

            W = jnp.linalg.solve(L, jnp.eye(L.shape[0], dtype=C.dtype))
            W_inv = L

            theta = None

        else:
            theta, G = jnp.linalg.eigh(C)

            sqrt_theta_inv = 1 / jnp.sqrt(theta)

            W = jnp.einsum(
                "i,ji->ij",
                sqrt_theta_inv,
                jnp.conj(G),
            )

            W_inv = jnp.einsum(
                "ij,j->ij",
                G,
                jnp.sqrt(theta),
            )

        if apply_mask:
            if max_features is not None:
                if W.shape[0] > max_features:
                    print(f"whiten: reducing dim to {max_features=}")
                    W = W[:max_features, :]

        print(f"{jnp.linalg.norm(W @ C @ jnp.conj(W).T - jnp.eye(W.shape[0]))=}")

        return W, W_inv, theta

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

        if self.rho_01 is not None:
            d = jnp.diag(self.rho_01) if not self.only_diag else self.rho_01
        else:
            assert self.rho_gen is not None
            d = jnp.exp(-jnp.diag(self.rho_gen)) if not self.only_diag else jnp.exp(-self.rho_gen)

        d = jnp.real(d)

        argsort = jnp.argsort(d[argmask], descending=True)
        argmask = argmask[argsort]

        if auto_cov_threshold is not None:
            b = d[argmask] > auto_cov_threshold
            argmask = argmask[b]

        if max_features is not None:
            if argmask.shape[0] > max_features:
                argmask = argmask[:max_features]

        def mask_rho(rho: jax.Array | None) -> jax.Array | None:
            if rho is None:
                return None
            if self.only_diag:
                return rho[argmask]
            return rho[argmask, :][:, argmask]

        self.rho_00 = mask_rho(self.rho_00)

        if self.rho_11 is not None:
            self.rho_11 = mask_rho(self.rho_11)
        if self.rho_01 is not None:
            self.rho_01 = mask_rho(self.rho_01)
        if self.rho_10 is not None:
            self.rho_10 = mask_rho(self.rho_10)

        if self.pi_s_0 is not None:
            self.pi_s_0 = self.pi_s_0[argmask]
        if self.pi_s_1 is not None:
            self.pi_s_1 = self.pi_s_1[argmask]

        self.sigma_0 = self.sigma_0[argmask]
        if self.sigma_1 is not None:
            self.sigma_1 = self.sigma_1[argmask]

        if self.rho_gen is not None:
            self.rho_gen = mask_rho(self.rho_gen)

        if self.rho_gen_sq is not None:
            self.rho_gen_sq = mask_rho(self.rho_gen_sq)

        if self.rho_00_sq is not None:
            self.rho_00_sq = mask_rho(self.rho_00_sq)

        if self.rho_bend is not None:
            self.rho_bend = mask_rho(self.rho_bend)

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
        scale_gen=1.0,
        angle=False,
    ):
        W_0, _, _ = self.whiten_rho("rho_00", epsilon=eps, apply_mask=apply_mask)

        print(f"{W_0.shape=}")

        if not self.symmetric:
            W_1, _, _ = self.whiten_rho("rho_11", epsilon=eps, apply_mask=apply_mask)
        else:
            W_1 = W_0

        print(f"{self.generator=}")

        if self.only_diag:
            if self.generator:
                T_tilde = jnp.einsum("i,i,i->i", self.rho_gen, W_0, jnp.conj(W_0))
            else:
                assert self.rho_01 is not None
                T_tilde = jnp.einsum("i,i,i->i", self.rho_01, W_1, jnp.conj(W_0))

        else:
            if self.generator:
                T_tilde = (W_0 @ (self.rho_gen) @ jnp.conj(W_0).T).T
            else:
                assert self.rho_01 is not None
                T_tilde = (W_0 @ self.rho_01 @ jnp.conj(W_1).T).T

        print(f"{T_tilde.shape=}")

        is_complex = jnp.iscomplexobj(T_tilde)

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
            idx = jnp.argsort(jnp.real(s), descending=True)[:n_modes]
            return idx, s

        # lobpcg does not handle complex operators; fall back to dense for complex T
        if n_modes + 1 < T_tilde.shape[0] / 5 and sparse and not is_complex:
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

                V = U

            else:
                l, V, n_iter = lobpcg_standard(
                    T_tilde @ T_tilde.T,
                    x0,
                    m=200,
                )

                print(n_iter)

                s = l ** (1 / 2)
                s_inv: jax.Array = safe_inv(s, cond=s > 1e-12)  # type: ignore
                # T_tilde = V diag(s) U^T  ->  U = T_tilde^T V diag(1/s)
                U = T_tilde.T @ V @ jnp.diag(s_inv)

        else:
            if self.symmetric and W_0.shape[0] == W_1.shape[0]:
                print("using eigh")
                # for complex data T_tilde is hermitian if the dynamics is reversible
                s, U = jax.numpy.linalg.eigh(
                    T_tilde.T,
                )
                V = U
            else:
                print("using svd")
                U, s, VT = jax.numpy.linalg.svd(T_tilde.T)
                V = jnp.conj(VT).T

        if self.generator:
            print(f"generator {s=}")
            s *= scale_gen
            idx = jnp.argsort(jnp.abs(s), descending=False)
        else:
            idx = jnp.argsort(jnp.real(s), descending=True)
        U = U[:, idx]
        s = s[idx]
        V = V[:, idx]

        W_0 = jnp.conj(U).T @ W_0
        W_1 = jnp.conj(V).T @ W_1

        if out_eps is not None:
            m = jnp.abs(1 - s) < out_eps

            U = U[:, m]
            V = V[:, m]
            s = s[m]

            print(f"{jnp.sum(m)=}")

        return W_0, W_1, s, idx

    def decompose_diag(
        self,
        out_dim: int | None = None,
        scale_gen=1.0,
        non_per: int | None = None,
        per: int | None = 0,
    ):
        tot = self.rho_00.shape[0]

        if non_per is not None:
            nonper = self.rho_00.shape[0] - 2 * per

        if out_dim is None:
            out_dim = 10

        if out_dim == -1:
            out_dim = self.rho_00.shape[0]

        if out_dim < 20:
            out_dim = 20

        s = self.rho_01 if self.rho_01 is not None else self.rho_gen

        if s.ndim > 1:
            s = jnp.diag(s)

        if per != 0:
            s_non_per = s[:non_per]
            s_per = (s[non_per::2] + s[non_per + 1 :: 2]) / 2

            s = jnp.concatenate([s_non_per, s_per])

        if self.generator:
            print(f"generator {s=}")
            s *= scale_gen
            idx = jnp.argsort(jnp.abs(s), descending=False)[:out_dim]
        else:
            idx = jnp.argsort(jnp.real(s), descending=True)[:out_dim]

        if per != 0:
            per_idx = jnp.arange(non_per, tot, 2)

            _A = self.rho_00[per_idx, per_idx]
            _B = self.rho_00[per_idx, per_idx + 1]
            _C = self.rho_00[per_idx + 1, per_idx]
            _D = self.rho_00[per_idx + 1, per_idx + 1]

            @jax.vmap
            def _get_W(A, B, C, D):
                C = jnp.array([[A, B], [C, D]])
                theta, G = jnp.linalg.eigh(C)
                sqrt_theta_inv = 1 / jnp.sqrt(theta)
                W = jnp.einsum(
                    "i,ji->ij",
                    sqrt_theta_inv,
                    jnp.conj(G),
                )

                return W

            W = _get_W(_A, _B, _C, _D)

        else:
            W = None

        return idx, s[idx], W

    @property
    def p(self):
        return self.rho_00.shape[0]

    @property
    def n_eff(self):
        return 1.0 / self.w2

    # ------------------------------------------------------------------ #
    #  shrinkage                                                         #
    # ------------------------------------------------------------------ #

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
                        [jnp.mean(jnp.diag(self.C00).real) * jnp.eye(self.C00.shape[0]), self.C01 * 0],
                        [self.C10 * 0, jnp.mean(jnp.diag(self.C11).real) * jnp.eye(self.C11.shape[0])],
                    ]
                )
            elif shrinkage == "biconstant":
                T = jnp.block(
                    [
                        [
                            jnp.mean(jnp.diag(self.C00).real) * jnp.eye(self.C00.shape[0]),
                            jnp.mean(jnp.diag(self.C01)) * jnp.eye(self.C01.shape[0]),
                        ],
                        [
                            jnp.mean(jnp.diag(self.C10)) * jnp.eye(self.C10.shape[0]),
                            jnp.mean(jnp.diag(self.C11).real) * jnp.eye(self.C11.shape[0]),
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
                T = jnp.mean(jnp.diag(C).real) * jnp.eye(C.shape[0])
            else:
                raise NotImplementedError

        assert self.w2 is not None
        assert self.w3 is not None

        gamma = 1.0 / (1.0 - self.w2)
        nu = 1 - self.w2 - 2 * self.w3
        eta = self.w2 + self.w2**2 - 2 * self.w3
        eps = self.w2

        C = C / gamma  # with bessel correction, C = gamma S

        # Shrinkage MMSE estimators of covariances beyond the zero-mean
        # and stationary variance assumptions

        def get_C(lambd):
            return (1 - lambd) * C + lambd * T

        def get_lambd(C):
            # hermitian-safe traces: tr(C C^H) is real for any C
            diag_sq = jnp.sum(jnp.abs(jnp.diag(C)) ** 2)
            tr_CS = jnp.real(jnp.trace(C @ jnp.conj(C).T)) - diag_sq
            tr_C = jnp.real(jnp.trace(C))
            tr_CS_diag = tr_C * tr_C - diag_sq

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

            return jnp.einsum("i,ij,j->ij", sigma_0_inv, X, sigma_1_inv)

        # shrinkage only touches the second moments; sigma is unchanged, so the
        # stored higher-moment estimates carry over unmodified
        common = dict(
            rho_gen=self.rho_gen,
            rho_gen_sq=self.rho_gen_sq,
            rho_00_sq=self.rho_00_sq,
            rho_bend=self.rho_bend,
            trans_f=self.trans_f,
            trans_g=self.trans_g,
            symmetric=self.symmetric,
            shrinkage_method=shrinkage,
            w2=self.w2,
            w3=self.w3,
            time_series=self.time_series,
            scale_deriv=self.scale_deriv,
            only_diag=self.only_diag,
        )

        if not self.time_series:
            return Covariances(
                rho_00=get_rho(0, 0),
                rho_01=None,
                rho_10=None,
                rho_11=None,
                pi_s_0=self.pi_s_0,
                pi_s_1=None,
                sigma_0=self.sigma_0,
                sigma_1=None,
                **common,
            )

        return Covariances(
            rho_00=get_rho(0, 0),
            rho_01=get_rho(0, 1),
            rho_10=get_rho(1, 0),
            rho_11=get_rho(1, 1),
            pi_s_0=self.pi_s_0,
            pi_s_1=self.pi_s_1,
            sigma_0=self.sigma_0,
            sigma_1=self.sigma_1,
            **common,
        )

    # ------------------------------------------------------------------ #
    #  symmetrization                                                    #
    # ------------------------------------------------------------------ #

    def symmetrize(self):
        if not self.time_series:
            return self

        _rho_00 = self.rho_00
        _rho_01 = self.rho_01
        _rho_10 = self.rho_10
        _rho_11 = self.rho_11
        _pi_s_0 = self.pi_s_0
        _pi_s_1 = self.pi_s_1

        # make sigma same

        _sigma = jnp.sqrt(self.sigma_0 * self.sigma_1)

        d_sigma_0 = self.sigma_0 * safe_inv(_sigma)
        d_sigma_1 = self.sigma_1 * safe_inv(_sigma)

        def get_rho(rho, f1, f2) -> jax.Array:
            if self.only_diag:
                return jnp.einsum("i,i->i", rho, jnp.conj(f1) * f2)
            return jnp.einsum("ij,i,j->ij", rho, jnp.conj(f1), f2)

        _rho_00 = get_rho(_rho_00, d_sigma_0, d_sigma_0)
        _rho_01 = get_rho(_rho_01, d_sigma_0, d_sigma_1)
        _rho_10 = get_rho(_rho_10, d_sigma_1, d_sigma_0)
        _rho_11 = get_rho(_rho_11, d_sigma_1, d_sigma_1)

        # make pi same

        calc_pi = self.pi_s_0 is not None

        def get_outer(pi1, pi2):
            if self.only_diag:
                return jnp.einsum("i,i->i", jnp.conj(pi1), pi2)
            return jnp.einsum("i,j->ij", jnp.conj(pi1), pi2)

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
        # for complex matrices the time-reversed lagged covariance is rho_10^H
        sym_rho_01 = (1 / 2) * (_rho_01 + jnp.conj(_rho_10).T if not self.only_diag else _rho_01 + jnp.conj(_rho_10))

        d_sigma = jnp.sqrt(jnp.real(jnp.diag(sym_rho_00))) if not self.only_diag else jnp.sqrt(jnp.real(sym_rho_00))
        d_sigma_inv = safe_inv(d_sigma)

        sym_rho_00 = get_rho(sym_rho_00, d_sigma_inv, d_sigma_inv)
        sym_rho_01 = get_rho(sym_rho_01, d_sigma_inv, d_sigma_inv)

        if calc_pi:
            _pi_s = jnp.einsum("i,i->i", _pi_s, d_sigma)
        else:
            _pi_s = None

        # carry the higher-moment estimates of the x0 leg into the new sigma units.
        # net per-index factor sigma_old / sigma_new = d_sigma_0 * d_sigma_inv
        f_h = d_sigma_0 * d_sigma_inv

        if self.rho_00_sq is not None:
            _rho_00_sq = get_rho(self.rho_00_sq, f_h**2, f_h**2)
        else:
            _rho_00_sq = None

        _rho_gen = get_rho(self.rho_gen, f_h, f_h) if self.rho_gen is not None else None
        _rho_gen_sq = get_rho(self.rho_gen_sq, f_h**2, f_h**2) if self.rho_gen_sq is not None else None

        _sigma = _sigma * d_sigma

        return Covariances(
            rho_00=sym_rho_00,
            rho_11=sym_rho_00,
            rho_01=sym_rho_01,
            rho_10=sym_rho_01,
            rho_gen=_rho_gen,
            rho_gen_sq=_rho_gen_sq,
            rho_00_sq=_rho_00_sq,
            rho_bend=self.rho_bend,
            pi_s_0=_pi_s,
            pi_s_1=_pi_s,
            sigma_0=_sigma,
            sigma_1=_sigma,
            only_diag=self.only_diag,
            symmetric=True,
            trans_f=self.trans_f,
            trans_g=self.trans_g,
            shrinkage_method=self.shrinkage_method,
            bc=self.bc,
            w2=self.w2,
            w3=self.w3,
            time_series=self.time_series,
            scale_deriv=self.scale_deriv,
        )

    # ------------------------------------------------------------------ #
    #  combination of two estimates                                      #
    # ------------------------------------------------------------------ #

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
                return jnp.einsum("i,i->i", jnp.conj(pi1), pi2)
            return jnp.einsum("i,j->ij", jnp.conj(pi1), pi2)

        # 1. Bring everything to absolute scale (scale by sigma)
        pi1_0 = cov1.pi_s_0 * cov1.sigma_0 if cov1.pi_s_0 is not None else 0
        pi1_1 = cov1.pi_s_1 * cov1.sigma_1 if cov1.pi_s_1 is not None else 0
        pi2_0 = cov2.pi_s_0 * cov2.sigma_0 if cov2.pi_s_0 is not None else 0
        pi2_1 = cov2.pi_s_1 * cov2.sigma_1 if cov2.pi_s_1 is not None else 0

        # 2. Compute the new weighted mean (raw scale)
        new_pi_abs_0 = w1 * pi1_0 + w2 * pi2_0
        new_pi_abs_1 = w1 * pi1_1 + w2 * pi2_1

        # 3. Combine second moments
        def combine_component(r1, s1_a, s1_b, p1_a, p1_b, r2, s2_a, s2_b, p2_a, p2_b):
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
        new_sigma_0 = jnp.sqrt(jnp.diag(res_00).real) if not cov1.only_diag else jnp.sqrt(jnp.real(res_00))
        new_sigma_1 = jnp.sqrt(jnp.diag(res_11).real) if not cov1.only_diag else jnp.sqrt(jnp.real(res_11))

        inv_sigma_0 = safe_inv(new_sigma_0)
        inv_sigma_1 = safe_inv(new_sigma_1)

        new_rho_00 = get_rho(res_00, inv_sigma_0, inv_sigma_0, cov1.only_diag)
        new_rho_11 = get_rho(res_11, inv_sigma_1, inv_sigma_1, cov1.only_diag)
        new_rho_01 = get_rho(res_01, inv_sigma_0, inv_sigma_1, cov1.only_diag)
        new_rho_10 = get_rho(res_10, inv_sigma_1, inv_sigma_0, cov1.only_diag)

        # 6. Final mean normalization (scaled back)
        final_pi_0 = new_pi_abs_0 * inv_sigma_0 if cov1.pi_s_0 is not None else None
        final_pi_1 = new_pi_abs_1 * inv_sigma_1 if cov1.pi_s_1 is not None else None

        # 7. Raw higher moments combine linearly at raw scale, then re-normalize
        new_rho_00_sq = None
        if cov1.rho_00_sq is not None and cov2.rho_00_sq is not None:
            m4_raw = w1 * cov1.C_00_sq + w2 * cov2.C_00_sq
            new_rho_00_sq = get_rho(m4_raw, inv_sigma_0**2, inv_sigma_0**2, cov1.only_diag)

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
            rho_00_sq=new_rho_00_sq,
            only_diag=cov1.only_diag,
            symmetric=cov1.symmetric and cov2.symmetric,
            trans_f=cov1.trans_f,
            trans_g=cov1.trans_g,
            shrinkage_method=cov1.shrinkage_method,
            bc=None,
            w2=(w1 * cov1.w2 + w2 * cov2.w2) if cov1.w2 is not None and cov2.w2 is not None else None,
            w3=(w1 * cov1.w3 + w2 * cov2.w3) if cov1.w3 is not None and cov2.w3 is not None else None,
            time_series=cov1.time_series and cov2.time_series,
        )
