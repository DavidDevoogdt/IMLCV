from functools import partial

import jax.numpy as jnp

from IMLCV.base.datastructures import Partial_decorator, vmap_decorator

# XYZ-to-sRGB matrix
_m = jnp.array(
    [
        [3.240969941904521, -1.537383177570093, -0.498610760293],
        [-0.96924363628087, 1.87596750150772, 0.041555057407175],
        [0.055630079696993, -0.20397695888897, 1.056971514242878],
    ]
)
# sRGB-to-XYZ matrix
_m_inv = jnp.array(
    [
        [0.41239079926595, 0.35758433938387, 0.18048078840183],
        [0.21263900587151, 0.71516867876775, 0.072192315360733],
        [0.019330818715591, 0.11919477979462, 0.95053215224966],
    ]
)
_ref_y = 1.0
_ref_u = 0.19783000664283
_ref_v = 0.46831999493879
_kappa = 903.2962962  # 24389/27 == (29/3)**3
_epsilon = 0.0088564516  # 216/24389 == (6/29)**3


def _normalize_output(conversion):
    # as in snapshot rev 4, the tolerance should be 1e-11
    normalize = vmap_decorator(Partial_decorator(jnp.round, decimals=11 - 1))

    def normalized(*args, **kwargs):
        color = conversion(*args, **kwargs)
        return normalize(color)

    return normalized


@vmap_decorator
def _distance_line_from_origin(slope, intercept):
    v = slope**2 + 1.0
    return jnp.abs(intercept) / jnp.sqrt(v)


@partial(vmap_decorator, in_axes=(None, 0, 0))
def _length_of_ray_until_intersect(theta, slope, intercept):
    return intercept / (jnp.sin(theta) - slope * jnp.cos(theta))


def _get_bounds(l):
    sub1 = ((l + 16) ** 3) / 1560896
    sub2 = jnp.where(sub1 > _epsilon, sub1, l / _kappa)

    @vmap_decorator
    def _c(c):
        m1 = _m[c][0]
        m2 = _m[c][1]
        m3 = _m[c][2]

        @vmap_decorator
        def _t(t):
            top1 = (284517.0 * m1 - 94839.0 * m3) * sub2
            top2 = (838422 * m3 + 769860.0 * m2 + 731718.0 * m1) * l * sub2 - (769860.0 * t) * l
            bottom = (632260.0 * m3 - 126452.0 * m2) * sub2 + 126452.0 * t

            return top1 / bottom, top2 / bottom

        return _t(jnp.array([0, 1, 2]))

    return _c(jnp.array([0, 1, 2]))


def _max_safe_chroma_for_l(l):
    slopes, intercepts = _get_bounds(l)

    return jnp.min(_distance_line_from_origin(slopes, intercepts))


def _max_chroma_for_lh(l, h):
    hrad = jnp.radians(h)

    slopes, intercepts = _get_bounds(l)

    lengths = _length_of_ray_until_intersect(hrad, slopes, intercepts)

    return jnp.min(jnp.where(lengths > 0, lengths, jnp.inf))  # type: ignore


@vmap_decorator
def _from_linear(c):
    return jnp.where(c <= 0.0031308, 12.92 * c, 1.055 * jnp.pow(c, 5 / 12) - 0.055)


@vmap_decorator
def _to_linear(c):
    return jnp.where(c > 0.04045, jnp.power((c + 0.055) / 1.055, 2.4), c / 12.92)


def _y_to_l(y):
    return jnp.where(y <= _epsilon, y / _kappa, 116.0 * jnp.pow(y / _ref_y, 1 / 3) - 16)


def _l_to_y(l):
    return jnp.where(l <= 8, _kappa * l, _ref_y * ((l + 16) / 116) ** 3)


def xyz_to_rgb(xyz):
    return _from_linear(_m @ xyz)


def rgb_to_xyz(rgb):
    return _m_inv @ _to_linear(rgb)


def xyz_to_luv(xyz):
    x, y, z = xyz[0], xyz[1], xyz[2]
    l = _y_to_l(y)

    divider = x + 15.0 * y + 3.0 * z

    var_u = 4.0 * x / divider
    var_v = 9.0 * y / divider
    u = 13.0 * l * (var_u - _ref_u)  # type: ignore
    v = 13.0 * l * (var_v - _ref_v)  # type: ignore

    return jnp.array([l, u, v])


def luv_to_xyz(luv):
    l, u, v = luv[0], luv[1], luv[2]

    var_u = u / (13.0 * l) + _ref_u
    var_v = v / (13.0 * l) + _ref_v
    y = _l_to_y(l)
    x = y * 9.0 * var_u / (4.0 * var_v)
    z = y * (12 - 3.0 * var_u - 20.0 * var_v) / (4.0 * var_v)

    return jnp.where(l > 0, jnp.array([x, y, z]), jnp.array([0.0, 0.0, 0.0]))


def luv_to_lch(luv):
    l, u, v = luv[0], luv[1], luv[2]
    c = jnp.hypot(u, v)

    hrad = jnp.atan2(v, u)
    h = jnp.degrees(hrad)

    h = jnp.where(c < 1e-08, 0, h)
    h = jnp.where(h < 0, h + 360, h)

    return jnp.array([l, c, h])


def lch_to_luv(lch):
    l, c, h = lch[0], lch[1], lch[2]
    hrad = jnp.radians(h)
    u = jnp.cos(hrad) * c
    v = jnp.sin(hrad) * c
    return jnp.array([l, u, v])


def hsluv_to_lch(hsl):
    h, s, l = hsl[0], hsl[1], hsl[2]
    _hx_max = _max_chroma_for_lh(l, h)
    c = _hx_max / 100.0 * s

    return jnp.select(
        [l > 100, l < 1e-08, True], [jnp.array([100.0, 0.0, h]), jnp.array([0.0, 0.0, h]), jnp.array([l, c, h])]
    )


def lch_to_hsluv(lsh):
    l, c, h = lsh[0], lsh[1], lsh[2]

    _hx_max = _max_chroma_for_lh(l, h)
    s = c / _hx_max * 100

    return jnp.select(
        [l > 100.0 - 1e-7, l < 1e-08, True],
        [jnp.array([h, 0.0, 100.0]), jnp.array([h, 0.0, 0.0]), jnp.array([h, s, l])],
    )


def hpluv_to_lch(hsl):
    h, s, l = hsl[0], hsl[1], hsl[2]

    _hx_max = _max_safe_chroma_for_l(l)
    c = _hx_max / 100.0 * s

    return jnp.select(
        [l > 100.0 - 1e-7, l < 1e-08, True],
        [jnp.array([100.0, 0.0, h]), jnp.array([0.0, 0.0, h]), jnp.array([l, c, h])],
    )


def lch_to_hpluv(lch):
    l, c, h = lch[0], lch[1], lch[2]

    _hx_max = _max_safe_chroma_for_l(l)
    s = c / _hx_max * 100.0

    return jnp.select(
        [l > 100.0 - 1e-7, l < 1e-08, True],
        [jnp.array([h, 0.0, 100.0]), jnp.array([h, 0.0, 0.0]), jnp.array([h, s, l])],
    )


def lch_to_rgb(lch):
    return xyz_to_rgb(luv_to_xyz(lch_to_luv(lch)))


def rgb_to_lch(rgb):
    return luv_to_lch(xyz_to_luv(rgb_to_xyz(rgb)))


def _hsluv_to_rgb(hsluv):
    return lch_to_rgb(hsluv_to_lch(hsluv))


def hsluv_to_rgb(hsluv):
    return _normalize_output(_hsluv_to_rgb)(hsluv)


def rgb_to_hsluv(rgb):
    return lch_to_hsluv(rgb_to_lch(rgb))


def _hpluv_to_rgb(hpluv):
    return lch_to_rgb(hpluv_to_lch(hpluv))


hpluv_to_rgb = _normalize_output(_hpluv_to_rgb)


def rgb_to_hpluv(_hx_tuple):
    return lch_to_hpluv(rgb_to_lch(_hx_tuple))
