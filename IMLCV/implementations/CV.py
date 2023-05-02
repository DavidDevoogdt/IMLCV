from __future__ import annotations

import dataclasses
import tempfile

# from IMLCV.base.CV import CV, CvTrans
from functools import partial
from importlib import import_module
from typing import TYPE_CHECKING

# import numpy as np
import jax.numpy as jnp
import numpy as np
import tensorflow
import tensorflow as tfl
from flax.linen.linear import Dense
from jax import Array, jit, vmap
from jax.experimental.jax2tf import call_tf
from keras.api._v2 import keras as KerasAPI

if TYPE_CHECKING:
    pass

keras: KerasAPI = import_module("tensorflow.keras")

import dataclasses

import distrax
import jax.numpy as jnp
import numba
import numpy as np
from tensorflow_probability.substrates import jax as tfp

from IMLCV.base.CV import (
    CV,
    CollectiveVariable,
    CvFlow,
    CvFunDistrax,
    CvFunInput,
    CvFunNn,
    CvMetric,
    CvTrans,
    NeighbourList,
    SystemParams,
)

######################################
#       CV transformations           #
######################################


class PeriodicLayer(keras.layers.Layer):
    def __init__(self, bbox, periodicity, **kwargs):
        super().__init__(**kwargs)

        self.bbox = tfl.Variable(np.array(bbox))
        self.periodicity = np.array(periodicity)

    def call(self, inputs):
        # maps to periodic box
        bbox = self.bbox

        inputs_mod = (
            tfl.math.mod(inputs - bbox[:, 0], bbox[:, 1] - bbox[:, 0]) + bbox[:, 0]
        )
        return tfl.where(self.periodicity, inputs_mod, inputs)

    def metric(self, r):
        # maps difference
        a = self.bbox[:, 1] - self.bbox[:, 0]

        r = tfl.math.mod(r, a)
        r = tfl.where(r > a / 2, r - a, r)
        return tfl.norm(r, axis=1)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "bbox": np.array(self.bbox),
                "periodicity": self.periodicity,
            }
        )
        return config


class KerasTrans(CvTrans):
    def __init__(self, encoder) -> None:
        self.encoder = encoder

    @partial(jit, static_argnums=(0,))
    def compute_cv_trans(self, cc: Array):
        out = call_tf(self.encoder.call)(cc)
        return out

    def __getstate__(self):
        # https://stackoverflow.com/questions/48295661/how-to-pickle-keras-model
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as fd:
            tensorflow.keras.models.save_model(self.encoder, fd.name, overwrite=True)
            model_str = fd.read()
        d = {"model_str": model_str}
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as fd:
            fd.write(state["model_str"])
            fd.flush()

            custom_objects = {"PeriodicLayer": PeriodicLayer}
            with keras.utils.custom_object_scope(custom_objects):
                model = keras.models.load_model(fd.name)

        self.encoder = model


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
            jnp.arange(n), jnp.arange(n)
        )

        return out[jnp.triu_indices_from(out, k=1)]

        # return out

    return h


def dihedral(numbers):
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
    from IMLCV.tools.soap_kernel import Kernel, p_i, p_inl_sb

    # @jit #jit makes it not pickable
    def f(sp: SystemParams, nl: NeighbourList):
        assert nl is not None, "provide neighbourlist for sb describport"

        def _reduce(a):
            a = vmap(
                vmap(
                    vmap(
                        lambda a: a[jnp.tril_indices_from(a)], in_axes=(0), out_axes=(0)
                    ),
                    in_axes=(1),
                    out_axes=(1),
                ),
                in_axes=(2),
                out_axes=(2),
            )(
                a
            )  # eliminate l>n
            a = vmap(
                vmap(lambda a: a[jnp.triu_indices_from(a)], in_axes=(0), out_axes=(0)),
                in_axes=(3),
                out_axes=(2),
            )(
                a
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
        assert references_nl is not None

        refs = (f)(references, references_nl)

        # @NeighbourList.vmap_x_nl
        # @partial(vmap, in_axes=(0, 0, None, None))
        def _f(refs, references_nl, val, nl):
            return Kernel(val, refs, nl, references_nl)

        if references.batched:
            _f = vmap(f, in_axes=(0, 0, None, None))

        _f = partial(_f, refs, references_nl)

        # @jit
        def sb_descriptor_distance2(sp: SystemParams, nl: NeighbourList):
            assert nl is not None, "provide neighbourlist for sb describport"

            val = f(sp=sp, nl=nl)
            com = _f(val, nl)

            y2 = 1 - com
            y2_safe = jnp.where(y2 <= 0, jnp.ones_like(y2), y2)
            y = jnp.where(y2 <= 0, 0.0, jnp.sqrt(y2_safe))

            return jnp.ravel(y)

        return CvFlow.from_function(sb_descriptor_distance2)  # type: ignore

    else:
        return CvFlow.from_function(f)  # type: ignore


def NoneCV() -> CollectiveVariable:
    return CollectiveVariable(
        f=CvFlow.from_function(lambda sp, nl: jnp.array([0.0])),
        metric=CvMetric(periodicities=[None]),
    )


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
                [[jnp.cos(alpha), jnp.sin(alpha)], [-jnp.sin(alpha), jnp.cos(alpha)]]
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
        self.s = Dense(features=self.latent_dim)
        self.t = Dense(features=self.latent_dim)

        # Alternating binary mask.
        self.bijector = distrax.as_bijector(
            tfp.bijectors.RealNVP(
                fraction_masked=0.5,
                shift_and_log_scale_fn=self.shift_and_scale,
            )
        )

    def shift_and_scale(self, x0, input_depth, **condition_kwargs):
        return self.s(x0), self.t(x0)


######################################
#           Test                     #
######################################


class MetricUMAP(CvMetric):
    def __init__(self, periodicities, bounding_box=None) -> None:
        super().__init__(periodicities=periodicities, bounding_box=bounding_box)

        bb = np.array(self.bounding_box)
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
