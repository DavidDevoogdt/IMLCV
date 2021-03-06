from __future__ import annotations

import os
from abc import ABC, abstractmethod
from functools import partial
from typing import Iterable, List, Optional

import dill
import jax.numpy as jnp
import jax.scipy as jsp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from IMLCV.base.CV import CV, SystemParams
from IMLCV.launch.parsl_conf.bash_app_python import bash_app_python
from jax import jacfwd, jit
from molmod.constants import boltzmann
from molmod.units import kjmol
from parsl.data_provider.files import File
from scipy.interpolate import RBFInterpolator


class Energy():
    """base class for biased Energy of MD simulation."""

    def __init__(self) -> None:
        pass

    def compute_coor(self, sp: SystemParams, gpos=False, vir=False):
        """Computes the bias, the gradient of the bias wrt the coordinates and
        the virial."""
        raise NotImplementedError

    def save(self, filename):
        with open(filename, 'wb') as f:
            dill.dump(self, f)

    @staticmethod
    def load(filename) -> Energy:
        with open(filename, 'rb') as f:
            self = dill.load(f)
        return self


class Bias(Energy, ABC):
    """base class for biased MD runs."""

    def __init__(self, cvs: CV, start=None, step=None) -> None:
        """"args:
                cvs: collective variables
                start: number of md steps before update is called
                step: steps between update is called"""
        super().__init__()

        self.cvs = cvs
        self.start = start
        self.step = step

        self.__comp()

        self.finalized = False

    def update_bias(self, sp: SystemParams):
        """update the bias.

        Can only change the properties from _get_args
        """

    class _HashableArrayWrapper:
        """#see https://github.com/google/jax/issues/4572"""

        def __init__(self, val):
            self.val = val

        def __hash__(self):
            return hash(self.val.tobytes())

        def __eq__(self, other):
            eq = isinstance(
                other, Bias._HashableArrayWrapper) and (self.__hash__()
                                                        == other.__hash__())
            return eq

    @staticmethod
    def _gnool_jit(fun, static_array_argnums=(), static_argnums=()):
        """#see https://github.com/google/jax/issues/4572"""

        @partial(jit, static_argnums=static_array_argnums + static_argnums)
        def callee(*args):
            args = list(args)
            for i in static_array_argnums:
                args[i] = args[i].val
            return fun(*args)

        def caller(*args):
            args = list(args)
            for i in static_array_argnums:
                args[i] = Bias._HashableArrayWrapper(args[i])
            return callee(*args)

        return caller

    @partial(jit, static_argnums=(0, 2, 3))
    def compute_coor(self, sp: SystemParams, gpos=False, vir=False):
        """Computes the bias, the gradient of the bias wrt the coordinates and
        the virial."""

        if sp.cell is not None:
            if not (sp.cell.shape == (0, 3) or sp.cell.shape == (3, 3)):
                raise NotImplementedError(
                    "other cell shapes not yet supported")

        # bv = (vir is not None)
        # bg = (gpos is not None)

        [cvs, jac] = self.cvs.compute(sp=sp,
                                      jac_p=gpos,
                                      jac_c=vir)
        [ener, de] = self.compute(cvs, diff=(gpos or vir))

        e_gpos = jnp.einsum('j,jkl->kl', de, jac.coordinates) if gpos else None
        e_vir = jnp.einsum('ji,k,kjl->il', sp.cell, de,
                           jac.cell) if vir else None

        return ener, e_gpos, e_vir

    def __comp(self):
        args = self.get_args()
        static_array_argnums = tuple(i + 1 for i in range(len(args)))

        self.e = Bias._gnool_jit(partial(self._compute),
                                 static_array_argnums=static_array_argnums)
        self.de = Bias._gnool_jit(jacfwd(self.e, argnums=(0,)),
                                  static_array_argnums=static_array_argnums)

    @partial(jit, static_argnums=(0, 2, 3))
    def compute(self, cvs, diff=False, map=True):
        """compute the energy and derivative.

        If map==False, the cvs are assumed to be already mapped
        """

        if map:
            cvs = self.cvs.metric.map(cvs)

        if diff:
            dcvs = jacfwd(self.cvs.metric.map)(cvs)

        E = self.e(cvs, *self.get_args())
        if diff:
            diffE = self.de(cvs, *self.get_args())[0]
            diffE = jnp.einsum('i,ij->j', diffE, dcvs)  # apply chain rule
        else:
            diffE = None

        return E, diffE

    @abstractmethod
    def _compute(self, cvs, *args):
        """function that calculates the bias potential."""
        raise NotImplementedError

    @abstractmethod
    def get_args(self):
        """function that return dictionary with kwargs of _compute."""
        return []

    def finalize(self):
        """Should be called at end of metadynamics simulation.

        Optimises compute
        """

        self.finalized = True

    @staticmethod
    def load(filename) -> Bias:
        return Energy.load(filename)

    def plot(self, name, n=50, traj=None, vmin=0, vmax=100, map=True, ):
        """plot bias."""

        assert self.cvs.n == 2

        bins = self.cvs.metric.grid(n=n, map=map, endpoints=True)
        mg = np.meshgrid(*bins, indexing='xy')

        xlim = [mg[0].min(), mg[0].max()]
        ylim = [mg[1].min(), mg[1].max()]

        bias, _ = jnp.apply_along_axis(self.compute,
                                       axis=0,
                                       arr=np.array(mg),
                                       diff=False,
                                       map=not map)

        extent = [xlim[0], xlim[1], ylim[0], ylim[1]]

        if map is False:
            mask = self.cvs.metric._get_mask(tol=0.01, interp_mg=mg)
            bias = bias*mask
        # normalise lowest point of bias
        bias -= bias[~np.isnan(bias)].min()

        # plt.clf()
        plt.switch_backend('PDF')

        # plt.rc('font', **{'family': 'sans-serif'})

        fig, ax = plt.subplots()

        p = ax.imshow(bias / (kjmol),
                      cmap=plt.get_cmap('rainbow'),
                      origin='lower',
                      extent=extent,
                      vmin=vmin,
                      vmax=vmax
                      )

        ax.set_xlabel('cv1', fontsize=16)
        ax.set_ylabel('cv2', fontsize=16)

        cbar = fig.colorbar(p)
        cbar.set_label('Bias [kJ/mol]', fontsize=16)

        if traj is not None:

            if not isinstance(traj, Iterable):
                traj = [traj]
            for tr in traj:
                # trajs are ij indexed
                ax.scatter(tr[:, 0], tr[:, 1], s=3)

        ax.set_title(name)

        os.makedirs(os.path.dirname(name), exist_ok=True)

        fig.set_size_inches([12, 8])

        print(f"name figure = {name}")

        fig.savefig(name)


@bash_app_python
def plot_app(bias: Bias,
             outputs: List[File],
             n: int = 50,
             vmin: float = 0,
             vmax: float = 100,
             map: bool = True,
             traj:  Optional[List[np.ndarray]] = None):
    bias.plot(name=outputs[0].filepath, n=n, traj=traj,
              vmin=vmin, vmax=vmax, map=map)


class CompositeBias(Bias):
    """Class that combines several biases in one single bias."""

    def __init__(self, biases: Iterable[Bias], fun=jnp.sum) -> None:

        self.init = True

        self.biases = []

        self.start_list = np.array([], dtype=np.int16)
        self.step_list = np.array([], dtype=np.int16)
        self.args_shape = np.array([0])
        self.cvs = None

        for bias in biases:
            self._append_bias(bias)

        self.fun = fun

        super().__init__(cvs=self.cvs, start=0, step=1)
        self.init = True

    def _append_bias(self, b: Bias):

        if isinstance(b, NoneBias):
            return

        self.biases.append(b)

        self.start_list = np.append(self.start_list, b.start if
                                    (b.start is not None) else -1)
        self.step_list = np.append(self.step_list, b.step if
                                   (b.step is not None) else -1)
        self.args_shape = np.append(self.args_shape,
                                    len(b.get_args()) + self.args_shape[-1])

        if self.cvs is None:
            self.cvs = b.cvs
        else:
            pass
            # assert self.cvs == b.cvs, "CV should be the same"

    def _compute(self, cvs, *args):

        e = jnp.array([
            self.biases[i]._compute(
                cvs, *args[self.args_shape[i]:self.args_shape[i + 1]])
            for i in range(len(self.biases))
        ])

        return self.fun(e)

    def finalize(self):
        for b in self.biases:
            b.finalize()

    def update_bias(self, sp: SystemParams):

        if self.finalized:
            return

        mask = (self.start_list == 0)

        self.start_list[mask] += self.step_list[mask]
        self.start_list -= 1

        for i in np.argwhere(mask):
            self.biases[int(i)].update_bias(sp=sp)

    def get_args(self):
        return [a for b in self.biases for a in b.get_args()]


class MinBias(CompositeBias):

    def __init__(self, biases: Iterable[Bias]) -> None:
        super().__init__(biases, fun=jnp.min)


class BiasF(Bias):
    """Bias according to CV."""

    def __init__(self, cvs: CV, f=None):

        self.f = f if (f is not None) else lambda _: jnp.zeros((1,))
        self.f = jit(self.f)
        super().__init__(cvs, start=None, step=None)

    def _compute(self, cvs):
        return self.f(cvs)[0]

    def get_args(self):
        return []


class NoneBias(BiasF):
    """dummy bias."""

    def __init__(self, cvs: CV):
        super().__init__(cvs)


class HarmonicBias(Bias):
    """Harmonic bias potential centered arround q0 with force constant k."""

    def __init__(self, cvs: CV, q0, k):
        """generate harmonic potentia;

        Args:
            cvs: CV
            q0: rest pos spring
            k: force constant spring
        """

        if isinstance(k, float):
            k = q0 * 0 + k
        else:
            assert k.shape == q0.shape
        assert np.all(k > 0)
        self.k = jnp.array(k)
        self.q0 = jnp.array(q0)

        super().__init__(cvs)

    def _compute(self, cvs, *args):
        r = self.cvs.metric.difference(cvs, self.q0)
        return jnp.einsum('i,i,i', self.k, r, r)

    def get_args(self):
        return []


class BiasMTD(Bias):
    r"""A sum of Gaussian hills, for instance used in metadynamics:
    Adapted from Yaff.

    V = \sum_{\\alpha} K_{\\alpha}} \exp{-\sum_{i}
    \\frac{(q_i-q_{i,\\alpha}^0)^2}{2\sigma^2}}

    where \\alpha loops over deposited hills and i loops over collective
    variables.
    """

    def __init__(self,
                 cvs: CV,
                 K,
                 sigmas,
                 tempering=0.0,
                 start=None,
                 step=None):
        """_summary_

        Args:
            cvs: _description_
            K: _description_
            sigmas: _description_
            start: _description_. Defaults to None.
            step: _description_. Defaults to None.
            tempering: _description_. Defaults to 0.0.
        """

        if isinstance(sigmas, float):
            sigmas = jnp.array([sigmas])
        if isinstance(sigmas, np.ndarray):
            sigmas = jnp.array(sigmas)

        self.ncv = cvs.n
        assert sigmas.ndim == 1
        assert sigmas.shape[0] == self.ncv
        assert jnp.all(sigmas > 0)
        self.sigmas = sigmas
        self.sigmas_isq = 1.0 / (2.0 * sigmas**2.0)

        self.Ks = jnp.zeros((0,))
        self.q0s = jnp.zeros((0, self.ncv))

        self.tempering = tempering
        self.K = K

        super().__init__(cvs, start, step)

    def _add_hill_cv(self, q0, K):
        """Deposit a single hill.

        Args:
            q0: A NumPy array [Ncv] specifying the Gaussian center for each
            collective variable, or a single float if there is only one
            collective variable
            K: the force constant of this hill
        """
        if isinstance(q0, float):
            assert self.ncv == 1
            q0 = jnp.array([q0])
        assert q0.ndim == 1
        assert q0.shape[0] == self.ncv
        self.q0s = jnp.append(self.q0s, jnp.array([q0]), axis=0)
        self.Ks = jnp.append(self.Ks, jnp.array([K]), axis=0)

    def _add_hills_cv(self, q0s, Ks):
        """Deposit multiple hills.

        Args:
        q0s: A NumPy array [Nhills,Ncv]. Each row represents a hill,
        specifying the Gaussian center for each collective variable
        K: A NumPy array [Nhills] providing the force constant of each hill.
        """
        assert q0s.ndim == 2
        assert q0s.shape[1] == self.ncv
        self.q0s = jnp.concatenate((self.q0s, q0s), axis=0)
        assert Ks.ndim == 1
        assert Ks.shape[0] == q0s.shape[0]
        self.Ks = jnp.concatenate((self.Ks, Ks), axis=0)

    def update_bias(self, sp: SystemParams):
        """_summary_

        Args:
            coordinates: _description_
            cell: _description_

        Raises:
            NotImplementedError: _description_
        """
        if self.finalized:
            return

        # Compute current CV values
        q0s, _, _ = self.cvs.compute(sp)
        # Compute force constant
        K = self.K
        if self.tempering != 0.0:
            raise NotImplementedError("untested")
            # K *= jnp.exp(-self.compute() / molmod.constants.boltzmann /
            #              self.tempering)
        # Add a hill
        self._add_hill_cv(q0s, K)

    def _compute(self, cvs, q0s, Ks):
        """Computes sum of hills."""

        deltas = jnp.apply_along_axis(
            self.cvs.metric.difference,
            axis=1,
            arr=q0s,
            x2=cvs,
        )

        exparg = jnp.einsum('ji,ji,i -> j', deltas, deltas, self.sigmas_isq)
        energy = jnp.sum(Ks * jnp.exp(-exparg))

        return energy

    def get_args(self):
        return [self.q0s, self.Ks]


class GridBias(Bias):
    """Bias interpolated from lookup table on uniform grid.

    values are caluclated in bin centers
    """

    def __init__(self,
                 cvs: CV,
                 vals,
                 bounds=None,
                 start=None,
                 step=None,
                 centers=True,
                 ) -> None:
        super().__init__(cvs, start, step)

        if not centers:
            raise NotImplementedError
        assert cvs.n == 2

        # extend periodically
        self.n = np.array(vals.shape)

        bias = np.zeros(np.array(vals.shape) + 2)
        bias[1:-1, 1:-1] = vals

        if self.cvs.metric.periodicities[0]:
            bias[0, :] = bias[-2, :]
            bias[-1, :] = bias[1, :]
        else:
            bias[0, :] = bias[1, :]
            bias[-1, :] = bias[-2, :]

        if self.cvs.metric.periodicities[1]:
            bias[:, 0] = bias[:, -2]
            bias[:, -1] = bias[:, 1]
        else:
            bias[:, 0] = bias[:, 1]
            bias[:, -1] = bias[:, -2]

        # do general interpolation
        x, y = np.indices(bias.shape)
        mask = np.isnan(bias)
        rbf = RBFInterpolator(np.array([x[~mask], y[~mask]]).T, bias[~mask])
        bias[mask] = rbf(np.array([x[mask], y[mask]]).T)

        self.vals = bias

        if bounds is not None:
            per = np.array(bounds)
        else:
            per = np.array(self.cvs.metric._boundaries)

        self.per = jnp.array(per)

    def _compute(self, cvs):
        per = self.per
        # map to halfway array and shift on for (for side rows)
        coords = jnp.array((cvs - per[:, 0]) / (per[:, 1] - per[:, 0]) *
                           (self.n)) + 0.5

        return jsp.ndimage.map_coordinates(self.vals,
                                           coords,
                                           mode='nearest',
                                           order=1)

    def get_args(self):
        return []


class FesBias(Bias):
    """FES bias wraps another (grid) Bias. The compute function properly accounts for the transformation formula F_{unmapped} = F_{mapped} + KT*ln( J(cvs))"""

    def __init__(self, bias: Bias, T) -> None:

        self.bias = bias
        self.T = T

        super().__init__(cvs=bias.cvs, start=bias.start, step=bias.step)

    def compute(self, cvs, diff=False, map=True):
        if map is True:
            return self.bias.compute(cvs, diff=diff, map=True)

        e, de = self.bias.compute(cvs, diff=diff, map=map)

        r = jit(lambda x: self.T * boltzmann *
                jnp.log(jnp.abs(jnp.linalg.det(jacfwd(self.bias.cvs.metric.map)(x)))))

        e += r(cvs)

        if diff:
            de += jit(jacfwd(r))(cvs)

        return e + r(cvs), de

    def update_bias(self, sp: SystemParams):
        self.bias.update_bias(sp=sp)

    def _compute(self, cvs, *args):
        """function that calculates the bias potential."""
        return self.bias._compute(cvs, *args)

    def get_args(self):
        """function that return dictionary with kwargs of _compute."""
        return self.bias.get_args()


class CvMonitor(BiasF):

    def __init__(self, cvs: CV, start=0, step=1):
        super().__init__(cvs, f=None)
        self.start = start
        self.step = step

        self.last_cv = np.nan
        self.transitions = np.zeros((0, self.cvs.metric.ndim, 2))

    def update_bias(self, sp: SystemParams):
        if self.finalized:
            return

        new_cv, _ = self.cvs.compute(sp=sp)

        if jnp.linalg.norm(new_cv - self.last_cv) > 1:
            new_trans = np.array([[new_cv, self.last_cv]])
            print(f"new trans {new_trans}")

            self.transitions = np.vstack(
                (self.transitions, new_trans))

            print(self.transitions)

        self.last_cv = new_cv


class BiasPlumed(Bias):
    pass
