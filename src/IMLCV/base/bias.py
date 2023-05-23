from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from collections.abc import Iterable
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import cloudpickle
import jax
import jax.numpy as jnp
import jax_dataclasses
import matplotlib.pyplot as plt
import numpy as np
import yaff
from IMLCV.base.CV import CollectiveVariable
from IMLCV.base.CV import CV
from IMLCV.base.CV import NeighbourList
from IMLCV.base.CV import SystemParams
from IMLCV.configs.bash_app_python import bash_app_python
from IMLCV.tools.tools import HashableArrayWrapper
from jax import Array
from jax import jit
from jax import value_and_grad
from jax import vmap
from molmod.units import angstrom
from molmod.units import electronvolt
from molmod.units import kjmol
from parsl.data_provider.files import File

yaff.log.set_level(yaff.log.silent)

if TYPE_CHECKING:
    from IMLCV.base.MdEngine import MDEngine


######################################
#              Energy                #
######################################


@jax_dataclasses.pytree_dataclass
class EnergyResult:
    energy: float
    gpos: Array | None = None
    vtens: Array | None = None

    def __post_init__(self):
        if isinstance(self.gpos, Array):
            self.__dict__["gpos"] = jnp.array(self.gpos)
        if isinstance(self.vtens, Array):
            self.__dict__["vtens"] = jnp.array(self.vtens)

    def __add__(self, other) -> EnergyResult:
        assert isinstance(other, EnergyResult)

        gpos = self.gpos
        if self.gpos is None:
            assert other.gpos is None
        else:
            assert other.gpos is not None
            gpos += other.gpos

        vtens = self.vtens

        if other.vtens is not None:
            if vtens is not None:
                vtens += other.vtens
            else:
                vtens = other.vtens

        return EnergyResult(energy=self.energy + other.energy, gpos=gpos, vtens=vtens)

    def __str__(self) -> str:
        str = f"energy [eV]: {self.energy/electronvolt}"
        if self.gpos is not None:
            str += f"\ndE/dx^i_j [eV/angstrom] \n {self.gpos[:]*angstrom/electronvolt}"
        if self.vtens is not None:
            str += f"\n  viriaal [eV] \n {self.vtens[:] / electronvolt }"

        return str


class BC:
    """base class for biased Energy of MD simulation."""

    def __init__(self) -> None:
        pass

    # def compute_from_system_params(
    #     self,
    #     gpos=False,
    #     vir=False,
    #     sp: SystemParams | None = None,
    #     nl: NeighbourList | None = None,
    # ) -> EnergyResult:
    #     """Computes the bias, the gradient of the bias wrt the coordinates and
    #     the virial."""
    #     raise NotImplementedError

    def save(self, filename: str | Path):
        if isinstance(filename, str):
            filename = Path(filename)
        if not filename.parent.exists():
            filename.parent.mkdir(parents=True, exist_ok=True)

        with open(filename, "wb") as f:
            cloudpickle.dump(self, f)

    @staticmethod
    def load(filename) -> BC:
        with open(filename, "rb") as f:
            self = cloudpickle.load(f)
        return self


class EnergyError(Exception):
    pass


class Energy(BC):
    @staticmethod
    def load(filename) -> Energy:
        energy = BC.load(filename=filename)
        assert isinstance(energy, Energy)
        return energy

    @property
    @abstractmethod
    def cell(self):
        pass

    @cell.setter
    @abstractmethod
    def cell(self, cell):
        pass

    @property
    @abstractmethod
    def coordinates(self):
        pass

    @coordinates.setter
    @abstractmethod
    def coordinates(self, coordinates):
        pass

    @property
    def sp(self) -> SystemParams:
        return SystemParams(coordinates=self.coordinates, cell=self.cell)

    @sp.setter
    def sp(self, sp: SystemParams):
        self.cell = sp.cell
        self.coordinates = sp.coordinates

    @abstractmethod
    def _compute_coor(self, gpos=False, vir=False) -> EnergyResult:
        pass

    def _handle_exception(self):
        return ""

    def compute_from_system_params(
        self,
        gpos=False,
        vir=False,
        sp: SystemParams | None = None,
        nl: NeighbourList | None = None,
    ) -> EnergyResult:
        if sp is not None:
            raise NotImplementedError("untested")
            self.sp = sp

        # try:
        return self._compute_coor(gpos=gpos, vir=vir)


class PlumedEnerg(Energy):
    pass


######################################
#       Biases                       #
######################################


class BiasError(Exception):
    pass


class Bias(BC, ABC):
    """base class for biased MD runs."""

    def __init__(
        self,
        collective_variable: CollectiveVariable,
        start=None,
        step=None,
    ) -> None:
        """args:
        cvs: collective variables
        start: number of md steps before update is called
        step: steps between update is called"""
        super().__init__()

        self.collective_variable = collective_variable
        self.start = start
        self.step = step
        self.couter = 0

        self.finalized = False

    def update_bias(
        self,
        md: MDEngine,
    ):
        """update the bias.

        Can only change the properties from _get_args
        """

    def _update_bias(self):
        """update the bias.

        Can only change the properties from _get_args
        """
        if self.finalized:
            return False

        if self.start is None or self.step is None:
            return False

        if self.start == 0:
            self.start += self.step - 1
            return True
        self.start -= 1
        return False

    # @partial(jit, static_argnums=(0, 2, 3))
    @partial(jax.jit, static_argnames=["self", "gpos", "vir"])
    def compute_from_system_params(
        self,
        sp: SystemParams,
        gpos=False,
        vir=False,
        nl: NeighbourList | None = None,
    ) -> tuple[CV, EnergyResult]:
        """Computes the bias, the gradient of the bias wrt the coordinates and
        the virial."""

        if sp.batched:
            if nl is not None:
                assert nl.batched
                return vmap(
                    self.compute_from_system_params,
                    in_axes=(0, None, None, 0),
                )(sp, gpos, vir, nl)
            else:
                return vmap(
                    self.compute_from_system_params,
                    in_axes=(0, None, None, None),
                )(sp, gpos, vir, nl)

        [cvs, jac] = self.collective_variable.compute_cv(
            sp=sp,
            nl=nl,
            jacobian=gpos or vir,
        )
        [ener, de] = self.compute_from_cv(cvs, diff=(gpos or vir))

        # def _resum(sp, jac, de):
        e_gpos = None
        if gpos:
            es = "nj,njkl->nkl"
            if not sp.batched:
                es = es.replace("n", "")
            e_gpos = jnp.einsum(es, de.cv, jac.cv.coordinates)

        e_vir = None
        if vir and sp.cell is not None:
            # transpose, see https://pubs.acs.org/doi/suppl/10.1021/acs.jctc.5b00748/suppl_file/ct5b00748_si_001.pdf s1.4 and S1.22
            es = "nji,nk,nkjl->nli"
            if not sp.batched:
                es = es.replace("n", "")
            e_vir = jnp.einsum(es, sp.cell, de.cv, jac.cv.cell)

        return cvs, EnergyResult(ener, e_gpos, e_vir)

    @partial(jax.jit, static_argnames=["self", "diff"])
    def compute_from_cv(self, cvs: CV, diff=False) -> CV:
        """compute the energy and derivative.

        If map==False, the cvs are assumed to be already mapped
        """
        assert isinstance(cvs, CV)

        # map compute command
        def f0(x):
            args = [HashableArrayWrapper(a) for a in self.get_args()]
            static_array_argnums = tuple(i + 1 for i in range(len(args)))

            # if jit:
            return jax.jit(self._compute, static_argnums=static_array_argnums)(
                x,
                *args,
            )

        def f1(x):
            return value_and_grad(f0)(x) if diff else (f0(x), None)

        def f2(cvs):
            return vmap(f1)(cvs) if cvs.batched else f1(cvs)

        return f2(cvs)

    @abstractmethod
    def _compute(self, cvs, *args):
        """function that calculates the bias potential. CVs live in mapped space"""
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

    # def __getstate__(self):
    #     return self.__dict__

    # def __setstate__(self, state):
    #     self.__init__(**state)
    #     return self

    @staticmethod
    def load(filename) -> Bias:
        bias = BC.load(filename=filename)
        assert isinstance(bias, Bias)
        return bias

    def plot(
        self,
        name,
        x_unit: str | None = None,
        y_unit: str | None = None,
        n=50,
        traj: list[CV] | None = None,
        vmin=0,
        vmax=100 * kjmol,
        map=False,
        inverted=False,
        margin=None,
        x_lim=None,
        y_lim=None,
        bins=None,
    ):
        """plot bias."""

        if self.collective_variable.n == 1:
            if bins is None:
                [bins] = self.collective_variable.metric.grid(
                    n=n,
                    endpoints=True,
                    margin=margin,
                )

            if x_unit is not None:
                if x_unit == "rad":
                    x_unit_label = "rad"
                    x_fact = 0
                elif x_unit == "ang":
                    x_unit_label = "Ang"
                    x_fact = angstrom
            else:
                x_fact = 1
                x_unit_label = "a.u."

            if x_lim is None:
                xlim = [bins.min() / x_fact, bins.max() / x_fact]

            extent = [xlim[0], xlim[1]]

            @jit
            def f(point):
                return self.compute_from_cv(
                    CV(cv=point),
                    diff=False,
                )

            bias, _ = jnp.apply_along_axis(f, axis=0, arr=jnp.array([bins]))

            if inverted:
                bias = -bias
            bias -= bias[~np.isnan(bias)].min()

            bias = jnp.reshape(bias, (-1,))

            # plt.switch_backend("PDF")
            fig, ax = plt.subplots()

            ax.set_xlim(*extent)
            ax.set_ylim(vmin / kjmol, vmax / kjmol)
            p = ax.plot(bins, bias / (kjmol))
            ax2 = ax.twinx()

            ax.set_xlabel(f"cv1 [{x_unit_label}]", fontsize=16)
            ax.set_ylabel("Bias [kJ/mol]]", fontsize=16)

            ax.tick_params(axis="both", which="major", labelsize=18)
            ax.tick_params(axis="both", which="minor", labelsize=16)

            if traj is not None:
                if not isinstance(traj, Iterable):
                    traj = [traj]
                for tr in traj:
                    # trajs are ij indexed
                    _ = ax2.hist(tr.cv, density=True, histtype="step")

        elif self.collective_variable.n == 2:
            if bins is None:
                bins = self.collective_variable.metric.grid(
                    n=n,
                    endpoints=True,
                    margin=margin,
                )
            mg = np.meshgrid(*bins, indexing="xy")

            if x_unit is not None:
                if x_unit == "rad":
                    x_unit_label = "rad"
                    x_fact = 0
                elif x_unit == "ang":
                    x_unit_label = "Ang"
                    x_fact = angstrom
            else:
                x_fact = 1
                x_unit_label = "a.u."

            if y_unit is not None:
                if y_unit == "rad":
                    y_unit_label = "rad"
                    y_fact = 0
                elif x_unit == "ang":
                    y_unit_label = "Ang"
                    y_fact = angstrom
            else:
                y_fact = 1
                y_unit_label = "a.u."

            if x_lim is None:
                xlim = [mg[0].min() / x_fact, mg[0].max() / x_fact]
            if y_lim is None:
                ylim = [mg[1].min() / y_fact, mg[1].max() / y_fact]

            extent = [xlim[0], xlim[1], ylim[0], ylim[1]]

            @jit
            def f(point):
                return self.compute_from_cv(
                    CV(cv=point),
                    diff=False,
                )

            bias, _ = jnp.apply_along_axis(f, axis=0, arr=np.array(mg))

            if inverted:
                bias = -bias
            bias -= bias[~np.isnan(bias)].min()

            # plt.switch_backend("PDF")
            fig, ax = plt.subplots()

            p = ax.imshow(
                bias / (kjmol),
                cmap=plt.get_cmap("rainbow"),
                origin="lower",
                extent=extent,
                vmin=vmin / kjmol,
                vmax=vmax / kjmol,
            )

            ax.set_xlabel(f"cv1 [{x_unit_label}]", fontsize=16)
            ax.set_ylabel(f"cv2 [{y_unit_label}]", fontsize=16)

            ax.tick_params(axis="both", which="major", labelsize=18)
            ax.tick_params(axis="both", which="minor", labelsize=16)

            cbar = fig.colorbar(p)
            cbar.set_label("Bias [kJ/mol]", size=18)

            if traj is not None:
                if not isinstance(traj, Iterable):
                    traj = [traj]
                for tr in traj:
                    # trajs are ij indexed
                    ax.scatter(tr.cv[:, 0], tr.cv[:, 1], s=3)

        else:
            raise ValueError

        Path(name).parent.mkdir(parents=True, exist_ok=True)

        plt.tight_layout()
        plt.savefig(name)

        plt.close(fig=fig)  # write out


@bash_app_python(executors=["default"])
def plot_app(
    bias: Bias,
    outputs: list[File],
    n: int = 50,
    vmin: float = 0,
    vmax: float = 100 * kjmol,
    map: bool = True,
    inverted=False,
    traj: list[CV] | None = None,
    margin=None,
    x_unit=None,
    y_unit=None,
    x_lim=None,
    y_lim=None,
    bins=None,
):
    bias.plot(
        name=outputs[0].filepath,
        n=n,
        traj=traj,
        vmin=vmin,
        vmax=vmax,
        map=map,
        inverted=inverted,
        margin=margin,
        x_unit=x_unit,
        y_unit=y_unit,
        x_lim=x_lim,
        y_lim=y_lim,
        bins=bins,
    )


class CompositeBias(Bias):
    """Class that combines several biases in one single bias."""

    def __init__(self, biases: Iterable[Bias], fun=jnp.sum) -> None:
        self.init = True

        self.biases: list[Bias] = []

        # self.start_list = np.array([], dtype=np.int16)
        # self.step_list = np.array([], dtype=np.int16)
        self.args_shape = np.array([0])
        self.collective_variable: CollectiveVariable = None  # type: ignore

        for bias in biases:
            self._append_bias(bias)

        if self.biases is None:
            assert biases[0] is NoneBias
            self.biases = bias[0]

        self.fun = fun

        super().__init__(collective_variable=self.collective_variable, start=0, step=1)
        self.init = True

    def _append_bias(self, b: Bias):
        if b is NoneBias:
            return

        self.biases.append(b)

        # self.start_list = np.append(
        #     self.start_list, b.start if (b.start is not None) else -1
        # )
        # self.step_list = np.append(
        #     self.step_list, b.step if (b.step is not None) else -1
        # )
        self.args_shape = np.append(
            self.args_shape,
            len(b.get_args()) + self.args_shape[-1],
        )

        if self.collective_variable is None:
            self.collective_variable = b.collective_variable
        else:
            pass
            # assert self.cvs == b.cvs, "CV should be the same"

    def _compute(self, cvs, *args):
        return self.fun(
            jnp.array(
                [
                    jnp.reshape(
                        self.biases[i]._compute(
                            cvs,
                            *args[self.args_shape[i] : self.args_shape[i + 1]],
                        ),
                        (),
                    )
                    for i in range(len(self.biases))
                ],
            ),
        )

    def finalize(self):
        for b in self.biases:
            b.finalize()

    def update_bias(
        self,
        md: MDEngine,
    ):
        for b in self.biases:
            b.update_bias(md=md)

    def get_args(self):
        return [a for b in self.biases for a in b.get_args()]


class BiasF(Bias):
    """Bias according to CV."""

    def __init__(self, cvs: CollectiveVariable, g=None):
        self.g = g if (g is not None) else lambda _: jnp.array(0.0)
        # self.g = jit(self.g) #leads to pickler issues
        super().__init__(cvs, start=None, step=None)

    def _compute(self, cvs):
        return self.g(cvs)

    def get_args(self):
        return []


class NoneBias(BiasF):
    """dummy bias."""

    def __init__(self, cvs: CollectiveVariable):
        super().__init__(cvs)
