from __future__ import annotations

import os
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import ase
import ase.calculators.calculator
import ase.cell
import ase.geometry
import ase.stress
import ase.units
import dill
import jax.numpy as jnp
import jax.scipy as jsp
import jax_dataclasses
import matplotlib.pyplot as plt
import numpy as np
import scipy
import yaff
from ase.calculators.cp2k import CP2K
from jax import jit, value_and_grad, vmap
from molmod.units import angstrom, electronvolt, kjmol, nanometer, picosecond
from parsl.data_provider.files import File

yaff.log.set_level(yaff.log.silent)
from configs.bash_app_python import bash_app_python
from IMLCV.base.CV import CV, CollectiveVariable, SystemParams
from IMLCV.base.tools._rbf_interp import RBFInterpolator
from IMLCV.base.tools.tools import HashableArrayWrapper

if TYPE_CHECKING:
    from IMLCV.base.MdEngine import MDEngine

import os
import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from IMLCV.base.CV import (
    CollectiveVariable,
    CvFlow,
    CvMetric,
    NeighbourList,
    SystemParams,
    Volume,
    dihedral,
)
from jax import Array
from molmod import units
from molmod.units import kelvin, kjmol

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
            dill.dump(self, f)

    @staticmethod
    def load(filename) -> BC:
        with open(filename, "rb") as f:
            self = dill.load(f)
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


#         except EnergyError as be:
#             raise EnergyError(
#                 f"""An error occured during the nergy calculation with {self.__class__}.
# The lates coordinates were {self.sp}.
# raised exception from calculator:{be}"""
#             )


class YaffEnergy(Energy):
    def __init__(self, f: Callable[[], yaff.ForceField]) -> None:
        super().__init__()
        self.f = f
        self.ff: yaff.ForceField = f()

    @property
    def cell(self):
        out = self.ff.system.cell.rvecs[
            :
        ]  # empty cell represented as array with shape (0,3)
        if out.size == 0:
            return None
        return out

    @cell.setter
    def cell(self, cell):
        if cell is not None:
            cell = np.array(cell, dtype=np.double)
        self.ff.update_rvecs(cell)

    @property
    def coordinates(self):
        return self.ff.system.pos[:]

    @coordinates.setter
    def coordinates(self, coordinates):
        self.ff.update_pos(coordinates)

    def _compute_coor(self, gpos=False, vir=False) -> EnergyResult:

        gpos_out = np.zeros_like(self.ff.gpos) if gpos else None
        vtens_out = np.zeros_like(self.ff.vtens) if vir else None

        try:
            ener = self.ff.compute(gpos=gpos_out, vtens=vtens_out)
        except BaseException as be:
            raise EnergyError(f"calculating yaff  energy raised execption:\n{be}\n")

        return EnergyResult(ener, gpos_out, vtens_out)

    def __getstate__(self):
        return {"f": self.f, "sp": self.sp}

    def __setstate__(self, state):

        self.f = state["f"]
        self.ff = self.f()
        self.sp = state["sp"]
        return self


class AseEnergy(Energy):
    """Conversion to ASE energy"""

    def __init__(
        self,
        atoms: ase.Atoms,
        calculator: ase.calculators.calculator.Calculator | None = None,
    ):

        self.atoms = atoms

        if calculator is not None:
            self.atoms.calc = self.calculator

    @property
    def cell(self):
        return self.atoms.get_cell()[:] * angstrom

    @cell.setter
    def cell(self, cell):
        self.atoms.set_cell(ase.geometry.Cell(np.array(cell[:]) / angstrom))

    @property
    def coordinates(self):
        return self.atoms.get_positions() * angstrom

    @coordinates.setter
    def coordinates(self, coordinates):
        self.atoms.set_positions(np.array(coordinates[:]) / angstrom)

    def _compute_coor(self, gpos=False, vir=False) -> EnergyResult:
        """use unit conventions of ASE"""

        if self.atoms.calc is None:
            self.atoms.calc = self._calculator()
            # self.atoms.calc.atoms = self.atoms

        try:
            energy = self.atoms.get_potential_energy() * electronvolt
        except:
            self._handle_exception()

        gpos_out = None
        vtens_out = None
        if gpos:
            forces = self.atoms.get_forces()
            gpos_out = -forces * electronvolt / angstrom

        if vir:
            cell = self.atoms.get_cell()
            volume = np.linalg.det(cell)
            stress = self.atoms.get_stress(voigt=False)
            vtens_out = volume * stress * electronvolt

        res = EnergyResult(energy, gpos_out, vtens_out)

        return res

    def _calculator(self) -> ase.calculators.calculator.Calculator:
        raise NotImplementedError

    def _handle_exception(self):
        raise EnergyError("Ase failed to provide an energy\n")

    def __getstate__(self):

        extra_args = {
            "label": self.calculator.label,
        }

        dict = {
            "cc": self.calculator.__class__,
            "calc_args": {**self.calculator.todict(), **extra_args},
            "atoms": self.atoms.todict(),
        }

        return dict

    def __setstate__(self, state):
        clss = state["cc"]
        calc_params = state["calc_args"]
        atom_params = state["atoms"]

        self.atoms = ase.Atoms.fromdict(**atom_params)
        self.calculator = clss(**calc_params)
        self.atoms.calc = self.calculator


class Cp2kEnergy(AseEnergy):
    # override default params, only if explicitly set
    default_parameters = dict(
        auto_write=False,
        basis_set=None,
        basis_set_file=None,
        charge=None,
        cutoff=None,
        force_eval_method=None,
        inp="",
        max_scf=None,
        potential_file=None,
        pseudo_potential=None,
        stress_tensor=True,
        uks=False,
        poisson_solver=None,
        xc=None,
        print_level="LOW",
    )

    def __init__(
        self,
        atoms: ase.Atoms,
        input_file,
        input_kwargs: dict,
        cp2k_path: Path | None = None,
        **kwargs,
    ):

        self.atoms = atoms
        self.cp2k_inp = os.path.abspath(input_file)
        self.input_kwargs = input_kwargs
        self.kwargs = kwargs
        super().__init__(atoms)

        self.rp = cp2k_path

    def _calculator(self):
        def relative(target: Path, origin: Path):
            """return path of target relative to origin"""
            try:
                return Path(target).resolve().relative_to(Path(origin).resolve())
            except ValueError as e:  # target does not start with origin
                # recursion with origin (eventually origin is root so try will succeed)
                return Path("..").joinpath(relative(target, Path(origin).parent))

        rp = Path.cwd()
        rp.mkdir(parents=True, exist_ok=True)
        print(f"saving CP2K output in {rp}")

        new_dict = {}
        for key, val in self.input_kwargs.items():
            assert Path(val).exists()
            new_dict[key] = relative(val, rp)

        with open(self.cp2k_inp) as f:
            inp = "".join(f.readlines()).format(**new_dict)
        params = self.default_parameters.copy()
        params.update(**{"inp": inp, **self.kwargs})

        if "label" in params:
            del params["label"]
            print("ignoring label for Cp2kEnergy")
        if "directory" in params:
            del params["directory"]
            print("ignoring directory for Cp2kEnergy")

        params["directory"] = "."

        calc = CP2K(**params)

        return calc

    def _handle_exception(self):
        p = f"{self.atoms.calc.directory}/cp2k.out"
        assert os.path.exists(p), "no cp2k output file after failure"
        with open(p) as f:
            lines = f.readlines()
        out = min(len(lines), 50)
        assert out != 0, "cp2k.out doesn't contain output"

        file = "\n".join(lines[-out:])

        raise EnergyError(
            f"The cp2k calculator failed to provide an energy. The end of the output from cp2k.out is { file}"
        )

    def __getstate__(self):
        return [
            self.atoms.todict(),
            self.cp2k_inp,
            self.input_kwargs,
            self.kwargs,
        ]

    def __setstate__(self, state):
        atoms_dict, cp2k_inp, input_kwargs, kwargs = state

        self.__init__(
            ase.Atoms.fromdict(atoms_dict),
            cp2k_inp,
            input_kwargs,
            **kwargs,
        )


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
        self, collective_variable: CollectiveVariable, start=None, step=None
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

    @partial(jit, static_argnums=(0, 2, 3))
    def compute_from_system_params(
        self, sp: SystemParams, gpos=False, vir=False, nl: NeighbourList | None = None
    ) -> tuple[CV, EnergyResult]:
        """Computes the bias, the gradient of the bias wrt the coordinates and
        the virial."""

        @NeighbourList.vmap_sp_nl
        def _compute_from_system_params(sp, nl):

            [cvs, jac] = self.collective_variable.compute_cv(
                sp=sp, nl=nl, jacobian=gpos or vir
            )
            [ener, de] = self.compute_from_cv(cvs, diff=(gpos or vir))

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

        return _compute_from_system_params(sp, nl)

    @partial(jit, static_argnums=(0, 2))
    def compute_from_cv(self, cvs: CV, diff=False) -> CV:
        """compute the energy and derivative.

        If map==False, the cvs are assumed to be already mapped
        """
        assert isinstance(cvs, CV)

        # map compute command
        def f0(x):
            args = [HashableArrayWrapper(a) for a in self.get_args()]
            static_array_argnums = tuple(i + 1 for i in range(len(args)))

            return jit(self._compute, static_argnums=static_array_argnums)(x, *args)

        def f2(x):
            return value_and_grad(f0)(x) if diff else (f0(x), None)

        return vmap(f2)(cvs) if cvs.batched else f2(cvs)

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
                    n=n, endpoints=True, margin=margin
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

            # plt.switch_backend("PDF")
            fig, ax = plt.subplots()

            ax.set_xlim(*extent)
            ax.set_ylim(vmin / kjmol, vmax / kjmol)
            p = ax.plot(bins, bias / (kjmol))
            ax2 = ax.twinx()

            ax.set_xlabel(f"cv1 [{x_unit_label}]", fontsize=16)
            ax.set_ylabel(f"Bias [kJ/mol]]", fontsize=16)

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
                    n=n, endpoints=True, margin=margin
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

        # ax.set_title(name)
        os.makedirs(os.path.dirname(name), exist_ok=True)

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

        self.fun = fun

        super().__init__(collective_variable=self.collective_variable, start=0, step=1)
        self.init = True

    def _append_bias(self, b: Bias):

        self.biases.append(b)

        # self.start_list = np.append(
        #     self.start_list, b.start if (b.start is not None) else -1
        # )
        # self.step_list = np.append(
        #     self.step_list, b.step if (b.step is not None) else -1
        # )
        self.args_shape = np.append(
            self.args_shape, len(b.get_args()) + self.args_shape[-1]
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
                    self.biases[i]._compute(
                        cvs, *args[self.args_shape[i] : self.args_shape[i + 1]]
                    )
                    for i in range(len(self.biases))
                ]
            )
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


class MinBias(CompositeBias):
    def __init__(self, biases: Iterable[Bias]) -> None:
        super().__init__(biases, fun=jnp.min)


class BiasF(Bias):
    """Bias according to CV."""

    def __init__(self, cvs: CollectiveVariable, g=None):

        self.g = g if (g is not None) else lambda _: jnp.zeros((cvs.n,))
        self.g = jit(self.g)
        super().__init__(cvs, start=None, step=None)

    def _compute(self, cvs):
        return self.g(cvs)

    def get_args(self):
        return []


class NoneBias(BiasF):
    """dummy bias."""

    def __init__(self, cvs: CollectiveVariable):
        super().__init__(cvs)


class HarmonicBias(Bias):
    """Harmonic bias potential centered arround q0 with force constant k."""

    def __init__(
        self, cvs: CollectiveVariable, q0: CV, k, k_max: Array | float | None = None
    ):
        """generate harmonic potentia;

        Args:
            cvs: CV
            q0: rest pos spring
            k: force constant spring
        """
        super().__init__(cvs)

        if isinstance(k, float):
            k = jnp.zeros_like(q0.cv) + k
        else:
            assert k.shape == q0.cv.shape

        if k_max is not None:
            if isinstance(k_max, float):
                k_max = jnp.zeros_like(q0.cv) + k_max
            else:
                assert k_max.shape == q0.cv.shape

        assert np.all(k > 0)
        self.k = jnp.array(k)
        self.q0 = q0

        self.k_max = k_max
        if k_max is not None:
            assert np.all(k_max > 0)
            self.r0 = k_max / k
            self.y0 = jnp.einsum("i,i,i", k, self.r0, self.r0) / 2

    def _compute(self, cvs: CV, *args):

        r = self.collective_variable.metric.difference(cvs, self.q0)

        def parabola(r):
            return jnp.einsum("i,i,i", self.k, r, r) / 2

        if self.k_max is None:
            return parabola(r)

        return jnp.where(
            jnp.linalg.norm(r) < self.r0,
            parabola(r),
            jnp.sqrt(
                jnp.einsum(
                    "i,i,i,i",
                    self.k_max,
                    self.k_max,
                    jnp.abs(r) - self.r0,
                    jnp.abs(r) - self.r0,
                )
            )
            + self.y0,
        )

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

    def __init__(
        self, cvs: CollectiveVariable, K, sigmas, tempering=0.0, start=None, step=None
    ):
        """_summary_

        Args:
            cvs: _description_
            K: _description_
            sigmas: _description_
            start: _description_. Defaults to None.
            step: _description_. Defaults to None.
            tempering: _description_. Defaults to 0.0.
        """

        # raise NotImplementedError

        if isinstance(sigmas, float):
            sigmas = jnp.array([sigmas])
        if isinstance(sigmas, Array):
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

    def update_bias(
        self,
        md: MDEngine,
    ):
        if not self._update_bias():
            return

        assert md.trajectory_info is not None
        sp = md.sp

        if self.finalized:
            return
        # Compute current CV values
        sp, nl = sp.get_neighbour_list(md.static_trajectory_info.r_cut)
        q0s = self.collective_variable.compute_cv(sp=sp, nl=nl)[0].cv
        K = self.K
        if self.tempering != 0.0:
            raise NotImplementedError("untested")

        self.q0s = jnp.vstack([self.q0s, q0s])
        self.Ks = jnp.array([*self.Ks, K])

    def _compute(self, cvs, q0s, Ks):
        """Computes sum of hills."""

        def f(x):
            return self.collective_variable.metric.difference(x1=CV(cv=x), x2=cvs)

        deltas = jnp.apply_along_axis(f, axis=1, arr=q0s.val)

        exparg = jnp.einsum("ji,ji,i -> j", deltas, deltas, self.sigmas_isq)
        energy = jnp.sum(jnp.exp(-exparg) * Ks.val)

        return energy

    def get_args(self):
        return [self.q0s, self.Ks]


class RbfBias(Bias):
    """Bias interpolated from lookup table on uniform grid.

    values are caluclated in bin centers
    """

    def __init__(
        self,
        cvs: CollectiveVariable,
        vals: Array,
        cv: CV,
        start=None,
        step=None,
        kernel="thin_plate_spline",
        epsilon=None,
        smoothing=0.0,
        degree=None,
    ) -> None:
        super().__init__(cvs, start, step)

        assert cv.batched
        assert cv.shape[1] == cvs.n
        assert len(vals.shape) == 1
        assert cv.shape[0] == vals.shape[0]

        self.rbf = jit(
            RBFInterpolator(
                y=cv,
                kernel=kernel,
                d=vals,
                metric=cvs.metric,
                smoothing=smoothing,
                epsilon=epsilon,
                degree=degree,
            )
        )

        # assert jnp.allclose(vals, self.rbf(cv), atol=1e-7)

    def _compute(self, cvs: CV, *args):

        out = self.rbf(cvs)
        if cvs.batched:
            return out
        return out[0]

    def get_args(self):
        return []


class GridBias(Bias):
    """Bias interpolated from lookup table on uniform grid.

    values are caluclated in bin centers
    """

    def __init__(
        self,
        cvs: CollectiveVariable,
        vals,
        bounds,
        start=None,
        step=None,
        centers=True,
    ) -> None:
        super().__init__(cvs, start, step)

        if not centers:
            raise NotImplementedError

        # extend periodically
        self.n = np.array(vals.shape)

        bias = vals
        for i, p in enumerate(self.collective_variable.metric.periodicities):
            # extend array and fill boundary values if periodic
            def sl(a, b):
                out = [slice(None) for _ in range(self.collective_variable.n)]
                out[a] = b

                return tuple(out)

            def get_ext(i):
                a = np.array(bias.shape)
                a[i] = 1
                part = bias[sl(i, 0)].reshape(a)

                return part * jnp.nan

            bias = np.concatenate((get_ext(i), bias, get_ext(i)), axis=i)

            if p:

                bias[sl(i, 0)] = bias[sl(i, -2)]
                bias[sl(i, -1)] = bias[sl(i, 1)]

        # do general interpolation
        inds_pairs = np.array(np.indices(bias.shape))
        mask = np.isnan(bias)
        inds_pairs[:, mask]

        rbf = scipy.interpolate.RBFInterpolator(
            np.array([i[~mask] for i in inds_pairs]).T,
            bias[~mask],
        )

        bias[mask] = rbf(np.array([i[mask] for i in inds_pairs]).T)

        self.vals = bias
        self.bounds = jnp.array(bounds)

    def _compute(self, cvs: CV, *args):
        # overview of grid points. stars are addded to allow out of bounds extension.
        # the bounds of the x square are per.
        #  ___ ___ ___ ___
        # |   |   |   |   |
        # | * | * | * | * |
        # |___|___|___|___|
        # |   |   |   |   |
        # | * | x | x | * |
        # |___|___|___|___|
        # |   |   |   |   |
        # | * | x | x | * |
        # |___|___|___|___|
        # |   |   |   |   |
        # | * | * | * | * |
        # |___|___|___|___|
        # gridpoints are in the middle of

        # map between vals 0 and 1
        # if self.bounds is not None:
        coords = (cvs.cv - self.bounds[:, 0]) / (self.bounds[:, 1] - self.bounds[:, 0])
        # else:
        #     coords = self.collective_variable.metric.__map(cvs).cv

        # map between vals matrix edges
        coords = (coords * self.n - 0.5) / (self.n - 1)
        # scale to array size and offset extra row
        coords = coords * (self.n - 1) + 1

        return jsp.ndimage.map_coordinates(self.vals, coords, mode="nearest", order=1)  # type: ignore

    def get_args(self):
        return []


class GridBiasNd(Bias):
    # inspiration fromhttps://github.com/stanbiryukov/Nyx/tree/main/nyx/jax

    def __init__(
        self,
        collective_variable: CollectiveVariable,
        vals,
        bounds=None,
        start=None,
        step=None,
    ) -> None:
        super().__init__(collective_variable, start, step)

    def _compute(self, cvs, *args):
        return super()._compute(cvs, *args)


class CvMonitor(BiasF):
    def __init__(self, cvs: CollectiveVariable, start=0, step=1):
        super().__init__(cvs, g=None)
        self.start = start
        self.step = step

        self.last_cv: CV | None = None
        self.transitions = np.zeros((0, self.collective_variable.metric.ndim, 2))

    def update_bias(self, md: MDEngine):

        if not self._update_bias():
            return

        raise NotImplementedError
        assert md.trajectory_info is not None

        sp = md.trajectory_info.sp

        new_cv, _ = self.collective_variable.compute_cv(sp=sp)

        if self.last_cv is not None:
            if self.collective_variable.metric.norm(new_cv, self.last_cv) > 0.1:
                # a = self.collective_variable.metric.__unmap(new_cv).cv
                # b = self.collective_variable.metric.__unmap(self.last_cv).cv
                a = new_cv.cv
                b = new_cv.cv

                new_trans = np.array([np.stack([a, b], axis=1)])
                self.transitions = np.vstack((self.transitions, new_trans))

        self.last_cv = new_cv


class PlumedBias(Bias):
    def __init__(
        self,
        collective_variable: CollectiveVariable,
        timestep,
        kernel=None,
        fn="plumed.dat",
        fn_log="plumed.log",
    ) -> None:

        super().__init__(
            collective_variable,
            start=0,
            step=1,
        )

        self.fn = fn
        self.kernel = kernel
        self.fn_log = fn_log
        self.plumedstep = 0
        self.hooked = False

        self.setup_plumed(timestep, 0)

        raise NotImplementedError("this is untested")

    def setup_plumed(self, timestep, restart):
        r"""Send commands to PLUMED to make it computation-ready.

        **Arguments:**

        timestep
            The timestep (in au) of the integrator

        restart
            Set to an integer value different from 0 to let PLUMED know that
            this is a restarted run
        """
        # Try to load the plumed Python wrapper, quit if not possible
        try:
            from plumed import Plumed
        except:
            raise ImportError

        self.plumed = Plumed(kernel=self.kernel)
        # Conversion between PLUMED internal units and YAFF internal units
        # Note that PLUMED output will follow the PLUMED conventions
        # concerning units
        self.plumed.cmd("setMDEnergyUnits", 1.0 / kjmol)
        self.plumed.cmd("setMDLengthUnits", 1.0 / nanometer)
        self.plumed.cmd("setMDTimeUnits", 1.0 / picosecond)
        # Initialize the system in PLUMED
        self.plumed.cmd("setPlumedDat", self.fn)
        self.plumed.cmd("setNatoms", self.system.natom)
        self.plumed.cmd("setMDEngine", "IMLCV")
        self.plumed.cmd("setLogFile", self.fn_log)
        self.plumed.cmd("setTimestep", timestep)
        self.plumed.cmd("setRestart", restart)
        self.plumed.cmd("init")

    def update_bias(self, md: MDEngine):
        r"""When this point is reached, a complete time integration step was
        finished and PLUMED should be notified about this.
        """
        if not self.hooked:
            self.setup_plumed(timestep=self.plumedstep, restart=int(md.step > 0))
            self.hooked = True

        # PLUMED provides a setEnergy command, which should pass the
        # current potential energy. It seems that this is never used, so we
        # don't pass anything for the moment.
        #        current_energy = sum([part.energy for part in iterative.ff.parts[:-1] if not isinstance(part, ForcePartPlumed)])
        #        self.plumed.cmd("setEnergy", current_energy)
        # Ensure the plumedstep is an integer and not a numpy data type
        self.plumedstep += 1
        self._internal_compute(None, None)
        self.plumed.cmd("update")

    def _internal_compute(self, gpos, vtens):
        self.plumed.cmd("setStep", self.plumedstep)
        self.plumed.cmd("setPositions", self.system.pos)
        self.plumed.cmd("setMasses", self.system.masses)
        if self.system.charges is not None:
            self.plumed.cmd("setCharges", self.system.charges)
        if self.system.cell.nvec > 0:
            rvecs = self.system.cell.rvecs.copy()
            self.plumed.cmd("setBox", rvecs)
        # PLUMED always needs arrays to write forces and virial to, so
        # provide dummy arrays if Yaff does not provide them
        # Note that gpos and forces differ by a minus sign, which has to be
        # corrected for when interacting with PLUMED
        if gpos is None:
            my_gpos = np.zeros(self.system.pos.shape)
        else:
            gpos[:] *= -1.0
            my_gpos = gpos
        self.plumed.cmd("setForces", my_gpos)
        if vtens is None:
            my_vtens = np.zeros((3, 3))
        else:
            my_vtens = vtens
        self.plumed.cmd("setVirial", my_vtens)
        # Do the actual calculation, without an update; this should
        # only be done at the end of a time step
        self.plumed.cmd("prepareCalc")
        self.plumed.cmd("performCalcNoUpdate")
        if gpos is not None:
            gpos[:] *= -1.0
        # Retrieve biasing energy
        energy = np.zeros((1,))
        self.plumed.cmd("getBias", energy)
        return energy[0]


######################################
#              test                  #
######################################


def test_harmonic():

    cvs = CollectiveVariable(
        f=(dihedral(numbers=[4, 6, 8, 14]) + dihedral(numbers=[6, 8, 14, 16])),
        metric=CvMetric(
            periodicities=[True, True], bounding_box=[[0, 2 * np.pi], [0, 2 * np.pi]]
        ),
    )

    bias = HarmonicBias(cvs, q0=np.array([np.pi, -np.pi]), k=1.0)

    x = np.random.rand(2)

    a1, _ = bias.compute_from_cv(np.array([np.pi, np.pi]) + x)
    a2, _ = bias.compute_from_cv(np.array([-np.pi, -np.pi] + x))
    a3, _ = bias.compute_from_cv(np.array([np.pi, -np.pi]) + x)
    a4, _ = bias.compute_from_cv(np.array([-np.pi, np.pi] + x))
    a5, _ = bias.compute_from_cv(np.array([np.pi, np.pi]) + x.T)

    assert pytest.approx(a1, abs=1e-5) == a2
    assert pytest.approx(a1, abs=1e-5) == a3
    assert pytest.approx(a1, abs=1e-5) == a4
    assert pytest.approx(a1, abs=1e-5) == a5


def test_virial():
    # virial for volume based CV is V*I(3)

    metric = CvMetric(periodicities=[False])
    cv0 = CollectiveVariable(f=Volume, metric=metric)
    coordinates = np.random.random((10, 3))
    cell = np.random.random((3, 3))
    vir = np.zeros((3, 3))

    def fun(x):
        return x.cv[0]

    bias = BiasF(cvs=cv0, g=fun)

    _, e_r = bias.compute_from_system_params(
        SystemParams(coordinates=coordinates, cell=cell), vir=True
    )
    vol = e_r.energy
    vir = e_r.vtens
    assert pytest.approx(vir, abs=1e-7) == vol * np.eye(3)


def test_grid_bias():

    # bounds = [[0, 3], [0, 3]]
    n = [4, 6]

    cv = CollectiveVariable(
        CvFlow(func=lambda x: x.coordinates),
        CvMetric(
            periodicities=[False, False],
            bounding_box=np.array([[-2, 2], [1, 5]]),
        ),
    )

    bins = [
        np.linspace(a, b, ni, endpoint=True, dtype=np.double)
        for ni, (a, b) in zip(n, cv.metric.bounding_box)
    ]

    def f(x, y):
        return x**3 + y

    # reevaluation of thermolib histo
    bin_centers1, bin_centers2 = 0.5 * (bins[0][:-1] + bins[0][1:]), 0.5 * (
        bins[1][:-1] + bins[1][1:]
    )
    xc, yc = np.meshgrid(bin_centers1, bin_centers2, indexing="ij")
    xcf = np.reshape(xc, (-1))
    ycf = np.reshape(yc, (-1))
    val = np.array([f(x, y) for x, y in zip(xcf, ycf)]).reshape(xc.shape)

    bias = RbfBias(cvs=cv, vals=val)

    def c(x, y):
        return bias.compute_from_cv(cvs=np.array([x, y]))[0]

    val2 = np.array([c(x, y) for x, y in zip(xcf, ycf)]).reshape(xc.shape)
    assert np.allclose(val, val2)


def test_combine_bias(full_name):
    from IMLCV.base.MdEngine import StaticTrajectoryInfo, YaffEngine
    from yaff.test.common import get_alaninedipeptide_amber99ff

    T = 300 * kelvin

    cv0 = CollectiveVariable(
        f=(dihedral(numbers=[4, 6, 8, 14]) + dihedral(numbers=[6, 8, 14, 16])),
        metric=CvMetric(
            periodicities=[True, True],
            bounding_box=[[-np.pi, np.pi], [-np.pi, np.pi]],
        ),
    )

    bias1 = BiasMTD(
        cvs=cv0, K=2.0 * units.kjmol, sigmas=np.array([0.35, 0.35]), start=25, step=500
    )
    bias2 = BiasMTD(
        cvs=cv0, K=0.5 * units.kjmol, sigmas=np.array([0.1, 0.1]), start=50, step=250
    )

    bias = CompositeBias(biases=[bias1, bias2])

    stic = StaticTrajectoryInfo(
        T=T,
        timestep=2.0 * units.femtosecond,
        timecon_thermo=100.0 * units.femtosecond,
        write_step=1,
        atomic_numbers=np.array(
            [1, 6, 1, 1, 6, 8, 7, 1, 6, 1, 6, 1, 1, 1, 6, 8, 7, 1, 6, 1, 1, 1],
            dtype=int,
        ),
    )

    mde = YaffEngine(
        energy=YaffEnergy(f=get_alaninedipeptide_amber99ff),
        bias=bias,
        static_trajectory_info=stic,
    )

    mde.run(int(1e2))


def test_bias_save(full_name):
    """save and load bias to disk."""
    from IMLCV.examples.example_systems import alanine_dipeptide_yaff

    yaffmd = alanine_dipeptide_yaff(
        bias=lambda cv0: BiasMTD(
            cvs=cv0,
            K=2.0 * units.kjmol,
            sigmas=np.array([0.35, 0.35]),
            start=25,
            step=500,
        )
    )
    yaffmd.run(int(1e3))

    yaffmd.bias.save("output/bias_test_2.xyz")
    bias = Bias.load("output/bias_test_2.xyz")

    from IMLCV.base.CV import CV

    cvs = CV(cv=jnp.array([0.0, 0.0]))

    [b, db] = yaffmd.bias.compute_from_cv(cvs=cvs, diff=True)
    [b2, db2] = bias.compute_from_cv(cvs=cvs, diff=True)

    assert pytest.approx(b) == b2
    assert pytest.approx(db.cv) == db2.cv


if __name__ == "__main__":
    test_harmonic()
    test_virial()
    test_grid_bias()
    test_virial()
    with tempfile.TemporaryDirectory() as tmp:
        test_combine_bias(full_name=f"{tmp}/combine.h5")
        test_bias_save(full_name=f"{tmp}/bias_save.h5")
