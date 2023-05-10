import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import ase.calculators.calculator
import ase.cell
import ase.geometry
import ase.stress
import ase.units
import numpy as np
import yaff
from ase.calculators.cp2k import CP2K
from IMLCV.base.bias import Energy
from IMLCV.base.bias import EnergyError
from IMLCV.base.bias import EnergyResult
from IMLCV.configs.config_general import get_cp2k
from molmod.units import angstrom
from molmod.units import electronvolt

yaff.log.set_level(yaff.log.silent)

if TYPE_CHECKING:
    pass


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
        out = self.ff.system.cell.rvecs[:]  # empty cell represented as array with shape (0,3)
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
        except BaseException:
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
            except ValueError:  # target does not start with origin
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

        params["command"] = get_cp2k()

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
            f"The cp2k calculator failed to provide an energy. The end of the output from cp2k.out is { file}",
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
