import os

# from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import ase
import jax
import jax.numpy as jnp
import numpy as np
from ase import geometry as ase_geometry
from ase.calculators.calculator import Calculator
from openmm import Context, State, System, Vec3
from openmm.app import Simulation, Topology

from IMLCV.base.bias import Energy, EnergyError, EnergyResult, EnergyFn
from IMLCV.base.CV import NeighbourList, SystemParams
from IMLCV.base.datastructures import MyPyTreeNode, field
from IMLCV.base.UnitsConstants import angstrom, electronvolt, kjmol, nanometer
from IMLCV.configs.config_general import REFERENCE_COMMANDS, ROOT_DIR


class OpenMmEnergy(Energy):
    system: System = field(
        pytree_node=False,
    )
    context: Context | None = field(
        pytree_node=False, default=None, hash=False
    )  # not important to compare energy functions

    @staticmethod
    def create(system: System):
        return OpenMmEnergy(system=system)

    def get_context(self):
        from openmm import LangevinIntegrator, System, VerletIntegrator

        self.context = Context(
            self.system,
            VerletIntegrator(
                0.0  # type:ignore
            ),
        )

    def _compute_coor(self, sp: SystemParams, nl: NeighbourList, gpos=False, vir=False) -> EnergyResult:
        if self.context is None:
            self.get_context()

        self.context.setPositions(sp.coordinates.__array__() / nanometer)
        if sp.cell is not None:
            self.context.setPeriodicBoxVectors(
                sp.cell[0, :].__array__() / nanometer,
                sp.cell[1, :].__array__() / nanometer,
                sp.cell[2, :].__array__() / nanometer,
            )

        state: State = self.context.getState(
            energy=True,
            forces=gpos,
        )

        assert not vir

        res = EnergyResult(
            energy=jnp.asarray(
                state.getPotentialEnergy()._value * kjmol,
            ),  # type:ignore
            gpos=-jnp.asarray(state.getForces(asNumpy=True)) * kjmol / nanometer if gpos else None,
        )

        return res

    def __getstate__(self):
        return {
            "system": self.system,
        }

    def __setstate__(self, state):
        sys = state["system"]
        if isinstance(sys, str):
            from openmm import XmlSerializer

            sys = XmlSerializer.deserialize(sys)

        self.system = sys

        return self


class AseEnergy(Energy):
    """Conversion to ASE energy"""

    atoms: ase.Atoms = field(pytree_node=False)
    # calculator: Calculator =

    @property
    def calculator(self) -> Calculator | None:
        return self.atoms.calc

    @calculator.setter
    def calculator(self, calc: Calculator):
        self.atoms.calc = calc
        return

    @staticmethod
    def create(
        atoms: ase.Atoms,
        calculator: Calculator | None = None,
    ):
        self = AseEnergy(
            atoms=atoms,
        )

        if calculator is not None:
            self.calculator = calculator

        else:
            self.calculator = self._calculator()

        return self

    def _compute_coor(self, sp, nl, gpos=False, vir=False) -> EnergyResult:
        """use unit conventions of ASE"""

        # self.sp = sp

        self.atoms.set_positions(sp.coordinates.__array__() / angstrom)
        if sp.cell is not None:
            self.atoms.set_cell(sp.cell.__array__() / angstrom)

        if self.calculator is None:
            print("ASE calculator is None, recreating")
            # sp_save = self.sp
            # assert sp_save is not None
            self.calculator = self._calculator()
            # self.sp = sp_save

        try:
            energy = self.atoms.get_potential_energy() * electronvolt
        except BaseException as e:
            self._handle_exception(e)

        gpos_out = None
        vtens_out = None
        if gpos:
            forces = self.atoms.get_forces()
            gpos_out = -forces * electronvolt / angstrom

        if vir:
            cell = self.atoms.get_cell()
            volume = cell.volume
            stress = self.atoms.get_stress(voigt=False)
            vtens_out = volume * stress * electronvolt

        return EnergyResult(
            energy=jnp.asarray(energy, dtype=jnp.float64),
            gpos=jnp.asarray(gpos_out, dtype=jnp.float64) if gpos else None,
            vtens=jnp.asarray(vtens_out, dtype=jnp.float64) if vir else None,
        )

    def _calculator(self):  # -> ase.calculators.calculator.Calculator:
        raise NotImplementedError

    def _handle_exception(self, e=None):
        raise EnergyError(f"Ase failed to provide an energy\nexception {e=}")

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

        import ase

        self.atoms = ase.Atoms.fromdict(**atom_params)
        print(f"setting {atom_params=}")

        self.calculator = clss(**calc_params)


class Cp2kEnergy(AseEnergy):
    cp2k_inp: str

    input_kwargs: dict = field(
        default_factory=lambda: dict(
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
    )

    kwargs: dict = field(default_factory=dict)

    @staticmethod
    def create(
        atoms,  # ase.Atoms,
        input_file,
        input_kwargs: dict,
        # cp2k_path: Path | None = None,
        **kwargs,
    ):
        self = Cp2kEnergy(
            atoms=atoms,
            # cp2k_path=cp2k_path,
            cp2k_inp=input_file,
            input_kwargs=input_kwargs,
            kwargs=kwargs,
        )
        if self.calculator is None:
            self.calculator = self._calculator()

    # def replace_paths(self,old_path:Path, new_path: Path):
    #     len_old = len(old_path.parts)

    #     self.cp2k_inp = new_path.joinpath( *new_path.parts , *Path(self.cp2k_inp).parts[len_old:])

    #     for key, val in self.input_kwargs.items():
    #         self.input_kwargs[key] = new_path.joinpath( *new_path.parts , *Path(val).parts[len_old:])

    @staticmethod
    def _relative(target: Path, origin: Path):
        """return path of target relative to origin"""
        try:
            return Path(target).resolve().relative_to(Path(origin).resolve())
        except ValueError:  # target does not start with origin
            # recursion with origin (eventually origin is root so try will succeed)
            return Path("..").joinpath(Cp2kEnergy._relative(target, Path(origin).parent))

    def _calculator(self):
        rp = Path.cwd()
        rp.mkdir(parents=True, exist_ok=True)
        print(f"saving CP2K output in {rp}")

        new_dict = {}
        for key, val in self.input_kwargs.items():
            p = ROOT_DIR.joinpath(*val.parts)

            assert p.exists(), f"recieved {val=},{key=}, resolved to {p}"
            new_dict[key] = Cp2kEnergy._relative(p, rp)

        inp = ROOT_DIR.joinpath(*Path(self.cp2k_inp).parts)

        print(f"{ROOT_DIR=} {inp=} {new_dict=} {rp=}")

        with open(inp) as f:
            inp = "".join(f.readlines()).format(**new_dict)

        params = self.input_kwargs
        params.update(**{"inp": inp, **self.kwargs})

        if "label" in params:
            del params["label"]
            print("ignoring label for Cp2kEnergy")
        if "directory" in params:
            del params["directory"]
            print("ignoring directory for Cp2kEnergy")

        params["directory"] = "."

        params["command"] = REFERENCE_COMMANDS["cp2k"]

        from ase.calculators.cp2k import CP2K

        calc = CP2K(**params)  # type: ignore

        return calc

    def _handle_exception(self, e=None):
        if self.calculator is None:
            raise Energy("no calculator")

        print(f"{e=}")
        p = f"{self.calculator.directory}/cp2k.out"
        assert os.path.exists(p), "no cp2k output file after failure"
        with open(p) as f:
            lines = f.readlines()
        out = min(len(lines), 50)
        assert out != 0, "cp2k.out doesn't contain output"

        file = "\n".join(lines[-out:])

        raise EnergyError(
            f"The cp2k calculator failed to provide an energy. The end of the output from cp2k.out is {file}",
        )

    def __getstate__(self):
        return [
            self.atoms.todict(),
            self.cp2k_inp,
            self.input_kwargs,
            self.kwargs,
        ]

    def __setstate__(self, state):
        # print(f"unpickling {self.__class__}")
        import ase

        atoms_dict, cp2k_inp, input_kwargs, kwargs = state

        if (p := Path(cp2k_inp)).is_absolute():
            n = p.parts.index("src")
            cp2k_inp = Path(*p.parts[n + 2 :])
            assert (ROOT_DIR / cp2k_inp).exists(), f"cannot find {ROOT_DIR / cp2k_inp}"

            print(f"setting {cp2k_inp}  instead of absoulte path {p}")

            for key, val in input_kwargs.items():
                n = val.parts.index("src")
                input_kwargs[key] = Path(*val.parts[n + 2 :])

        self = Cp2kEnergy(
            atoms=ase.Atoms.fromdict(atoms_dict),
            input_kwargs=input_kwargs,
            cp2k_inp=cp2k_inp,
            kwargs=kwargs,
        )

        self.calculator = self._calculator()

        return self


# @dataclass
class MACEASE(AseEnergy):
    model: str | Path = field(pytree_node=False, default="medium")
    dtype: str = field(pytree_node=False, default="float32")

    def _calculator(self):
        import torch
        from mace.calculators import mace_mp

        torch.set_num_threads(jax.device_count())
        print(f"{torch.get_num_threads()=}")
        print(f"{torch.cuda.is_available()=}")

        print(f"loading MACE model from {self.model} with dtype {self.dtype}")

        calc = mace_mp(
            model=self.model,
            dispersion=False,
            default_dtype=self.dtype,
        )

        return calc

    def __getstate__(self):
        dict = {
            "atoms": self.atoms.todict(),
        }

        return dict

    def __setstate__(self, state):
        import ase

        if isinstance(state["atoms"], ase.Atoms):
            self.atoms = state["atoms"]
        else:
            self.atoms = ase.Atoms.fromdict(state["atoms"])


class MACEJax(EnergyFn):
    model: str | Path = field(pytree_node=False, default="medium")
    dtype: str = field(pytree_node=False, default="float32")

    def load(self):
        from mace_jax import modules

        # Load a foundation model (this handles downloading and JAX initialization)
        # 'medium' refers to the standard MACE-MP-0-Medium checkpoint
        model, params = modules.load_foundation_model(self.model, dtype=self.dtype)

        print(f"loaded MACE model from {self.model} with dtype {self.dtype}")

    def f(self, sp: SystemParams, nl: NeighbourList, gpos=False, vir=False) -> EnergyResult:
        raise NotImplementedError
