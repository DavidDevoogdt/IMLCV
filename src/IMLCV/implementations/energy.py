import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import ase
import jax
import jax.numpy as jnp
import numpy as np
from ase import geometry as ase_geometry
from openmm import State, System, Vec3
from openmm.app import Simulation, Topology

from IMLCV.base.bias import Energy, EnergyError, EnergyResult
from IMLCV.base.CV import NeighbourList, SystemParams
from IMLCV.base.UnitsConstants import angstrom, electronvolt, kjmol, nanometer
from IMLCV.configs.config_general import REFERENCE_COMMANDS, ROOT_DIR


@dataclass
class OpenMmEnergy(Energy):
    topology: Topology
    system: System

    _simul: Simulation | None = None

    @property
    def nl(self) -> NeighbourList | None:
        return None

    @nl.setter
    def nl(self, nl: NeighbourList):
        return

    @property
    def cell(self) -> jax.Array | None:
        if self.topology._periodicBoxVectors is None:
            return None

        print(f"{self.topology._periodicBoxVectors=}")

        if self._simul is None:
            self._get_simul()

        state: State = self._simul.context.getState()

        if (c := state.getPeriodicBoxVectors(asNumpy=True)) is None:
            return None

        print(f"{c=}")

        cell = jnp.array(c) * nanometer

        return cell

    @cell.setter
    def cell(self, cell: jax.Array | None):
        if cell is None or self.topology._periodicBoxVectors is None:
            return

        if self._simul is None:
            self._get_simul()

        self._simul.context.setPeriodicBoxVectors(
            Vec3(*(cell[0, :].__array__() / nanometer)),
            Vec3(*(cell[1, :].__array__() / nanometer)),
            Vec3(*(cell[2, :].__array__() / nanometer)),
        )

    @property
    def coordinates(self) -> jax.Array | None:
        state: State = self._simul.context.getState(positions=True)

        coor = jnp.array(state.getPositions(asNumpy=True)) * nanometer

        return coor

    @coordinates.setter
    def coordinates(self, coordinates: jax.Array):
        self._simul.context.setPositions(
            coordinates.__array__() / nanometer,
        )

    @staticmethod
    def to_jax_vec(thing):
        return jnp.array([[a._value.x, a._value.y, a._value.z] for a in thing])

    def get_info(self):
        if self._simul is None:
            self._get_simul()

        assert self._simul is not None

        atomic_numbers = jnp.array(
            [a.element.atomic_number for a in self._simul.topology.atoms()],
            dtype=int,
        )

        sp = self.sp

        return sp, atomic_numbers

    def _compute_coor(self, sp: SystemParams, nl: NeighbourList, gpos=False, vir=False) -> EnergyResult:
        if self._simul is None:
            self._get_simul()

        # print(f"computing energy {sp/ nanometer=}")

        assert self._simul is not None

        self.sp = sp

        state: State = self._simul.context.getState(
            energy=True,
            forces=gpos,
        )

        assert not vir

        res = EnergyResult(
            energy=state.getPotentialEnergy()._value * kjmol,  # type:ignore
            gpos=-jnp.array(state.getForces(asNumpy=True)) * kjmol / nanometer if gpos else None,
        )

        return res

    def _get_simul(self):
        import openmm.unit as openmm_unit
        from openmm import LangevinIntegrator, System
        from openmm.app import (
            Simulation,
        )

        # pdb = PDBFile(str(self.pdb))
        topo = self.topology
        system: System = self.system

        # values don't matter, a integrator is needed
        integrator = LangevinIntegrator(
            300 * openmm_unit.kelvin,  # type:ignore
            1 / openmm_unit.picosecond,  # type:ignore
            0.004 * openmm_unit.picoseconds,  # type:ignore
        )
        simulation = Simulation(topo, system, integrator)
        # simulation.context.setPositions(pdb.positions)

        if (c := topo.getPeriodicBoxVectors()) is not None:
            simulation.context.setPeriodicBoxVectors(*c)

        self._simul = simulation

    def __getstate__(self):
        return dict(
            topology=self.topology,
            system=self.system,
        )

    def __setstate__(self, state):
        if "pdb" in state:
            self.topology = state["topology"]

        if "topology" in state:
            self.topology = state["topology"]

        if "system" in state:
            self.system = state["system"]
        else:
            from openmm.app import (
                ForceField,
            )

            forcefield = ForceField("amber14-all.xml")
            system: System = forcefield.createSystem(self.topology)

            self.system = system

    def get_bonds(self):
        bonds = []

        for b in self.topology.bonds():
            bonds.append([b.atom1.index, b.atom2.index])

        b_arr = jnp.array(bonds)

        # n_atoms = sp_boat.coordinates.shape[0]
        n_atoms = len(list(self.topology.atoms()))

        _, num_bonds = jnp.unique(b_arr, return_counts=True)

        max_bonds = int(jnp.max(num_bonds))

        @jax.vmap
        def _match(i):
            b = jnp.argwhere(jnp.any(b_arr == i, axis=1), size=max_bonds, fill_value=-1).reshape(-1)

            out = jnp.where(b == -1, -1, jnp.where(b_arr[b, 0] == i, b_arr[b, 1], b_arr[b, 0]))

            return out

        return _match(jnp.arange(n_atoms))


class AseEnergy(Energy):
    """Conversion to ASE energy"""

    def __init__(
        self,
        atoms: ase.Atoms,  # , : ase.Atoms,
        calculator=None,  #: ase.calculators.calculator.Calculator | None = None,
    ):
        self.atoms = atoms

        if calculator is not None:
            self.atoms.calc = self.calculator
        else:
            self.atoms.calc = self._calculator()

    @property
    def cell(self):
        cell = jnp.asarray(np.asarray(self.atoms.cell)) * angstrom

        if cell.ndim == 0:
            return None

        return cell

    @cell.setter
    def cell(self, cell):
        if cell is None:
            return

        self.atoms.set_cell(ase_geometry.Cell(np.array(cell) / angstrom))

    @property
    def coordinates(self):
        # return jnp.asarray(self.atoms.get_positions() * angstrom)

        return jnp.asarray(self.atoms.arrays["positions"]) * angstrom

    @coordinates.setter
    def coordinates(self, coordinates):
        self.atoms.set_positions(np.array(coordinates / angstrom))

    def _compute_coor(self, sp, nl, gpos=False, vir=False) -> EnergyResult:
        """use unit conventions of ASE"""

        self.sp = sp

        if self.atoms.calc is None:
            sp_save = self.sp
            assert sp_save is not None
            self.atoms.calc = self._calculator()
            self.sp = sp_save

            # self.atoms.calc.atoms = self.atoms

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
        atoms,  # ase.Atoms,
        input_file,
        input_kwargs: dict,
        cp2k_path: Path | None = None,
        **kwargs,
    ):
        self.atoms = atoms
        self.cp2k_inp = input_file  # = os.path.abspath(input_file)
        self.input_kwargs = input_kwargs
        self.kwargs = kwargs
        super().__init__(atoms)

        self.rp = cp2k_path

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

        params = self.default_parameters.copy()
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
        print(f"{e=}")
        p = f"{self.atoms.calc.directory}/cp2k.out"
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

        self.__init__(
            atoms=ase.Atoms.fromdict(atoms_dict),
            input_file=cp2k_inp,
            input_kwargs=input_kwargs,
            **kwargs,
        )


class MACEASE(AseEnergy):
    def _calculator(self):
        from mace.calculators import mace_mp

        return mace_mp(
            model="medium",
            dispersion=False,
            default_dtype="float64",
            device="cpu",
            # compile_mode="default", #doens't compute stress
        )

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

        self.calculator = self._calculator()
        self.atoms.calc = self.calculator
