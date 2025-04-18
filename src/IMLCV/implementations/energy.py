import os
from pathlib import Path

import ase
import jax.numpy as jnp
import numpy as np

from IMLCV.base.bias import Energy, EnergyError, EnergyResult
from IMLCV.base.UnitsConstants import angstrom, electronvolt
from IMLCV.configs.config_general import REFERENCE_COMMANDS, ROOT_DIR


class YaffEnergy(Energy):
    def __init__(
        self,
        f,  #: Callable[[], yaff.ForceField],
    ) -> None:
        # import yaff.ForceField

        super().__init__()
        self.f = f
        self.ff = f()

    @property
    def cell(self):
        out = self.ff.system.cell.rvecs[:]  # empty cell represented as array with shape (0,3)
        if out.size == 0:
            return None
        return jnp.asarray(out)

    @cell.setter
    def cell(self, cell):
        if cell is None:
            return

        cell = np.asarray(cell, dtype=np.double)

        self.ff.update_rvecs(cell)

    @property
    def coordinates(self):
        return jnp.asarray(self.ff.system.pos[:])

    @coordinates.setter
    def coordinates(self, coordinates):
        self.ff.update_pos(np.array(coordinates))

    def _compute_coor(self, gpos=False, vir=False) -> EnergyResult:
        gpos_out = np.zeros_like(self.ff.gpos) if gpos else None
        vtens_out = np.zeros_like(self.ff.vtens) if (vir and self.cell is not None) else None

        try:
            ener = self.ff.compute(gpos=gpos_out, vtens=vtens_out)
        except BaseException as be:
            raise EnergyError(f"calculating yaff  energy raised execption:\n{be}\n")

        return EnergyResult(
            ener,
            jnp.array(gpos_out) if gpos_out is not None else None,
            jnp.array(vtens_out) if vtens_out is not None else None,
        )

    def __getstate__(self):
        return {"f": self.f, "sp": self.sp}

    def __setstate__(self, state):
        # print(f"unpickling {self.__class__}")
        self.f = state["f"]
        self.ff = self.f()
        self.sp = state["sp"]
        return self


class AseEnergy(Energy):
    """Conversion to ASE energy"""

    def __init__(
        self,
        atoms,  # , : ase.Atoms,
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

        self.atoms.set_cell(ase.geometry.Cell(np.array(cell) / angstrom))

    @property
    def coordinates(self):
        # return jnp.asarray(self.atoms.get_positions() * angstrom)

        return jnp.asarray(self.atoms.arrays["positions"]) * angstrom

    @coordinates.setter
    def coordinates(self, coordinates):
        self.atoms.set_positions(np.array(coordinates / angstrom))

    def _compute_coor(self, gpos=False, vir=False) -> EnergyResult:
        """use unit conventions of ASE"""

        if self.atoms.calc is None:
            sp_save = self.sp
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
            jnp.asarray(energy, dtype=jnp.float64),
            jnp.asarray(gpos_out, dtype=jnp.float64) if gpos else None,
            jnp.asarray(vtens_out, dtype=jnp.float64) if vir else None,
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

        calc = CP2K(**params)

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
        # print(f"unpickling {self.__class__}")
        import ase

        atoms_dict, cp2k_inp, input_kwargs, kwargs = state

        if (p := Path(cp2k_inp)).is_absolute():
            n = p.parts.index("src")
            cp2k_inp = Path(*p.parts[n + 2 :])
            assert (ROOT_DIR / cp2k_inp).exists(), f"cannot find {ROOT_DIR/cp2k_inp}"

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
            default_dtype="float32",
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
