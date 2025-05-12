from __future__ import annotations

import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import ase
import ase.units
import jax.numpy as jnp
import numpy as np
from ase import Atoms, units

import new_yaff

from IMLCV.base.bias import Bias, Energy
from IMLCV.base.CV import SystemParams
from IMLCV.base.MdEngine import MDEngine, StaticMdInfo, TrajectoryInfo, time
from IMLCV.base.UnitsConstants import angstrom, bar, electronvolt, femtosecond, kelvin
import new_yaff.npt

# from IMLCV.base.MdEngine import StaticMdInfo

if TYPE_CHECKING:
    import yaff
    import yaff.analysis.biased_sampling
    import yaff.external
    import yaff.log
    import yaff.pes.bias
    import yaff.pes.ext
    import yaff.sampling.iterative
    from yaff.sampling.verlet import VerletIntegrator


@dataclass
class YaffEngine(MDEngine):
    """MD engine with YAFF as backend.

    Args:
        ff (yaff.pes.ForceField)
    """

    _verlet_initialized: bool = False
    _verlet: yaff.sampling.VerletIntegrator | None = None
    _yaff_ener: Any | None = None
    # additional_parts: list[Any] = field(default_factory=list)

    @classmethod
    def create(  # type: ignore[override]
        self,
        bias: Bias,
        energy: Energy,
        static_trajectory_info: StaticMdInfo,
        sp: SystemParams,
        trajectory_info: TrajectoryInfo | None = None,
        trajectory_file=None,
        additional_parts=[],
        **kwargs,
    ) -> YaffEngine:
        cont = False

        create_kwargs = {}

        if trajectory_file is not None:
            trajectory_file = Path(trajectory_file)
            # continue with existing file if it exists
            if Path(trajectory_file).exists():
                trajectory_info = TrajectoryInfo.load(trajectory_file)
                cont = True

            if trajectory_info._size is None:
                cont = False

        if not cont:
            create_kwargs["step"] = 1

        else:
            create_kwargs["step"] = trajectory_info._size
            sp = trajectory_info.sp[-1]
            if trajectory_info.t is not None:
                create_kwargs["time0"] = time()

        kwargs.update(create_kwargs)

        return YaffEngine(
            bias=bias,
            energy=energy,
            sp=sp,
            static_trajectory_info=static_trajectory_info,
            trajectory_info=trajectory_info,
            trajectory_file=trajectory_file,
            additional_parts=additional_parts,
            **kwargs,
        )

    def _setup_verlet(self):
        hooks = []

        import yaff

        yaff.log.set_level(0)

        # myhook calls save_step
        class myHook(yaff.sampling.iterative.Hook):
            def __init__(self, md_engine: YaffEngine):
                self.md_engine = md_engine

                super().__init__()

            def __call__(self, iterative: VerletIntegrator):
                kwargs = dict(t=iterative.time, T=iterative.temp, err=iterative.cons_err)

                if hasattr(iterative, "press"):
                    kwargs["P"] = iterative.press

                self.md_engine.save_step(**kwargs)

            def expects_call(self, counter):
                return True

        class YaffFF(yaff.pes.ForceField):
            def __init__(
                self,
                md_engine: YaffEngine,
                name="IMLCV_YAFF_forcepart",
                additional_parts=[],
            ):
                self.md_engine = md_engine

                super().__init__(system=self.system, parts=additional_parts)

            @property
            def system(self):
                return self.md_engine.yaff_system

            @system.setter
            def system(self, sys):
                assert sys == self.system

            @property
            def sp(self):
                return self.md_engine.energy.sp

            def update_rvecs(self, rvecs):
                self.clear()
                self.system.cell.rvecs = rvecs

            def update_pos(self, pos):
                self.clear()
                self.system.pos = pos

            def _internal_compute(self, gpos, vtens):
                energy = self.md_engine.get_energy(
                    gpos is not None,
                    vtens is not None,
                )

                cv, bias = self.md_engine.get_bias(
                    gpos is not None,
                    vtens is not None,
                )

                res = energy + bias

                self.md_engine.last_ener = energy
                self.md_engine.last_bias = bias
                self.md_engine.last_cv = cv

                if res.gpos is not None:
                    gpos[:] += np.array(res.gpos, dtype=np.float64)
                if res.vtens is not None:
                    vtens[:] += np.array(res.vtens, dtype=np.float64)

                return res.energy

        self._yaff_ener = YaffFF(
            md_engine=self,
            additional_parts=self.additional_parts,
        )

        if self.static_trajectory_info.thermostat:
            hooks.append(
                yaff.sampling.LangevinThermostat(
                    self.static_trajectory_info.T,
                    timecon=self.static_trajectory_info.timecon_thermo,
                ),
            )
        if self.static_trajectory_info.barostat:
            hooks.append(
                yaff.sampling.LangevinBarostat(
                    self._yaff_ener,
                    self.static_trajectory_info.T,
                    self.static_trajectory_info.P,
                    timecon=self.static_trajectory_info.timecon_baro,
                    anisotropic=True,
                ),
            )

        # # plumed hook
        # for i in self.additional_parts:
        #     if isinstance(i, yaff.sampling.iterative.Hook):
        #         hooks.append(i)

        hooks.append(myHook(self))

        self._verlet = yaff.sampling.VerletIntegrator(
            self._yaff_ener,
            self.static_trajectory_info.timestep,
            temp0=self.static_trajectory_info.T,
            hooks=hooks,
        )

        i_screenlog = None

        for i, h in enumerate(self._verlet.hooks):
            if isinstance(h, yaff.sampling.VerletScreenLog):
                i_screenlog = i

                break

        if i_screenlog is not None:
            self._verlet.hooks.pop(i_screenlog)

        self._verlet_initialized = True

    @staticmethod
    def load(file, **kwargs) -> MDEngine:
        return MDEngine.load(file, **kwargs)

    def _run(self, steps):
        if not self._verlet_initialized:
            self._setup_verlet()
        self._verlet.run(int(steps))

    @property
    def yaff_system(self) -> YaffSys:
        return YaffSys(self.energy, self.static_trajectory_info)


@dataclass
class YaffSys:
    _md: MDEngine
    _tic: StaticMdInfo

    @dataclass
    class YaffCell:
        _md: MDEngine

        @property
        def rvecs(self):
            if self._md.sp.cell is None:
                return np.zeros((0, 3))
            return np.array(self._md.sp.cell)

        @rvecs.setter
        def rvecs(self, rvecs):
            self._md.sp.cell = jnp.array(rvecs)

        def update_rvecs(self, rvecs):
            self.rvecs = rvecs

        @property
        def nvec(self):
            return self.rvecs.shape[0]

        @property
        def volume(self):
            if self.nvec == 0:
                return np.nan

            vol_unsigned = np.linalg.det(self.rvecs)
            if vol_unsigned < 0:
                print("cell volume was negative")

            return np.abs(vol_unsigned)

    def __post_init__(self):
        self._cell = self.YaffCell(_md=self._md)

    @property
    def numbers(self):
        return self._tic.atomic_numbers

    @property
    def masses(self):
        return np.array(self._tic.masses)

    @property
    def charges(self):
        return None

    @property
    def cell(self):
        return self._cell

    @property
    def pos(self):
        return np.array(self._md.sp.coordinates)

    @pos.setter
    def pos(self, pos):
        self._md.sp.coordinates = jnp.array(pos)

    @property
    def natom(self):
        return self.pos.shape[0]


@dataclass
class AseEngine(MDEngine):
    """MD engine with ASE as backend."""

    _verlet_initialized: bool = False
    _verlet: ase.md.md.MolecularDynamics | None = None
    langevin: bool = False

    @staticmethod
    def create(
        bias: Bias,
        energy: Energy,
        static_trajectory_info: StaticMdInfo,
        sp: SystemParams | None = None,
        trajectory_info: TrajectoryInfo | None = None,
        trajectory_file=None,
        langevin=True,
        **kwargs,
    ) -> AseEngine:
        cont = False

        if trajectory_file is not None:
            trajectory_file = Path(trajectory_file)
            # continue with existing file if it exists
            if Path(trajectory_file).exists():
                trajectory_info = TrajectoryInfo.load(trajectory_file)
                cont = True

            if trajectory_info._size is None:
                cont = False

        if not cont:
            kwargs["step"] = 1
            # if sp is not None:
            # energy.sp = sp

        else:
            kwargs["step"] = trajectory_info._size
            sp = trajectory_info.sp[-1]
            if trajectory_info.t is not None:
                kwargs["time0"] = time()

        self = AseEngine(
            bias=bias,
            energy=energy,
            static_trajectory_info=static_trajectory_info,
            trajectory_info=trajectory_info,
            trajectory_file=trajectory_file,
            sp=sp,
            # langevin=langevin,
            **kwargs,
        )

        self.langevin = langevin

        return self

    def update_sp(self, atoms: Atoms):
        """Update the system parameters with the current atoms object."""

        # print(f"before {atoms=} {self.sp=}")

        self.sp = SystemParams(
            coordinates=jnp.array(atoms.get_positions() / ase.units.Ang * angstrom),
            cell=jnp.array(atoms.get_cell() / ase.units.Ang * angstrom) if atoms.pbc.all() else None,
        )

        # print(f"after {self.sp=}")

    def _setup_verlet(self):
        # everyhting stays the same, except the position as linked to the true positions

        from ase.calculators.calculator import Calculator
        from ase.md.langevin import Langevin
        from ase.md.nose_hoover_chain import NoseHooverChainNVT
        from ase.md.npt import NPT
        from ase.md.nptberendsen import NPTBerendsen
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

        self.atoms = Atoms(
            numbers=self.static_trajectory_info.atomic_numbers,
            positions=self.sp.coordinates * ase.units.Ang / angstrom,
            cell=self.sp.cell * ase.units.Ang / angstrom if self.sp.cell is not None else None,
            pbc=self.sp.cell is not None,
        )

        MaxwellBoltzmannDistribution(
            self.atoms,
            temperature_K=self.static_trajectory_info.T / kelvin,  # ase kelvin=1)
        )

        if self.static_trajectory_info.barostat:
            if self.langevin:
                print(f"langevin cannot be combined with npt")

            dyn = NPT(
                atoms=self.atoms,
                timestep=self.static_trajectory_info.timestep / femtosecond * ase.units.fs,
                pfactor=(self.static_trajectory_info.timecon_baro / femtosecond * ase.units.fs) ** 2 * 0.6,
                ttime=self.static_trajectory_info.timecon_thermo / femtosecond * ase.units.fs,
                externalstress=self.static_trajectory_info.P / bar * ase.units.bar,
                temperature_K=self.static_trajectory_info.T / kelvin,  # ase kelvin =1
            )

            # dyn = NPTBerendsen(
            #     atoms=self.atoms,
            #     timestep=self.static_trajectory_info.timestep / femtosecond * ase.units.fs,
            #     temperature_K=self.static_trajectory_info.T / kelvin,  # ase kelvin=1
            #     pressure=self.static_trajectory_info.P / bar * ase.units.bar,
            #     compressibility_au=4.57e-5 / ase.units.bar,
            #     taut=self.static_trajectory_info.timecon_thermo / femtosecond * ase.units.fs,
            #     taup=self.static_trajectory_info.timecon_baro / femtosecond * ase.units.fs,
            # )

        else:
            if self.langevin:
                dyn = Langevin(
                    atoms=self.atoms,
                    timestep=self.static_trajectory_info.timestep / femtosecond * ase.units.fs,
                    temperature_K=self.static_trajectory_info.T / kelvin,  # ase kelvin=1
                    friction=0.01 / ase.units.fs,
                    fixcm=False,
                )

            else:
                dyn = NoseHooverChainNVT(
                    atoms=self.atoms,
                    timestep=self.static_trajectory_info.timestep / femtosecond * ase.units.fs,
                    temperature_K=self.static_trajectory_info.T / kelvin,  # ase kelvin=1
                    tdamp=self.static_trajectory_info.timecon_thermo / femtosecond * ase.units.fs,
                )

        class AseCalculator(Calculator):
            implemented_properties = ["energy", "forces", "stress"]

            def __init__(self, engine: AseEngine):
                self.engine = weakref.proxy(engine)
                super().__init__(atoms=self.engine.atoms)

            def calculate(self, atoms, properties, system_changes):
                gpos = "forces" in properties
                vtens = "stress" in properties

                self.engine.update_sp(atoms)

                energy = self.engine.get_energy(
                    gpos=gpos,
                    vtens=vtens,
                )

                cv, bias = self.engine.get_bias(
                    gpos=gpos,
                    vtens=vtens,
                )

                # print(f"{energy=} {bias=} {cv=}")

                res = energy + bias

                # print(f"{res=}")

                self.engine.last_ener = energy
                self.engine.last_bias = bias
                self.engine.last_cv = cv

                self.results = {
                    "energy": res.energy * ase.units.eV / electronvolt,
                }

                if res.gpos is not None:
                    self.results["forces"] = -res.gpos * (ase.units.eV / electronvolt) * (angstrom / ase.units.Ang)

                if res.vtens is not None:
                    vol = self.engine.atoms.get_volume()  # vol already in ase units
                    self.results["stress"] = res.vtens * (ase.units.eV / electronvolt) / vol

        calc = AseCalculator(self)
        self.calc = calc

        class MDLogger:
            def __init__(
                self,
                engine: AseEngine,
            ):
                self.engine = weakref.proxy(engine)

            def __del__(self):
                self.close()

            def close(self):
                pass

            def __call__(self):
                temp = self.engine.atoms.get_temperature()

                t = self.engine._verlet.get_time() * ase.units.s / (1000 * units.fs)

                kwargs = dict(t=t, T=temp, err=0.0)

                # TODO
                # if hasattr(iterative, "press"):
                #     kwargs["P"] = iterative.press

                self.engine.save_step(**kwargs)

        dyn.attach(
            MDLogger(self),
            interval=1,
        )

        self._verlet = dyn

        self._verlet_initialized = True

    def _run(self, steps):
        if not self._verlet_initialized:
            self._setup_verlet()

        self._verlet.run(int(steps))


@dataclass
class NewYaffEngine(MDEngine):
    """MD engine with YAFF as backend.

    Args:
        ff (yaff.pes.ForceField)
    """

    _verlet_initialized: bool = False
    _verlet: yaff.sampling.VerletIntegrator | None = None
    _yaff_ener: Any | None = None
    # additional_parts: list[Any] = field(default_factory=list)

    @classmethod
    def create(  # type: ignore[override]
        self,
        bias: Bias,
        energy: Energy,
        static_trajectory_info: StaticMdInfo,
        sp: SystemParams,
        trajectory_info: TrajectoryInfo | None = None,
        trajectory_file=None,
        # additional_parts=[],
        **kwargs,
    ) -> YaffEngine:
        cont = False

        create_kwargs = {}

        if trajectory_file is not None:
            trajectory_file = Path(trajectory_file)
            # continue with existing file if it exists
            if Path(trajectory_file).exists():
                trajectory_info = TrajectoryInfo.load(trajectory_file)
                cont = True

            if trajectory_info._size is None:
                cont = False

        if not cont:
            create_kwargs["step"] = 1

        else:
            create_kwargs["step"] = trajectory_info._size
            sp = trajectory_info.sp[-1]
            if trajectory_info.t is not None:
                create_kwargs["time0"] = time()

        kwargs.update(create_kwargs)

        return NewYaffEngine(
            bias=bias,
            energy=energy,
            sp=sp,
            static_trajectory_info=static_trajectory_info,
            trajectory_info=trajectory_info,
            trajectory_file=trajectory_file,
            # additional_parts=additional_parts,
            **kwargs,
        )

    def _setup_verlet(self):
        hooks = []

        class myHook(new_yaff.iterative.Hook):
            def __init__(self, md_engine: YaffEngine):
                self.md_engine = md_engine

                super().__init__()

            def __call__(self, iterative: VerletIntegrator):
                kwargs = dict(t=iterative.time, T=iterative.temp, err=iterative.cons_err)

                if hasattr(iterative, "press"):
                    kwargs["P"] = iterative.press

                self.md_engine.save_step(**kwargs)

            def expects_call(self, counter):
                return True

        class YaffFF:
            def __init__(
                self,
                md_engine: YaffEngine,
                name="IMLCV_YAFF_forcepart",
                additional_parts=[],
            ):
                self.md_engine = md_engine

                self.energy = 0.0
                self.gpos = np.zeros((self.system.natom, 3), float)
                self.vtens = np.zeros((3, 3), float)
                self.clear()

            @property
            def system(self):
                return self.md_engine.yaff_system

            @system.setter
            def system(self, sys):
                assert sys == self.system

            @property
            def sp(self):
                return self.md_engine.sp

            def update_rvecs(self, rvecs):
                self.clear()
                self.system.cell.rvecs = rvecs

            def update_pos(self, pos):
                self.clear()
                self.system.pos = pos

            def _internal_compute(self, gpos, vtens):
                # print(f"inside _internal_compute {gpos=} {vtens=} {self.sp=}  ")

                energy = self.md_engine.get_energy(
                    gpos is not None,
                    vtens is not None and self.md_engine.sp.cell is not None,
                )

                cv, bias = self.md_engine.get_bias(
                    gpos is not None,
                    vtens is not None and self.md_engine.sp.cell is not None,
                )

                res = energy + bias

                self.md_engine.last_ener = energy
                self.md_engine.last_bias = bias
                self.md_engine.last_cv = cv

                if res.gpos is not None:
                    gpos[:] += np.array(res.gpos, dtype=np.float64)
                if res.vtens is not None:
                    vtens[:] += np.array(res.vtens, dtype=np.float64)

                return res.energy

            def clear(self):
                self.energy = np.nan
                self.gpos[:] = np.nan
                self.vtens[:] = np.nan

            def compute(self, gpos=None, vtens=None):
                """Compute the energy and optionally some derivatives for this FF (part)

                The only variable inputs for the compute routine are the atomic
                positions and the cell vectors, which can be changed through the
                ``update_rvecs`` and ``update_pos`` methods. All other aspects of
                a force field are considered to be fixed between subsequent compute
                calls. If changes other than positions or cell vectors are needed,
                one must construct new ``ForceField`` and/or ``ForcePart`` objects.

                **Optional arguments:**

                gpos
                        The derivatives of the energy towards the Cartesian coordinates
                        of the atoms. ('g' stands for gradient and 'pos' for positions.)
                        This must be a writeable numpy array with shape (N, 3) where N
                        is the number of atoms.

                vtens
                        The force contribution to the pressure tensor. This is also
                        known as the virial tensor. It represents the derivative of the
                        energy towards uniform deformations, including changes in the
                        shape of the unit cell. (v stands for virial and 'tens' stands
                        for tensor.) This must be a writeable numpy array with shape (3,
                        3). Note that the factor 1/V is not included.

                The energy is returned. The optional arguments are Fortran-style
                output arguments. When they are present, the corresponding results
                are computed and **added** to the current contents of the array.
                """
                if gpos is None:
                    my_gpos = None
                else:
                    my_gpos = self.gpos
                    my_gpos[:] = 0.0
                if vtens is None:
                    my_vtens = None
                else:
                    my_vtens = self.vtens
                    my_vtens[:] = 0.0
                self.energy = self._internal_compute(my_gpos, my_vtens)
                if np.isnan(self.energy):
                    raise ValueError("The energy is not-a-number (nan).")
                if gpos is not None:
                    if np.isnan(my_gpos).any():
                        raise ValueError("Some gpos element(s) is/are not-a-number (nan).")
                    gpos += my_gpos
                if vtens is not None:
                    if np.isnan(my_vtens).any():
                        raise ValueError("Some vtens element(s) is/are not-a-number (nan).")
                    vtens += my_vtens

                # print(f"inside compute {self.energy=} {gpos=} {vtens=}")

                return self.energy

        self._yaff_ener = YaffFF(
            md_engine=self,
            # additional_parts=self.additional_parts,
        )

        if self.static_trajectory_info.thermostat:
            from new_yaff.nvt import LangevinThermostat

            hooks.append(
                LangevinThermostat(
                    self.static_trajectory_info.T,
                    timecon=self.static_trajectory_info.timecon_thermo,
                ),
            )
        if self.static_trajectory_info.barostat:
            from new_yaff.npt import LangevinBarostat

            hooks.append(
                LangevinBarostat(
                    self._yaff_ener,
                    self.static_trajectory_info.T,
                    self.static_trajectory_info.P,
                    timecon=self.static_trajectory_info.timecon_baro,
                    anisotropic=True,
                ),
            )

        hooks.append(myHook(self))

        from new_yaff.verlet import VerletIntegrator

        self._verlet = VerletIntegrator(
            self._yaff_ener,
            self.static_trajectory_info.timestep,
            temp0=self.static_trajectory_info.T,
            hooks=hooks,
        )

        self._verlet_initialized = True

    @staticmethod
    def load(file, **kwargs) -> MDEngine:
        return MDEngine.load(file, **kwargs)

    def _run(self, steps):
        if not self._verlet_initialized:
            self._setup_verlet()
        self._verlet.run(int(steps))

    @property
    def yaff_system(self) -> YaffSys:
        return YaffSys(self, self.static_trajectory_info)
