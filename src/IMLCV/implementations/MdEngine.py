from __future__ import annotations

import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ase
import ase.md.md
import ase.units
import jax.numpy as jnp

# import numpy as np
from ase import Atoms, units

import IMLCV.new_yaff
import IMLCV.new_yaff.iterative
import IMLCV.new_yaff.verlet
from IMLCV.base.bias import Bias, Energy
from IMLCV.base.CV import SystemParams
from IMLCV.base.datastructures import field
from IMLCV.base.MdEngine import FullTrajectoryInfo, MDEngine, StaticMdInfo, time
from IMLCV.base.UnitsConstants import angstrom, bar, electronvolt, femtosecond, kelvin


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
        trajectory_info: FullTrajectoryInfo | None = None,
        trajectory_file=None,
        langevin=True,
        **kwargs,
    ) -> AseEngine:
        cont = False

        create_kwargs = {}

        if trajectory_file is not None:
            trajectory_file = Path(trajectory_file)
            # continue with existing file if it exists
            assert Path(trajectory_file).exists()
            trajectory_info = FullTrajectoryInfo.load(trajectory_file)
            cont = True

        if not cont:
            create_kwargs["step"] = 1

        else:
            assert trajectory_info is not None

            create_kwargs["step"] = trajectory_info.size
            sp = trajectory_info.sp[-1]
            if trajectory_info.t is not None:
                create_kwargs["time0"] = time()

        assert sp is not None, "SystemParams must be provided"

        kwargs.update(create_kwargs)

        self = AseEngine(
            bias=bias,
            energy=energy,
            static_trajectory_info=static_trajectory_info,
            trajectory_info=trajectory_info,
            trajectory_file=trajectory_file,
            sp=sp,
            **kwargs,
        )

        self.langevin = langevin

        return self

    def update_sp(self, atoms: Atoms):
        """Update the system parameters with the current atoms object."""

        # print(f"before {atoms=} {self.sp=}")

        self.sp = SystemParams(
            coordinates=jnp.array(atoms.get_positions() / ase.units.Ang * angstrom),
            cell=jnp.array(atoms.get_cell().array / ase.units.Ang * angstrom) if atoms.pbc.all() else None,
        )

        # print(f"after {self.sp=}")

    def _setup_verlet(self):
        # everyhting stays the same, except the position as linked to the true positions

        from ase.calculators.calculator import Calculator
        from ase.md.langevin import Langevin
        from ase.md.nose_hoover_chain import NoseHooverChainNVT
        from ase.md.npt import NPT
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
                print("langevin cannot be combined with npt")

            assert self.static_trajectory_info.timecon_baro is not None
            assert self.static_trajectory_info.P is not None

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
        assert self._verlet is not None

        self._verlet.run(int(steps))


class NewYaffEngine(MDEngine):
    """MD engine with YAFF as backend.

    Args:
        ff (yaff.pes.ForceField)
    """

    _verlet_initialized: bool = False
    _verlet: IMLCV.new_yaff.verlet.VerletIntegrator | None = None
    _verlet_hook: IMLCV.new_yaff.verlet.VerletHook | None = None
    _yaff_ener: Any | None = None

    @staticmethod
    def create(  # type: ignore[override]
        bias: Bias,
        energy: Energy,
        static_trajectory_info: StaticMdInfo,
        sp: SystemParams | None = None,
        trajectory_info: FullTrajectoryInfo | None = None,
        trajectory_file=None,
        # additional_parts=[],
        **kwargs,
    ) -> NewYaffEngine:
        cont = trajectory_info is not None

        create_kwargs = {}

        if trajectory_file is not None:
            trajectory_file = Path(trajectory_file)
            # continue with existing file if it exists
            assert Path(trajectory_file).exists()
            trajectory_info = FullTrajectoryInfo.load(trajectory_file)
            cont = True

        if not cont:
            create_kwargs["step"] = 1

        else:
            assert trajectory_info is not None

            create_kwargs["step"] = trajectory_info.size
            sp = trajectory_info.sp[-1]
            if trajectory_info.t is not None:
                create_kwargs["time0"] = time()

        assert sp is not None, "SystemParams must be provided"

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
        print("Setting up YAFF verlet integrator")

        hooks = []

        class myHook(IMLCV.new_yaff.iterative.Hook):
            md_engine: MDEngine = field(pytree_node=False)

            def __call__(self, iterative: IMLCV.new_yaff.verlet.VerletIntegrator):
                assert iterative.epot is not None, "Potential energy must be computed"
                assert iterative.e_bias is not None, "e bias must be set"
                assert iterative.temp is not None, "Temperature must be set"

                kwargs = dict(
                    t=iterative.time,
                    T=iterative.temp,
                    err=iterative.cons_err,
                    cv=iterative.cv,
                    e_bias=iterative.e_bias,
                    e_pot=iterative.epot - iterative.e_bias,
                    sp=iterative.sp,
                )

                if hasattr(iterative, "press"):
                    kwargs["P"] = iterative.press

                self.md_engine.save_step(**kwargs)  # type:ignore

            def expects_call(self, counter):
                return True

        from IMLCV.new_yaff.ff import YaffFF

        self._yaff_ener = YaffFF.create(
            energy=self.energy,
            bias=self.bias,
            permanent_bias=self.permanent_bias,
            sp=self.sp,
            tic=self.static_trajectory_info,
        )

        self._yaff_ener.system.update_nl()

        thermo = None

        if self.static_trajectory_info.thermostat:
            from IMLCV.new_yaff.nvt import LangevinThermostat

            thermo = LangevinThermostat(
                temp=jnp.array(self.static_trajectory_info.T),
                timecon=jnp.array(self.static_trajectory_info.timecon_thermo),
            )

        baro = None
        if self.static_trajectory_info.barostat:
            from IMLCV.new_yaff.npt import LangevinBarostat

            assert self.static_trajectory_info.P is not None
            assert self.static_trajectory_info.timecon_baro is not None

            baro = LangevinBarostat(
                temp=jnp.array(self.static_trajectory_info.T),
                press=jnp.array(self.static_trajectory_info.P),
                timecon=jnp.array(self.static_trajectory_info.timecon_baro),
                anisotropic=True,
            )

        hooks.append(
            myHook(md_engine=self),
        )

        from IMLCV.new_yaff.verlet import VerletIntegrator

        self._verlet, self._verlet_hook = VerletIntegrator.create(
            ff=self._yaff_ener,
            timestep=self.static_trajectory_info.timestep,
            temp0=self.static_trajectory_info.T,
            other_hooks=hooks,
            thermostat=thermo,
            barostat=baro,
        )

        self._verlet_initialized = True

    @staticmethod
    def load(file, **kwargs) -> MDEngine:
        return MDEngine.load(file, **kwargs)

    def run(self, steps):
        if not self._verlet_initialized:
            self._setup_verlet()

        return super().run(steps)

    def _step(self):
        self._verlet, self._verlet_hook = self._verlet.step(self._verlet_hook)
