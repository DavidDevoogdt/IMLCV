from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import yaff.analysis.biased_sampling
import yaff.external
import yaff.log
import yaff.pes.bias
import yaff.pes.ext
import yaff.sampling.iterative
from IMLCV.base.bias import Bias
from IMLCV.base.bias import Energy
from IMLCV.base.CV import SystemParams
from IMLCV.base.MdEngine import MDEngine
from IMLCV.base.MdEngine import StaticMdInfo
from IMLCV.base.MdEngine import time
from IMLCV.base.MdEngine import TrajectoryInfo
from typing_extensions import Self
from yaff.external import libplumed
from yaff.log import log
from yaff.sampling.verlet import VerletIntegrator


@dataclass
class YaffEngine(MDEngine, yaff.sampling.iterative.Hook):
    """MD engine with YAFF as backend.

    Args:
        ff (yaff.pes.ForceField)
    """

    start = 0

    _verlet_initialized: bool = False
    _verlet: yaff.sampling.VerletIntegrator | None = None
    _yaff_ener: YaffFF | None = None
    name = "YaffEngineIMLCV"
    additional_parts: list[yaff.ForcePart] = field(default_factory=list)

    @classmethod
    def create(  # type: ignore[override]
        self,
        bias: Bias,
        energy: Energy,
        static_trajectory_info: StaticMdInfo,
        trajectory_info: TrajectoryInfo | None = None,
        trajectory_file=None,
        sp: SystemParams | None = None,
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
            if sp is not None:
                energy.sp = sp

        else:
            create_kwargs["step"] = trajectory_info._size
            energy.sp = trajectory_info.sp[-1]
            if trajectory_info.t is not None:
                create_kwargs["time0"] = time() - trajectory_info.t[-1]

        kwargs.update(create_kwargs)

        return YaffEngine(
            bias=bias,
            energy=energy,
            static_trajectory_info=static_trajectory_info,
            trajectory_info=trajectory_info,
            trajectory_file=trajectory_file,
            additional_parts=additional_parts,
            **kwargs,
        )

    def __call__(self, iterative: VerletIntegrator):
        if not self._verlet_initialized:
            return

        kwargs = dict(t=iterative.time, T=iterative.temp, err=iterative.cons_err)

        if hasattr(iterative, "press"):
            kwargs["P"] = iterative.press

        self.save_step(**kwargs)

    def expects_call(self, counter):
        return True

    def _setup_verlet(self):
        hooks = [self]

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

        # plumed hook
        for i in self.additional_parts:
            if isinstance(i, yaff.sampling.iterative.Hook):
                hooks.append(i)

        self._verlet = yaff.sampling.VerletIntegrator(
            self._yaff_ener,
            self.static_trajectory_info.timestep,
            temp0=self.static_trajectory_info.T,
            hooks=hooks,
        )

        self._verlet_initialized = True

    @staticmethod
    def load(file, **kwargs) -> MDEngine:
        return super().load(file, **kwargs)

    def _run(self, steps):
        if not self._verlet_initialized:
            self._setup_verlet()
        self._verlet.run(int(steps))

    @property
    def yaff_system(self) -> YaffSys:
        return YaffSys(self.energy, self.static_trajectory_info)


class YaffFF(yaff.pes.ForceField):
    def __init__(
        self,
        md_engine: MDEngine,
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


@dataclass
class YaffSys:
    _ener: Energy
    _tic: StaticMdInfo

    @dataclass
    class YaffCell:
        _ener: Energy

        @property
        def rvecs(self):
            if self._ener.cell is None:
                return np.zeros((0, 3))
            return np.array(self._ener.cell)

        @rvecs.setter
        def rvecs(self, rvecs):
            self._ener.cell = jnp.array(rvecs)

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
        self._cell = self.YaffCell(_ener=self._ener)

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
        return np.array(self._ener.coordinates)

    @pos.setter
    def pos(self, pos):
        self._ener.coordinates = jnp.array(pos)

    @property
    def natom(self):
        return self.pos.shape[0]


# class PlumedEngine(YaffEngine):
#     # Energy - kJ/mol
#     # Length - nanometers
#     # Time - picoseconds

#     # https://github.com/giorginolab/plumed2-pycv/tree/v2.8-pycv/src/pycv
#     # pybias  https://raimis.github.io/plumed-pybias/

#     # pycv doesn't work with cell vectors?
#     # this does ? https://github.com/giorginolab/plumed2-pycv/blob/v2.8-pycv/src/pycv/PythonFunction.cpp
#     # see https://github.com/giorginolab/plumed2-pycv/blob/v2.6-pycv-devel/regtest/pycv/rt-f2/plumed.dat

#     plumed_dat = """
#     LOAD FILE=libpybias.so

#     dist: DISTANCE ATOMS=1,2


#     rc: PYTHONCV ATOMS=1,4,3 IMPORT=curvature FUNCTION=r

#     # Creat a PyBias action, which executes "bias.py"
#     PYBIAS ARG=rc


#     RESTRAINT ARG=rc AT=0 KAPPA=0 SLOPE=1
#     """

#     def __init__(
#         self,
#         bias: Bias,
#         static_trajectory_info: StaticMdInfo,
#         energy: Energy,
#         trajectory_file=None,
#         sp: SystemParams | None = None,
#     ) -> None:
#         super().__init__(
#             bias,
#             static_trajectory_info,
#             energy,
#             trajectory_file,
#             sp,
#             additional_parts=[
#                 libplumed.ForcePartPlumed(timestep=static_trajectory_info.timestep),
#             ],
#         )
