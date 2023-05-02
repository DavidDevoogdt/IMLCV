"""MD engine class peforms MD simulations in a given NVT/NPT ensemble.

Currently, the MD is done with YAFF/OpenMM
"""
from __future__ import annotations

import numpy as np
import yaff.analysis.biased_sampling
import yaff.external
import yaff.log
import yaff.pes
import yaff.pes.bias
import yaff.pes.ext
import yaff.sampling
import yaff.sampling.iterative
from yaff.external import libplumed
from yaff.log import log
from yaff.sampling.verlet import VerletIntegrator

from IMLCV.base.CV import SystemParams

yaff.log.set_level(yaff.log.silent)


# if TYPE_CHECKING:
from IMLCV.base.bias import Bias, Energy
from IMLCV.base.MdEngine import MDEngine, StaticTrajectoryInfo


class YaffEngine(MDEngine, yaff.sampling.iterative.Hook):
    """MD engine with YAFF as backend.

    Args:
        ff (yaff.pes.ForceField)
    """

    def __init__(
        self,
        bias: Bias,
        static_trajectory_info: StaticTrajectoryInfo,
        energy: Energy,
        trajectory_file=None,
        sp: SystemParams | None = None,
        additional_parts=[],
    ) -> None:
        yaff.log.set_level(log.silent)
        self.start = 0
        self.name = "YaffEngineIMLCV"

        self._verlet: yaff.sampling.VerletIntegrator | None = None
        self._yaff_ener: YaffEngine.YaffFF | None = None

        self._verlet_initialized = False

        super().__init__(
            energy=energy,
            bias=bias,
            static_trajectory_info=static_trajectory_info,
            trajectory_file=trajectory_file,
            sp=sp,
        )
        self.initializing = False

        self.additional_parts = additional_parts

    def __call__(self, iterative: VerletIntegrator):
        if not self._verlet_initialized:
            return

        kwargs = dict(t=iterative.time, T=iterative.temp, err=iterative.cons_err)

        if hasattr(iterative, "press"):
            kwargs["P"] = iterative.press

        self.save_step(**kwargs)

    def _setup_verlet(self):
        # hooks = [self, VerletScreenLog(step=self.static_trajectory_info.screen_log)]

        hooks = [self]

        self._yaff_ener = YaffEngine.YaffFF(
            md_engine=self,
            additional_parts=self.additional_parts,
        )

        if self.static_trajectory_info.thermostat:
            hooks.append(
                # yaff.sampling.NHCThermostat(
                #     self.static_trajectory_info.T,
                #     timecon=self.static_trajectory_info.timecon_thermo,
                # )
                yaff.sampling.LangevinThermostat(
                    self.static_trajectory_info.T,
                    timecon=self.static_trajectory_info.timecon_thermo,
                )
            )
        if self.static_trajectory_info.barostat:
            hooks.append(
                yaff.sampling.LangevinBarostat(
                    self._yaff_ener,
                    self.static_trajectory_info.T,
                    self.static_trajectory_info.P,
                    timecon=self.static_trajectory_info.timecon_baro,
                    anisotropic=True,
                )
                # yaff.sampling.MTKBarostat(
                #     self._yaff_ener,
                #     self.static_trajectory_info.T,
                #     self.static_trajectory_info.P,
                #     timecon=self.static_trajectory_info.timecon_baro,
                #     anisotropic=True,
                # )
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

            if gpos is not None:
                gpos[:] += np.array(res.gpos)
            if vtens is not None:
                vtens[:] += np.array(res.vtens)

            return res.energy


class PlumedEngine(YaffEngine):
    # Energy - kJ/mol
    # Length - nanometers
    # Time - picoseconds

    # https://github.com/giorginolab/plumed2-pycv/tree/v2.8-pycv/src/pycv
    # pybias  https://raimis.github.io/plumed-pybias/

    # pycv doesn't work with cell vectors?
    # this does ? https://github.com/giorginolab/plumed2-pycv/blob/v2.8-pycv/src/pycv/PythonFunction.cpp
    # see https://github.com/giorginolab/plumed2-pycv/blob/v2.6-pycv-devel/regtest/pycv/rt-f2/plumed.dat

    plumed_dat = """
    LOAD FILE=libpybias.so

    dist: DISTANCE ATOMS=1,2
    

    rc: PYTHONCV ATOMS=1,4,3 IMPORT=curvature FUNCTION=r

    # Creat a PyBias action, which executes "bias.py"
    PYBIAS ARG=rc


    RESTRAINT ARG=rc AT=0 KAPPA=0 SLOPE=1
    """

    def __init__(
        self,
        bias: Bias,
        static_trajectory_info: StaticTrajectoryInfo,
        energy: Energy,
        trajectory_file=None,
        sp: SystemParams | None = None,
    ) -> None:
        super().__init__(
            bias,
            static_trajectory_info,
            energy,
            trajectory_file,
            sp,
            additional_parts=[
                libplumed.ForcePartPlumed(timestep=static_trajectory_info.timestep)
            ],
        )
