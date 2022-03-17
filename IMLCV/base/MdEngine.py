import numpy as np
import IMLCV.base.CV


class MDEngine:
    def __init__(
        self,
        cv: IMLCV.base.CV.CV,
        T,
        P,
        ES="None",
        timestep=None,
        timecon_thermo=None,
        timecon_baro=None,
    ) -> None:
        """
            cvs: list of cvs

            T: temp

            P: press

            ES: enhanced sampling method, choice = "None","MTD"

            timestep: step verlet integrator

            timecon_thermo: thermostat time constant

            timecon_baro: baroostat time constant
        
        """
        self.cv = cv
        self.ES = ES

        self.timestep = timestep
        self.thermostat = T is not None and timecon_thermo is not None
        self.T = T
        self.timecon_thermo = timecon_thermo
        self.barostat = P is not None and timecon_baro is not None
        self.P = P
        self.timecon_baro = timecon_baro

    def run(self, steps):
        raise NotImplementedError

    def write_output():
        raise NotImplementedError


import yaff.external
import yaff.pes
import yaff.system

from yaff import log
from yaff.sampling import MTKBarostat, NHCThermostat, TBCombination, VerletIntegrator, HDF5Writer, VerletScreenLog
import h5py as h5


class YaffEngine(MDEngine):
    def __init__(self,
                 ff: yaff.pes.ForceField,
                 cv: IMLCV.base.CV.CV,
                 ES="None",
                 T=None,
                 P=None,
                 timestep=None,
                 timecon_thermo=None,
                 timecon_baro=None,
                 **kwargs) -> None:
        """
        ES: "None:,
            "MTD" kwargs= K, sigmas and periodicities
        """

        super().__init__(cv=cv,
                         T=T,
                         P=P,
                         timestep=timestep,
                         timecon_thermo=timecon_thermo,
                         timecon_baro=timecon_baro,
                         ES=ES)

        self.ff = ff

        # Setup the integrator
        vsl = VerletScreenLog(step=100)
        self.fh5 = h5.File('traj.h5', 'w')
        h5writer = HDF5Writer(self.fh5, step=50)
        hooks = [vsl, h5writer]

        if self.thermostat:
            nhc = NHCThermostat(self.T,
                                start=0,
                                timecon=self.timecon_thermo,
                                chainlength=3)

        if self.barostat:
            mtk = MTKBarostat(ff,
                              self.T,
                              self.P,
                              start=0,
                              timecon=self.timecon_baro,
                              anisotropic=False)

        if self.thermostat and self.barostat:
            tbc = TBCombination(nhc, mtk)
            hooks.append(tbc)
        elif self.thermostat and not self.barostat:
            hooks.append(nhc)
        elif not self.thermostat and self.barostat:
            hooks.append(mtk)
        else:
            raise NotImplementedError

        self.hooks = hooks

        if self.ES == "MTD":
            self.cvs = [self._convert_cv(cvy, ff=self.ff) for cvy in cv]
            if not ({"K", "sigmas", "periodicities", "step"} <= kwargs.keys()):
                raise ValueError(
                    "provide argumetns K, sigmas and periodicities")
            self._mtd(K=kwargs["K"],
                      sigmas=kwargs["sigmas"],
                      periodicities=kwargs["periodicities"],
                      step=kwargs["step"])
        elif self.ES == "MTD_plumed":
            plumed = yaff.external.ForcePartPlumed(ff.system, fn='plumed.dat')
            ff.add_part(plumed)
            self.hooks.append(plumed)

        self.verlet = VerletIntegrator(self.ff,
                                       self.timestep,
                                       temp0=self.T,
                                       hooks=self.hooks)

    def _mtd(self, K, sigmas, periodicities, step):
        self.hooks.append(
            yaff.sampling.MTDHook(self.ff,
                                  self.cvs,
                                  sigma=sigmas,
                                  K=K,
                                  periodicities=periodicities,
                                  f=self.fh5,
                                  start=500,
                                  step=step))

    def _convert_cv(self, cv, ff):
        """
        convert generic CV class to a yaff CollectiveVariable
        """
        class YaffCv(yaff.pes.CollectiveVariable):
            def __init__(self, system):
                super().__init__("YaffCV", system)
                self.cv = cv

            def compute(self, gpos=None, vtens=None):

                coordinates = self.system.pos
                cell = self.system.cell.rvecs

                self.value = self.cv.compute(coordinates,
                                             cell,
                                             grad=gpos,
                                             vir=vtens)

                return self.value

        return YaffCv(ff.system)

    def run(self, steps):
        self.verlet.run(steps)


from openmm import *
from openmm.app import PDBFile, ForceField, PME, Simulation
from openmm.openmm import System


class OpenMMEngine(MDEngine):
    def __init__(
        self,
        system: System,
        cv: IMLCV.base.CV.CV,
        T,
        P,
        ES="None",
        timestep=None,
        timecon_thermo=None,
        timecon_baro=None,
    ) -> None:
        super().__init__(cv, T, P, ES, timestep, timecon_thermo, timecon_baro)

        if self.thermostat:
            self.integrator = LangevinMiddleIntegrator(temperature=T,
                                                       frictionCoeff=1 /
                                                       picosecond,
                                                       stepSize=timestep)
        if self.barostat:
            system.addForce(MonteCarloBarostat(P, T))

        if self.ES == "MTD":
            self.cvs = [self._convert_cv(cvy, ff=self.ff) for cvy in cv]
            raise NotImplementedError

        elif self.ES == "MTD_plumed":
            raise NotImplementedError

    def _mtd(self, K, sigmas, periodicities, step):
        raise NotImplementedError

    def _convert_cv(self, cv, ff):
        """
        convert generic CV class to an OpenMM CV
        """
        raise NotImplementedError

    def run(self, steps):
        self.integrator.step(steps)
