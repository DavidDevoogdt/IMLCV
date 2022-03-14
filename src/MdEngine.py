from tkinter.messagebox import NO
import yaff.sampling
import yaff.pes
import numpy as np

from yaff import log
from yaff.sampling import MTKBarostat, NHCThermostat, TBCombination, VerletIntegrator, HDF5Writer, VerletScreenLog
from molmod.units import femtosecond
import h5py as h5


class MDEngine:
    def __init__(self,
                 cvs: yaff.pes.CollectiveVariable,
                 T,
                 P,
                 timestep=2.0 * femtosecond,
                 timecon_thermo=100.0 * femtosecond,
                 timecon_baro=1000.0 * femtosecond) -> None:
        self.cvs = cvs

        self.timestep = timestep
        self.thermostat = T is not None and timecon_thermo is not None
        self.T = T
        self.timecon_thermo = timecon_thermo
        self.barostat = P is not None and timecon_baro is not None
        self.P = P
        self.timecon_baro = timecon_baro

    def step(self, x_old: yaff.sampling.DOF) -> yaff.sampling.DOF:
        x_old = self.convert_coordinates(x_old, forward=True)
        x_new = self._step(x_old)
        return self.convert_coordinates(x_new, forward=False)

    def _run(self, steps):
        NotImplementedError

    def convert_coordinates(self, x, forward=True):
        # if the engine uses a different system, convert here
        return x

    def write_output():
        NotImplementedError


class YaffEngine(MDEngine):
    def __init__(self, ff: yaff.pes.ForceField, sigmas, K,
                 periodicities) -> None:

        super.__init__()

        # Setup the integrator
        vsl = VerletScreenLog(step=100)
        nhc = NHCThermostat(T,
                            start=0,
                            timecon=self.timecon_thermo,
                            chainlength=3)
        mtk = MTKBarostat(ff,
                          self.T,
                          self.P,
                          start=0,
                          timecon=self.timecon_baro,
                          anisotropic=False)
        tbc = TBCombination(nhc, mtk)
        hooks = [vsl, tbc]

        hdf5 = h5.File('output.h5', mode='w')

        #do metadynamics
        mtd = yaff.sampling.MTDHook(ff,
                                    self.cvs,
                                    sigma=sigmas,
                                    K=K,
                                    periodicities=periodicities,
                                    f=hdf5,
                                    start=500,
                                    step=500)

        hooks.append(mtd)

        # Production run
        self.verlet = VerletIntegrator(ff,
                                       self.timestep,
                                       temp0=self.T,
                                       hooks=hooks + [hdf5])

    def _step(self, steps):
        self.verlet.run(steps)


class OpenMMEngine(MDEngine):
    pass