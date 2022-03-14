# setup and start simulation here

from __future__ import division
from __future__ import print_function

import numpy as np

np.random.seed(3)
import h5py as h5

from yaff.sampling import VerletScreenLog, HDF5Writer, NHCThermostat, MTDHook,\
    VerletIntegrator
from yaff.external import ForcePartPlumed
from yaff.pes import CVInternalCoordinate, DihedAngle
from yaff.test.common import get_alaninedipeptide_amber99ff
from yaff.log import log

log.set_level(log.medium)
from molmod.units import *

plumed = False

T = 300 * kelvin

if __name__ == '__main__':
    ff = get_alaninedipeptide_amber99ff()
    vsl = VerletScreenLog(step=1000)
    fh5 = h5.File('traj.h5', 'w')
    h5writer = HDF5Writer(fh5, step=1000)
    thermo = NHCThermostat(T, timecon=100 * femtosecond)
    hooks = [thermo, vsl, h5writer]
    if plumed:
        plumed = ForcePartPlumed(ff.system, fn='plumed.dat')
        ff.add_part(plumed)
        hooks.append(plumed)
    else:
        cv0 = CVInternalCoordinate(ff.system, DihedAngle(4, 6, 8, 14))
        cv1 = CVInternalCoordinate(ff.system, DihedAngle(6, 8, 14, 16))
        sigmas = np.array([0.35, 0.35])
        periodicities = np.array([2.0 * np.pi, 2.0 * np.pi])
        mtd = MTDHook(ff, [cv0, cv1],
                      sigmas,
                      1.2 * kjmol,
                      periodicities=periodicities,
                      f=fh5,
                      start=500,
                      step=500)
        hooks.append(mtd)
    verlet = VerletIntegrator(ff, 0.5 * femtosecond, temp0=2 * T, hooks=hooks)
    verlet.run(1000000)