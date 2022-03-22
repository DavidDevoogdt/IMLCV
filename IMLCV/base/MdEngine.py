"""MD engine class peforms MD simulations in a given NVT/NPT ensemble.

Currently, the MD is done with YAFF/OpenMM
"""
import openmm
import openmm.openmm
import openmm.unit
import h5py
import yaff.sampling
from yaff import log
import yaff.system
import yaff.pes
import yaff.external
import numpy as np
from .CV import CV
import molmod
import ase
from abc import ABC, abstractmethod


class MDEngine(ABC):
    """Base class for MD engine.

    Args:
        cvs: list of cvs
        T:  temp
        P:  press
        ES:  enhanced sampling method, choice = "None","MTD"
        timestep:  step verlet integrator
        timecon_thermo: thermostat time constant
        timecon_baro:  barostat time constant
    """

    def __init__(self,
                 cv: CV,
                 T,
                 P,
                 step,
                 start,
                 ES="None",
                 timestep=None,
                 timecon_thermo=None,
                 timecon_baro=None,
                 filename="traj.h5") -> None:
        self.cv = cv
        self.ES = ES
        self.filename = filename
        self.step = step
        self.start = start

        self.timestep = timestep
        self.thermostat = T is not None and timecon_thermo is not None
        self.T = T
        self.timecon_thermo = timecon_thermo
        self.barostat = P is not None and timecon_baro is not None
        self.P = P
        self.timecon_baro = timecon_baro

    @abstractmethod
    def run(self, steps):
        """run the integrator for a given number of steps.

        Args:
            steps: number of MD steps
        """
        raise NotImplementedError

    @abstractmethod
    def to_ASE_traj(self):
        """convert the MD run to ASE trajectory."""
        raise NotImplementedError


from ase.io import read, write
import ase.geometry
import ase.stress


class YaffEngine(MDEngine):
    """MD engine with YAFF as backend.

    Args:
        ff (yaff.pes.ForceField)
        cv (IMLCV.base.CV.CV):
        ES (str, optional):
    """

    def __init__(self,
                 ff: yaff.pes.ForceField,
                 cv: CV,
                 ES="None",
                 T=None,
                 P=None,
                 timestep=None,
                 timecon_thermo=None,
                 timecon_baro=None,
                 step=1,
                 start=1,
                 filename="traj.h5",
                 **kwargs) -> None:
        super().__init__(cv=cv,
                         T=T,
                         P=P,
                         timestep=timestep,
                         timecon_thermo=timecon_thermo,
                         timecon_baro=timecon_baro,
                         ES=ES,
                         step=step,
                         start=start,
                         filename=filename)
        self.ff = ff

        # setup the the logging
        vsl = yaff.sampling.VerletScreenLog(step=100)
        self.hooks = [vsl]

        if self.filename.endswith(".h5"):
            self.fh5 = h5py.File(self.filename, 'w')
            h5writer = yaff.sampling.HDF5Writer(self.fh5, step=step)
            self.hooks.append(h5writer)
        elif self.filename.endswith(".xyz"):
            xyzhook = yaff.XYZWriter(self.filename, step=step, start=start)
            self.hooks.append(xyzhook)
        else:
            raise NotImplemented("only h5 and xyz are supported as filename")

        # setup barp/thermostat
        if self.thermostat:
            nhc = yaff.sampling.NHCThermostat(self.T, start=0, timecon=self.timecon_thermo, chainlength=3)

        if self.barostat:
            mtk = yaff.sampling.MTKBarostat(ff, self.T, self.P, start=0, timecon=self.timecon_baro, anisotropic=False)

        if self.thermostat and self.barostat:
            tbc = yaff.sampling.TBCombination(nhc, mtk)
            self.hooks.append(tbc)
        elif self.thermostat and not self.barostat:
            self.hooks.append(nhc)
        elif not self.thermostat and self.barostat:
            self.hooks.append(mtk)
        else:
            raise NotImplementedError

        # setup metadynamics
        if self.ES == "MTD":
            self.cvs = [self._convert_cv(cvy, ff=self.ff) for cvy in cv]
            if not ({"K", "sigmas", "periodicities", 'step_hills'} <= kwargs.keys()):
                raise ValueError("provide argumetns K, sigmas and periodicities")
            self._mtd_Yaff(K=kwargs["K"],
                           sigmas=kwargs["sigmas"],
                           periodicities=kwargs["periodicities"],
                           step_hills=kwargs["step_hills"])
        elif self.ES == "MTD_plumed":
            plumed = yaff.external.ForcePartPlumed(ff.system, fn='plumed.dat')
            ff.add_part(plumed)
            self.hooks.append(plumed)

        class GposContribStateItem(yaff.sampling.iterative.StateItem):
            """Keeps track of all the contributions to the forces."""

            def __init__(self):
                yaff.sampling.iterative.StateItem.__init__(self, 'gpos_contribs')

            def get_value(self, iterative):
                n = len(iterative.ff.parts)
                natom, _ = iterative.gpos.shape
                gpos_contribs = np.zeros((n, natom, 3))
                for i in range(n):
                    gpos_contribs[i, :, :] = iterative.ff.parts[i].gpos
                return gpos_contribs

            def iter_attrs(self, iterative):
                yield 'gpos_contrib_names', np.array([part.name for part in iterative.ff.parts], dtype='S')

        self.verlet = yaff.sampling.VerletIntegrator(
            self.ff,
            self.timestep,
            temp0=self.T,
            hooks=self.hooks,
            # add forces as state item
            state=[GposContribStateItem()])

    @staticmethod
    def create_forcefield_from_ASE(atoms, calculator) -> yaff.pes.ForceField:
        """Creates force field from ASE atoms instance."""

        class ForcePartASE(yaff.pes.ForcePart):
            """YAFF Wrapper around an ASE calculator.

            args:
                   system (yaff.System): system object
                   atoms (ase.Atoms): atoms object with calculator included.
            """

            def __init__(self, system, atoms, calculator):
                yaff.pes.ForcePart.__init__(self, 'ase', system)
                self.system = system  # store system to obtain current pos and box
                self.atoms = atoms
                self.calculator = calculator

            def _internal_compute(self, gpos=None, vtens=None):
                self.atoms.set_positions(self.system.pos / molmod.units.angstrom)
                self.atoms.set_cell(ase.geometry.Cell(self.system.cell._get_rvecs() / molmod.units.angstrom))
                energy = self.atoms.get_potential_energy() * molmod.units.electronvolt
                if gpos is not None:
                    forces = self.atoms.get_forces()
                    gpos[:] = -forces * molmod.units.electronvolt / molmod.units.angstrom
                if vtens is not None:
                    volume = np.linalg.det(self.atoms.get_cell())
                    stress = ase.stress.voigt_6_to_full_3x3_stress(self.atoms.get_stress())
                    vtens[:] = volume * stress * molmod.units.electronvolt
                return energy

        system = yaff.System(
            numbers=atoms.get_atomic_numbers(),
            pos=atoms.get_positions() * molmod.units.angstrom,
            rvecs=atoms.get_cell() * molmod.units.angstrom,
        )
        system.set_standard_masses()
        part_ase = ForcePartASE(system, atoms, calculator)

        return yaff.pes.ForceField(system, [part_ase])

    def _mtd_Yaff(self, K, sigmas, periodicities, step_hills):
        self.hooks.append(
            yaff.sampling.MTDHook(self.ff,
                                  self.cvs,
                                  sigma=sigmas,
                                  K=K,
                                  periodicities=periodicities,
                                  f=self.fh5,
                                  start=self.start,
                                  step=step_hills))

    def _mtd_Plumed(self, K, sigmas, periodicities, step):
        Warning("sigmas and peridocities ignorde in plumed, todo write custom .dat file")

        plumed = yaff.external.ForcePartPlumed(self.ff.system, fn='plumed.dat', timestep=self.timestep)
        self.ff.add_part(plumed)
        self.hooks.append(plumed)

    def _convert_cv(self, cv, ff):
        """convert generic CV class to a yaff CollectiveVariable."""

        class YaffCv(yaff.pes.CollectiveVariable):

            def __init__(self, system):
                super().__init__("YaffCV", system)
                self.cv = cv

            def compute(self, gpos=None, vtens=None):

                coordinates = self.system.pos
                cell = self.system.cell.rvecs

                self.value = self.cv.compute(coordinates, cell, grad=gpos, vir=vtens)

                return self.value

        return YaffCv(ff.system)

    def to_ASE_traj(self):
        if self.filename.endswith(".h5"):
            with h5py.File(self.filename, 'r') as f:
                at_numb = f['system']['numbers']
                traj = []
                for frame, energy_au in enumerate(f['trajectory']['epot']):
                    pos_A = f['trajectory']['pos'][frame, :, :] / molmod.units.angstrom
                    #vel_x = f['trajectory']['vel'][frame,:,:] / x
                    energy_eV = energy_au / molmod.units.electronvolt
                    forces_eVA = -f['trajectory']['gpos_contribs'][
                        frame, 0, :, :] * molmod.units.angstrom / molmod.units.electronvolt  # forces = -gpos

                    pbc = 'cell' in f['trajectory']
                    if pbc:
                        cell_A = f['trajectory']['cell'][frame, :, :] / molmod.units.angstrom

                        vol_A3 = f['trajectory']['volume'][frame] / molmod.units.angstrom**3
                        vtens_eV = f['trajectory']['vtens'][frame, :, :] / molmod.units.electronvolt
                        stresses_eVA3 = vtens_eV / vol_A3

                        atoms = ase.Atoms(
                            numbers=at_numb,
                            positions=pos_A,
                            pbc=pbc,
                            cell=cell_A,
                        )
                        atoms.info['stress'] = stresses_eVA3
                    else:
                        atoms = ase.Atoms(numbers=at_numb, positions=pos_A)

                    # atoms.set_velocities(vel_x)
                    atoms.arrays['forces'] = forces_eVA

                    atoms.info['energy'] = energy_eV
                    traj.append(atoms)
            return traj
        else:
            raise NotImplementedError("only for h5, impl this")

    def run(self, steps):
        self.verlet.run(steps)


class OpenMMEngine(MDEngine):

    def __init__(
        self,
        system: openmm.openmm.System,
        cv: CV,
        T,
        P,
        ES="None",
        timestep=None,
        timecon_thermo=None,
        timecon_baro=None,
    ) -> None:
        super().__init__(cv, T, P, ES, timestep, timecon_thermo, timecon_baro)

        if self.thermostat:
            self.integrator = openmm.openmm.LangevinMiddleIntegrator(temperature=T,
                                                                     frictionCoeff=1 / picosecond,
                                                                     stepSize=timestep)
        if self.barostat:
            system.addForce(openmm.openmm.MonteCarloBarostat(P, T))

        if self.ES == "MTD":
            self.cvs = [self._convert_cv(cvy, ff=self.ff) for cvy in cv]
            raise NotImplementedError

        elif self.ES == "MTD_plumed":
            raise NotImplementedError

    def _mtd(self, K, sigmas, periodicities, step):
        raise NotImplementedError

    def _convert_cv(self, cv, ff):
        """convert generic CV class to an OpenMM CV."""
        raise NotImplementedError

    def run(self, steps):
        self.integrator.step(steps)
