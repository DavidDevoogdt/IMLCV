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
import yaff.analysis.biased_sampling
from yaff.sampling.io import XYZWriter
import numpy as np
from .CV import CV
import molmod
import ase
from abc import ABC, abstractmethod

from ase.io import read, write
import ase.geometry
import ase.stress


class MDEngineBias(ABC):
    """base class for bias."""

    def __init__(self, cvs: CV) -> None:
        self.cvs = cvs

    def add_engine(self, mdengine):
        """function is called from MDEngine instance."""
        self.mdengine = mdengine

    @abstractmethod
    def get_bias(self, cvs):
        raise NotImplementedError


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
                 MDEngineBias: MDEngineBias,
                 T,
                 P,
                 timestep=None,
                 timecon_thermo=None,
                 timecon_baro=None,
                 filename="traj.h5",
                 write_step=100) -> None:
        self.MDEngineBias = MDEngineBias
        self.filename = filename
        self.write_step = write_step

        self.timestep = timestep

        self.T = T
        self.thermostat = (T is not None)
        if self.thermostat:
            assert timecon_thermo is not None
            self.timecon_thermo = timecon_thermo

        self.P = P
        self.barostat = (P is not None)
        if self.barostat:
            assert timecon_baro is not None
            self.timecon_baro = timecon_baro

        MDEngineBias.add_engine(mdengine=self)

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


class YaffBias(MDEngineBias, ABC):
    """Definition of bias engines for Yaff MD Engine."""

    def add_engine(self, mdengine):
        """function is called from MDEngine instance."""
        self.mdengine = mdengine

    @abstractmethod
    def append_hook(self):
        """gets hook for given bias."""


class YaffBiasNone(YaffBias):

    def append_hook(self):
        pass

    def get_bias(self, cvs):
        return 0.0


class YaffBiasMTD(YaffBias):
    """(Well-tempered) metadynamics for Yaff Engine."""

    def __init__(self, cvs: CV, K, sigmas, step_hills=100, start=100) -> None:
        super().__init__(cvs)
        self.K = K
        self.sigmas = sigmas
        self.step_hills = step_hills
        self.start = start

    def append_hook(self):
        self.yaff_cvs = self._convert_cv(self.cvs)
        self.mdengine.hooks.append(
            yaff.sampling.MTDHook(self.mdengine.ff,
                                  self.yaff_cvs,
                                  sigma=self.sigmas,
                                  K=self.K,
                                  periodicities=self.cvs.periodicity[:, 1] - self.cvs.periodicity[:, 0],
                                  f=self.mdengine.fh5,
                                  start=self.start,
                                  step=self.step_hills))

    def _convert_cv(self, cvs: CV):
        """convert generic CV class to a yaff CollectiveVariable."""

        if not isinstance(cvs, list):
            cvs = [cvs]
        cv2 = []

        class YaffCv(yaff.pes.CollectiveVariable):

            def __init__(self, system, cv):
                super().__init__("YaffCV", system)
                self.cv = cv

            def compute(self, gpos=None, vtens=None):
                self.value = self.cv.compute(coordinates=self.system.pos,
                                             cell=self.system.cell.rvecs,
                                             gpos=gpos,
                                             vir=vtens)
                return self.value

        for cv in cvs:
            if cv.n == 1:
                cv2.append(YaffCv(self.mdengine.ff.system, cv))
            else:
                cvs_split = cv.split_cv()
                for cv_split in cvs_split:
                    cv2.append(YaffCv(self.mdengine.ff.system, cv_split))

        return cv2

    def get_bias(self, cvs):

        mtd = yaff.analysis.biased_sampling.SumHills(cvs)
        mtd.load_hdf5('traj.h5')
        fes = mtd.compute_fes()


class YaffBiasPlumed(YaffBias):

    def append_hook(self):
        self.plumed_cvs = self._convert_cvs()

        plumed = yaff.external.ForcePartPlumed(self.mdengine.ff.system, fn='plumed.dat')
        self.mdengine.ff.add_part(plumed)
        self.mdengine.hooks.append(plumed)

    def _convert_cvs(self):
        raise NotImplementedError(
            "convert python CV to plumed script, see https://giorginolab.github.io/plumed2-pycv/ ")

    def get_bias(self, cvs):
        raise NotImplementedError


class YaffEngine(MDEngine):
    """MD engine with YAFF as backend.

    Args:
        ff (yaff.pes.ForceField)
    """

    def __init__(
        self,
        ff: yaff.pes.ForceField,
        MDEngineBias: YaffBias,
        T=None,
        P=None,
        timestep=None,
        timecon_thermo=None,
        timecon_baro=None,
        filename="traj.h5",
        write_step=100,
    ) -> None:
        super().__init__(T=T,
                         P=P,
                         MDEngineBias=MDEngineBias,
                         timestep=timestep,
                         timecon_thermo=timecon_thermo,
                         timecon_baro=timecon_baro,
                         write_step=write_step,
                         filename=filename)
        self.ff = ff

        # setup the the logging
        vsl = yaff.sampling.VerletScreenLog(step=write_step)
        self.hooks = [vsl]

        #setup writer to collect results
        if self.filename.endswith(".h5"):
            self.fh5 = h5py.File(self.filename, 'w')
            h5writer = yaff.sampling.HDF5Writer(self.fh5, step=write_step)
            self.hooks.append(h5writer)
        elif self.filename.endswith(".xyz"):
            xyzhook = XYZWriter(self.filename, step=write_step)
            self.hooks.append(xyzhook)
        else:
            raise NotImplemented("only h5 and xyz are supported as filename")

        # setup baro/thermostat
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

        #add the enhanced dynamics hook
        MDEngineBias.append_hook()

        self.verlet = yaff.sampling.VerletIntegrator(
            self.ff,
            self.timestep,
            temp0=self.T,
            hooks=self.hooks,
            # add forces as state item
            state=[self._GposContribStateItem()],
        )

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

    class _GposContribStateItem(yaff.sampling.iterative.StateItem):
        """Keeps track of all the contributions to the forces."""

        def __init__(self):
            yaff.sampling.iterative.StateItem.__init__(self, 'gpos_contribs')

        def get_value(self, iterative):
            n = len(iterative.ff.parts)
            natom, _ = iterative.gpos.shape
            gpos_contribs = np.zeros((n, natom, 3))
            for i in range(n):
                # gpos_contribs = gpos_contribs.at[i, :, :].set(iterative.ff.parts[i].gpos)
                gpos_contribs[i, :, :] = iterative.ff.parts[i].gpos
            return gpos_contribs

        def iter_attrs(self, iterative):
            yield 'gpos_contrib_names', np.array([part.name for part in iterative.ff.parts], dtype='S')


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
