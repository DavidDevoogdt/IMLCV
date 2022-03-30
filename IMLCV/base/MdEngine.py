"""MD engine class peforms MD simulations in a given NVT/NPT ensemble.

Currently, the MD is done with YAFF/OpenMM
"""
from copyreg import pickle
import openmm
import openmm.openmm
import openmm.unit
import h5py
import yaff.sampling
import yaff.system
import yaff.pes
import yaff.pes.bias
import yaff.external
import yaff.analysis.biased_sampling
from yaff.sampling.io import XYZWriter
import yaff.sampling.iterative

import molmod.constants
import molmod.units

import numpy as np

import molmod
import ase
from abc import ABC, abstractmethod

import ase.geometry
import ase.stress

from IMLCV.base.bias import Bias

import jax.numpy as jnp

import dill

from yaff.log import log, timer


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
                 bias: Bias,
                 T,
                 P,
                 timestep=None,
                 timecon_thermo=None,
                 timecon_baro=None,
                 filename="traj.h5",
                 write_step=100,
                 screenlog=1000) -> None:
        if isinstance(bias, list):
            raise NotImplementedError("only single bias atm")

        self.bias = bias
        self.filename = filename
        self.write_step = write_step

        self.timestep = timestep

        self.T = T
        self.thermostat = (T is not None)
        self.timecon_thermo = timecon_thermo
        if self.thermostat:
            assert timecon_thermo is not None

        self.P = P
        self.barostat = (P is not None)
        self.timecon_baro = timecon_baro
        if self.barostat:
            assert timecon_baro is not None

        self.screenlog = screenlog

    def save(self, filename):

        md_dict = {
            'T': self.T,
            'P': self.P,
            'timestep': self.timestep,
            'timecon_thermo': self.timecon_thermo,
            'timecon_baro': self.timecon_baro,
            'filename': self.filename,
            'write_step': self.write_step,
            'screenlog': self.screenlog,
        }

        with open(filename, 'wb') as f:
            dill.dump(md_dict, f)
            dill.dump(self.bias, f)

    @abstractmethod
    def load(filename):
        pass

    @abstractmethod
    def run(self, steps):
        """run the integrator for a given number of steps.

        Args:
            steps: number of MD steps
        """

    @abstractmethod
    def to_ASE_traj(self):
        """convert the MD run to ASE trajectory."""
        raise NotImplementedError


class YaffEngine(MDEngine):
    """MD engine with YAFF as backend.

    Args:
        ff (yaff.pes.ForceField)
    """

    def __init__(
        self,
        ff: yaff.pes.ForceField,
        bias: Bias,
        T=None,
        P=None,
        timestep=None,
        timecon_thermo=None,
        timecon_baro=None,
        filename="traj.h5",
        write_step=100,
        screenlog=1000,
    ) -> None:
        super().__init__(T=T,
                         P=P,
                         bias=bias,
                         timestep=timestep,
                         timecon_thermo=timecon_thermo,
                         timecon_baro=timecon_baro,
                         write_step=write_step,
                         filename=filename,
                         screenlog=screenlog)
        self.ff = ff

        # setup the the logging
        vsl = yaff.sampling.VerletScreenLog(step=screenlog)
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
            nhc = yaff.sampling.NHCThermostat(self.T, timecon=self.timecon_thermo)
        if self.barostat:
            mtk = yaff.sampling.MTKBarostat(ff, self.T, self.P, timecon=self.timecon_baro, anisotropic=False)
        if self.thermostat and self.barostat:
            tbc = yaff.sampling.TBCombination(nhc, mtk)
            self.hooks.append(tbc)
        elif self.thermostat and not self.barostat:
            self.hooks.append(nhc)
        elif not self.thermostat and self.barostat:
            self.hooks.append(mtk)
        else:
            raise NotImplementedError

        yb = YaffEngine._YaffBias(self.ff, bias)
        part = yaff.pes.ForcePartBias(ff.system)
        part.add_term(yb)
        ff.add_part(part)

        if yb.hook:
            self.hooks.append(yb)

        self.verlet = yaff.sampling.VerletIntegrator(
            self.ff,
            self.timestep,
            temp0=self.T,
            hooks=self.hooks,
            # add forces as state item
            state=[self._GposContribStateItem()],
        )

        # #remove jit from timing
        # self.run(1)
        # timer.reset()

    @staticmethod
    def create_forcefield_from_ASE(atoms, calculator) -> yaff.pes.ForceField:
        """Creates force field from ASE atoms instance."""

        class ForcePartASE(yaff.pes.ForcePart):
            """YAFF Wrapper around an ASE calculator.

            args:
                   system (yaff.System): system object
                   atoms (ase.Atoms): atoms object with calculator included.
            """

            def __init__(self, system: yaff.system.System, atoms, calculator):
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

    @staticmethod
    def load(
        filename,
        ff: yaff.pes.ForceField,
    ):
        with open(filename, 'rb') as f:
            md_dict = dill.load(f)
            bias = dill.load(f)

        return YaffEngine(ff, bias=bias, **md_dict)

    class _YaffBias(yaff.sampling.iterative.Hook, yaff.pes.bias.BiasPotential):
        """placeholder for all classes which work with yaff."""

        def __init__(
            self,
            ff: yaff.pes.ForceField,
            bias: Bias,
        ) -> None:

            self.ff = ff
            self.bias = bias

            #not all biases have a hook
            self.hook = (self.bias.start is not None) and (self.bias.step is not None)
            if self.hook:
                super().__init__(start=self.bias.start, step=self.bias.step)

        def compute(self, gpos=None, vtens=None):
            return self.bias.compute_coor(coordinates=self.ff.system.pos,
                                          cell=self.ff.system.cell.rvecs,
                                          gpos=gpos,
                                          vir=vtens)

        def __call__(self, iterative):
            coordinates = self.ff.system.pos
            cell = self.ff.system.cell.rvecs

            self.bias.update_bias(coordinates, cell)

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


# class OpenMMEngine(MDEngine):

#     def __init__(
#         self,
#         system: openmm.openmm.System,
#         cv: CV,
#         T,
#         P,
#         ES="None",
#         timestep=None,
#         timecon_thermo=None,
#         timecon_baro=None,
#     ) -> None:
#         super().__init__(cv, T, P, ES, timestep, timecon_thermo, timecon_baro)

#         if self.thermostat:
#             self.integrator = openmm.openmm.LangevinMiddleIntegrator(temperature=T,
#                                                                      frictionCoeff=1 / picosecond,
#                                                                      stepSize=timestep)
#         if self.barostat:
#             system.addForce(openmm.openmm.MonteCarloBarostat(P, T))

#         if self.ES == "MTD":
#             self.cvs = [self._convert_cv(cvy, ff=self.ff) for cvy in cv]
#             raise NotImplementedError

#         elif self.ES == "MTD_plumed":
#             raise NotImplementedError

#     def _mtd(self, K, sigmas, periodicities, step):
#         raise NotImplementedError

#     def _convert_cv(self, cv, ff):
#         """convert generic CV class to an OpenMM CV."""
#         raise NotImplementedError

#     def run(self, steps):
#         self.integrator.step(steps)
