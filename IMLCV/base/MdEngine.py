"""MD engine class peforms MD simulations in a given NVT/NPT ensemble.

Currently, the MD is done with YAFF/OpenMM
"""
from audioop import bias
from copyreg import pickle
from functools import partial
import openmm
import openmm.openmm
import openmm.unit
import h5py
import yaff.sampling
from yaff import log
import yaff.system
import yaff.pes
import yaff.pes.bias
import yaff.external
import yaff.analysis.biased_sampling
from yaff.sampling.io import XYZWriter
import yaff.sampling.iterative

import molmod.constants

import numpy as np
import jax.numpy as jnp
from jax import jit, grad, jacfwd

# from .CV import CV, YaffCv
from .CV import CV
import molmod
import ase
from abc import ABC, abstractmethod

from ase.io import read, write
import ase.geometry
import ase.stress

import pickle


class Bias(ABC):
    """base class for biased MD runs."""

    def __init__(self, cvs: CV, start=None, step=None) -> None:
        """"args:
                cvs: collective variables
                start: number of md steps before update is called
                step: steps between update is called"""

        self.cvs = cvs
        self.start = start
        self.step = step

        self._update_bias_f()

    def compute_coor(self, coordinates, cell, gpos=None, vir=None):
        """Computes the bias for arbitrary coordinates+cell."""

        if not (cell.shape == (0, 3) or cell.shape == (3, 3)):
            raise NotImplementedError("other cell shapes not yet supported")

        bv = (vir is not None)
        bg = (gpos is not None)

        [cvs, jac_p_val, jac_c_val] = self.cvs.compute(coordinates, cell, jac_p=bg, jac_c=bv)
        ener = self._comp(cvs)

        if bv or bg:
            de = self._dcomp(cvs)

            if bg:
                gpos = np.einsum('ij,jkl->kl', de, jac_p_val)
                #gpos = gpos.at[:].set(jnp.einsum('ij,jkl->kl', de, jac_p_val))

            if bv:
                vir[:] = np.einsum('ji,kz,kjl->il', cell, de, jac_c_val)
                #vir = vir.at[:].set(jnp.einsum('ji,kz,kjl->il', cell, de, jac_c_val))

        return ener

    def compute(self, cvs):
        """function that calculates the bias potential"""
        raise NotImplementedError

    def update(self, coordinates, cell):
        """update the bias. Used in metadyanmics to deposit hills."""
        self._update_bias_f()

    def _update_bias_f(self):
        self._comp = jit(lambda x: (jnp.atleast_1d(self.compute(x))))
        self._dcomp = jit(jacfwd(self._comp))


class BiasF(Bias):
    """Bias according to CV"""

    def __init__(self, cvs: CV, f=None):
        self.compute = f if (f is not None) else lambda x: 0.0

        super().__init__(cvs, start=None, step=None)


class BiasMTD(Bias):
    '''A sum of Gaussian hills, for instance used in metadynamics: Adapted from Yaff

       V = \sum_{\\alpha} K_{\\alpha}} \exp{-\sum_{i} \\frac{(q_i-q_{i,\\alpha}^0)^2}{2\sigma^2}}

       where \\alpha loops over deposited hills and i loops over collective
       variables.
    '''

    def __init__(self, cvs: CV, K, sigmas, start=None, step=None, tempering=0.0, f=None):
        '''
           args:
                sigmas:  The width of the Gaussian or a NumPy array [Ncv] specifying thewidths of the Gaussians
                periodicity: todo
        '''
        super().__init__(cvs, start, step)

        if isinstance(sigmas, float):
            sigmas = np.array([sigmas])
        # if isinstance(periodicities, float):
        #     periodicities = np.array([periodicities])

        periodicities = cvs.periodicity

        self.ncv = self.cvs.n
        assert sigmas.ndim == 1
        assert sigmas.shape[0] == self.ncv
        assert np.all(sigmas > 0)
        self.sigmas = sigmas
        self.sigmas_isq = 1.0 / (2.0 * sigmas**2.0)
        self.Ks = np.zeros((0,))
        self.q0s = np.zeros((0, self.ncv))
        if periodicities is not None:
            assert periodicities.shape[0] == self.ncv

        self.q_filt = ~np.isnan(periodicities).any(axis=1)
        self.d_q = periodicities[self.q_filt, 1] - periodicities[self.q_filt, 0]
        self.q_min = periodicities[self.q_filt, 0]

        self.tempering = tempering
        self.K = K

        self.f = f

        self._update_cv_fun()

    def add_hill_cv(self, q0, K):
        '''
            Deposit a single hill

            args: 
                q0: 
                A NumPy array [Ncv] specifying the Gaussian center for each
                collective variable, or a single float if there is only one
                collective variable

                K:
                The force constant of this hill
        '''
        if isinstance(q0, float):
            assert self.ncv == 1
            q0 = np.array([q0])
        assert q0.ndim == 1
        assert q0.shape[0] == self.ncv
        self.q0s = np.append(self.q0s, [q0], axis=0)
        self.Ks = np.append(self.Ks, [K], axis=0)

        self._update_cv_fun()

    def add_hills_cv(self, q0s, Ks):
        '''
            Deposit multiple hills

            **Arguments:**

            q0s
                A NumPy array [Nhills,Ncv]. Each row represents a hill,
                specifying the Gaussian center for each collective variable

            K
                A NumPy array [Nhills] providing the force constant of each
                hill.
        '''
        assert q0s.ndim == 2
        assert q0s.shape[1] == self.ncv
        self.q0s = np.concatenate((self.q0s, q0s), axis=0)
        assert Ks.ndim == 1
        assert Ks.shape[0] == q0s.shape[0]
        self.Ks = np.concatenate((self.Ks, Ks), axis=0)

        self._update_cv_fun()

    def _update_cv_fun(self):
        self.compute = partial(BiasMTD._compute,
                               q0s=self.q0s,
                               q_filt=self.q_filt,
                               d_q=self.d_q,
                               sigmas_isq=self.sigmas_isq,
                               Ks=self.Ks)

    def update(self, coordinates, cell):
        """hook to update the bias. Used in metadyanmics to deposit hills"""
        # Compute current CV values
        q0s, _, _ = self.cvs.compute(coordinates, cell)
        # Compute force constant
        K = self.K
        if self.tempering != 0.0:
            K *= np.exp(-self.compute() / molmod.constants.boltzmann / self.tempering)
        # Add a hill
        self.add_hill_cv(q0s, K)

        self._update_cv_fun()

        super().update(coordinates, cell)

    def _compute(cvs, q0s, q_filt, d_q, sigmas_isq, Ks):
        deltas = cvs - q0s
        diff = jnp.floor(0.5 + deltas[:, q_filt] / d_q) * d_q
        deltas = deltas.at[:, q_filt].set(deltas[:, q_filt] - diff)

        exparg = deltas * deltas
        exparg = jnp.multiply(exparg, sigmas_isq)
        exparg = jnp.sum(exparg, axis=1)
        # Compute the bias energy
        exponents = jnp.exp(-exparg)
        energy = jnp.sum(Ks * exponents)

        return energy


class BiasPlumed(Bias):
    pass


class BiasThermolib(Bias):
    """use FES from thermolib as bias potential."""
    pass


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
                 write_step=100) -> None:
        if isinstance(bias, list):
            raise NotImplementedError("only single bias atm")

        self.bias = bias
        self.init_bias = False
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
        screenlog=500,
    ) -> None:
        super().__init__(T=T,
                         P=P,
                         bias=bias,
                         timestep=timestep,
                         timecon_thermo=timecon_thermo,
                         timecon_baro=timecon_baro,
                         write_step=write_step,
                         filename=filename)
        self.ff = ff

        # setup the the logging
        vsl = yaff.sampling.VerletScreenLog(step=screenlog)
        self.hooks = [vsl]

        #setup writer to collect results
        # if self.filename.endswith(".h5"):
        #     self.fh5 = h5py.File(self.filename, 'w')
        #     h5writer = yaff.sampling.HDF5Writer(self.fh5, step=write_step)
        #     self.hooks.append(h5writer)
        # elif self.filename.endswith(".xyz"):
        #     xyzhook = XYZWriter(self.filename, step=write_step)
        #     self.hooks.append(xyzhook)
        # else:
        #     raise NotImplemented("only h5 and xyz are supported as filename")

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

        class YaffBias(yaff.sampling.iterative.Hook, yaff.pes.bias.BiasPotential):
            """placeholder for all classes which work with yaff"""

            def __init__(
                self,
                ff: yaff.pes.ForceField,
                bias: Bias,
            ) -> None:

                self.ff = ff
                self.bias = bias

                self.hook = (self.bias.start is not None) and (self.bias.step is not None)
                if self.hook:
                    super().__init__(start=self.bias.start, step=self.bias.step)

            def compute(self, gpos=None, vtens=None):
                coordinates = self.ff.system.pos
                cell = self.ff.system.cell.rvecs

                return self.bias.compute_coor(coordinates=coordinates, cell=cell, gpos=gpos, vir=vtens)

            def __call__(self, iterative):
                coordinates = self.ff.system.pos
                cell = self.ff.system.cell.rvecs

                bias.update(coordinates, cell)

        yb = YaffBias(self.ff, bias)

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
        super().run(steps)
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
