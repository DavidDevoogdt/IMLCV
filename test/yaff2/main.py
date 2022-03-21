import sys
import torch
import time
import numpy as np
from pathlib import Path
import h5py
import yaff
import molmod
import ase

from ase.io import read, write
from ase.geometry import Cell
from ase.stress import voigt_6_to_full_3x3_stress
from ase.calculators.cp2k import CP2K

import functools

print = functools.partial(print, flush=True)

# when evaluating a model that was converted to double precision, the default
# dtype of torch must be changed as well; otherwise the data loading within
# the calculator will still be performed in single precision
# torch.set_default_dtype(torch.float64)


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
        yield 'gpos_contrib_names', np.array(
            [part.name for part in iterative.ff.parts], dtype='S')


class ForcePartASE(yaff.pes.ForcePart):
    """YAFF Wrapper around an ASE calculator"""

    def __init__(self, system, atoms, calculator):
        """Constructor

        Parameters
        ----------

        system : yaff.System
            system object

        atoms : ase.Atoms
            atoms object with calculator included.

        """
        yaff.pes.ForcePart.__init__(self, 'ase', system)
        self.system = system  # store system to obtain current pos and box
        self.atoms = atoms
        self.calculator = calculator

    def _internal_compute(self, gpos=None, vtens=None):
        self.atoms.set_positions(self.system.pos / molmod.units.angstrom)
        self.atoms.set_cell(
            Cell(self.system.cell._get_rvecs() / molmod.units.angstrom))
        energy = self.atoms.get_potential_energy() * molmod.units.electronvolt
        if gpos is not None:
            forces = self.atoms.get_forces()
            gpos[:] = -forces * molmod.units.electronvolt / \
                molmod.units.angstrom
        if vtens is not None:
            volume = np.linalg.det(self.atoms.get_cell())
            stress = voigt_6_to_full_3x3_stress(self.atoms.get_stress())
            vtens[:] = volume * stress * molmod.units.electronvolt
        return energy


def create_forcefield(atoms, calculator):
    """Creates force field from ASE atoms instance"""
    system = yaff.System(
        numbers=atoms.get_atomic_numbers(),
        pos=atoms.get_positions() * molmod.units.angstrom,
        rvecs=atoms.get_cell() * molmod.units.angstrom,
    )
    system.set_standard_masses()
    part_ase = ForcePartASE(system, atoms, calculator)
    return yaff.pes.ForceField(system, [part_ase])


def simulate(steps,
             step,
             start,
             atoms,
             calculator,
             temperature,
             pressure=None):
    """Samples the phase space using Langevin dynamics"""
    # set default output paths
    path_h5 = Path.cwd() / 'md.h5'
    path_xyz = Path.cwd() / 'md.xyz'
    #path_h5  = None
    #path_xyz = None

    # create forcefield from atoms
    ff = create_forcefield(atoms, calculator)

    # hooks
    hooks = []
    loghook = yaff.VerletScreenLog(step=step, start=0)
    hooks.append(loghook)
    if path_h5 is not None:
        h5file = h5py.File(path_h5, 'w')
        h5hook = yaff.HDF5Writer(h5file, step=step, start=start)
        hooks.append(h5hook)
    if path_xyz is not None:
        xyzhook = yaff.XYZWriter(str(path_xyz), step=step, start=start)
        hooks.append(xyzhook)

    # temperature / pressure control
    thermo = yaff.LangevinThermostat(temperature,
                                     timecon=100 * molmod.units.femtosecond)
    if pressure is None:
        print('CONSTANT TEMPERATURE, CONSTANT VOLUME')
        #vol_constraint = True
        # pressure = 0 # dummy pressure
        hooks.append(thermo)
    else:
        print('CONSTANT TEMPERATURE, CONSTANT PRESSURE')
        vol_constraint = False
        baro = yaff.LangevinBarostat(
            ff,
            temperature,
            pressure * 1e6 * molmod.units.pascal,  # pressure in MPa
            timecon=molmod.units.picosecond,
            anisotropic=True,
            vol_constraint=vol_constraint,
        )
        tbc = yaff.TBCombination(thermo, baro)
        hooks.append(tbc)

    # integration
    verlet = yaff.VerletIntegrator(
        ff,
        state=[GposContribStateItem()],
        timestep=0.5 * molmod.units.femtosecond,
        hooks=hooks,
        temp0=temperature,  # initialize velocities to correct temperature
    )
    yaff.log.set_level(yaff.log.medium)
    verlet.run(steps)
    yaff.log.set_level(yaff.log.silent)


def convert_h5_to_asetraj(file_path_h5):
    with h5py.File(file_path_h5, 'r') as f:
        at_numb = f['system']['numbers']
        traj = []
        for frame, energy_au in enumerate(f['trajectory']['epot']):
            pos_A = f['trajectory']['pos'][frame, :, :] / molmod.units.angstrom
            cell_A = f['trajectory']['cell'][
                frame, :, :] / molmod.units.angstrom
            #vel_x = f['trajectory']['vel'][frame,:,:] / x
            energy_eV = energy_au / molmod.units.electronvolt
            forces_eVA = -f['trajectory']['gpos_contribs'][
                frame,
                0, :, :] * molmod.units.angstrom / molmod.units.electronvolt  # forces = -gpos
            vol_A3 = f['trajectory']['volume'][frame] / \
                molmod.units.angstrom**3
            vtens_eV = f['trajectory']['vtens'][
                frame, :, :] / molmod.units.electronvolt
            stresses_eVA3 = vtens_eV / vol_A3
            atoms = ase.Atoms(
                numbers=at_numb,
                positions=pos_A,
                pbc=True,
                cell=cell_A,
            )
            # atoms.set_velocities(vel_x)
            atoms.arrays['forces'] = forces_eVA
            atoms.info['stress'] = stresses_eVA3
            atoms.info['energy'] = energy_eV
            traj.append(atoms)
    return traj


if __name__ == '__main__':
    steps = 1000
    step = 50
    start = 0
    #temperature = float(sys.argv[1])
    temperature = 800
    pressure = None
    path_atoms = Path.cwd() / 'atoms.xyz'
    with open(path_atoms, 'r') as f:
        atoms = read(f)

    #path_model  = Path.cwd() / 'model.pth'
    #calc_neq = NequIPCalculator.from_deployed_model(path_model, device='cuda')

    path_source = Path('/data/gent/vo/000/gvo00003/vsc42365/Libraries')
    path_potentials = path_source / 'GTH_POTENTIALS'
    path_basis = path_source / 'BASIS_SETS'
    path_dispersion = path_source / 'dftd3.dat'

    with open("CP2K_para.inp", "r") as f:
        additional_input = f.read().format(path_basis, path_potentials,
                                           path_dispersion)

    calc_cp2k = CP2K(
        atoms=atoms,
        auto_write=True,
        basis_set=None,
        command='mpirun cp2k_shell.popt',
        cutoff=800 * ase.units.Rydberg,
        stress_tensor=False,
        print_level='LOW',
        inp=additional_input,
        pseudo_potential=None,
        max_scf=None,  # disable
        xc=None,  # disable
        basis_set_file=None,  # disable
        charge=None,  # disable
        potential_file=None,  # disable
        debug=False)

    atoms.calc = calc_cp2k
    simulate(steps, step, start, atoms, calc_cp2k, temperature, pressure)

    traj = convert_h5_to_asetraj('md.h5')
    write('md_ext.xyz', traj, format='extxyz', append=True)
