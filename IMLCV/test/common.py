from IMLCV.base.CV import CV, CVUtils, CombineCV
from IMLCV.base.MdEngine import YaffEngine
from IMLCV.base.bias import BiasMTD, Energy

from yaff.test.common import get_alaninedipeptide_amber99ff
import numpy as np
from molmod import units
from yaff.system import System
from yaff import ForceField
from ase.calculators.cp2k import CP2K
import ase.io
import ase.units
from pathlib import Path


def ala_yaff(write=1000):

    T = 600 * units.kelvin
    # ff = get_alaninedipeptide_amber99ff()

    # ff = get_alaninedipeptide_amber99ff()

    cvs = CombineCV([
        CV(CVUtils.dihedral, numbers=[4, 6, 8, 14], periodicity=[-np.pi, np.pi]),
        CV(CVUtils.dihedral, numbers=[6, 8, 14, 16], periodicity=[-np.pi, np.pi]),
    ])
    bias = BiasMTD(cvs=cvs, K=2.0 * units.kjmol, sigmas=np.array([0.35, 0.35]), start=500, step=500)

    yaffmd = YaffEngine(
        ener=get_alaninedipeptide_amber99ff,
        bias=bias,
        write_step=write,
        T=T,
        P=None,
        timestep=2.0 * units.femtosecond,
        timecon_thermo=100.0 * units.femtosecond,
        filename="output/aladipep.h5",
    )

    return yaffmd


def mil53_yaff():
    T = 300 * units.kelvin
    P = 1 * units.atm

    system = System.from_file("data/MIL53.chk")
    ff = ForceField.generate(system, 'data/MIL53_pars.txt')
    cvs = CV(CVUtils.Volume)
    bias = BiasMTD(cvs=cvs, K=1.2 * units.kjmol, sigmas=np.array([0.35]), start=50, step=50)

    yaffmd = YaffEngine(
        ener=ff,
        bias=bias,
        write_step=100,
        T=T,
        P=P,
        timestep=1.0 * units.femtosecond,
        timecon_thermo=100.0 * units.femtosecond,
        timecon_baro=100.0 * units.femtosecond,
        filename="output/mil53.h5",
    )

    return yaffmd


def todo_ASE_yaff():

    #make CP2K ase calculator
    path_atoms = Path.cwd() / 'atoms.xyz'
    with open(path_atoms, 'r') as f:
        atoms = ase.io.read(f)

    path_source = Path('/data/gent/vo/000/gvo00003/vsc42365/Libraries')

    path_potentials = path_source / 'GTH_POTENTIALS'
    path_basis = path_source / 'BASIS_SETS'
    path_dispersion = path_source / 'dftd3.dat'

    with open("CP2K_para.inp", "r") as f:
        additional_input = f.read().format(path_basis, path_potentials, path_dispersion)

    calc_cp2k = CP2K(atoms=atoms,
                     auto_write=True,
                     basis_set=None,
                     command='mpirun cp2k_shell.popt',
                     cutoff=800 * ase.units.Rydberg,
                     stress_tensor=False,
                     print_level='LOW',
                     inp=additional_input,
                     pseudo_potential=None,
                     max_scf=None,
                     xc=None,
                     basis_set_file=None,
                     charge=None,
                     potential_file=None,
                     debug=False)

    atoms.calc = calc_cp2k

    #do yaff MD
    ff = YaffEngine.create_forcefield_from_ASE(atoms, calc_cp2k)
    cvs = CV(CVUtils.Volume)
    bias = BiasMTD(cvs=cvs, K=1.2 * units.kjmol, sigmas=np.array([0.35]), start=50, step=50)
    yaffmd = YaffEngine(
        ener=ff,
        bias=bias,
        write_step=100,
        T=600 * units.kelvin,
        timestep=1.0 * units.femtosecond,
        timecon_thermo=100.0 * units.femtosecond,
        timecon_baro=100.0 * units.femtosecond,
        filename="output/ase.h5",
    )

    return yaffmd


def villin_OpenMM():

    pdb = PDBFile('villin.pdb')
    forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
    system = forcefield.createSystem(pdb.topology,
                                     nonbondedMethod=PME,
                                     nonbondedCutoff=1 * nanometer,
                                     constraints=HBonds)

    simulation = Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)
    T = 300 * kelvin
    timestep = 0.004 * picoseconds
