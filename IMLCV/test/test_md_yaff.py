from __future__ import division

import os

from IMLCV.base.CV import CV, CVUtils, CombineCV
from IMLCV.base.MdEngine import YaffEngine, YaffBiasMTD
from yaff.test.common import get_alaninedipeptide_amber99ff
from yaff.log import log
import numpy as np
from molmod import units
from yaff.system import System
from yaff import ForceField
from ase.calculators.cp2k import CP2K
import ase.io
import ase.units
from pathlib import Path
import pytest

log.set_level(log.medium)
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


def test_yaff_md_ala_dipep():
    T = 600 * units.kelvin
    ff = get_alaninedipeptide_amber99ff()

    cvs = CombineCV([
        CV(CVUtils.dihedral, numbers=[4, 6, 8, 14], periodicity=2.0 * np.pi),
        CV(CVUtils.dihedral, numbers=[6, 8, 14, 16], periodicity=2.0 * np.pi),
    ])

    bias = YaffBiasMTD(cvs=cvs, K=1.2 * units.kjmol, sigmas=np.array([0.35, 0.35]), step_hills=50)
    # bias = YaffBiasNone(cvs=None)

    yaffmd = YaffEngine(
        ff=ff,
        MDEngineBias=bias,
        write_step=1000,
        T=T,
        P=None,
        timestep=2.0 * units.femtosecond,
        timecon_thermo=100.0 * units.femtosecond,
        filename="output/aladipep.h5",
    )

    yaffmd.run(int(1e2))


def test_yaff_md_mil53():
    T = 300 * units.kelvin
    P = 1 * units.atm

    system = System.from_file("data/MIL53.chk")
    ff = ForceField.generate(system, 'data/MIL53_pars.txt')

    cvs = CV(CVUtils.Volume)

    bias = YaffBiasMTD(cvs=cvs, K=1.2 * units.kjmol, sigmas=np.array([0.35]), step_hills=50)

    yaffmd = YaffEngine(
        ff=ff,
        MDEngineBias=bias,
        write_step=100,
        T=T,
        P=P,
        timestep=1.0 * units.femtosecond,
        timecon_thermo=100.0 * units.femtosecond,
        timecon_baro=100.0 * units.femtosecond,
        filename="output/mil53.h5",
    )

    yaffmd.run(int(1e2))


@pytest.mark.skip(reason="path+files not ready")
def test_yaff_ase():
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
    bias = YaffBiasMTD(cvs=cvs, K=1.2 * units.kjmol, sigmas=np.array([0.35]), step_hills=50)
    yaffmd = YaffEngine(
        ff=ff,
        MDEngineBias=bias,
        write_step=100,
        T=600 * units.kelvin,
        timestep=1.0 * units.femtosecond,
        timecon_thermo=100.0 * units.femtosecond,
        timecon_baro=100.0 * units.femtosecond,
        filename="output/ase.h5",
    )

    yaffmd.run(1000)


if __name__ == '__main__':
    test_yaff_md_ala_dipep()
    test_yaff_md_mil53()
    # test_yaff_ase()
