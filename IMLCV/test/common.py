import os
import pathlib
import shutil
from importlib import import_module

import ase
import ase.io
import ase.units
import jax.numpy as jnp
import numpy as np
from ase.calculators.cp2k import CP2K
from keras.api._v2 import keras as KerasAPI
from molmod import units
from molmod.units import kelvin

import yaff
from IMLCV import ROOT_DIR
from IMLCV.base.bias import AseEnergy, BiasMTD, NoneBias
from IMLCV.base.CV import CV, SystemParams, Volume, cvflow, dihedral
from IMLCV.base.CVDiscovery import CVDiscovery
from IMLCV.base.MdEngine import MDEngine, YaffEngine
from IMLCV.base.metric import Metric
from IMLCV.scheme import Scheme
from yaff.test.common import get_alaninedipeptide_amber99ff

keras: KerasAPI = import_module("tensorflow.keras")  # type: ignore


abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


def cleancopy(base):

    if not os.path.exists(f"{base}"):
        os.mkdir(base)
        return

    if not os.path.exists(f"{base}_orig"):
        assert os.path.exists(f"{base}"), "folder not found"
        shutil.copytree(f"{base}", f"{base}_orig")

    if os.path.exists(f"{base}"):

        shutil.rmtree(f"{base}")
    shutil.copytree(f"{base}_orig", f"{base}")


def alanine_dipeptide_yaff():
    T = 300 * kelvin

    cv0 = CV(
        f=(dihedral(numbers=[4, 6, 8, 14]) + dihedral(numbers=[6, 8, 14, 16])),
        metric=Metric(
            periodicities=[True, True],
            bounding_box=[[-np.pi, np.pi], [-np.pi, np.pi]],
        ),
    )

    mde = YaffEngine(
        ener=get_alaninedipeptide_amber99ff,
        T=T,
        timestep=2.0 * units.femtosecond,
        timecon_thermo=100.0 * units.femtosecond,
        write_step=1,
        bias=NoneBias(cv0),
    )

    return mde


def mil53_yaff():
    T = 300 * units.kelvin
    P = 1 * units.atm

    system = yaff.System.from_file("data/MIL53.chk")
    ff = yaff.ForceField.generate(system, "data/MIL53_pars.txt")
    cvs = CV(f=Volume)
    bias = BiasMTD(
        cvs=cvs, K=1.2 * units.kjmol, sigmas=np.array([0.35]), step=50, start=50
    )

    yaffmd = YaffEngine(
        ener=ff,
        bias=bias,
        write_step=10,
        T=T,
        P=P,
        timestep=1.0 * units.femtosecond,
        timecon_thermo=100.0 * units.femtosecond,
        timecon_baro=100.0 * units.femtosecond,
    )

    return yaffmd


def ase_yaff():

    base = pathlib.Path(ROOT_DIR) / "test/data/CsPbI_3"

    # make CP2K ase calculator
    path_atoms = base / "Pos.xyz"
    with open(path_atoms) as f:
        atoms = ase.io.read(f)

    path_source = base / "Libraries"

    path_potentials = os.path.relpath(path_source / "GTH_POTENTIALS")
    path_basis = os.path.relpath(path_source / "BASIS_SETS")
    path_dispersion = os.path.relpath(path_source / "dftd3.dat")

    with open(base / "cp2k.inp") as f:
        additional_input = f.read().format(path_basis, path_potentials, path_dispersion)

    calc_cp2k = CP2K(
        atoms=atoms,
        auto_write=True,
        basis_set=None,
        command="mpirun cp2k_shell.psmp",
        cutoff=800 * ase.units.Rydberg,
        stress_tensor=True,
        print_level="LOW",
        inp=additional_input,
        pseudo_potential=None,
        max_scf=None,
        xc=None,
        basis_set_file=None,
        charge=None,
        potential_file=None,
        debug=False,
        directory=".CP2K",
    )

    @cvflow
    def f(sp: SystemParams):
        l = jnp.linalg.norm(sp.cell, axis=0)
        return jnp.array([l.min(), l.max()])

    cv = CV(
        f=f,
        metric=Metric(
            periodicities=[False, False],
            bounding_box=jnp.array([[4.0, 6.0], [5.0, 9.0]]),
        ),
    )

    bias = NoneBias(cvs=cv)

    # do yaff MD
    ener = AseEnergy(atoms=atoms, calculator=calc_cp2k)
    yaffmd = YaffEngine(
        ener=ener,
        bias=bias,
        write_step=1,
        T=300 * units.kelvin,
        timestep=1.0 * units.femtosecond,
        timecon_thermo=100.0 * units.femtosecond,
        timecon_baro=100.0 * units.femtosecond,
    )

    return yaffmd


def get_FES(name, engine: MDEngine, cvd: CVDiscovery, recalc=False) -> Scheme:
    """calculate some rounds, and perform long run. Starting point for cv discovery methods"""

    full_name = f"output/{name}"
    full_name_orig = f"output/{name}_orig"
    pe = os.path.exists(full_name)
    pe_orig = os.path.exists(full_name_orig)

    if recalc or (not pe and not pe_orig):
        if pe:
            shutil.rmtree(full_name)
        if pe_orig:
            shutil.rmtree(full_name_orig)

        scheme0 = Scheme(cvd=None, Engine=engine, folder=full_name)

        scheme0.round(rnds=3, steps=1e3, n=4)

        scheme0.rounds.run(NoneBias(scheme0.rounds.get_bias().cvs), steps=1e5)
        scheme0.rounds.save()

        del scheme0

    cleancopy(full_name)

    return Scheme.from_rounds(folder=full_name, cvd=cvd)
