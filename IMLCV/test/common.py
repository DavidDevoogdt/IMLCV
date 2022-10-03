import os
import shutil
from importlib import import_module

import ase
import ase.io
import ase.units
import jax.numpy as jnp
import numpy as np
from keras.api._v2 import keras as KerasAPI
from molmod import units
from molmod.units import angstrom, kelvin, kjmol

import yaff
from IMLCV import CP2K_COMMAND, ROOT_DIR
from IMLCV.base.bias import Cp2kEnergy, HarmonicBias, NoneBias, YaffEnergy
from IMLCV.base.CV import CV, SystemParams, Volume, cvflow, dihedral
from IMLCV.base.CVDiscovery import CVDiscovery
from IMLCV.base.MdEngine import MDEngine, StaticTrajectoryInfo, YaffEngine
from IMLCV.base.metric import Metric
from IMLCV.base.rounds import RoundsMd
from IMLCV.external.parsl_conf.config import config
from IMLCV.scheme import Scheme
from yaff.test.common import get_alaninedipeptide_amber99ff

keras: KerasAPI = import_module("tensorflow.keras")  # type: ignore


abspath = __file__
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

    tic = StaticTrajectoryInfo(
        T=T,
        timestep=2.0 * units.femtosecond,
        timecon_thermo=100.0 * units.femtosecond,
        write_step=10,
        atomic_numbers=np.array(
            [1, 6, 1, 1, 6, 8, 7, 1, 6, 1, 6, 1, 1, 1, 6, 8, 7, 1, 6, 1, 1, 1],
            dtype=int,
        ),
    )

    mde = YaffEngine(
        energy=YaffEnergy(f=get_alaninedipeptide_amber99ff),
        static_trajectory_info=tic,
        bias=NoneBias(cv0),
        trajectory_file="test.h5",
    )

    return mde


def mil53_yaff():

    T = 300 * units.kelvin
    P = 1 * units.atm

    def f():
        rd = ROOT_DIR / "IMLCV" / "test" / "data" / "MIL53"
        system = yaff.System.from_file(str(rd / "MIL53.chk"))
        ff = yaff.ForceField.generate(system, str(rd / "MIL53_pars.txt"))
        return ff

    cvs = CV(
        f=Volume,
        metric=Metric(
            periodicities=[False],
            bounding_box=jnp.array(
                [850, 1500],
            )
            * angstrom**3,
        ),
    )

    bias = HarmonicBias(
        cvs=cvs,
        q0=np.array(
            [1400 * angstrom],
        ),
        k=jnp.array(
            [5 * kjmol / angstrom**2],
        ),
    )

    energy = YaffEnergy(f=f)

    st = StaticTrajectoryInfo(
        T=T,
        P=P,
        timestep=1.0 * units.femtosecond,
        timecon_thermo=100.0 * units.femtosecond,
        timecon_baro=100.0 * units.femtosecond,
        write_step=10,
        atomic_numbers=energy.ff.system.numbers,
    )

    yaffmd = YaffEngine(
        energy=energy,
        bias=bias,
        static_trajectory_info=st,
    )

    return yaffmd


def ase_yaff():

    base = ROOT_DIR / "IMLCV" / "test/data/CsPbI_3"

    # make CP2K ase calculator
    path_atoms = base / "Pos.xyz"
    with open(path_atoms) as f:
        atoms = ase.io.read(f)

    path_source = base / "Libraries"

    path_potentials = path_source / "GTH_POTENTIALS"
    path_basis = path_source / "BASIS_SETS"
    path_dispersion = path_source / "dftd3.dat"

    input_params = {
        "PATH_DISPERSION": path_dispersion,
        "BASIS_SET_FILE_NAME": path_basis,
        "POTENTIAL_FILE_NAME": path_potentials,
    }

    energy = Cp2kEnergy(
        atoms=atoms,
        input_file=base / "cp2k.inp",
        input_kwargs=input_params,
        auto_write=True,
        basis_set=None,
        command=CP2K_COMMAND,
        cutoff=400 * ase.units.Rydberg,
        stress_tensor=True,
        print_level="LOW",
        pseudo_potential=None,
        max_scf=None,
        xc=None,
        basis_set_file=None,
        charge=None,
        potential_file=None,
        debug=False,
    )

    @cvflow
    def f(sp: SystemParams):
        import jax.numpy as jnp

        l = jnp.linalg.norm(sp.cell, axis=1)

        return jnp.array([(l[0] - l[1]) / 2, (l[0] + l[1]) / 2])

    cv = CV(
        f=f,
        metric=Metric(
            periodicities=[False, False],
            bounding_box=jnp.array([[0.0, 3.0], [6.0, 7.0]]) * angstrom,
        ),
    )

    print(f"{cv.metric.bounding_box}")

    bias = NoneBias(cvs=cv)

    tic = StaticTrajectoryInfo(
        write_step=1,
        T=300 * units.kelvin,
        P=1.0 * units.bar,
        timestep=2.0 * units.femtosecond,
        timecon_thermo=100.0 * units.femtosecond,
        timecon_baro=1000.0 * units.femtosecond,
        atomic_numbers=energy.atoms.get_atomic_numbers(),
        equilibration=0.0,
    )

    yaffmd = YaffEngine(
        energy=energy,
        bias=bias,
        static_trajectory_info=tic,
        sp=energy.sp,
    )

    return yaffmd


def get_FES(
    name,
    engine: MDEngine,
    cvd: CVDiscovery,
    recalc=False,
    steps=5e3,
    K=5 * kjmol,
) -> Scheme:
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

        scheme0.round(
            rnds=3,
            steps=steps,
            n=4,
            K=5 * kjmol,
        )

        scheme0.rounds.run(
            NoneBias(scheme0.rounds.get_bias().cvs),
            steps=1e5,
        )
        scheme0.rounds.save()

        del scheme0

    cleancopy(full_name)

    return Scheme.from_rounds(folder=full_name, cvd=cvd)


if __name__ == "__main__":

    config(cluster="doduo", max_blocks=10)

    # md = mil53_yaff()
    md = ase_yaff()
    # md = alanine_dipeptide_yaff()

    scheme = RoundsMd(folder="mdtest")
    scheme.new_round(md=md)

    scheme.run(steps=100, bias=None)

    # md.run(100)

    # print(md.get_trajectory().sp.shape)

    # md.trajectory_info.save("test.h5")
    # ti2 = TrajectoryInfo.load("test.h5")

    # print(ti2)
