import os
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
from IMLCV.base.CV import CollectiveVariable, CvMetric, SystemParams, Volume, dihedral
from IMLCV.base.MdEngine import StaticTrajectoryInfo, YaffEngine
from yaff.test.common import get_alaninedipeptide_amber99ff

keras: KerasAPI = import_module("tensorflow.keras")  # type: ignore


abspath = __file__
dname = os.path.dirname(abspath)
os.chdir(dname)


def alanine_dipeptide_yaff(bias=lambda cv0: NoneBias(cvs=cv0)):
    T = 300 * kelvin

    cv0 = CollectiveVariable(
        f=(dihedral(numbers=[4, 6, 8, 14]) + dihedral(numbers=[6, 8, 14, 16])),
        metric=CvMetric(
            periodicities=[True, True],
            bounding_box=[[-np.pi, np.pi], [-np.pi, np.pi]],
        ),
    )

    tic = StaticTrajectoryInfo(
        T=T,
        timestep=2.0 * units.femtosecond,
        timecon_thermo=100.0 * units.femtosecond,
        write_step=1000,
        atomic_numbers=np.array(
            [1, 6, 1, 1, 6, 8, 7, 1, 6, 1, 6, 1, 1, 1, 6, 8, 7, 1, 6, 1, 1, 1],
            dtype=int,
        ),
        screen_log=1000,
        equilibration=0 * units.femtosecond,
    )

    mde = YaffEngine(
        energy=YaffEnergy(f=get_alaninedipeptide_amber99ff),
        static_trajectory_info=tic,
        bias=bias(cv0),
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

    cvs = CollectiveVariable(
        f=Volume,
        metric=CvMetric(
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
            [1400 * angstrom**3],
        ),
        k=jnp.array(
            [10 * kjmol],
        ),
    )

    print(f"{1400 * angstrom**3}")

    energy = YaffEnergy(f=f)

    st = StaticTrajectoryInfo(
        T=T,
        P=P,
        timestep=1.0 * units.femtosecond,
        timecon_thermo=100.0 * units.femtosecond,
        timecon_baro=200.0 * units.femtosecond,
        write_step=1,
        atomic_numbers=energy.ff.system.numbers,
    )

    yaffmd = YaffEngine(
        energy=energy,
        bias=bias,
        static_trajectory_info=st,
    )

    return yaffmd


def ase_yaff(small=True):

    base = ROOT_DIR / "IMLCV" / "examples" / "data" / "CsPbI_3"

    fb = base / "small" if small else "large"

    path_atoms = fb / "Pos.xyz"

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
        input_file=fb / "cp2k.inp",
        input_kwargs=input_params,
        command=CP2K_COMMAND,
        stress_tensor=True,
        debug=False,
    )

    from IMLCV.base.CV import CvFlow

    @CvFlow.from_function
    def f(sp: SystemParams):

        import jax.numpy as jnp

        assert sp.cell is not None

        sp = sp.minkowski_reduce()

        l = jnp.linalg.norm(sp.cell, axis=1)
        l0 = jnp.max(l)
        l1 = jnp.min(l)

        return jnp.array([(l0 - l1) / 2, (l0 + l1) / 2])

    cv = CollectiveVariable(
        f=f,
        metric=CvMetric(
            periodicities=[False, False],
            bounding_box=jnp.array([[0.0, 2.0], [5.5, 7.5]]) * angstrom,
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
        timecon_baro=500.0 * units.femtosecond,
        atomic_numbers=energy.atoms.get_atomic_numbers(),
        equilibration=0 * units.femtosecond,
        screen_log=1,
    )

    yaffmd = YaffEngine(
        energy=energy,
        bias=bias,
        static_trajectory_info=tic,
    )

    return yaffmd
