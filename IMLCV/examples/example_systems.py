import ase
import ase.io
import ase.units
import jax.numpy as jnp
import numpy as np

# from keras.api._v2 import keras as KerasAPI
from molmod import units
from molmod.units import angstrom, kelvin, kjmol

import yaff

yaff.log.set_level(yaff.log.silent)
from configs.config_general import ROOT_DIR
from IMLCV.base.bias import Cp2kEnergy, HarmonicBias, NoneBias, YaffEnergy
from IMLCV.base.CV import (
    CollectiveVariable,
    CvMetric,
    SystemParams,
    Volume,
    dihedral,
    sb_descriptor,
)
from IMLCV.base.MdEngine import StaticTrajectoryInfo, YaffEngine
from yaff.test.common import get_alaninedipeptide_amber99ff

# keras: KerasAPI = import_module("tensorflow.keras")  # type: ignore


def alanine_dipeptide_yaff(bias=lambda cv0: NoneBias(cvs=cv0), cv="backbone_dihedrals"):

    T = 300 * kelvin

    tic = StaticTrajectoryInfo(
        T=T,
        timestep=2.0 * units.femtosecond,
        timecon_thermo=100.0 * units.femtosecond,
        write_step=1,
        atomic_numbers=np.array(
            [1, 6, 1, 1, 6, 8, 7, 1, 6, 1, 6, 1, 1, 1, 6, 8, 7, 1, 6, 1, 1, 1],
            dtype=int,
        ),
        screen_log=1,
        equilibration=0 * units.femtosecond,
    )

    if cv == "backbone_dihedrals":

        cv0 = CollectiveVariable(
            f=(dihedral(numbers=[4, 6, 8, 14]) + dihedral(numbers=[6, 8, 14, 16])),
            metric=CvMetric(
                periodicities=[True, True],
                bounding_box=[[-np.pi, np.pi], [-np.pi, np.pi]],
            ),
        )

        r_cut = None

    elif cv == "soap_dist":

        sp0 = SystemParams(
            coordinates=jnp.array(
                [
                    [20.50227153, 27.10140184, -3.34817412],
                    [21.0461908, 28.73129116, -4.66992425],
                    [20.17228194, 30.57272672, -4.0893886],
                    [20.14818037, 28.16647569, -6.52168564],
                    [23.82704211, 28.83771713, -5.06690266],
                    [25.21294108, 29.12963401, -3.26870028],
                    [24.67461403, 28.29676979, -7.34257064],
                    [23.43931697, 27.73683338, -8.56029431],
                    [27.31917609, 28.48822462, -8.31024573],
                    [27.18293128, 28.07572215, -10.25881302],
                    [28.20667299, 31.17867019, -8.05517514],
                    [28.15270379, 31.34484467, -6.10398861],
                    [30.10577592, 31.41386418, -8.72179289],
                    [26.78317318, 32.42531988, -8.70454539],
                    [29.26880641, 26.51927963, -7.31688488],
                    [31.05035494, 26.03676003, -8.61645022],
                    [28.86765813, 25.66566859, -4.93866559],
                    [27.55284067, 26.2896754, -3.88117847],
                    [30.78413712, 24.1634122, -3.64298108],
                    [32.69302594, 25.03796984, -3.87937406],
                    [30.35952141, 24.21711956, -1.72405888],
                    [30.56663205, 22.31834421, -4.22471714],
                ]
            ),
            cell=None,
        )

        sp1 = SystemParams(
            coordinates=jnp.array(
                [
                    [24.69990139, 33.84927338, 5.42635469],
                    [26.53899516, 33.29837707, 4.70366138],
                    [26.98863002, 31.38799068, 5.36750364],
                    [28.07709282, 34.54155216, 5.17525499],
                    [26.63874039, 33.52967014, 1.80371664],
                    [25.22416562, 34.92847471, 0.66178309],
                    [28.29460859, 32.0235449, 0.70817748],
                    [29.42794161, 30.68295555, 1.62889435],
                    [28.43154657, 31.97814732, -2.11177037],
                    [26.56321828, 32.39407073, -2.96774345],
                    [30.42354434, 34.01280049, -2.91045157],
                    [32.40071882, 33.67656865, -2.29656292],
                    [30.49405861, 34.4244646, -4.96722918],
                    [30.12557626, 35.78451246, -1.60718247],
                    [28.9822668, 29.35940349, -3.21128742],
                    [30.54452352, 27.89097314, -2.30283724],
                    [27.67467624, 28.86964093, -5.343489],
                    [26.22595247, 30.16449899, -5.88131758],
                    [27.70562333, 26.61900678, -6.88377585],
                    [28.54046656, 25.03126151, -5.78970519],
                    [25.69007045, 26.11709574, -7.42231063],
                    [28.9082386, 26.94822982, -8.54864114],
                ]
            ),
            cell=None,
        )

        r_cut = 3 * angstrom

        cv0 = CollectiveVariable(
            f=sb_descriptor(
                r_cut=r_cut, sti=tic, n_max=5, l_max=5, references=sp0 + sp1
            ),
            metric=CvMetric(
                periodicities=[False, False],
                bounding_box=jnp.array([[0.0, 1.5], [0.0, 1.5]]),
            ),
        )

    else:
        raise ValueError("unknown value for cv")

    mde = YaffEngine(
        energy=YaffEnergy(f=get_alaninedipeptide_amber99ff),
        static_trajectory_info=tic,
        bias=bias(cv0),
        trajectory_file="test.h5",
        r_cut=r_cut,
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


def CsPbI3(unit_cells: list[int] = [1], cv="cell_vec"):

    base = ROOT_DIR / "IMLCV" / "examples" / "data" / "CsPbI_3"

    assert isinstance(unit_cells, list)

    if len(unit_cells) == 3:
        [x, y, z] = unit_cells
    elif len(unit_cells) == 1:
        [n] = unit_cells
        x = n
        y = n
        z = n
    else:
        raise ValueError(
            f"provided unit cell {unit_cells}, please provide 1 or 3 arguments "
        )
    fb = base / f"{x}x{y}x{z}"

    assert (p := fb).exists(), f"cannot find {p}"
    atoms = []

    for a in fb.glob("*.xyz"):
        atoms.append(ase.io.read(str(a)))
    assert len(atoms) != 0, "no xyz file found"

    path_source = base / "Libraries"

    assert (p := path_source).exists(), f"cannot find {p}"

    path_potentials = path_source / "GTH_POTENTIALS"
    path_basis = path_source / "BASIS_SETS"
    path_dispersion = path_source / "dftd3.dat"

    assert (p := fb / "cp2k.inp").exists(), f"cannot find {p}"

    for p in [path_potentials, path_basis, path_dispersion]:
        assert p.exists(), f"cannot find {p}"

    input_params = {
        "PATH_DISPERSION": path_dispersion,
        "BASIS_SET_FILE_NAME": path_basis,
        "POTENTIAL_FILE_NAME": path_potentials,
    }

    energy = Cp2kEnergy(
        atoms=atoms[0],
        input_file=fb / "cp2k.inp",
        input_kwargs=input_params,
        command=f"mpirun cp2k_shell.psmp",
        stress_tensor=True,
        debug=False,
    )

    from IMLCV.base.CV import CvFlow

    if cv == "cell_vec":

        @CvFlow.from_function
        def f(sp: SystemParams):

            import jax.numpy as jnp

            assert sp.cell is not None

            sp = sp.minkowski_reduce()

            l = jnp.linalg.norm(sp.cell, axis=1)
            l0 = jnp.max(l)
            l1 = jnp.min(l)

            return jnp.array([(l0 - l1) / 2, (l0 + l1) / 2])

        assert x == y
        assert x == z

        cv = CollectiveVariable(
            f=f,
            metric=CvMetric(
                periodicities=[False, False],
                bounding_box=jnp.array([[0.0, 3.0 * x], [5.5 * x, 8.0 * x]]) * angstrom,
            ),
        )
    else:
        raise NotImplementedError(f"cv {cv} unrecognized for CsPbI3")

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


if __name__ == "__main__":

    # sys = alanine_dipeptide_yaff(cv="backbone_dihedrals")
    # sys.run(100)

    sys = alanine_dipeptide_yaff(cv="soap_dist")
    sys.run(100)
