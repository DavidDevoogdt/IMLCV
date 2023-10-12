from typing import Callable

import ase.io
import ase.units
import jax.numpy as jnp
import numpy as np
import yaff
from IMLCV.base.bias import Bias
from IMLCV.base.bias import NoneBias
from IMLCV.base.CV import CollectiveVariable
from IMLCV.base.CV import CV
from IMLCV.base.CV import CvMetric
from IMLCV.base.CV import NeighbourList
from IMLCV.base.CV import SystemParams
from IMLCV.base.MdEngine import StaticMdInfo
from IMLCV.configs.config_general import ROOT_DIR
from IMLCV.implementations.bias import HarmonicBias
from IMLCV.implementations.CV import dihedral
from IMLCV.implementations.CV import NoneCV
from IMLCV.implementations.CV import Volume
from IMLCV.implementations.energy import Cp2kEnergy
from IMLCV.implementations.energy import YaffEnergy
from IMLCV.implementations.MdEngine import YaffEngine
from molmod import units
from molmod.units import angstrom
from molmod.units import kelvin
from molmod.units import kjmol
from yaff.test.common import get_alaninedipeptide_amber99ff

yaff.log.set_level(yaff.log.silent)

DATA_ROOT = ROOT_DIR / "data"


def alanine_dipeptide_yaff(
    cv="backbone_dihedrals",
    bias: Callable[[CollectiveVariable], Bias] | None = None,
):
    T = 300 * kelvin

    tic = StaticMdInfo(
        T=T,
        timestep=2.0 * units.femtosecond,
        timecon_thermo=100.0 * units.femtosecond,
        write_step=200,
        atomic_numbers=jnp.array(
            [1, 6, 1, 1, 6, 8, 7, 1, 6, 1, 6, 1, 1, 1, 6, 8, 7, 1, 6, 1, 1, 1],
            dtype=int,
        ),
        screen_log=10,
        equilibration=0 * units.femtosecond,
        r_cut=None,
    )

    if cv == "backbone_dihedrals":
        cv0 = CollectiveVariable(
            f=(dihedral(numbers=[4, 6, 8, 14]) + dihedral(numbers=[6, 8, 14, 16])),
            metric=CvMetric(
                periodicities=[True, True],
                bounding_box=[[-np.pi, np.pi], [-np.pi, np.pi]],
            ),
        )

    elif cv is None:
        cv0 = NoneCV()
    else:
        raise ValueError(
            f"unknown value {cv} for cv choos 'soap_dist' or 'backbone_dihedrals'",
        )

    if bias is None:
        bias_cv0 = NoneBias(cvs=cv0)
    else:
        bias_cv0 = bias(cv0)

    mde = YaffEngine(
        energy=YaffEnergy(f=get_alaninedipeptide_amber99ff),
        static_trajectory_info=tic,
        bias=bias_cv0,
    )

    return mde


def alanine_dipeptide_refs():
    sp0 = SystemParams(
        coordinates=jnp.array(
            [
                [26.77741932, 35.69692667, 0.15117809],
                [26.90970015, 33.69381697, -0.30235618],
                [25.20894663, 32.76785116, 0.43463701],
                [28.66714545, 33.01351556, 0.51022606],
                [26.68293301, 33.05131008, -3.09915086],
                [25.9081453, 34.69537182, -4.55423998],
                [27.26874811, 30.7458442, -3.93063036],
                [28.21361118, 29.61200852, -2.72120563],
                [27.00418645, 30.06554279, -6.55734968],
                [25.19004937, 30.84033051, -7.14316479],
                [28.95060437, 31.21827573, -8.35258951],
                [30.78363872, 30.27341267, -8.25810321],
                [28.23250844, 31.12378943, -10.28011017],
                [29.36634412, 33.20248817, -7.95574702],
                [27.06087824, 27.19315907, -6.82191134],
                [28.38368653, 25.92704256, -5.40461674],
                [26.07822065, 26.26719326, -8.93840461],
                [25.37902198, 27.51441251, -10.22341838],
                [25.6624809, 23.64047394, -9.59980876],
                [25.83255625, 23.43260406, -11.64071298],
                [26.85300836, 22.29876838, -8.59825391],
                [23.73496024, 23.13024788, -9.07068544],
            ],
        ),
        cell=None,
    )

    sp1 = SystemParams(
        coordinates=jnp.array(
            [
                [23.931, 32.690, -5.643],
                [24.239, 31.818, -3.835],
                [22.314, 31.227, -3.153],
                [25.100, 33.275, -2.586],
                [25.835, 29.525, -3.858],
                [27.425, 29.164, -2.258],
                [25.638, 27.991, -5.861],
                [24.292, 28.473, -7.216],
                [27.221, 25.765, -6.438],
                [26.509, 24.957, -8.255],
                [29.991, 26.660, -6.699],
                [30.753, 27.301, -4.872],
                [30.920, 25.078, -7.447],
                [30.233, 28.236, -8.053],
                [26.856, 23.398, -4.858],
                [27.483, 21.402, -5.810],
                [25.732, 23.673, -2.608],
                [25.785, 25.535, -1.850],
                [25.227, 21.564, -0.916],
                [26.860, 20.494, -0.570],
                [24.444, 22.298, 0.859],
                [23.648, 20.454, -1.497],
            ],
        ),
        cell=None,
    )

    return sp0 + sp1


def mil53_yaff():
    T = 300 * units.kelvin
    P = 1 * units.atm

    def f():
        rd = ROOT_DIR / "data" / "MIL53"
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
        q0=CV(cv=jnp.array([3000 * angstrom**3])),
        k=jnp.array([0.1 * kjmol]),
    )

    bias = NoneBias(cvs=cvs)

    energy = YaffEnergy(f=f)

    st = StaticMdInfo(
        T=T,
        P=P,
        timestep=1.0 * units.femtosecond,
        timecon_thermo=100.0 * units.femtosecond,
        timecon_baro=200.0 * units.femtosecond,
        write_step=1,
        screen_log=1,
        atomic_numbers=energy.ff.system.numbers,
    )

    yaffmd = YaffEngine(
        energy=energy,
        bias=bias,
        static_trajectory_info=st,
    )

    return yaffmd


def CsPbI3(cv=None, unit_cells=[2]):
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
            f"provided unit cell {unit_cells}, please provide 1 or 3 arguments ",
        )

    fb = DATA_ROOT / "CsPbI_3" / f"{x}x{y}x{z}"

    path_source = DATA_ROOT / "CsPbI_3" / "Libraries"

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

    refs, z_array, atoms = CsPbI3_refs(x, y, z)

    energy = Cp2kEnergy(
        atoms=atoms[0],
        input_file=fb / "cp2k.inp",
        input_kwargs=input_params,
        stress_tensor=True,
        debug=False,
    )

    from IMLCV.base.CV import CvFlow

    r_cut = 5 * angstrom

    tic = StaticMdInfo(
        write_step=1,
        T=300 * units.kelvin,
        P=1.0 * units.bar,
        timestep=2.0 * units.femtosecond,
        timecon_thermo=100.0 * units.femtosecond,
        timecon_baro=500.0 * units.femtosecond,
        atomic_numbers=z_array,
        equilibration=0 * units.femtosecond,
        screen_log=1,
        r_cut=r_cut,
    )

    if cv == "cell_vec":

        @CvFlow.from_function
        def f(sp: SystemParams, _: NeighbourList | None):
            import jax.numpy as jnp

            assert sp.cell is not None

            sp = sp.minkowski_reduce()[0]

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

    elif cv is None:
        cv = NoneCV()
    else:
        raise ValueError(f"unknown value {cv} for cv choose 'cell_vec'")

    bias = NoneBias(cvs=cv)

    yaffmd = YaffEngine(
        energy=energy,
        bias=bias,
        static_trajectory_info=tic,
    )

    return yaffmd


def CsPbI3_refs(x, y, z, input_atoms=None):
    fb = DATA_ROOT / "CsPbI_3" / f"{x}x{y}x{z}"

    from ase import Atoms

    assert (p := fb).exists(), f"cannot find {p}"
    atoms: list[Atoms] = []

    if input_atoms is None:
        input_atoms = fb.glob("*.xyz")
    else:
        o = []
        for a in input_atoms:
            assert (p := fb / f"{a}.xyz").exists()
            o.append(p)
        input_atoms = o

    for a in input_atoms:
        atoms.append(ase.io.read(str(a)))

    assert len(atoms) != 0, "no xyz file found"
    z_arr = None
    refs: SystemParams | None = None

    for a in atoms:
        sp_a, _ = SystemParams(
            coordinates=a.positions * angstrom,
            cell=a.cell * angstrom,
        ).canonicalize()

        if z_arr is None:
            z_arr = a.get_atomic_numbers()

            refs = sp_a
        else:
            assert (z_arr == a.get_atomic_numbers()).all()

            refs += sp_a

    return refs, z_arr, atoms
