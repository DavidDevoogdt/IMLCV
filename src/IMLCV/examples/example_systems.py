from typing import Callable

import ase.io
import ase.units
import jax
import jax.numpy as jnp
import numpy as np

from IMLCV.base.bias import Bias, BiasF, NoneBias, Energy, EnergyFn
from IMLCV.base.CV import CV, CollectiveVariable, CvMetric, NeighbourList, SystemParams, CvFlow
from IMLCV.base.MdEngine import StaticMdInfo
from IMLCV.base.UnitsConstants import angstrom, electronvolt, atm, bar, femtosecond, kelvin, kjmol, boltzmann
from IMLCV.configs.config_general import ROOT_DIR
from IMLCV.implementations.bias import HarmonicBias
from IMLCV.implementations.CV import NoneCV, Volume, dihedral, position_index
from IMLCV.implementations.energy import MACEASE, Cp2kEnergy, YaffEnergy
from IMLCV.implementations.MdEngine import AseEngine, YaffEngine, NewYaffEngine


DATA_ROOT = ROOT_DIR / "data"


def alanine_dipeptide_yaff(
    cv="backbone_dihedrals",
    bias: Callable[[CollectiveVariable], Bias] | None = None,
    r_cut=None,
):
    T = 300 * kelvin

    tic = StaticMdInfo(
        T=T,
        timestep=2.0 * femtosecond,
        timecon_thermo=100.0 * femtosecond,
        write_step=100,
        atomic_numbers=jnp.array(
            [1, 6, 1, 1, 6, 8, 7, 1, 6, 1, 6, 1, 1, 1, 6, 8, 7, 1, 6, 1, 1, 1],
            dtype=int,
        ),
        screen_log=100,
        equilibration=0 * femtosecond,
        r_cut=r_cut,
    )

    if cv == "backbone_dihedrals":
        cv0 = CollectiveVariable(
            f=(dihedral(numbers=(4, 6, 8, 14)) + dihedral(numbers=(6, 8, 14, 16))),
            metric=CvMetric.create(
                periodicities=[True, True],
                bounding_box=[[-np.pi, np.pi], [-np.pi, np.pi]],
            ),
        )
    elif cv == "backbone_dihedrals_theta":
        cv0 = CollectiveVariable(
            f=(dihedral(numbers=(4, 6, 8, 14)) + dihedral(numbers=(6, 8, 14, 16)) + dihedral(numbers=(1, 4, 6, 8))),
            metric=CvMetric.create(
                periodicities=[True, True, True],
                bounding_box=[[-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi]],
            ),
        )
    elif cv is None:
        cv0 = NoneCV()
    else:
        raise ValueError(
            f"unknown value {cv} for cv 'backbone_dihedrals'",
        )

    if bias is None:
        bias_cv0 = NoneBias.create(collective_variable=cv0)
    else:
        bias_cv0 = bias(cv0)

    print(bias_cv0)

    from yaff.test.common import get_alaninedipeptide_amber99ff

    mde = YaffEngine.create(
        energy=YaffEnergy(f=get_alaninedipeptide_amber99ff),
        static_trajectory_info=tic,
        bias=bias_cv0,
    )

    return mde


def alanine_dipeptide_ase(
    cv="backbone_dihedrals",
    bias: Callable[[CollectiveVariable], Bias] | None = None,
    r_cut=None,
):
    T = 300 * kelvin

    tic = StaticMdInfo(
        T=T,
        timestep=2.0 * femtosecond,
        timecon_thermo=100.0 * femtosecond,
        write_step=100,
        atomic_numbers=jnp.array(
            [1, 6, 1, 1, 6, 8, 7, 1, 6, 1, 6, 1, 1, 1, 6, 8, 7, 1, 6, 1, 1, 1],
            dtype=int,
        ),
        screen_log=100,
        equilibration=0 * femtosecond,
        r_cut=r_cut,
    )

    if cv == "backbone_dihedrals":
        cv0 = CollectiveVariable(
            f=(dihedral(numbers=(4, 6, 8, 14)) + dihedral(numbers=(6, 8, 14, 16))),
            metric=CvMetric.create(
                periodicities=[True, True],
                bounding_box=[[-np.pi, np.pi], [-np.pi, np.pi]],
            ),
        )
    elif cv == "backbone_dihedrals_theta":
        cv0 = CollectiveVariable(
            f=(dihedral(numbers=(4, 6, 8, 14)) + dihedral(numbers=(6, 8, 14, 16)) + dihedral(numbers=(1, 4, 6, 8))),
            metric=CvMetric.create(
                periodicities=[True, True, True],
                bounding_box=[[-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi]],
            ),
        )
    elif cv is None:
        cv0 = NoneCV()
    else:
        raise ValueError(
            f"unknown value {cv} for cv 'backbone_dihedrals'",
        )

    if bias is None:
        bias_cv0 = NoneBias.create(collective_variable=cv0)
    else:
        bias_cv0 = bias(cv0)

    print(bias_cv0)

    from yaff.test.common import get_alaninedipeptide_amber99ff

    mde = AseEngine.create(
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
    T = 300 * kelvin
    P = 1 * atm

    import yaff

    def f():
        rd = ROOT_DIR / "data" / "MIL53"
        system = yaff.System.from_file(str(rd / "MIL53.chk"))
        ff = yaff.ForceField.generate(system, str(rd / "MIL53_pars.txt"))
        return ff

    cvs = CollectiveVariable(
        f=Volume,
        metric=CvMetric.create(
            periodicities=[False],
            bounding_box=jnp.array(
                [850, 1500],
            )
            * angstrom**3,
        ),
    )

    bias = HarmonicBias.create(
        cvs=cvs,
        q0=CV(cv=jnp.array([3000 * angstrom**3])),
        k=jnp.array([0.1 * kjmol]),
    )

    bias = NoneBias.create(collective_variable=cvs)

    energy = YaffEnergy(f=f)

    st = StaticMdInfo(
        T=T,
        P=P,
        timestep=1.0 * femtosecond,
        timecon_thermo=100.0 * femtosecond,
        timecon_baro=200.0 * femtosecond,
        write_step=1,
        screen_log=1,
        atomic_numbers=energy.ff.system.numbers,
    )

    yaffmd = YaffEngine.create(
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
        T=300 * kelvin,
        P=1.0 * bar,
        timestep=2.0 * femtosecond,
        timecon_thermo=100.0 * femtosecond,
        timecon_baro=100.0 * femtosecond,
        atomic_numbers=z_array,
        equilibration=0 * femtosecond,
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

            return CV(cv=jnp.array([(l0 - l1) / 2, (l0 + l1) / 2]))

        assert x == y
        assert x == z

        cv = CollectiveVariable(
            f=f,
            metric=CvMetric.create(
                periodicities=[False, False],
                bounding_box=jnp.array([[0.0, 3.0 * x], [5.5 * x, 8.0 * x]]) * angstrom,
            ),
        )

    elif cv is None:
        cv = NoneCV()
    else:
        raise ValueError(f"unknown value {cv} for cv choose 'cell_vec'")

    bias = NoneBias.create(collective_variable=cv)

    yaffmd = YaffEngine.create(
        energy=energy,
        bias=bias,
        static_trajectory_info=tic,
    )

    return yaffmd


def CsPbI3_MACE(unit_cells=[2]):
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

    refs, z_array, atoms = CsPbI3_refs(x, y, z)

    energy = MACEASE(
        atoms=atoms[0],
    )

    r_cut = 6 * angstrom

    tic = StaticMdInfo(
        write_step=50,
        T=300 * kelvin,
        P=1.0 * bar,
        timestep=2.0 * femtosecond,
        timecon_thermo=100.0 * femtosecond,
        timecon_baro=500.0 * femtosecond,
        atomic_numbers=z_array,
        equilibration=0 * femtosecond,
        screen_log=50,
        r_cut=r_cut,
    )

    cv = NoneCV()
    bias = NoneBias.create(collective_variable=cv)

    yaffmd = YaffEngine.create(
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


def _ener_3d_muller_brown(cvs, *_):
    x, y = cvs.cv

    A_i = jnp.array([-200, -100, -170, 15])
    a = jnp.array([-1, -1, -6.5, 0.7])
    b = jnp.array([0, 0, 11, 0.6])
    c = jnp.array([-10, -10, -6.5, 0.7])
    _x = jnp.array([1, 0, -0.5, -1])
    _y = jnp.array([0, 0.5, 1.5, 1])

    def V_2d(x, y):
        return jnp.sum(A_i * jnp.exp((a * (x - _x) ** 2 + b * (x - _x) * (y - _y) + c * (y - _y) ** 2)))

    ener = V_2d(x, y)

    return ener.reshape(()) * kjmol


def _3d_muller_brown_cvs(sp: SystemParams, _nl, _c, shmap, shmap_kwargs):
    # x1, x2, x3 = sp.coordinates[0, :]
    # x4, x5 = sp.coordinates[1, :2]

    # x = jnp.sqrt(x1**2 + x2**2 + 1e-7 * x5**2)
    # y = jnp.sqrt(x3**2 + x4**2)

    x = sp.coordinates[0, 0]
    y = sp.coordinates[0, 1]

    return CV(cv=jnp.array([x, y]))


def f_3d_Muller_Brown(sp: SystemParams, _):
    cvs = _3d_muller_brown_cvs(sp, _, _, _, _)

    return _ener_3d_muller_brown(cvs)


def toy_1d():
    # 1 carbon

    tic = StaticMdInfo(
        T=300 * kelvin,
        timestep=2.0 * femtosecond,
        timecon_thermo=100.0 * femtosecond,
        write_step=100,
        atomic_numbers=jnp.array([6], dtype=int),
        screen_log=100,
        equilibration=0 * femtosecond,
        r_cut=None,
    )

    sp0 = SystemParams(
        coordinates=jnp.array([[-0.5, 1.5, 0]]),
        cell=None,
    )

    sp1 = SystemParams(
        coordinates=jnp.array([[0.6, 0.0, 0]]),
        cell=None,
    )

    cv0 = CollectiveVariable(
        f=CvFlow.from_function(_3d_muller_brown_cvs),
        metric=CvMetric.create(
            periodicities=[False, False],
            bounding_box=jnp.array([[-1.5, 1.0], [-0.5, 2.0]]),
        ),
    )

    fes_0 = BiasF.create(cvs=cv0, g=_ener_3d_muller_brown)

    bias_cv0 = NoneBias.create(collective_variable=cv0)

    energy = EnergyFn(
        f=f_3d_Muller_Brown,
    )

    # 1D

    mde = NewYaffEngine.create(
        energy=energy,
        static_trajectory_info=tic,
        bias=bias_cv0,
        sp=sp0,
    )

    return mde, [sp0, sp1], fes_0


def _toy_periodic_cvs(sp: SystemParams, _nl, _c, shmap, shmap_kwargs):
    x = sp.volume()

    return CV(cv=jnp.array([x]))


def f_toy_periodic(sp: SystemParams, nl: NeighbourList, _nl0: NeighbourList):
    # sigma = 2 ** (-1 / 6) * (1.5 * angstrom)

    r0 = 1.501 * angstrom

    def f(r_ij, _):
        r2 = jnp.sum(r_ij**2)

        r2_safe = jnp.where(r2 < 1e-10, 1e-10, r2)
        r = jnp.where(r2 > 1e-10, jnp.sqrt(r2_safe), 0.0)

        return -10 * kjmol * jnp.log(jnp.exp(-5 * (r - r0) ** 2) + jnp.exp(-5 * (r - 2 * r0) ** 2))

    _, ener_bond = _nl0.apply_fun_neighbour(sp, f, r_cut=jnp.inf, exclude_self=True)

    V = sp.volume()

    E_vol = (
        -10.0 * kjmol * jnp.log(jnp.exp(-0.001 * (V - (2 * r0) ** 3) ** 2) + jnp.exp(-0.001 * (V - (4 * r0) ** 3) ** 2))
    )

    # print(f"V: {V}, E_vol: {E_vol}")

    return jnp.sum(ener_bond) + E_vol + 0.1 * kjmol


def toy_periodic_phase_trans():
    tic = StaticMdInfo(
        T=300 * kelvin,
        P=1 * atm,
        timestep=2.0 * femtosecond,
        timecon_thermo=100.0 * femtosecond,
        timecon_baro=500.0 * femtosecond,
        write_step=100,
        atomic_numbers=jnp.array(
            [6] * 8,
            dtype=int,
        ),
        screen_log=100,
        equilibration=0 * femtosecond,
        r_cut=None,
    )

    # from jax import Array
    from jax.numpy import float64

    Array = jax.numpy.array

    r0 = 1.5 * angstrom

    sp0 = SystemParams(
        coordinates=jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        )
        * r0,
        cell=jnp.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        * 2
        * r0,
    )

    from IMLCV.base.CV import NeighbourListInfo

    _nl0 = sp0.get_neighbour_list(
        info=NeighbourListInfo.create(
            r_cut=r0 * 1.1,
            z_array=tic.atomic_numbers,
        )
    )

    sp1 = SystemParams(
        coordinates=sp0.coordinates * 2,
        cell=sp0.cell * 2,
    )

    cv0 = CollectiveVariable(
        f=CvFlow.from_function(_toy_periodic_cvs),
        metric=CvMetric.create(
            periodicities=[False],
            bounding_box=jnp.array(
                [
                    [0.8 * (2 * r0) ** 3, 1.2 * (4 * r0) ** 3],
                ]
            ),
        ),
    )

    bias_cv0 = NoneBias.create(collective_variable=cv0)

    energy = EnergyFn(f=f_toy_periodic, kwargs={"_nl0": _nl0})

    # 1D

    mde = NewYaffEngine.create(
        energy=energy,
        static_trajectory_info=tic,
        bias=bias_cv0,
        sp=sp0,
    )

    return mde, [sp0, sp1]
