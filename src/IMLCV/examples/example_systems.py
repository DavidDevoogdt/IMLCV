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
from IMLCV.base.MdEngine import StaticTrajectoryInfo
from IMLCV.configs.config_general import ROOT_DIR
from IMLCV.implementations.bias import HarmonicBias
from IMLCV.implementations.CV import dihedral
from IMLCV.implementations.CV import NoneCV
from IMLCV.implementations.CV import project_distances
from IMLCV.implementations.CV import sb_descriptor
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


def alanine_dipeptide_yaff(
    bias: None | Callable[[CollectiveVariable], Bias] = None,
    cv="backbone_dihedrals",
    k=5 * kjmol,
    project=True,
    lda_steps=500,
    num_kfda=1,
    kernel=True,
    harmonic=True,
    folder=None,
    kernel_type=None,
    alpha_rematch=1e-1,
):
    T = 300 * kelvin

    if cv == "backbone_dihedrals":
        r_cut = None
    elif cv == "soap_dist" or cv == "soap_lda":
        r_cut = 3.0 * angstrom
    else:
        r_cut = None

    tic = StaticTrajectoryInfo(
        T=T,
        timestep=2.0 * units.femtosecond,
        timecon_thermo=100.0 * units.femtosecond,
        write_step=10,
        atomic_numbers=jnp.array(
            [1, 6, 1, 1, 6, 8, 7, 1, 6, 1, 6, 1, 1, 1, 6, 8, 7, 1, 6, 1, 1, 1],
            dtype=int,
        ),
        screen_log=10,
        equilibration=0 * units.femtosecond,
        r_cut=r_cut,
        # max_grad=max_grad,
    )

    if cv == "backbone_dihedrals":
        cv0 = CollectiveVariable(
            f=(dihedral(numbers=[4, 6, 8, 14]) + dihedral(numbers=[6, 8, 14, 16])),
            metric=CvMetric(
                periodicities=[True, True],
                bounding_box=[[-np.pi, np.pi], [-np.pi, np.pi]],
            ),
        )

        # sp = None

    elif cv == "soap_dist" or cv == "soap_lda":
        raise NotImplementedError("todo couple CV discovery")

    else:
        raise ValueError(
            f"unknown value {cv} for cv choos 'soap_dist' or 'backbone_dihedrals'",
        )

    if bias is None:
        bias = NoneBias(cvs=cv0)
    elif bias == "harm":
        bias = HarmonicBias(cvs=cv0, q0=CV(cv=jnp.array([1.0])), k=k)
    elif isinstance(bias, Callable):
        bias = bias(cv0)
    else:
        raise ValueError

    mde = YaffEngine(
        energy=YaffEnergy(f=get_alaninedipeptide_amber99ff),
        static_trajectory_info=tic,
        bias=bias,
    )

    return mde


def mil53_yaff():
    T = 300 * units.kelvin
    P = 1 * units.atm

    def f():
        rd = ROOT_DIR / "IMLCV" / "data" / "MIL53"
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

    st = StaticTrajectoryInfo(
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


def CsPbI3(cv, unit_cells, folder=None, input_atoms=None, project=True, lda_steps=500):
    base = ROOT_DIR / "IMLCV" / "data" / "CsPbI_3"

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
    fb = base / f"{x}x{y}x{z}"

    from ase import Atoms

    assert (p := fb).exists(), f"cannot find {p}"
    atoms: list[Atoms] = []

    if input_atoms is None:
        input_atoms = fb.glob("*.xyz")
    else:
        input_atoms = [fb / x for x in input_atoms]

    for a in input_atoms:
        atoms.append(ase.io.read(str(a)))

    print(f"{atoms=}")

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
        stress_tensor=True,
        debug=False,
    )

    from IMLCV.base.CV import CvFlow

    if cv == "cell_vec":
        r_cut = None
    else:
        r_cut = 5 * angstrom

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
    elif cv == "soap_dist":
        r_cut = 5 * angstrom

        z_arr = None
        refs: SystemParams | None = None

        for a in atoms:
            if z_arr is None:
                z_arr = a.get_atomic_numbers()
                refs = SystemParams(
                    coordinates=a.positions * angstrom,
                    cell=a.cell * angstrom,
                )

            else:
                assert (z_arr == a.get_atomic_numbers()).all()
                refs += SystemParams(
                    coordinates=a.positions * angstrom,
                    cell=a.cell * angstrom,
                )

        assert refs.batched is True

        refs_nl = refs.get_neighbour_list(r_cut=r_cut, z_array=jnp.array(z_arr))

        sbd = sb_descriptor(
            r_cut=r_cut,
            n_max=3,
            l_max=3,
            references=refs,
            references_nl=refs_nl,
        )

        cv_ref = sbd.compute_cv_flow(refs, refs_nl)

        if project:
            assert refs.shape[0] == 2, "option --project needs 2D CV (2 input_atoms)"
            cv_ref = sbd.compute_cv_flow(refs, refs_nl)

            a = float(cv_ref.cv[0, 1])
            cv = CollectiveVariable(
                f=sbd * project_distances(a),
                metric=CvMetric(
                    periodicities=[False, False],
                    bounding_box=jnp.array([[-0.1, 1.1], [0.0, 1.0]]),
                ),
            )

        else:
            raise "use --project"

        o = cv.compute_cv(refs, refs_nl)[0].cv
        assert jnp.allclose(jnp.array([[1.0, 0.0], [0.0, 0.0]]) - o, 0.0)

    elif cv == "soap_lda":
        # r_cut = 5 * angstrom

        # z_arr = None
        # refs: SystemParams | None = None

        # for a in atoms:
        #     if z_arr is None:
        #         z_arr = a.get_atomic_numbers()
        #         refs = SystemParams(
        #             coordinates=a.positions * angstrom,
        #             cell=a.cell * angstrom,
        #         )

        #     else:
        #         assert (z_arr == a.get_atomic_numbers()).all()
        #         refs += SystemParams(
        #             coordinates=a.positions * angstrom,
        #             cell=a.cell * angstrom,
        #         )

        # assert refs.batched is True

        # bias = NoneBias(cvs=NoneCV())
        # tic = StaticTrajectoryInfo(
        #     write_step=1,
        #     T=300 * units.kelvin,
        #     P=1.0 * units.bar,
        #     timestep=2.0 * units.femtosecond,
        #     timecon_thermo=100.0 * units.femtosecond,
        #     timecon_baro=500.0 * units.femtosecond,
        #     atomic_numbers=jnp.array(energy.atoms.get_atomic_numbers()),
        #     equilibration=0 * units.femtosecond,
        #     screen_log=1,
        #     r_cut=r_cut,
        # )

        # yaffmd = YaffEngine(
        #     energy=energy,
        #     bias=bias,
        #     static_trajectory_info=tic,
        # )

        # sbd = sb_descriptor(
        #     r_cut=r_cut,
        #     n_max=3,
        #     l_max=3,
        # )

        raise NotImplementedError("todo")

    else:
        raise NotImplementedError(f"cv {cv} unrecognized for CsPbI3,")

    bias = NoneBias(cvs=cv)

    yaffmd = YaffEngine(
        energy=energy,
        bias=bias,
        static_trajectory_info=tic,
    )

    return yaffmd
