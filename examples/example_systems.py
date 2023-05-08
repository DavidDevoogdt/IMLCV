import ase
import ase.io
import ase.units
import jax.numpy as jnp
import numpy as np
import yaff
from molmod import units
from molmod.units import angstrom, kelvin, kjmol

from IMLCV.base.rounds import Rounds

yaff.log.set_level(yaff.log.silent)

from typing import Callable

from yaff.test.common import get_alaninedipeptide_amber99ff

from configs.config_general import ROOT_DIR
from IMLCV.base.bias import Bias, NoneBias
from IMLCV.base.CV import CV, CollectiveVariable, CvMetric, NeighbourList, SystemParams
from IMLCV.base.MdEngine import StaticTrajectoryInfo
from IMLCV.implementations.bias import HarmonicBias
from IMLCV.implementations.CV import (
    NoneCV,
    Volume,
    dihedral,
    get_lda_cv,
    project_distances,
    sb_descriptor,
)
from IMLCV.implementations.energy import Cp2kEnergy, YaffEnergy
from IMLCV.implementations.MdEngine import YaffEngine


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

        sp = None

    elif cv == "soap_dist" or cv == "soap_lda":
        # ref pos
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
                ]
            ),
            cell=None,
        )

        sp1_a = jnp.array(
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
            ]
        )

        sp1_b = jnp.array(
            [
                [35.704, 38.625, 1.7931],
                [33.729, 38.815, 1.0493],
                [33.579, 40.608, 0.0085435],
                [32.332, 38.65, 2.5276],
                [33.25, 36.637, -0.72336],
                [33.164, 34.435, 0.1553],
                [32.915, 37.228, -3.1684],
                [32.659, 39.045, -3.7444],
                [32.223, 35.498, -5.2083],
                [31.92, 36.566, -6.9957],
                [34.786, 34.127, -5.7689],
                [35.467, 32.9, -4.2401],
                [34.518, 32.891, -7.4366],
                [36.197, 35.553, -6.162],
                [29.809, 33.938, -5.0311],
                [28.183, 34.313, -6.6841],
                [29.38, 32.502, -3.0923],
                [30.957, 32.54, -2.0214],
                [27.446, 30.501, -2.9781],
                [26.278, 30.575, -4.7043],
                [28.288, 28.613, -2.9197],
                [26.157, 30.858, -1.3152],
            ]
        )

        sp1 = SystemParams(
            coordinates=sp1_a,
            cell=None,
        )

        refs = sp0 + sp1

        mde = YaffEngine(
            energy=YaffEnergy(f=get_alaninedipeptide_amber99ff),
            static_trajectory_info=tic,
            bias=NoneBias(cvs=NoneCV()),
        )

        sb = sb_descriptor(r_cut=r_cut, n_max=2, l_max=2)

        needs_sim = True

        if folder.exists():
            pre_round = Rounds(folder=folder, new_folder=False)
            needs_sim = pre_round.round != -1
        else:
            pre_round = Rounds(folder=folder)
        if needs_sim:
            pre_round.add_round_from_md(mde)

            biases = []
            for _ in refs:
                biases.append(NoneBias(cvs=NoneCV()))

            pre_round.run_par(biases=biases, steps=lda_steps, sp0=refs)

        cv0 = get_lda_cv(
            num_kfda,
            folder=folder,
            descriptor=sb,
            kernel=kernel,
            harmonic=harmonic,
            r_cut=r_cut,
            z_array=tic.atomic_numbers,
            kernel_type=kernel_type,
            execution_folder=folder.parent if folder is not None else None,
            alpha_rematch=alpha_rematch,
        )

    else:
        raise ValueError(
            f"unknown value {cv} for cv choos 'soap_dist' or 'backbone_dihedrals'"
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
        rd = ROOT_DIR / "IMLCV" / "examples" / "data" / "MIL53"
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
                    coordinates=a.positions * angstrom, cell=a.cell * angstrom
                )

            else:
                assert (z_arr == a.get_atomic_numbers()).all()
                refs += SystemParams(
                    coordinates=a.positions * angstrom, cell=a.cell * angstrom
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
        r_cut = 5 * angstrom

        z_arr = None
        refs: SystemParams | None = None

        for a in atoms:
            if z_arr is None:
                z_arr = a.get_atomic_numbers()
                refs = SystemParams(
                    coordinates=a.positions * angstrom, cell=a.cell * angstrom
                )

            else:
                assert (z_arr == a.get_atomic_numbers()).all()
                refs += SystemParams(
                    coordinates=a.positions * angstrom, cell=a.cell * angstrom
                )

        assert refs.batched is True

        bias = NoneBias(cvs=NoneCV())
        tic = StaticTrajectoryInfo(
            write_step=1,
            T=300 * units.kelvin,
            P=1.0 * units.bar,
            timestep=2.0 * units.femtosecond,
            timecon_thermo=100.0 * units.femtosecond,
            timecon_baro=500.0 * units.femtosecond,
            atomic_numbers=jnp.array(energy.atoms.get_atomic_numbers()),
            equilibration=0 * units.femtosecond,
            screen_log=1,
            r_cut=r_cut,
        )

        yaffmd = YaffEngine(
            energy=energy,
            bias=bias,
            static_trajectory_info=tic,
        )

        sbd = sb_descriptor(
            r_cut=r_cut,
            n_max=3,
            l_max=3,
        )

        cv = get_lda_cv(
            1,
            folder=folder,
            mde=yaffmd,
            descriptor=sbd,
            n_steps=lda_steps,
            r_cut=tic.r_cut,
            z_array=tic.atomic_numbers,
            execution_folder=folder,
        )

    else:
        raise NotImplementedError(f"cv {cv} unrecognized for CsPbI3,")

    bias = NoneBias(cvs=cv)

    yaffmd = YaffEngine(
        energy=energy,
        bias=bias,
        static_trajectory_info=tic,
    )

    return yaffmd
