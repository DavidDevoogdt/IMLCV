import ase.io
import ase.units
import jax
import jax.numpy as jnp
import numpy as np
from openmm import System
from openmm.app import ForceField, PDBFile

from IMLCV.base.bias import Bias, BiasF, EnergyFn, NoneBias
from IMLCV.base.CV import CV, CollectiveVariable, CvMetric, CvTrans, NeighbourList, NeighbourListInfo, SystemParams
from IMLCV.base.MdEngine import StaticMdInfo
from IMLCV.base.UnitsConstants import angstrom, atm, bar, femtosecond, kelvin, kjmol
from IMLCV.configs.config_general import ROOT_DIR
from IMLCV.implementations.bias import HarmonicBias
from IMLCV.implementations.CV import (
    LatticeInvariants,
    NoneCV,
    Volume,
    dihedral,
    mean_trans,
    pair,
    quad,
    rotate_2d,
    trip,
)
from IMLCV.implementations.energy import MACEASE, Cp2kEnergy, OpenMmEnergy
from IMLCV.implementations.MdEngine import NewYaffEngine

DATA_ROOT = ROOT_DIR / "data"


def alanine_dipeptide_openmm(
    # cv: str | None = "backbone_dihedrals",
    cv_phi: bool = True,
    cv_psi: bool = True,
    cv_theta_1: bool = False,
    cv_theta_2: bool = False,
    bias: Bias | None = None,
    save_step=5,
):
    pdb = DATA_ROOT / "ala" / "alanine-dipeptide.pdb"

    assert pdb.exists(), f"cannot find {pdb}"

    topo = PDBFile(str(pdb)).topology

    forcefield = ForceField("amber14-all.xml")
    system: System = forcefield.createSystem(topo)

    energy = OpenMmEnergy(
        topology=topo,
        system=system,
    )

    _, atomic_numbers = energy.get_info()

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
                [30.233, 28.236, -8.053],  ## import required packages
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

    refs = sp0 + sp1

    f = None
    periodicities = []
    bounding_box = []
    names = []

    if cv_phi:
        f += dihedral(numbers=(4, 6, 8, 14))
        periodicities.append(True)
        bounding_box.append([-jnp.pi, jnp.pi])
        names.append("φ")

    if cv_psi:
        f += dihedral(numbers=(6, 8, 14, 16))
        periodicities.append(True)
        bounding_box.append([-jnp.pi, jnp.pi])
        names.append("ψ")

    if cv_theta_1:
        f += dihedral(numbers=(1, 4, 6, 8))
        periodicities.append(True)
        bounding_box.append([-jnp.pi, jnp.pi])
        names.append("θ1")

    if cv_theta_2:
        f += dihedral(numbers=(8, 14, 16, 18))
        periodicities.append(True)
        bounding_box.append([-jnp.pi, jnp.pi])
        names.append("θ2")

    if f is not None:
        cv0 = CollectiveVariable(
            f=f,
            metric=CvMetric.create(
                periodicities=periodicities,
                bounding_box=bounding_box,
            ),
            cvs_name=tuple(names),
        )
    else:
        cv0 = NoneCV()

    bias = NoneBias.create(collective_variable=cv0)

    tic = StaticMdInfo(
        T=300 * kelvin,
        timestep=2 * femtosecond,
        timecon_thermo=100.0 * femtosecond,
        write_step=1000,
        save_step=save_step,
        atomic_numbers=atomic_numbers,
        screen_log=1000,
        equilibration=0 * femtosecond,
        r_cut=None,
    )

    engine = NewYaffEngine(
        bias=bias,
        energy=energy,
        sp=refs[0],
        static_trajectory_info=tic,
    )

    return engine, refs


def butane(save_step=10, CV=True):
    import openmm.app as omm_app

    psf = omm_app.CharmmPsfFile(str(DATA_ROOT / "butane" / "butane.psf"))
    topo = omm_app.PDBFile(str(DATA_ROOT / "butane" / "butane.pdb")).topology

    params = omm_app.CharmmParameterSet(
        str(DATA_ROOT / "butane" / "top_all35_ethers.rtf"),
        str(DATA_ROOT / "butane" / "par_all35_ethers.prm"),
    )

    system: System = psf.createSystem(params, nonbondedMethod=omm_app.NoCutoff)
    # 3, 6, 9, 13

    energy = OpenMmEnergy(
        topology=topo,
        system=system,
    )

    from jax.numpy import float64

    sp_gauche_1 = SystemParams(
        coordinates=jnp.array(
            [
                [2.15703615, 1.56241502, 0.2855983],
                [1.98902398, -1.45477068, -1.04414907],
                [3.72412497, -0.72114134, 1.89025943],
                [3.35438037, -0.14439145, 0.02248188],
                [4.85064669, 0.5925671, -3.58561873],
                [6.63841114, 2.480373, -1.18370997],
                [5.7073375, 0.70564265, -1.63979844],
                [8.40386285, -1.80486496, -0.27301189],
                [9.02017102, -0.77540604, -3.37037211],
                [7.55912958, -1.29061105, -2.13670307],
                [5.05302663, -4.55737119, -1.78523772],
                [5.13546015, -3.19381872, -4.89337193],
                [7.77161377, -5.19325172, -3.67891667],
                [6.42227882, -3.666439, -3.19428947],
            ],
            dtype=float64,
        ),
        cell=None,
    )

    sp_gauche_2 = SystemParams(
        coordinates=jnp.array(
            [
                [-4.25627582, -0.80851557, -5.32877152],
                [-1.56220456, -1.21466246, -7.1658655],
                [-2.14104165, -3.52219106, -4.69780741],
                [-2.24645047, -1.46261627, -5.30020942],
                [-1.20312824, 1.88888633, -3.10670941],
                [-0.70056235, -1.25758393, -1.39051604],
                [-0.50457241, -0.05036129, -3.3143475],
                [3.23291176, 0.47419084, -2.03164796],
                [2.5795043, 1.59552406, -5.25495941],
                [2.26625037, -0.02161397, -3.8465963],
                [2.97659818, -4.02486853, -3.52053269],
                [3.06401843, -2.88287415, -6.59734388],
                [5.95752335, -2.44401742, -4.45481468],
                [3.81350026, -2.45176556, -4.59651993],
            ],
            dtype=float64,
        ),
        cell=None,
    )

    sp_chair = SystemParams(
        coordinates=jnp.array(
            [
                [0.31217485, 4.30747718, -2.35561618],
                [-3.0130301, 3.22509444, -2.44461085],
                [-2.0281344, 4.98140872, 0.25302459],
                [-1.35741616, 3.56421956, -1.16327537],
                [-2.18045573, 0.76684422, 1.59277161],
                [0.01511773, -0.1746373, -0.6448474],
                [-0.48480415, 1.38138251, 0.51353031],
                [3.37474654, 1.43746045, 1.93203223],
                [1.46398436, 4.09126858, 2.916611],
                [1.56343925, 2.03910965, 2.5431388],
                [0.62406391, -1.29840078, 4.78305274],
                [2.34578069, 0.95295519, 6.4473915],
                [-0.73534707, 1.54878882, 5.74220872],
                [0.86332793, 0.75572427, 4.99051189],
            ],
            dtype=float64,
        ),
        cell=None,
    )

    sps = sp_gauche_1 + sp_gauche_2 + sp_chair

    _, atomic_numbers = energy.get_info()

    if CV:
        cv0 = CollectiveVariable(
            f=dihedral(numbers=(3, 6, 9, 13)),
            metric=CvMetric.create(
                periodicities=[True],
                bounding_box=[
                    [-jnp.pi, jnp.pi],
                ],
            ),
        )
    else:
        cv0 = NoneCV()

    bias = NoneBias.create(collective_variable=cv0)

    tic = StaticMdInfo(
        T=300 * kelvin,
        timestep=2 * femtosecond,
        timecon_thermo=100.0 * femtosecond,
        write_step=1000,
        atomic_numbers=atomic_numbers,
        screen_log=1000,
        equilibration=0 * femtosecond,
        r_cut=None,
    )

    engine = NewYaffEngine(
        bias=bias,
        energy=energy,
        sp=sps[0],
        static_trajectory_info=tic,
        step=1,
    )

    return engine, sps


from functools import partial
from itertools import combinations, permutations

import jax
import jax.numpy as jnp
from jax import Array
from jaxopt.linear_solve import solve_normal_cg

from IMLCV.base.CV import (
    CV,
    CollectiveVariable,
    CvMetric,
    CvTrans,
    NeighbourList,
    NeighbourListInfo,
    SystemParams,
)
from IMLCV.base.datastructures import vmap_decorator
from IMLCV.base.UnitsConstants import angstrom


def _ring_distance(
    sp: SystemParams,
    nl,
    shmap,
    shmap_kwargs,
    idx: Array,
):
    # Extract coordinates of the atoms in the ring
    coords = sp.coordinates[idx, :]

    # Center the coordinates
    centroid = jnp.mean(coords, axis=0)
    coords_centered = coords - centroid

    # Compute covariance matrix and perform SVD
    _, s, vh = jnp.linalg.svd(coords_centered, full_matrices=False)

    # jax.debug.print("Singular values: {s}", s=s)

    normal = vh[-1]  # Normal vector to the best-fit plane

    # Compute signed distances to the plane
    distances = jnp.dot(coords_centered, normal)

    out = []
    # output some invariants
    # out.append( jnp.sum(distances   ) ) # should be zero

    # out.append( jnp.sum(distances**2)  )

    # roll up to 3 times, afterwards it is not independent anymore
    # out.append( jnp.sum(distances * jnp.roll(distances,1) ) )
    # out.append( jnp.sum(distances * jnp.roll(distances,2) ) )
    out.append(jnp.mean(distances * jnp.roll(distances, 3)))

    return CV(cv=jnp.array(out))


ring_distance = CvTrans.from_cv_function(_ring_distance, idx=jnp.array([0, 1, 2, 3, 4, 5]))


def cyclohexane(CV=True):
    from openff.toolkit.topology import Molecule
    from openmm.app import ForceField, PDBFile
    from openmmforcefields.generators import GAFFTemplateGenerator

    # Load your molecule (from SMILES or SDF, not PDB)
    mol = Molecule.from_smiles("C1CCCCC1")  # cyclohexane
    mol.generate_conformers(n_conformers=1)

    # Write to PDB if needed
    mol.to_file("cyclohexane.sdf", file_format="SDF")

    # Set up GAFF generator
    gaff = GAFFTemplateGenerator(molecules=mol)
    forcefield = ForceField("amber14/protein.ff14SB.xml", "amber14/tip3p.xml")
    forcefield.registerTemplateGenerator(gaff.generator)

    # Load topology from OpenFF molecule, not from PDB!
    top = mol.to_topology().to_openmm()
    system = forcefield.createSystem(top)
    energy = OpenMmEnergy(
        topology=top,
        system=system,
    )

    boat_at: ase.Atoms = ase.io.read(ROOT_DIR / "data" / "cyclohexane" / "boat.xyz")  # type: ignore
    chair_at: ase.Atoms = ase.io.read(ROOT_DIR / "data" / "cyclohexane" / "chair.xyz")  # type: ignore

    boat_arr = jnp.array(boat_at.arrays["positions"]) * angstrom
    chair_arr = jnp.array(chair_at.arrays["positions"]) * angstrom

    sp_chair_1 = SystemParams(
        coordinates=jnp.array(
            [
                [0.14340799, -5.5372373, 5.34681911],
                [0.9568797, -4.34122565, 2.8962363],
                [1.1149015, -1.45663735, 2.98579955],
                [2.86872198, -0.62206835, 5.10053147],
                [2.50200018, -1.97847043, 7.74353036],
                [2.0419397, -4.89994324, 7.50705148],
                [-1.48072997, -4.69531493, 6.078978],
                [0.06260481, -7.66620182, 5.32951227],
                [-0.32391459, -4.76528068, 1.26129106],
                [3.00467527, -4.83333736, 2.60268685],
                [1.93431338, -0.92145111, 1.04109729],
                [-0.51627553, -0.36884488, 3.45334254],
                [2.73536712, 1.44740621, 5.45218237],
                [4.78993623, -0.90174677, 4.24019788],
                [3.85158187, -1.60115884, 9.12383317],
                [0.77439239, -1.46164296, 8.56725176],
                [1.69298074, -5.71357011, 9.43240747],
                [3.82923522, -5.83319959, 7.30832802],
            ]
        ),
        cell=None,
    )
    sp_chair_2 = SystemParams(
        coordinates=jnp.array(
            [
                [5.35441023, -4.70819595, 6.41865159],
                [4.04211194, -3.64086112, 3.94300161],
                [2.09217629, -1.50039002, 4.73210238],
                [0.40473182, -2.24873483, 6.81806505],
                [1.69140691, -3.32278052, 9.207577],
                [3.49180702, -5.52927396, 8.28809331],
                [6.53003465, -6.24416939, 5.67732876],
                [6.76369033, -3.45010061, 7.24454657],
                [3.58352181, -4.80543457, 2.3507693],
                [5.5723133, -2.53781245, 3.03394903],
                [3.23103678, 0.03490877, 5.48694605],
                [1.04470013, -0.73265091, 3.20538319],
                [-0.71450764, -3.95829218, 6.13002214],
                [-0.9574964, -0.80094665, 7.3101736],
                [2.26373786, -2.09024162, 10.76916016],
                [0.03039064, -4.32678488, 10.0310324],
                [2.02216756, -6.71721661, 7.28297305],
                [4.24766686, -6.71865322, 9.73890222],
            ]
        ),
        cell=None,
    )
    sp_boat_1 = SystemParams(
        coordinates=jnp.array(
            [
                [6.23430978, 0.80105139, 0.48504902],
                [9.053529, 1.46816163, 0.28888063],
                [10.43357183, 0.86590955, 2.73076558],
                [8.71926481, 1.17977965, 5.14344244],
                [6.34182613, 3.11311203, 4.76937591],
                [4.87828094, 2.63150712, 2.19680632],
                [5.2382716, 0.94624833, -1.40137538],
                [6.01991953, -1.12545573, 1.20750946],
                [9.24198632, 3.45298957, -0.46347865],
                [10.00354669, 0.42621852, -1.33882473],
                [11.12829779, -1.10930555, 2.82581887],
                [12.23582477, 1.90998409, 2.95916797],
                [9.90363198, 1.35287764, 6.87138157],
                [7.91851494, -0.80643757, 5.52546991],
                [5.06244734, 3.43785814, 6.31866243],
                [7.14318756, 5.06192419, 4.59889832],
                [4.56742581, 4.4412369, 1.2562198],
                [2.95924271, 2.16135193, 2.55708493],
            ]
        ),
        cell=None,
    )

    sps = sp_chair_1 + sp_chair_2 + sp_boat_1
    # sps = sp_chair + sp_boat_chair_flip

    _, atomic_numbers = energy.get_info()

    cv0, _ = ring_distance.compute_cv(sps)

    b = jnp.array([[jnp.min(cv0.cv), jnp.max(cv0.cv)]])
    s = b[0, 1] - b[0, 0]
    b = b + jnp.array([-0.1 * s, 0.1 * s])

    # colvar = CollectiveVariable(
    #     f=ring_distance,
    #     metric=CvMetric.create(
    #         periodicities=[False],
    #         bounding_box=b,
    #     ),
    # )

    if not CV:
        colvar = NoneCV()
    else:
        colvar = CollectiveVariable(
            f=(dihedral(numbers=(0, 1, 2, 3)) + dihedral(numbers=(3, 4, 5, 0))) * rotate_2d(-jnp.pi / 4),
            metric=CvMetric.create(
                periodicities=[False, False],
                bounding_box=jnp.array(
                    [
                        [-jnp.pi / 3, jnp.pi / 3],
                        [-jnp.pi / 3, jnp.pi / 3],
                    ]
                )
                * jnp.sqrt(2),
            ),
            cvs_name=("$\\phi$", "$\\psi$"),
        )

    bias = NoneBias.create(collective_variable=colvar)

    tic = StaticMdInfo(
        T=300 * kelvin,
        timestep=2 * femtosecond,
        timecon_thermo=100.0 * femtosecond,
        write_step=500,
        atomic_numbers=atomic_numbers,
        screen_log=500,
        equilibration=0 * femtosecond,
        r_cut=None,
    )

    engine = NewYaffEngine(
        bias=bias,
        energy=energy,
        sp=sps[0],
        static_trajectory_info=tic,
        step=1,
    )

    return engine, sps


# def mil53_yaff():
#     T = 300 * kelvin
#     P = 1 * atm

#     import yaff

#     def f():
#         rd = ROOT_DIR / "data" / "MIL53"
#         system = yaff.System.from_file(str(rd / "MIL53.chk"))
#         ff = yaff.ForceField.generate(system, str(rd / "MIL53_pars.txt"))
#         return ff

#     cvs = CollectiveVariable(
#         f=Volume,
#         metric=CvMetric.create(
#             periodicities=[False],
#             bounding_box=jnp.array(
#                 [850, 1500],
#             )
#             * angstrom**3,
#         ),
#     )

#     bias = HarmonicBias.create(
#         cvs=cvs,
#         q0=CV(cv=jnp.array([3000 * angstrom**3])),
#         k=jnp.array([0.1 * kjmol]),
#     )

#     bias = NoneBias.create(collective_variable=cvs)

#     energy = YaffEnergy(f=f)

#     st = StaticMdInfo(
#         T=T,
#         P=P,
#         timestep=1.0 * femtosecond,
#         timecon_thermo=100.0 * femtosecond,
#         timecon_baro=200.0 * femtosecond,
#         write_step=1,
#         screen_log=1,
#         atomic_numbers=energy.ff.system.numbers,
#     )

#     yaffmd = NewYaffEngine.create(
#         energy=energy,
#         bias=bias,
#         static_trajectory_info=st,
#     )

#     return yaffmd


# def CsPbI3(cv=None, unit_cells=[2]):
#     assert isinstance(unit_cells, list)

#     if len(unit_cells) == 3:
#         [x, y, z] = unit_cells
#     elif len(unit_cells) == 1:
#         [n] = unit_cells
#         x = n
#         y = n
#         z = n
#     else:
#         raise ValueError(
#             f"provided unit cell {unit_cells}, please provide 1 or 3 arguments ",
#         )

#     fb = DATA_ROOT / "CsPbI_3" / f"{x}x{y}x{z}"

#     path_source = DATA_ROOT / "CsPbI_3" / "Libraries"

#     assert (p := path_source).exists(), f"cannot find {p}"

#     path_potentials = path_source / "GTH_POTENTIALS"
#     path_basis = path_source / "BASIS_SETS"
#     path_dispersion = path_source / "dftd3.dat"

#     assert (p := fb / "cp2k.inp").exists(), f"cannot find {p}"

#     for p in [path_potentials, path_basis, path_dispersion]:
#         assert p.exists(), f"cannot find {p}"

#     input_params = {
#         "PATH_DISPERSION": path_dispersion,
#         "BASIS_SET_FILE_NAME": path_basis,
#         "POTENTIAL_FILE_NAME": path_potentials,
#     }

#     refs, z_array, atoms = CsPbI3_refs(x, y, z)

#     energy = Cp2kEnergy(
#         atoms=atoms[0],
#         input_file=fb / "cp2k.inp",
#         input_kwargs=input_params,
#         stress_tensor=True,
#         debug=False,
#     )

#     from IMLCV.base.CV import CvTrans

#     r_cut = 5 * angstrom

#     tic = StaticMdInfo(
#         write_step=1,
#         T=300 * kelvin,
#         P=1.0 * bar,
#         timestep=2.0 * femtosecond,
#         timecon_thermo=100.0 * femtosecond,
#         timecon_baro=100.0 * femtosecond,
#         atomic_numbers=z_array,
#         equilibration=0 * femtosecond,
#         screen_log=1,
#         r_cut=r_cut,
#     )

#     if cv == "cell_vec":

#         @CvTrans.from_function
#         def f(sp: SystemParams, _: NeighbourList | None):
#             import jax.numpy as jnp

#             assert sp.cell is not None

#             sp = sp.minkowski_reduce()[0]

#             l = jnp.linalg.norm(sp.cell, axis=1)
#             l0 = jnp.max(l)
#             l1 = jnp.min(l)

#             return CV(cv=jnp.array([(l0 - l1) / 2, (l0 + l1) / 2]))

#         assert x == y
#         assert x == z

#         cv = CollectiveVariable(
#             f=f,
#             metric=CvMetric.create(
#                 periodicities=[False, False],
#                 bounding_box=jnp.array([[0.0, 3.0 * x], [5.5 * x, 8.0 * x]]) * angstrom,
#             ),
#         )

#     elif cv is None:
#         cv = NoneCV()
#     else:
#         raise ValueError(f"unknown value {cv} for cv choose 'cell_vec'")

#     bias = NoneBias.create(collective_variable=cv)

#     yaffmd = NewYaffEngine.create(
#         energy=energy,
#         bias=bias,
#         static_trajectory_info=tic,
#     )

#     return yaffmd


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

    yaffmd = NewYaffEngine.create(
        energy=energy,
        bias=bias,
        static_trajectory_info=tic,
    )

    return yaffmd


def CsPbI3_MACE_lattice(unit_cells=[2], save_step=1):
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
        save_step=save_step,
    )

    from IMLCV.implementations.CV import scale_cv_trans

    flow = LatticeInvariants

    cvs, _ = flow.compute_cv(SystemParams.stack(*refs))
    scale_trans = scale_cv_trans(cvs, lower=0.2, upper=0.8)

    colvar = CollectiveVariable(
        f=LatticeInvariants * scale_trans,
        metric=CvMetric.create(
            bounding_box=jnp.array(
                [
                    [0.0, 1.0],
                ]
                * 3
            )
        ),
    )

    bias = NoneBias.create(collective_variable=colvar)

    yaffmd = NewYaffEngine.create(
        energy=energy,
        bias=bias,
        static_trajectory_info=tic,
        sp=refs[0],
    )

    return yaffmd, refs


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
        atoms.append(ase.io.read(str(a)))  # type: ignore

    assert len(atoms) != 0, "no xyz file found"
    z_arr = None
    refs: SystemParams | None = None

    for a in atoms:
        sp_a, _ = SystemParams(
            coordinates=jnp.array(a.positions) * angstrom,
            cell=jnp.array(a.cell) * angstrom,
        ).canonicalize()

        if z_arr is None:
            z_arr = jnp.array(a.get_atomic_numbers())

            refs = sp_a
        else:
            assert refs is not None

            assert (z_arr == jnp.array(a.get_atomic_numbers())).all()

            refs += sp_a

    assert z_arr is not None, "z_array must be defined"
    assert refs is not None, "refs must be defined"

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


def _3d_muller_brown_cvs(sp: SystemParams, _nl, shmap, shmap_kwargs):
    # x1, x2, x3 = sp.coordinates[0, :]
    # x4, x5 = sp.coordinates[1, :2]

    # x = jnp.sqrt(x1**2 + x2**2 + 1e-7 * x5**2)
    # y = jnp.sqrt(x3**2 + x4**2)

    x = sp.coordinates[0, 0]
    y = sp.coordinates[0, 1]

    return CV(cv=jnp.array([x, y]))


def f_3d_Muller_Brown(sp: SystemParams, _):
    cvs = _3d_muller_brown_cvs(sp, _, _, _)

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
        f=CvTrans.from_cv_function(_3d_muller_brown_cvs),
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


def _toy_periodic_cvs(sp: SystemParams, _nl, shmap, shmap_kwargs):
    x = sp.volume()

    return CV(cv=jnp.array([x]))


def f_toy_periodic(sp: SystemParams, nl: NeighbourList, _nl0: NeighbourList):
    # sigma = 2 ** (-1 / 6) * (1.5 * angstrom)

    # print(f"inside f toy {sp=} {_nl0=}")

    r0 = 1.501 * angstrom

    def gauss_1d(r, r0, a=1.0):
        return jnp.exp(-a * ((r - r0) / (r0)) ** 2) * (jnp.sqrt(a) / (r0))

    def f(r_ij, _):
        r2 = jnp.sum(r_ij**2)

        r2_safe = jnp.where(r2 < 1e-10, 1e-10, r2)
        r = jnp.where(r2 > 1e-10, jnp.sqrt(r2_safe), 0.0)

        return -0.5 * kjmol * jnp.log(gauss_1d(r, r0, a=20.0) + gauss_1d(r, 2 * r0, a=20.0))

    _, ener_bond = _nl0.apply_fun_neighbour(sp, f, r_cut=jnp.inf, exclude_self=True)

    V = sp.volume()

    def gauss_3d(v, r0, a=1.0, offset=0.0):
        return jnp.exp(-a * ((v - r0**3) / (r0**3)) ** 2 + jnp.log(jnp.sqrt(a) / (r0**3)) - offset)

    def f_V(V):
        return (-5.0 * jnp.log(gauss_3d(V, 2 * r0, 2) + gauss_3d(V, 4 * r0, 16, -5.0)) - 24) * kjmol

    return jnp.sum(ener_bond) + f_V(V)


def toy_periodic_phase_trans():
    tic = StaticMdInfo(
        T=300 * kelvin,
        P=1 * atm,
        timestep=2.0 * femtosecond,
        timecon_thermo=100.0 * femtosecond,
        timecon_baro=200.0 * femtosecond,
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

    assert sp0.cell is not None, "cell must be defined for periodic systems"

    sp1 = SystemParams(
        coordinates=sp0.coordinates * 2,
        cell=sp0.cell * 2,
    )

    cv0 = CollectiveVariable(
        f=CvTrans.from_cv_function(_toy_periodic_cvs),
        metric=CvMetric.create(
            periodicities=[False],
            bounding_box=jnp.array(
                [
                    [0.1 * (2 * r0) ** 3, 1.5 * (4 * r0) ** 3],
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
        sp=sp1,
    )

    return mde, [sp0, sp1]
