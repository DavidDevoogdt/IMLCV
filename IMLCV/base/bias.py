from __future__ import annotations

import os
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Iterable
from functools import partial
from pathlib import Path
from typing import Callable

import ase
import ase.calculators.calculator
import ase.cell
import ase.geometry
import ase.stress
import ase.units
import dill
import jax.numpy as jnp
import jax.scipy as jsp
import jax_dataclasses
import matplotlib.pyplot as plt
import numpy as np
from ase.calculators.cp2k import CP2K
from jax import jacrev, jit, value_and_grad, vmap
from molmod.constants import boltzmann
from molmod.units import angstrom, electronvolt, kjmol
from parsl.data_provider.files import File
from scipy.interpolate import RBFInterpolator

import yaff
from IMLCV import ROOT_DIR
from IMLCV.base.CV import CV, SystemParams
from IMLCV.base.tools import HashableArrayWrapper
from IMLCV.external.parsl_conf.bash_app_python import bash_app_python


@jax_dataclasses.pytree_dataclass
class EnergyResult:
    energy: float
    gpos: jnp.ndarray | None = None
    vtens: jnp.ndarray | None = None

    def __post_init__(self):
        if isinstance(self.gpos, np.ndarray):
            self.__dict__["gpos"] = jnp.array(self.gpos)
        if isinstance(self.vtens, np.ndarray):
            self.__dict__["vtens"] = jnp.array(self.vtens)

    def __add__(self, other) -> EnergyResult:
        assert isinstance(other, EnergyResult)

        gpos = self.gpos
        if self.gpos is None:
            assert other.gpos is None
        else:
            assert other.gpos is not None
            gpos += other.gpos

        vtens = self.vtens
        if self.vtens is None:
            assert other.vtens is None
        else:
            assert other.vtens is not None
            vtens += other.vtens

        return EnergyResult(energy=self.energy + other.energy, gpos=gpos, vtens=vtens)


class BC:
    """base class for biased Energy of MD simulation."""

    def __init__(self) -> None:
        pass

    def compute_coor(self, sp: SystemParams, gpos=False, vir=False) -> EnergyResult:
        """Computes the bias, the gradient of the bias wrt the coordinates and
        the virial."""
        raise NotImplementedError

    def save(self, filename: str | Path):
        if isinstance(filename, str):
            filename = Path(filename)
        if not filename.parent.exists():
            filename.parent.mkdir(parents=True, exist_ok=True)

        with open(filename, "wb") as f:
            dill.dump(self, f)

    @staticmethod
    def load(filename) -> BC:
        with open(filename, "rb") as f:
            self = dill.load(f)
        return self


class EnergyError(Exception):
    pass


class Energy(BC):
    @staticmethod
    def load(filename) -> Energy:
        energy = BC.load(filename=filename)
        assert isinstance(energy, Energy)
        return energy

    @property
    @abstractmethod
    def sp(self):
        pass

    @sp.setter
    @abstractmethod
    def sp(self):
        pass

    @abstractmethod
    def _compute_coor(self, gpos=False, vir=False) -> EnergyResult:
        pass

    def _handle_exception(self):
        return ""

    def compute_coor(self, sp: SystemParams, gpos=False, vir=False) -> EnergyResult:

        self.sp = sp

        try:
            return self._compute_coor(gpos=gpos, vir=vir)
        except EnergyError as be:
            raise EnergyError(
                f"""An error occured during the nergy calculation with {self.__class__}.
The lates coordinates were {self.sp}.                  
raised exception from calculator:{be}"""
            )


class YaffEnergy(Energy):
    def __init__(self, f: Callable[[], yaff.ForceField]) -> None:
        super().__init__()
        self.f = f
        self.ff: yaff.ForceField = f()

    @property
    def sp(self):
        return SystemParams(
            coordinates=jnp.array(self.system.pos[:]),
            cell=jnp.array(self.system.cell.rvecs[:]),
        )

    @sp.setter
    def sp(self, sp: SystemParams):
        self.ff.update_pos(np.array(sp.coordinates, dtype=np.double))
        self.ff.update_rvecs(np.array(sp.cell, dtype=np.double))

    def _compute_coor(self, gpos=False, vir=False) -> EnergyResult:

        gpos_out = np.zeros_like(self.ff.gpos) if gpos else None
        vtens_out = np.zeros_like(self.ff.vtens) if vir else None

        try:
            ener = self.ff.compute(gpos=gpos_out, vtens=vtens_out)
        except BaseException as be:
            raise EnergyError(f"calculating yaff  energy raise execption:\n{be}")

        return EnergyResult(ener, gpos_out, vtens_out)

    def __getstate__(self):
        return {"f": self.f}

    def __setstate__(self, state):

        self.f = state["f"]
        self.ff = self.f()
        return self


class AseError(EnergyError):
    pass


class AseEnergy(Energy):
    """Conversion to ASE energy"""

    def __init__(
        self,
        atoms: ase.Atoms,
        calculator: ase.calculators.calculator.Calculator | None = None,
    ):

        self.atoms = atoms

        if calculator is not None:
            self.atoms.calc = self.calculator

    @property
    def sp(self):
        return SystemParams(
            coordinates=jnp.array(self.atoms.get_positions()) * angstrom,
            cell=jnp.array(self.atoms.get_cell().array[:]) * angstrom,
        )

    @sp.setter
    def sp(self, sp: SystemParams):
        self.atoms.set_positions(np.array(sp.coordinates) / angstrom)
        self.atoms.set_cell(ase.geometry.Cell(np.array(sp.cell) / angstrom))

    def _compute_coor(self, gpos=False, vir=False) -> EnergyResult:
        """use unit conventions of ASE"""

        if self.atoms.calc is None:
            self.atoms.calc = self._calculator()

        try:
            energy = self.atoms.get_potential_energy() * electronvolt
        except:
            self._handle_exception()

        gpos_out = None
        vtens_out = None
        if gpos:
            forces = self.atoms.get_forces()
            gpos_out = -forces * electronvolt / angstrom

        if vir:
            cell = self.atoms.get_cell().array
            volume = np.linalg.det(cell)
            stress = self.atoms.get_stress(voigt=False)
            vtens_out = volume * stress * electronvolt

        res = EnergyResult(energy, gpos_out, vtens_out)

        return res

    def _calculator(self):
        raise NotImplementedError

    def _handle_exception(self):
        raise AseError

    def __getstate__(self):

        extra_args = {
            "label": self.calculator.label,
        }

        dict = {
            "cc": self.calculator.__class__,
            "calc_args": {**self.calculator.todict(), **extra_args},
            "atoms": self.atoms.todict(),
        }

        return dict

    def __setstate__(self, state):
        clss = state["cc"]
        calc_params = state["calc_args"]
        atom_params = state["atoms"]

        self.atoms = ase.Atoms.fromdict(**atom_params)
        self.calculator = clss(**calc_params)
        self.atoms.calc = self.calculator


class Cp2kEnergy(AseEnergy):
    default_parameters = dict(
        auto_write=False,
        basis_set=None,
        basis_set_file=None,
        charge=0,
        cutoff=400 * ase.units.Rydberg,
        force_eval_method="Quickstep",
        inp="",
        max_scf=50,
        potential_file=None,
        pseudo_potential="auto",
        stress_tensor=True,
        uks=False,
        poisson_solver="auto",
        xc="LDA",
        print_level="LOW",
    )

    def __init__(self, atoms: ase.Atoms, input_file, input_kwargs: dict, **kwargs):

        self.atoms = atoms
        self.cp2k_inp = os.path.abspath(input_file)
        self.input_kwargs = input_kwargs
        self.kwargs = kwargs
        super().__init__(atoms)

        rp = Path(ROOT_DIR) / "IMLCV" / ".ase_calculators" / "cp2k"
        rp.mkdir(parents=True, exist_ok=True)
        # if not os.path.exists(rp):
        #     os.makedirs(rp, exist_ok=True)

    def _calculator(self):

        new_dict = {}
        for key, val in self.input_kwargs.items():
            if Path(val).exists():
                new_dict[key] = os.path.relpath(Path(val))
            else:
                new_dict[key] = val

        with open(self.cp2k_inp) as f:
            inp = "".join(f.readlines()).format(**new_dict)
        params = self.default_parameters.copy()
        params.update(**{"inp": inp, **self.kwargs})

        if "label" in params:
            del params["label"]
            print("ignoring label for Cp2kEnergy")
        if "directory" in params:
            del params["directory"]
            print("ignoring directory for Cp2kEnergy")

        rp = Path(ROOT_DIR) / "IMLCV" / ".ase_calculators" / "cp2k"

        directory = tempfile.mkdtemp(dir=rp)
        print(f"saving CP2K output in {directory}")
        params["directory"] = os.path.relpath(directory)

        calc = CP2K(**params)

        return calc

    def _handle_exception(self):
        p = f"{self.atoms.calc.directory}/cp2k.out"
        assert os.path.exists(p), "no cp2k output file after failure"
        with open(p) as f:
            lines = f.readlines()
        out = min(len(lines), 20)
        assert out != 0, "cp2k.out doesn't contain output"

        file = "\\n".join(lines[-out:])

        raise AseError(
            f"The cp2k calculator failed to provide an energy. The end of the output from cp2k.out is { file}"
        )

    def __getstate__(self):
        return [
            self.atoms.todict(),
            self.cp2k_inp,
            self.input_kwargs,
            self.kwargs,
        ]

    def __setstate__(self, state):
        atoms_dict, cp2k_inp, input_kwargs, kwargs = state

        self.__init__(
            ase.Atoms.fromdict(atoms_dict),
            cp2k_inp,
            input_kwargs,
            **kwargs,
        )


class BiasError(Exception):
    pass


class Bias(BC, ABC):
    """base class for biased MD runs."""

    def __init__(self, cvs: CV, start=None, step=None) -> None:
        """args:
        cvs: collective variables
        start: number of md steps before update is called
        step: steps between update is called"""
        super().__init__()

        self.cvs = cvs
        self.start = start
        self.step = step

        self.finalized = False

    def update_bias(self, sp: SystemParams):
        """update the bias.

        Can only change the properties from _get_args
        """

    @partial(jit, static_argnums=(0, 2, 3))
    def compute_coor(self, sp: SystemParams, gpos=False, vir=False) -> EnergyResult:
        """Computes the bias, the gradient of the bias wrt the coordinates and
        the virial."""

        [cvs, jac] = self.cvs.compute(sp=sp, jacobian=gpos or vir)
        [ener, de] = self.compute(cvs, diff=(gpos or vir), batched=sp.batched)

        e_gpos = None
        if gpos:
            es = "nj,njkl->nkl"
            if not sp.batched:
                es = es.replace("n", "")
            e_gpos = jnp.einsum(es, de, jac.coordinates)

        e_vir = None
        if vir:
            es = "nji,nk,nkjl->nil"
            if not sp.batched:
                es = es.replace("n", "")
            e_vir = jnp.einsum(es, sp.cell, de, jac.cell)

        return EnergyResult(ener, e_gpos, e_vir)

    @partial(jit, static_argnums=(0, 2, 3, 4))
    def compute(self, cvs, diff=False, map=True, batched=True):
        """compute the energy and derivative.

        If map==False, the cvs are assumed to be already mapped
        """

        assert not (
            not map and diff
        ), "cannot retreive gradient from already mapped CVs"

        # map compute command
        def f0(x):
            args = [HashableArrayWrapper(a) for a in self.get_args()]
            static_array_argnums = tuple(i + 1 for i in range(len(args)))

            return jit(self._compute, static_argnums=static_array_argnums)(x, *args)

        def f1(x):
            if map:
                x = self.cvs.metric.map(x)
            return f0(x)

        def f2(x):
            return value_and_grad(f1)(x) if diff else (f1(x), None)

        return vmap(f2)(cvs) if batched else f2(cvs)

    @abstractmethod
    def _compute(self, cvs, *args):
        """function that calculates the bias potential. CVs live in mapped space"""
        raise NotImplementedError

    @abstractmethod
    def get_args(self):
        """function that return dictionary with kwargs of _compute."""
        return []

    def finalize(self):
        """Should be called at end of metadynamics simulation.

        Optimises compute
        """

        self.finalized = True

    @staticmethod
    def load(filename) -> Bias:
        bias = BC.load(filename=filename)
        assert isinstance(bias, Bias)
        return bias

    def plot(
        self,
        name,
        n=50,
        traj=None,
        vmin=0,
        vmax=100,
        map=True,
        inverted=False,
    ):
        """plot bias."""

        assert self.cvs.n == 2

        bins = self.cvs.metric.grid(n=n, map=map, endpoints=True)
        mg = np.meshgrid(*bins, indexing="xy")

        xlim = [mg[0].min(), mg[0].max()]
        ylim = [mg[1].min(), mg[1].max()]
        extent = [xlim[0], xlim[1], ylim[0], ylim[1]]

        bias, _ = jnp.apply_along_axis(
            self.compute,
            axis=0,
            arr=np.array(mg),
            diff=False,
            batched=False,
            map=not map,
        )

        if map is False:
            mask = self.cvs.metric._get_mask(tol=0.01, interp_mg=mg)
            bias = bias * mask

        # normalise lowest point of bias
        if inverted:
            bias = -bias
        bias -= bias[~np.isnan(bias)].min()

        plt.switch_backend("PDF")
        fig, ax = plt.subplots()

        p = ax.imshow(
            bias / (kjmol),
            cmap=plt.get_cmap("rainbow"),
            origin="lower",
            extent=extent,
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_xlabel("cv1", fontsize=16)
        ax.set_ylabel("cv2", fontsize=16)

        cbar = fig.colorbar(p)
        cbar.set_label("Bias [kJ/mol]", fontsize=16)

        if traj is not None:

            if not isinstance(traj, Iterable):
                traj = [traj]
            for tr in traj:
                # trajs are ij indexed
                ax.scatter(tr[:, 0], tr[:, 1], s=3)

        ax.set_title(name)
        os.makedirs(os.path.dirname(name), exist_ok=True)
        fig.set_size_inches([12, 8])
        fig.savefig(name)
        plt.close(fig=fig)  # write out


@bash_app_python()
def plot_app(
    bias: Bias,
    outputs: list[File],
    n: int = 50,
    vmin: float = 0,
    vmax: float = 100,
    map: bool = True,
    inverted=False,
    traj: list[np.ndarray] | None = None,
):
    bias.plot(
        name=outputs[0].filepath,
        n=n,
        traj=traj,
        vmin=vmin,
        vmax=vmax,
        map=map,
        inverted=inverted,
    )


class CompositeBias(Bias):
    """Class that combines several biases in one single bias."""

    def __init__(self, biases: Iterable[Bias], fun=jnp.sum) -> None:

        self.init = True

        self.biases: list[Bias] = []

        self.start_list = np.array([], dtype=np.int16)
        self.step_list = np.array([], dtype=np.int16)
        self.args_shape = np.array([0])
        self.cvs: CV = None  # type: ignore

        for bias in biases:
            self._append_bias(bias)

        self.fun = fun

        super().__init__(cvs=self.cvs, start=0, step=1)
        self.init = True

    def _append_bias(self, b: Bias):

        if isinstance(b, NoneBias):
            return

        self.biases.append(b)

        self.start_list = np.append(
            self.start_list, b.start if (b.start is not None) else -1
        )
        self.step_list = np.append(
            self.step_list, b.step if (b.step is not None) else -1
        )
        self.args_shape = np.append(
            self.args_shape, len(b.get_args()) + self.args_shape[-1]
        )

        if self.cvs is None:
            self.cvs = b.cvs
        else:
            pass
            # assert self.cvs == b.cvs, "CV should be the same"

    def _compute(self, cvs, *args):

        e = jnp.array(
            [
                self.biases[i]._compute(
                    cvs, *args[self.args_shape[i] : self.args_shape[i + 1]]
                )
                for i in range(len(self.biases))
            ]
        )

        return self.fun(e)

    def finalize(self):
        for b in self.biases:
            b.finalize()

    def update_bias(self, sp: SystemParams):

        if self.finalized:
            return

        mask = self.start_list == 0

        self.start_list[mask] += self.step_list[mask]
        self.start_list -= 1

        for i in np.argwhere(mask):
            self.biases[int(i)].update_bias(sp=sp)

    def get_args(self):
        return [a for b in self.biases for a in b.get_args()]


class MinBias(CompositeBias):
    def __init__(self, biases: Iterable[Bias]) -> None:
        super().__init__(biases, fun=jnp.min)


class BiasF(Bias):
    """Bias according to CV."""

    def __init__(self, cvs: CV, g=None):

        self.g = g if (g is not None) else lambda _: jnp.array(0.0)
        self.g = jit(self.g)
        super().__init__(cvs, start=None, step=None)

    def _compute(self, cvs):
        return self.g(cvs)

    def get_args(self):
        return []


class NoneBias(BiasF):
    """dummy bias."""

    def __init__(self, cvs: CV):
        super().__init__(cvs)


class HarmonicBias(Bias):
    """Harmonic bias potential centered arround q0 with force constant k."""

    def __init__(self, cvs: CV, q0, k):
        """generate harmonic potentia;

        Args:
            cvs: CV
            q0: rest pos spring
            k: force constant spring
        """

        if isinstance(k, float):
            k = q0 * 0 + k
        else:
            assert k.shape == q0.shape
        assert np.all(k > 0)
        self.k = jnp.array(k)
        self.q0 = jnp.array(q0)

        super().__init__(cvs)

    def _compute(self, cvs, *args):
        r = self.cvs.metric.difference(cvs, self.q0)
        return jnp.einsum("i,i,i", self.k, r, r)

    def get_args(self):
        return []


class BiasMTD(Bias):
    r"""A sum of Gaussian hills, for instance used in metadynamics:
    Adapted from Yaff.

    V = \sum_{\\alpha} K_{\\alpha}} \exp{-\sum_{i}
    \\frac{(q_i-q_{i,\\alpha}^0)^2}{2\sigma^2}}

    where \\alpha loops over deposited hills and i loops over collective
    variables.
    """

    def __init__(self, cvs: CV, K, sigmas, tempering=0.0, start=None, step=None):
        """_summary_

        Args:
            cvs: _description_
            K: _description_
            sigmas: _description_
            start: _description_. Defaults to None.
            step: _description_. Defaults to None.
            tempering: _description_. Defaults to 0.0.
        """

        if isinstance(sigmas, float):
            sigmas = jnp.array([sigmas])
        if isinstance(sigmas, np.ndarray):
            sigmas = jnp.array(sigmas)

        self.ncv = cvs.n
        assert sigmas.ndim == 1
        assert sigmas.shape[0] == self.ncv
        assert jnp.all(sigmas > 0)
        self.sigmas = sigmas
        self.sigmas_isq = 1.0 / (2.0 * sigmas**2.0)

        self.Ks = jnp.zeros((0,))
        self.q0s = jnp.zeros((0, self.ncv))

        self.tempering = tempering
        self.K = K

        super().__init__(cvs, start, step)

    def update_bias(self, sp: SystemParams):

        if self.finalized:
            return
        # Compute current CV values
        q0s, _ = self.cvs.compute(sp)
        K = self.K
        if self.tempering != 0.0:
            raise NotImplementedError("untested")

        self.q0s = jnp.vstack([self.q0s, q0s])
        self.Ks = jnp.array([*self.Ks, K])

    def _compute(self, cvs, q0s, Ks):
        """Computes sum of hills."""

        deltas = jnp.apply_along_axis(
            self.cvs.metric.difference,
            axis=1,
            arr=q0s.val,
            x2=cvs,
        )

        exparg = jnp.einsum("ji,ji,i -> j", deltas, deltas, self.sigmas_isq)
        energy = jnp.sum(jnp.exp(-exparg) * Ks.val)

        return energy

    def get_args(self):
        return [self.q0s, self.Ks]


class GridBias(Bias):
    """Bias interpolated from lookup table on uniform grid.

    values are caluclated in bin centers
    """

    def __init__(
        self,
        cvs: CV,
        vals,
        #  bounds=None,
        start=None,
        step=None,
        centers=True,
    ) -> None:
        super().__init__(cvs, start, step)

        if not centers:
            raise NotImplementedError
        assert cvs.n == 2

        # extend periodically
        self.n = np.array(vals.shape)

        bias = np.zeros(np.array(vals.shape) + 2) * np.nan
        bias[1:-1, 1:-1] = vals

        if self.cvs.metric.periodicities[0]:
            bias[0, :] = bias[-2, :]
            bias[-1, :] = bias[1, :]

        if self.cvs.metric.periodicities[1]:
            bias[:, 0] = bias[:, -2]
            bias[:, -1] = bias[:, 1]

        # do general interpolation
        x, y = np.indices(bias.shape)
        mask = np.isnan(bias)
        rbf = RBFInterpolator(np.array([x[~mask], y[~mask]]).T, bias[~mask])
        bias[mask] = rbf(np.array([x[mask], y[mask]]).T)

        self.vals = bias

    def _compute(self, cvs):
        # overview of grid points. stars are addded to allow out of bounds extension.
        # the bounds of the x square are per.
        # ___ ___ ___ ___
        # |   |   |   |   |
        # | * | * | * | * |
        # |___|___|___|___|
        # |   |   |   |   |
        # | * | x | x | * |
        # |___|___|___|___|
        # |   |   |   |   |
        # | * | x | x | * |
        # |___|___|___|___|
        # |   |   |   |   |
        # | * | * | * | * |
        # |___|___|___|___|
        # gridpoints are in the middle of

        # map between vals 0 and 1
        coords = (cvs * self.n - 0.5) / (self.n - 1)
        # scale to array size and offset extra row
        coords = coords * (self.n - 1) + 1

        return jsp.ndimage.map_coordinates(self.vals, coords, mode="nearest", order=1)

    def get_args(self):
        return []


class FesBias(Bias):
    """FES bias wraps another (grid) Bias. The compute function properly accounts for the transformation formula F_{unmapped} = F_{mapped} + KT*ln( J(cvs))"""

    def __init__(self, bias: Bias, T) -> None:

        self.bias = bias
        self.T = T

        super().__init__(cvs=bias.cvs, start=bias.start, step=bias.step)

    def compute(self, cvs, diff=False, map=True):
        if map is True:
            return self.bias.compute(cvs, diff=diff, map=True)

        e, de = self.bias.compute(cvs, diff=diff, map=map)

        r = jit(
            lambda x: self.T
            * boltzmann
            * jnp.log(jnp.abs(jnp.linalg.det(jacrev(self.bias.cvs.metric.map)(x))))
        )

        e += r(cvs)

        if diff:
            de += jit(jacrev(r))(cvs)

        return e + r(cvs), de

    def update_bias(self, sp: SystemParams):
        self.bias.update_bias(sp=sp)

    def _compute(self, cvs, *args):
        """function that calculates the bias potential."""
        return self.bias._compute(cvs, *args)

    def get_args(self):
        """function that return dictionary with kwargs of _compute."""
        return self.bias.get_args()


class CvMonitor(BiasF):
    def __init__(self, cvs: CV, start=0, step=1):
        super().__init__(cvs, g=None)
        self.start = start
        self.step = step

        self.last_cv: jnp.ndarray | None = None
        self.transitions = np.zeros((0, self.cvs.metric.ndim, 2))

    def update_bias(self, sp: SystemParams):
        if self.finalized:
            return

        new_cv, _ = self.cvs.compute(sp=sp)

        if self.last_cv is not None:
            if jnp.linalg.norm(new_cv - self.last_cv) > 1:
                new_trans = np.array([[new_cv, self.last_cv]])
                self.transitions = np.vstack((self.transitions, new_trans))

        self.last_cv = new_cv


class BiasPlumed(Bias):
    pass
