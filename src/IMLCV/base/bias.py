from __future__ import annotations

import warnings
from abc import ABC
from abc import abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from dataclasses import fields
from dataclasses import KW_ONLY
from functools import partial
from pathlib import Path
from typing import Callable
from typing import TYPE_CHECKING

import cloudpickle
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import yaff
from flax.serialization import from_state_dict
from flax.serialization import to_state_dict
from flax.struct import field
from flax.struct import PyTreeNode
from hsluv import hsluv_to_rgb
from IMLCV.base.CV import chunk_map
from IMLCV.base.CV import CollectiveVariable
from IMLCV.base.CV import CV
from IMLCV.base.CV import NeighbourList
from IMLCV.base.CV import SystemParams
from IMLCV.configs.bash_app_python import bash_app_python
from IMLCV.tools.tools import HashableArrayWrapper
from jax import Array
from jax import jit
from jax import value_and_grad
from jax import vmap
from jax.tree_util import Partial
from molmod.units import angstrom
from molmod.units import electronvolt
from molmod.units import kjmol
from parsl.data_provider.files import File
from typing_extensions import Self


yaff.log.set_level(yaff.log.silent)

if TYPE_CHECKING:
    from IMLCV.base.MdEngine import MDEngine


######################################
#              Energy                #
######################################


class EnergyResult(PyTreeNode):
    energy: float
    gpos: Array | None = field(default=None)
    vtens: Array | None = field(default=None)

    def __post_init__(self):
        if isinstance(self.gpos, Array):
            self.__dict__["gpos"] = jnp.array(self.gpos)
        if isinstance(self.vtens, Array):
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

    def __str__(self) -> str:
        str = f"energy [eV]: {self.energy/electronvolt}"
        if self.gpos is not None:
            str += f"\ndE/dx^i_j [eV/angstrom] \n {self.gpos[:]*angstrom/electronvolt}"
        if self.vtens is not None:
            str += f"\n  viriaal [eV] \n {self.vtens[:] / electronvolt }"

        return str

    # def __getstate__(self ):
    #     print(f"pickling {self.__class__}")
    #     return  to_state_dict(self)

    # def __setstate__(self, state):
    #     print(f"unpickling {self.__class__}")
    #     self = from_state_dict(self, state)


class EnergyError(Exception):
    pass


class Energy:
    @property
    @abstractmethod
    def cell(self):
        pass

    @cell.setter
    @abstractmethod
    def cell(self, cell):
        pass

    @property
    @abstractmethod
    def coordinates(self):
        pass

    @coordinates.setter
    @abstractmethod
    def coordinates(self, coordinates):
        pass

    @property
    def sp(self) -> SystemParams:
        return SystemParams(coordinates=self.coordinates, cell=self.cell)

    @sp.setter
    def sp(self, sp: SystemParams):
        self.cell = sp.cell
        self.coordinates = sp.coordinates

    @abstractmethod
    def _compute_coor(self, gpos=False, vir=False) -> EnergyResult:
        pass

    def _handle_exception(self):
        return ""

    def compute_from_system_params(
        self,
        gpos=False,
        vir=False,
        sp: SystemParams | None = None,
        nl: NeighbourList | None = None,
    ) -> EnergyResult:
        if sp is not None:
            self.sp = sp

        # try:
        return self._compute_coor(gpos=gpos, vir=vir)

    def save(self, filename: str | Path):
        if isinstance(filename, str):
            filename = Path(filename)
        if not filename.parent.exists():
            filename.parent.mkdir(parents=True, exist_ok=True)

        with open(filename, "wb") as f:
            cloudpickle.dump(self, f)

    @staticmethod
    def load(filename) -> Energy:
        with open(filename, "rb") as f:
            self = cloudpickle.load(f)
        return self


class PlumedEnerg(Energy):
    pass


######################################
#       Biases                       #
######################################


class BiasError(Exception):
    pass


class Bias(PyTreeNode, ABC):
    """base class for biased MD runs."""

    __: KW_ONLY

    collective_variable: CollectiveVariable = field(pytree_node=False)
    start: int | None = field(pytree_node=False, default=0)
    step: int | None = field(pytree_node=False, default=1)
    finalized: bool = field(pytree_node=False, default=False)

    @classmethod
    def create(clz, *args, **kwargs) -> Self:
        return clz(*args, **kwargs)

    def update_bias(
        self,
        md: MDEngine,
    ) -> Bias:
        """update the bias.

        Can only change the properties from _get_args
        """

        return self

    def _update_bias(self) -> tuple[bool, Self]:
        """update the bias.

        Can only change the properties from _get_args
        """
        if self.finalized:
            return False, self

        if self.start is None or self.step is None:
            return False, self

        if self.start == 0:
            return True, self.replace(start=self.start + self.step - 1)
        return False, self.replace(start=self.start - 1)

    @partial(jax.jit, static_argnames=["gpos", "vir", "chunk_size"])
    def compute_from_system_params(
        self,
        sp: SystemParams,
        nl: NeighbourList | None = None,
        gpos=False,
        vir=False,
        chunk_size: int | None = None,
    ) -> tuple[CV, EnergyResult]:
        """Computes the bias, the gradient of the bias wrt the coordinates and
        the virial."""

        if sp.batched:
            if nl is not None:
                assert nl.batched
                return chunk_map(
                    vmap(
                        Partial(
                            self.compute_from_system_params,
                            gpos=gpos,
                            vir=vir,
                        ),
                    ),
                    chunk_size=chunk_size,
                )(sp, nl)
            else:
                return chunk_map(
                    vmap(
                        Partial(
                            self.compute_from_system_params,
                            gpos=gpos,
                            vir=vir,
                            nl=None,
                        ),
                    ),
                    chunk_size=chunk_size,
                )(sp)

        [cvs, jac] = self.collective_variable.compute_cv(
            sp=sp,
            nl=nl,
            jacobian=gpos or vir,
        )
        [ener, de] = self.compute_from_cv(cvs, diff=(gpos or vir))

        # def _resum(sp, jac, de):
        e_gpos = None
        if gpos:
            es = "nj,njkl->nkl"
            if not sp.batched:
                es = es.replace("n", "")
            e_gpos = jnp.einsum(es, de.cv, jac.cv.coordinates)

        e_vir = None
        if vir and sp.cell is not None:
            # transpose, see https://pubs.acs.org/doi/suppl/10.1021/acs.jctc.5b00748/suppl_file/ct5b00748_si_001.pdf s1.4 and S1.22
            es = "nji,nk,nkjl->nli"
            if not sp.batched:
                es = es.replace("n", "")
            e_vir = jnp.einsum(es, sp.cell, de.cv, jac.cv.cell)

        return cvs, EnergyResult(ener, e_gpos, e_vir)

    @partial(jax.jit, static_argnames=["diff", "chunk_size"])
    def compute_from_cv(self, cvs: CV, diff=False, chunk_size=None) -> tuple[CV, CV | None]:
        """compute the energy and derivative.

        If map==False, the cvs are assumed to be already mapped
        """

        if cvs.batched:
            return chunk_map(
                vmap(
                    Partial(
                        self.compute_from_cv,
                        chunk_size=chunk_size,
                        diff=diff,
                    ),
                ),
                chunk_size=chunk_size,
            )(cvs)

        if diff:
            return value_and_grad(self._compute)(cvs)

        return self._compute(cvs), None

    @abstractmethod
    def _compute(self, cvs):
        """function that calculates the bias potential. CVs live in mapped space"""
        raise NotImplementedError

    def plot(
        self,
        name,
        x_unit: str | None = None,
        y_unit: str | None = None,
        n=50,
        traj: list[CV] | None = None,
        vmin=0,
        vmax=100 * kjmol,
        map=False,
        inverted=False,
        margin=None,
        x_lim=None,
        y_lim=None,
        bins=None,
        label="bias [kJ/mol]",
    ):
        """plot bias."""
        if bins is None:
            bins = self.collective_variable.metric.grid(
                n=n,
                endpoints=True,
                margin=margin,
            )
        mg = np.meshgrid(*bins, indexing="xy")

        cv_grid = CV.combine(*[CV(cv=j.reshape(-1, 1)) for j in jnp.meshgrid(*bins)])

        bias, _ = self.compute_from_cv(cv_grid)

        if inverted:
            bias = -bias

        bias -= bias[~np.isnan(bias)].min()
        bias = bias.reshape([len(mg_i) for mg_i in mg])

        if self.collective_variable.n == 1:
            if x_unit is not None:
                if x_unit == "rad":
                    x_unit_label = "rad"
                    x_fact = 1
                elif x_unit == "ang":
                    x_unit_label = "Ang"
                    x_fact = angstrom
            else:
                x_fact = 1
                x_unit_label = "a.u."

            if x_lim is None:
                x_lim = [bins.min() / x_fact, bins.max() / x_fact]

            extent = [x_lim[0], x_lim[1]]

            plt.switch_backend("PDF")
            fig, ax = plt.subplots()

            ax.set_xlim(*extent)
            ax.set_ylim(vmin / kjmol, vmax / kjmol)
            p = ax.plot(bins, bias / (kjmol))

            ax2 = ax.twinx()

            ax.set_xlabel(f"cv1 [{x_unit_label}]", fontsize=16)
            ax.set_ylabel(label, fontsize=16)

            ax.tick_params(axis="both", which="major", labelsize=18)
            ax.tick_params(axis="both", which="minor", labelsize=16)

            if traj is not None:
                if not isinstance(traj, Iterable):
                    traj = [traj]
                for tr in traj:
                    # trajs are ij indexed
                    _ = ax2.hist(tr.cv[:, 0], density=True, histtype="step")

        elif self.collective_variable.n == 2:
            if x_unit is not None:
                if x_unit == "rad":
                    x_unit_label = "rad"
                    x_fact = 0
                elif x_unit == "ang":
                    x_unit_label = "Ang"
                    x_fact = angstrom
            else:
                x_fact = 1
                x_unit_label = "a.u."

            if y_unit is not None:
                if y_unit == "rad":
                    y_unit_label = "rad"
                    y_fact = 0
                elif x_unit == "ang":
                    y_unit_label = "Ang"
                    y_fact = angstrom
            else:
                y_fact = 1
                y_unit_label = "a.u."

            if x_lim is None:
                x_lim = [mg[0].min() / x_fact, mg[0].max() / x_fact]
            if y_lim is None:
                ylim = [mg[1].min() / y_fact, mg[1].max() / y_fact]

            extent = [x_lim[0], x_lim[1], ylim[0], ylim[1]]

            print("styling plot")

            # plt.switch_backend("PDF")

            # import matplotlib.gridspec as gridspec

            # plt.rcdefaults()

            plt.rc("text", usetex=False)
            plt.rc("font", family="DejaVu Sans", size=18)

            # plt.switch_backend("PDF")
            fig = plt.figure(layout="constrained")

            if traj is not None:
                gs = gridspec.GridSpec(
                    nrows=2,
                    ncols=3,
                    width_ratios=[4, 1, 0.2],
                    height_ratios=[1, 4],
                    figure=fig,
                )

                ax = fig.add_subplot(gs[1, 0])
            else:
                gs = gridspec.GridSpec(
                    nrows=1,
                    ncols=2,
                    width_ratios=[4, 0.2],
                    height_ratios=[1],
                    figure=fig,
                )
                ax = fig.add_subplot(gs[0, 0])

            ax.set_xlabel(f"cv1 [{x_unit_label}]")
            ax.set_ylabel(f"cv2 [{y_unit_label}]")

            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])

            p = ax.imshow(
                bias / (kjmol),
                cmap=plt.get_cmap("rainbow"),
                origin="lower",
                extent=extent,
                vmin=vmin / kjmol,
                vmax=vmax / kjmol,
                aspect="auto",
            )

            if traj is not None:
                ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
                ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

                ax_histx.tick_params(axis="x", labelbottom=False, labelleft=False)
                ax_histy.tick_params(axis="y", labelleft=False, labelbottom=False)

                # ax_histx.set_ylabel(f"n")
                # ax_histy.set_xlabel(f"n")

                ax_cbar = fig.add_subplot(gs[1, 2])
            else:
                ax_cbar = fig.add_subplot(gs[0, 1])

            if traj is not None:
                if not isinstance(traj, Iterable):
                    traj = [traj]

                n = len(traj)

                n_sqrt = jnp.ceil(jnp.sqrt(n))

                x_list = []
                y_list = []
                c_list = []

                n_points = 0

                for tr_i, tr in enumerate(traj):
                    a = tr_i // n_sqrt
                    b = tr_i - a * n_sqrt

                    col = hsluv_to_rgb(
                        [
                            float(a) / float(n_sqrt - 1) * 360 if n != 1 else 180,
                            75,
                            20 + float(b) / float(n_sqrt - 1) * 60 if n != 1 else 70,
                        ],
                    )

                    # trajs are ij indexed
                    ax.scatter(tr.cv[:, 0], tr.cv[:, 1], s=2, color=col)

                    x_list.append(tr.cv[:, 0])
                    y_list.append(tr.cv[:, 1])
                    c_list.append(col)
                    n_points += tr.shape[0]

                n_bins = 3 * int(1 + jnp.ceil(jnp.log2(n_points)))

                ax_histx.hist(
                    x_list,
                    bins=n_bins,
                    color=c_list,
                    stacked=True,
                    histtype="step",
                )
                ax_histy.hist(
                    y_list,
                    bins=n_bins,
                    color=c_list,
                    histtype="step",
                    stacked=True,
                    orientation="horizontal",
                )

                fig.align_ylabels([ax, ax_histx])

            cbar = fig.colorbar(p, cax=ax_cbar)
            cbar.set_label(label)

            # plt.title(label)

        else:
            raise ValueError

        Path(name).parent.mkdir(parents=True, exist_ok=True)

        # fig.set_tig
        # gs.tight_layout(fig)

        plt.savefig(name)
        plt.close(fig=fig)  # write out

    def resample(self, cv_grid: CV | None = None, n=None, margin=0.2) -> Bias:
        from IMLCV.implementations.bias import RbfBias

        if cv_grid is None:
            grid = self.collective_variable.metric.grid(n=n, margin=margin)
            cv_grid = CV.combine(*[CV(cv=j.reshape(-1, 1)) for j in jnp.meshgrid(*grid)])

        bias, _ = self.compute_from_cv(cv_grid)

        return RbfBias.create(cv=cv_grid, cvs=self.collective_variable, vals=bias, kernel="thin_plate_spline")

    def save(self, filename: str | Path):
        if isinstance(filename, str):
            filename = Path(filename)
        if not filename.parent.exists():
            filename.parent.mkdir(parents=True, exist_ok=True)

        with open(filename, "wb") as f:
            cloudpickle.dump(self, f)

    @staticmethod
    def load(filename) -> Bias:
        with open(filename, "rb") as f:
            self = cloudpickle.load(f)
        return self

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, statedict: dict):
        try:
            f_names = [f.name for f in fields(self.__class__)]

            removed = []

            for k in statedict.keys():
                if k not in f_names:
                    removed.append(k)

            for k in removed:
                del statedict[k]

            self.__dict__.update(**statedict)
        except Exception as e:
            print(
                f"tried to initialize {self.__class__} with from {statedict=} {f'{removed=}' if len(removed) == 0  else ''} but got exception",
            )
            raise e


@bash_app_python(executors=["default"])
def plot_app(
    bias: Bias,
    outputs: list[File],
    n: int = 50,
    vmin: float = 0,
    vmax: float = 100 * kjmol,
    map: bool = True,
    inverted=False,
    traj: list[CV] | None = None,
    margin=None,
    x_unit=None,
    y_unit=None,
    x_lim=None,
    y_lim=None,
    bins=None,
    label="bias [kJ/mol]",
):
    bias.plot(
        name=outputs[0].filepath,
        n=n,
        traj=traj,
        vmin=vmin,
        vmax=vmax,
        map=map,
        inverted=inverted,
        margin=margin,
        x_unit=x_unit,
        y_unit=y_unit,
        x_lim=x_lim,
        y_lim=y_lim,
        bins=bins,
        label=label,
    )


class CompositeBias(Bias):
    """Class that combines several biases in one single bias."""

    biases: list[Bias]
    fun: Callable = field(pytree_node=False)

    @classmethod
    def create(clz, biases: Iterable[Bias], fun=jnp.sum) -> Self:
        collective_variable: CollectiveVariable = None  # type: ignore

        biases_new = []

        for b in biases:
            if collective_variable is None:
                collective_variable = b.collective_variable

            if b is NoneBias:
                continue

            biases_new.append(b)

        if biases_new is None:
            assert biases[0] is NoneBias
            biases_new = biases[0]

        assert collective_variable is not None

        return clz(
            collective_variable=collective_variable,
            biases=biases_new,
            fun=fun,
            start=0,
            step=1,
            finalized=False,
        )

    def _compute(self, cvs):
        return self.fun(jnp.array([jnp.reshape(self.biases[i]._compute(cvs), ()) for i in range(len(self.biases))]))

    def update_bias(
        self,
        md: MDEngine,
    ) -> Bias:
        return self.replace(biases=[a.update_bias(md) for a in self.biases])


class BiasF(Bias):
    """Bias according to CV."""

    g: Callable = field(pytree_node=False, default=lambda _: jnp.array(0.0))

    @classmethod
    def create(clz, cvs: CollectiveVariable, g: Callable):
        return clz(
            collective_variable=cvs,
            g=g,
            start=None,
            step=None,
            finalized=False,
        )

    def _compute(self, cvs):
        return self.g(cvs)


class NoneBias(BiasF):
    @classmethod
    def create(clz, collective_variable: CollectiveVariable) -> Self:  # type: ignore[override]
        return clz(
            collective_variable=collective_variable,
            start=None,
            step=None,
            finalized=False,
            g=lambda _: jnp.array(0.0),
        )
