from __future__ import annotations
from abc import ABC
from abc import abstractmethod
from collections.abc import Iterable
from dataclasses import fields
from dataclasses import KW_ONLY
from functools import partial
from pathlib import Path
from typing import Callable
from typing import TYPE_CHECKING
import cloudpickle
import jax
import jax.numpy as jnp
import jsonpickle
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import yaff
from flax.struct import field
from flax.struct import PyTreeNode
from IMLCV import unpickler
from IMLCV.base.CV import chunk_map
from IMLCV.base.CV import CollectiveVariable
from IMLCV.base.CV import CV
from IMLCV.base.CV import NeighbourList
from IMLCV.base.CV import SystemParams
from jax import Array
from jax import value_and_grad
from jax import vmap
from jax.tree_util import Partial
from molmod.units import angstrom
from molmod.units import electronvolt
from molmod.units import kjmol
from molmod.constants import boltzmann
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
        filename = Path(filename)

        if not filename.parent.exists():
            filename.parent.mkdir(parents=True, exist_ok=True)

        if filename.suffix == ".json":
            with open(filename, "w") as f:
                f.writelines(jsonpickle.encode(self, indent=1, use_base85=True))
        else:
            with open(filename, "wb") as f:
                cloudpickle.dump(self, f)

    @staticmethod
    def load(filename) -> Energy:
        filename = Path(filename)

        if filename.suffix == ".json":
            with open(filename) as f:
                self = jsonpickle.decode(f.read(), context=unpickler)
        else:
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

    collective_variable: CollectiveVariable = field(pytree_node=True)
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
        """function that calculates the bias potential."""
        raise NotImplementedError

    @staticmethod
    def static_plot(bias, **kwargs):
        bias.plot(**kwargs)

    def plot(
        self,
        name: str | None = None,
        x_unit: str | None = None,
        y_unit: str | None = None,
        n=100,
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
        plot_bias=True,
        colors: Array | None = None,
        offset=False,
    ):
        """plot bias."""
        if bins is None:
            bins, _, _ = self.collective_variable.metric.grid(
                n=n,
                endpoints=True,
                margin=margin,
            )
        mg = np.meshgrid(*bins, indexing="xy")
        cv_grid = CV.combine(*[CV(cv=j.reshape(-1, 1)) for j in jnp.meshgrid(*bins)])
        bias, _ = self.compute_from_cv(cv_grid)

        if offset:
            bias -= bias[~np.isnan(bias)].min()
        else:
            vrange = vmax - vmin

            vmax = bias[~np.isnan(bias)].max()
            vmin = vmax - vrange

        if inverted:
            bias = -bias
            vmin, vmax = -vmax, -vmin

        bias = bias.reshape([len(mg_i) for mg_i in mg])

        plt.rc("text", usetex=False)
        plt.rc("font", family="DejaVu Sans", size=18)

        # plt.switch_backend("PDF")
        fig = plt.figure(layout="constrained")

        if self.collective_variable.n == 1:
            bins = bins[0]

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

            ax = fig.add_subplot()
            ax.set_xlim(*extent)
            ax.set_ylim(vmin / kjmol, vmax / kjmol)

            p = ax.plot(bins, bias / (kjmol))

            ax.set_xlabel(f"cv_1 [{x_unit_label}]")
            ax.set_ylabel(label)

            ax.tick_params(axis="both", which="major")
            ax.tick_params(axis="both", which="minor")

            if traj is not None:
                ax2 = ax.twinx()
                ax2.set_ylabel("count")

                if not isinstance(traj, Iterable):
                    traj = [traj]

                n = len(traj)
                print(f"plotting {n=} trajectories")

                x_list = []
                c_list = []

                n_points = 0

                if colors is None:
                    from IMLCV.base.CVDiscovery import Transformer

                    colors = [
                        a.cv[0]
                        for a in Transformer._get_color_data(
                            a=CV.stack(*traj),
                            dim=1,
                            color_trajectories=True,
                        ).unstack()
                    ]

                for col, tr in zip(colors, traj):
                    col = np.array(col)

                    x_list.append(tr.cv[:, 0])
                    c_list.append(col)

                    in_xlim = jnp.logical_and(tr.cv[:, 0] > x_lim[0], tr.cv[:, 0] < x_lim[1])

                    n_points += jnp.sum(in_xlim)

                if n_points != 0:
                    n_bins = 3 * int(1 + jnp.ceil(jnp.log2(n_points)))
                else:
                    n_bins = 10

                ax2.hist(
                    x_list,
                    range=x_lim,
                    bins=n_bins,
                    color=c_list,
                    stacked=True,
                    histtype="step",
                )

                fig.align_ylabels([ax, ax2])

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
                y_lim = [mg[1].min() / y_fact, mg[1].max() / y_fact]

            extent = [x_lim[0], x_lim[1], y_lim[0], y_lim[1]]

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

            ax.set_xlabel(f"cv_1 [{x_unit_label}]")
            ax.set_ylabel(f"cv_2 [{y_unit_label}]")

            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])

            if plot_bias:
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

            if traj is not None:
                if not isinstance(traj, Iterable):
                    traj = [traj]

                n = len(traj)
                print(f"plotting {n=} trajectories")

                x_list = []
                y_list = []
                c_list = []

                n_points = 0

                if colors is None:
                    from IMLCV.base.CVDiscovery import Transformer

                    colors = [
                        a.cv[0]
                        for a in Transformer._get_color_data(
                            a=CV.stack(*traj),
                            dim=2,
                            color_trajectories=True,
                            min_val=jnp.array([extent[0], extent[2]]),
                            max_val=jnp.array([extent[1], extent[3]]),
                        ).unstack()
                    ]

                for col, tr in zip(colors, traj):
                    col = np.array(col)

                    # print(tr.cv)
                    # print(col)

                    # trajs are ij indexed
                    ax.scatter(tr.cv[:, 0], tr.cv[:, 1], s=2, color=col)

                    x_list.append(tr.cv[:, 0])
                    y_list.append(tr.cv[:, 1])
                    c_list.append(col)
                    in_xlim = jnp.logical_and(tr.cv[:, 0] > x_lim[0], tr.cv[:, 0] < x_lim[1])
                    in_ylim = jnp.logical_and(tr.cv[:, 1] > y_lim[0], tr.cv[:, 1] < y_lim[1])

                    n_points += jnp.sum(jnp.logical_and(in_xlim, in_ylim))

                if n_points != 0:
                    n_bins = 3 * int(1 + jnp.ceil(jnp.log2(n_points)))
                else:
                    n_bins = 10

                ax_histx.hist(
                    x_list,
                    range=x_lim,
                    bins=n_bins,
                    color=c_list,
                    stacked=True,
                    histtype="step",
                )
                ax_histy.hist(
                    y_list,
                    range=y_lim,
                    bins=n_bins,
                    color=c_list,
                    histtype="step",
                    stacked=True,
                    orientation="horizontal",
                )

                ax_histy.tick_params(axis="x", rotation=-90)

                fig.align_ylabels([ax, ax_histx])

            if plot_bias:
                if traj is not None:
                    ax_cbar = fig.add_subplot(gs[1, 2])
                else:
                    ax_cbar = fig.add_subplot(gs[0, 1])

                cbar = fig.colorbar(p, cax=ax_cbar)
                cbar.set_label(label)

            # plt.title(label)

        else:
            raise ValueError

        if name is not None:
            Path(name).parent.mkdir(parents=True, exist_ok=True)

            plt.savefig(name)
            plt.close(fig=fig)  # write out
        else:
            plt.show()

    def resample(self, cv_grid: CV | None = None, n=None, margin=0.2) -> Bias:
        from IMLCV.implementations.bias import RbfBias

        if cv_grid is None:
            _, cv_grid, _ = self.collective_variable.metric.grid(n=n, margin=margin)

        bias, _ = self.compute_from_cv(cv_grid)

        return RbfBias.create(
            cv=cv_grid,
            cvs=self.collective_variable,
            vals=bias,
            kernel="thin_plate_spline",
        )

    def save(self, filename: str | Path):
        if isinstance(filename, str):
            filename = Path(filename)
        if not filename.parent.exists():
            filename.parent.mkdir(parents=True, exist_ok=True)

        if filename.suffix == ".json":
            with open(filename, "w") as f:
                f.writelines(jsonpickle.encode(self, indent=1, use_base85=True))
        else:
            with open(filename, "wb") as f:
                cloudpickle.dump(self, f)

    @staticmethod
    def load(filename) -> Bias:
        filename = Path(filename)
        if filename.suffix == ".json":
            with open(filename) as f:
                self = jsonpickle.decode(f.read(), context=unpickler)
        else:
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

            self.__init__(**statedict)

        except Exception as e:
            print(
                f"tried to initialize {self.__class__} with from {statedict=} {f'{removed=}' if len(removed) == 0  else ''} but got exception",
            )
            raise e

    def bounds_from_bias(self, T, sign=1.0, margin=1e-10, n=50):
        margin = 1e-10

        colvar = self.collective_variable
        bins, grid, _ = colvar.metric.grid(n=50, margin=0.1)

        beta = 1 / (boltzmann * T)

        probs = jnp.exp(beta * self.compute_from_cv(grid)[0])
        probs /= jnp.sum(probs)
        probs = probs.reshape((n,) * colvar.n)

        limits = []

        for i in range(colvar.n):
            p_i = jax.vmap(jnp.sum, in_axes=i)(probs)

            cum_p_i = jnp.cumsum(p_i)

            index_0 = jnp.min(jnp.argwhere(cum_p_i >= margin))
            index_1 = jnp.max(jnp.argwhere(cum_p_i <= 1 - margin))

            limits.append([bins[i][index_0], bins[i][index_1]])

        bounds = jnp.array(limits)

        return bounds

    def kl_divergence(self, other: Bias, T: float, symmetric=True, sign=1.0, n=50):
        _, cvs, _ = self.collective_variable.metric.grid(n=n, margin=0.1)

        p_self = jnp.exp(sign * self.compute_from_cv(cvs)[0] / (T * boltzmann))
        p_self /= jnp.sum(p_self)

        p_other = jnp.exp(sign * other.compute_from_cv(cvs)[0] / (T * boltzmann))
        p_other /= jnp.sum(p_other)

        @vmap
        def f(x, y):
            return jnp.where(x == 0, 0, x * jnp.log(x / y))

        kl = jnp.sum(f(p_self, p_other))

        if symmetric:
            kl += jnp.sum(f(p_other, p_self))
            kl *= 0.5

        return kl

    # def __eq__(self, other):
    #     if not isinstance(other, Bias):
    #         return False

    #     self_val, self_tree = tree_flatten(self)
    #     other_val, other_tree = tree_flatten(other)

    #     if not self_tree == other_tree:
    #         return False

    #     for a, b in zip(self_val, other_val):
    #         a = jnp.array(a)
    #         b = jnp.array(b)

    #         if not a.shape == b.shape:
    #             return False

    #         if not a.dtype == b.dtype:
    #             return False

    #         if not jnp.allclose(a, b):
    #             return False

    #     return True


class CompositeBias(Bias):
    """Class that combines several biases in one single bias."""

    biases: list[Bias]
    fun: Callable = field(pytree_node=False)

    @classmethod
    def create(clz, biases: Iterable[Bias], fun=jnp.sum) -> Self:
        collective_variable: CollectiveVariable = None  # type: ignore

        biases_new = []

        for b in biases:
            if b is NoneBias:
                continue

            if collective_variable is None:
                collective_variable = b.collective_variable
            else:
                assert (
                    collective_variable == b.collective_variable
                ), f"encountered 2 different collective variables {collective_variable=}  {b.collective_variable=} "

            biases_new.append(b)

        if biases_new is None:
            assert biases[0] is NoneBias
            biases_new = biases[0]

        assert collective_variable is not None

        return CompositeBias(
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


def _zero_fun(cvs: CV):
    return jnp.array(0.0)


def _constant(cvs: CV, val: float = 0.0):
    return jnp.array(val)


class BiasModify(Bias):
    """Bias according to CV."""

    fun: Callable = field(pytree_node=False)
    bias: Bias
    kwargs: dict = field(default_factory=dict)
    static_kwargs: dict = field(pytree_node=False, default_factory=dict)

    @classmethod
    def create(clz, fun: Callable, bias: Bias, kwargs: dict = {}, static_kwargs: dict = {}) -> Self:  # type: ignore[override]
        return BiasModify(
            collective_variable=bias.collective_variable,
            fun=fun,
            start=None,
            step=None,
            finalized=False,
            kwargs=kwargs,
            static_kwargs=static_kwargs,
            bias=bias,
        )

    def _compute(self, cvs):
        return jnp.reshape(self.fun(self.bias._compute(cvs), **self.kwargs, **self.static_kwargs), ())

    def update_bias(
        self,
        md: MDEngine,
    ) -> Bias:
        return self.replace(bias=self.bias.update_bias(md))

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, statedict: dict):
        if "pytree_kwargs" in statedict:
            kw = statedict.pop("pytree_kwargs")
            statedict["static_kwargs"] = kw

        super().__setstate__(statedict)


class BiasF(Bias):
    """Bias according to CV."""

    g: Callable = field(pytree_node=False, default=_constant)
    static_kwargs: dict = field(pytree_node=False, default_factory=dict)
    kwargs: dict = field(default_factory=dict)

    @classmethod
    def create(
        clz,
        cvs: CollectiveVariable,
        g: Callable = _constant,
        kwargs: dict = {},
        static_kwargs: dict = {},
    ) -> Self:  # type: ignore[override]
        return BiasF(
            collective_variable=cvs,
            g=g,
            start=None,
            step=None,
            finalized=False,
            kwargs=kwargs,
            static_kwargs=static_kwargs,
        )

    def _compute(self, cvs):
        return self.g(cvs, **self.kwargs, **self.static_kwargs)


class NoneBias(BiasF):
    @classmethod
    def create(clz, collective_variable: CollectiveVariable) -> Self:  # type: ignore[override]
        return super().create(
            cvs=collective_variable,
            g=_zero_fun,
        )
