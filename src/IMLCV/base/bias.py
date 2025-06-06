from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import fields
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp
import jsonpickle
from jax import Array, value_and_grad
from typing_extensions import Self

from IMLCV import unpickler
from IMLCV.base.CV import (
    CV,
    CollectiveVariable,
    NeighbourList,
    ShmapKwargs,
    SystemParams,
    padded_shard_map,
    padded_vmap,
)
from IMLCV.base.datastructures import MyPyTreeNode, Partial_decorator, field, jit_decorator, vmap_decorator
from IMLCV.base.UnitsConstants import boltzmann, kelvin, kjmol

if TYPE_CHECKING:
    from IMLCV.base.MdEngine import MDEngine


######################################
#              Energy                #
######################################


class EnergyResult(MyPyTreeNode):
    energy: Array
    gpos: Array | None = field(default=None)
    vtens: Array | None = field(default=None)

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


class EnergyError(Exception):
    pass


class Energy:
    external_callback = True
    manual_vtens = False

    @property
    def nl(self):
        return None

    @nl.setter
    def nl(self, nl):
        return

    @property
    @abstractmethod
    def cell(self) -> jax.Array | None:
        pass

    @cell.setter
    @abstractmethod
    def cell(self, cell):
        pass

    @property
    @abstractmethod
    def coordinates(self) -> jax.Array:
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
    def _compute_coor(self, sp: SystemParams, nl: NeighbourList | None, gpos=False, vir=False) -> EnergyResult:
        pass

    def _handle_exception(self, e=None):
        return f"{e=}"

    def get_vtens_finite_difference(
        self,
        sp: SystemParams,
        nl: NeighbourList | None,
        eps=1e-5,
        gpos=False,
    ):
        assert sp.cell is not None

        print(f"{eps=}")

        # rotate cell such that it only has 6 components
        Q, R = jnp.linalg.qr(sp.cell)

        reduced, _ = sp.to_relative()

        print(f"{R=}")

        ener = self.compute_from_system_params(sp=sp, nl=nl, gpos=True, vir=False)

        e0 = ener.energy

        dEdR = jnp.zeros((3, 3))

        for i, j in [[0, 0], [1, 1], [2, 2], [1, 0], [2, 0], [2, 1]]:
            # basis = jnp.zeros((3, 3))
            # basis = basis.at[j, i].set(1)

            # print(f"{basis=}")

            full = reduced.replace(cell=Q @ R.at[j, i].add(eps)).to_absolute()
            dener = self.compute_from_system_params(sp=full, nl=nl, gpos=False, vir=False).energy
            dEdR = dEdR.at[j, i].set((dener - e0) / eps)

        virial = R.T @ dEdR

        print(f"{virial=}")
        # print(f"{(virial+ jnp.einsum('ni,nj->ij', sp.coordinates, ener.gpos))=} ")

        virial = 0.5 * (virial + virial.T)

        return EnergyResult(
            energy=ener.energy,
            gpos=ener.gpos if gpos else None,
            vtens=virial,
        )

    def compute_from_system_params(
        self,
        sp: SystemParams,
        gpos=False,
        vir=False,
        nl: NeighbourList | None = None,
        manual_vir=None,
        shmap=False,
        shmap_kwarg=ShmapKwargs.create(),
    ) -> EnergyResult:
        if manual_vir is None:
            manual_vir = self.manual_vtens

        if vir and manual_vir:
            return self.get_vtens_finite_difference(
                sp,
                nl,
                gpos=gpos,
            )

        return self._compute_coor(sp, nl, gpos=gpos, vir=vir)

    def save(self, filename: str | Path):
        filename = Path(filename)

        if not filename.parent.exists():
            filename.parent.mkdir(parents=True, exist_ok=True)

        if filename.suffix == ".json":
            with open(filename, "w") as f:
                f.writelines(jsonpickle.encode(self, indent=1, use_base85=True))  # type: ignore
        else:
            import cloudpickle

            with open(filename, "wb") as f:
                cloudpickle.dump(self, f)

    @staticmethod
    def load(filename) -> Energy:
        filename = Path(filename)

        if filename.suffix == ".json":
            with open(filename) as f:
                self = jsonpickle.decode(f.read(), context=unpickler)
        else:
            import cloudpickle

            with open(filename, "rb") as f:
                self = cloudpickle.load(f)

        return self  # type: ignore


class EnergyFn(Energy):
    external_callback = False

    f: Callable = field(pytree_node=False)
    _sp: SystemParams | None = None
    _nl: NeighbourList | None = None
    kwargs: dict = field(pytree_node=True, default_factory=dict)
    static_kwargs: dict = field(pytree_node=False, default_factory=dict)

    @property
    def nl(self):
        return self._nl

    @nl.setter
    def nl(self, nl):
        self._nl = nl

    @property
    def cell(self) -> jax.Array | None:
        if self._sp is None:
            return None

        return self._sp.cell

    # @cell.setter
    # def cell(self, cell):
    #     if self._sp is None:
    #         return

    #     self._sp.cell = cell

    @property
    def coordinates(self):
        if self._sp is None:
            return None

        return self._sp.coordinates

    # @coordinates.setter
    # def coordinates(self, coordinates):
    #     self._sp.coordinates = coordinates

    @property
    def sp(self) -> SystemParams | None:
        return self._sp

    @sp.setter
    def sp(self, sp: SystemParams):
        self._sp = sp

    @partial(jit_decorator, static_argnames=["gpos", "vir"])
    def _compute_coor(self, sp: SystemParams, nl: NeighbourList, gpos=False, vir=False) -> EnergyResult:
        def _energy(sp, nl):
            return self.f(sp, nl, **self.static_kwargs, **self.kwargs)

        # print(f"in energy {sp=} {nl=}")

        e = _energy(sp, nl)

        e_gpos = None
        e_vir = None

        if gpos or vir:
            dedsp: SystemParams = jax.jacrev(_energy)(sp, nl)

            if gpos:
                e_gpos = dedsp.coordinates

            if vir:
                e_vir = jnp.einsum("ij,il->jl", sp.cell, dedsp.cell) + jnp.einsum(
                    "ni,nl->il", sp.coordinates, dedsp.coordinates
                )

        res = EnergyResult(
            energy=e,
            gpos=e_gpos,
            vtens=e_vir,
        )

        return res


class PlumedEnerg(Energy):
    pass


######################################
#       Biases                       #
######################################


class BiasError(Exception):
    pass


class Bias(ABC, MyPyTreeNode):
    """base class for biased MD runs."""

    # __: KW_ONLY

    collective_variable: CollectiveVariable = field(pytree_node=True)
    start: int | None = field(pytree_node=False, default=0)
    step: int | None = field(pytree_node=False, default=1)
    finalized: bool = field(pytree_node=False, default=False)
    slice_exponent: float = field(pytree_node=False, default=1.0)
    log_exp_slice: bool = field(pytree_node=False, default=True)
    slice_mean: bool = field(pytree_node=False, default=False)

    @classmethod
    def create(cls, *args, **kwargs) -> Self:
        return cls(*args, **kwargs)

    def update_bias(
        self,
        md: MDEngine,
    ) -> Bias:
        """update the bias.

        Can only change the properties from _get_args
        """

        return self._update_bias()[1]

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

    @partial(
        jit_decorator,
        static_argnames=[
            "gpos",
            "vir",
            "chunk_size",
            "shmap",
            "use_jac",
            "push_jac",
            "rel",
            "shmap_kwargs",
            "return_cv",
        ],
    )
    def compute_from_system_params(
        self,
        sp: SystemParams,
        nl: NeighbourList | None = None,
        gpos=False,
        vir=False,
        chunk_size: int | None = None,
        shmap=False,
        use_jac=False,
        push_jac=False,
        rel=False,
        shmap_kwargs=ShmapKwargs.create(),
        return_cv=False,
    ) -> tuple[CV, EnergyResult]:
        """Computes the bias, the gradient of the bias wrt the coordinates and
        the virial."""

        if sp.batched:
            if nl is not None:
                assert nl.batched

            f = padded_vmap(
                Partial_decorator(
                    self.compute_from_system_params,
                    gpos=gpos,
                    vir=vir,
                    shmap=False,
                    shmap_kwargs=None,
                ),
                chunk_size=chunk_size,
            )

            if shmap:
                f = padded_shard_map(f, shmap_kwargs)

            return f(sp, nl)

        e_gpos = None
        e_vir = None

        if not use_jac:
            # for the virial computation, we need to work in relative coordinates
            # as such, all the positions are scaled when the unit cell is scaled

            @jit_decorator
            def _compute(sp_rel: SystemParams, nl: NeighbourList | None):
                if rel:
                    sp = sp_rel.to_absolute()
                else:
                    sp = sp_rel

                cvs, _ = self.collective_variable.compute_cv(
                    sp=sp,
                    nl=nl,
                    jacobian=False,
                    shmap=shmap,
                    shmap_kwargs=shmap_kwargs,
                )
                ener = self._compute(cvs)

                return ener, cvs

            if gpos or vir:
                if rel:
                    sp_rel, c_inv = sp.to_relative()
                else:
                    sp_rel = sp

                (ener, cvs), de = value_and_grad(_compute, has_aux=True)(sp_rel, nl)

                if gpos:
                    if rel:
                        e_gpos = jnp.einsum("jk, nk->nj ", c_inv, de.coordinates)
                    else:
                        e_gpos = de.coordinates

                if vir and sp.cell is not None:
                    if rel:
                        e_vir = jnp.einsum("ji,jl->il", sp.cell, de.cell)
                    else:
                        e_vir = jnp.einsum("ji,jl->il", sp.cell, de.cell) + jnp.einsum(
                            "ni,nl->il", sp.coordinates, de.coordinates
                        )

                        # jax.debug.print("e_vir_xorr {}", jnp.einsum("ni,nl->il", sp.coordinates, de.coordinates))

            else:
                ener, cvs = _compute(sp, nl)

        else:
            # raise NotImplementedError("adapt to relative coordinates")

            [cvs, jac] = self.collective_variable.compute_cv(
                sp=sp,
                nl=nl,
                jacobian=gpos or vir,
                shmap=shmap,
                push_jac=push_jac,
            )

            [ener, de] = self.compute_from_cv(
                cvs,
                diff=(gpos or vir),
                shmap=shmap,
            )

            if gpos:
                assert de is not None
                assert jac is not None
                e_gpos = jnp.einsum("j,jkl->kl", de.cv, jac.coordinates)

            e_vir = None
            if vir and sp.cell is not None:
                # transpose, see https://pubs.acs.org/doi/suppl/10.1021/acs.jctc.5b00748/suppl_file/ct5b00748_si_001.pdf s1.4 and S1.22

                assert de is not None
                assert jac is not None

                e_vir = jnp.einsum("ij,k,klj->il", sp.cell, de.cv, jac.cell) + jnp.einsum(
                    "ni,j,jnl->il", sp.coordinates, de.cv, jac.coordinates
                )

        ener_out = EnergyResult(
            energy=ener,
            gpos=e_gpos,
            vtens=e_vir,
        )

        # if return_cv:
        return cvs, ener_out

    @partial(jit_decorator, static_argnames=["diff", "chunk_size", "shmap", "shmap_kwargs"])
    def compute_from_cv(
        self,
        cvs: CV,
        diff=False,
        chunk_size=None,
        shmap=False,
        shmap_kwargs=ShmapKwargs.create(),
    ) -> tuple[Array, CV | None]:
        """compute the energy and derivative.

        If map==False, the cvs are assumed to be already mapped
        """

        if cvs.batched:
            f = padded_vmap(
                Partial_decorator(
                    self.compute_from_cv,
                    chunk_size=chunk_size,
                    diff=diff,
                    shmap=False,
                ),
                chunk_size=chunk_size,
            )

            if shmap:
                f = padded_shard_map(f, shmap_kwargs)

            return f(cvs)

        if diff:
            e, de = value_and_grad(self._compute)(cvs)

        else:
            e, de = self._compute(cvs), None

        return e, de

    @abstractmethod
    def _compute(self, cvs) -> jax.Array:
        """function that calculates the bias potential."""
        raise NotImplementedError

    @staticmethod
    def static_plot(bias, **kwargs):
        bias.plot(**kwargs)

    def plot(
        self,
        name: str | None = None,
        traj: list[CV] | None = None,
        dlo_kwargs=None,
        dlo=None,
        vmax=100 * kjmol,
        map=False,
        inverted=False,
        margin=0.1,
        dpi=300,
        T=300 * kelvin,
        **kwargs,
    ):
        from IMLCV.base.CVDiscovery import Transformer

        # option: if every weight is None, then we can use the bias to compute the weights
        # this might be a bad idea if there is not enough data

        if traj is None and dlo is not None:
            assert dlo_kwargs is not None
            traj = dlo.data_loader(**dlo_kwargs)[0].cv

        bias = self

        if inverted:
            print("inverting bias")
            bias = BiasModify.create(fun=lambda x: -x, bias=bias)

        Transformer.plot_app(
            collective_variables=[self.collective_variable],
            cv_data=[[traj]] if traj is not None else None,
            duplicate_cv_data=False,
            name=name,
            color_trajectories=True,
            margin=margin,
            plot_FES=True,
            biases=[[bias]],
            vmax=vmax,
            dpi=dpi,
            T=T,
            indicate_plots=None,
            cv_titles=True,
            **kwargs,
        )

    def resample(self, cv_grid: CV | None = None, n=40, margin=0.3) -> Bias:
        # return same bias, but as gridded bias
        from IMLCV.implementations.bias import GridBias

        return GridBias.create(
            cvs=self.collective_variable,
            bias=self,
            n=n,
            bounds=None,
            margin=margin,
        )

    def save(self, filename: str | Path, cv_file: Path | None = None):
        if isinstance(filename, str):
            filename = Path(filename)
        if not filename.parent.exists():
            filename.parent.mkdir(parents=True, exist_ok=True)

        # if cv_file is not None:
        #     colvar_backup = self.collective_variable
        #     self.collective_variable = cv_file

        if filename.suffix == ".json":
            with open(filename, "w") as f:
                f.writelines(jsonpickle.encode(self, indent=1, use_base85=True))  # type: ignore
        else:
            import cloudpickle

            with open(filename, "wb") as f:
                cloudpickle.dump(self, f)

        # if cv_file is not None:
        #     self.collective_variable = colvar_backup

    @staticmethod
    def load(filename) -> Bias:
        filename = Path(filename)
        if filename.suffix == ".json":
            with open(filename) as f:
                self = jsonpickle.decode(f.read(), context=unpickler)
        else:
            import cloudpickle

            with open(filename, "rb") as f:
                self = cloudpickle.load(f)

        assert isinstance(self, Bias)

        # substitute real CV
        if isinstance(self.collective_variable, Path):
            self.collective_variable = CollectiveVariable.load(self.collective_variable)

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
                f"tried to initialize {self.__class__} with from {statedict=} {f'{removed=}' if len(removed) == 0 else ''} but got exception",
            )
            raise e

    def bounds_from_bias(self, T, sign=1.0, margin=1e-10, n=50):
        margin = 1e-10

        colvar = self.collective_variable
        bins, grid, _, _ = colvar.metric.grid(n=50, margin=0.1)

        beta = 1 / (boltzmann * T)

        probs = jnp.exp(beta * self.compute_from_cv(grid)[0])
        probs /= jnp.sum(probs)
        probs = probs.reshape((n,) * colvar.n)

        limits = []

        for i in range(colvar.n):
            p_i = vmap_decorator(jnp.sum, in_axes=i)(probs)

            cum_p_i = jnp.cumsum(p_i)

            index_0 = jnp.min(jnp.argwhere(cum_p_i >= margin))
            index_1 = jnp.max(jnp.argwhere(cum_p_i <= 1 - margin))

            limits.append([bins[i][index_0], bins[i][index_1]])

        bounds = jnp.array(limits)

        return bounds

    def kl_divergence(
        self,
        other: Bias,
        T: float,
        symmetric=True,
        margin=0.2,
        sign=1.0,
        n=100,
    ):
        _, cvs, _, _ = self.collective_variable.metric.grid(n=n, margin=margin)

        u0 = sign * self.apply([cvs])[0] / (T * boltzmann)
        u0 -= jnp.max(u0)

        p_self = jnp.exp(u0)
        p_self /= jnp.sum(p_self)

        u1 = sign * other.apply([cvs])[0] / (T * boltzmann)
        u1 -= jnp.max(u1)

        p_other = jnp.exp(u1)
        p_other /= jnp.sum(p_other)

        @vmap_decorator
        def f(x, y):
            return jnp.where(x == 0, 0, x * jnp.log(x / y))

        kl = jnp.sum(f(p_self, p_other))  # type: ignore

        if symmetric:
            kl += jnp.sum(f(p_other, p_self))  # type: ignore
            kl *= 0.5

        return kl

    def slice(
        self,
        T,
        inverted=True,
        vmax=None,
        n_max_bias=1e5,
        margin=0.2,
        macro_chunk=10000,
        offset=True,
    ) -> dict[int, dict[tuple[int], Bias]]:
        free_energies = []

        n_grid = int(n_max_bias ** (1 / self.collective_variable.n))

        _, cv, _, bounds = self.collective_variable.metric.grid(
            n=n_grid,
            margin=margin,
        )

        x = self.apply([cv], macro_chunk_size=macro_chunk)[0]

        if not self.log_exp_slice:
            x /= boltzmann * T
        else:
            x /= boltzmann * T

        # if self.log_exp_slice:
        #     w = jnp.exp(w)

        # print(f"{x=}")

        x = x.reshape((n_grid,) * self.collective_variable.n)

        cvi = self.collective_variable

        free_energies = dict()

        from itertools import combinations

        # free_energies[cvi.n] = dict()

        for nd in range(cvi.n):
            free_energies[nd + 1] = dict()

            for tup in combinations(range(cvi.n), nd + 1):
                dims = jnp.arange(cvi.n)
                dims = jnp.delete(dims, jnp.array(tup))
                dims = tuple([int(a) for a in dims])

                # print(f"{tup=} {dims=} ")

                # print(f"{w_max=}")

                # print(f"{self.log_exp_slice=}")

                if self.log_exp_slice:

                    def _f(x):
                        x_max = jnp.max(x)

                        return (
                            jnp.log(jnp.nansum(jnp.exp((x - x_max) * self.slice_exponent)))
                            + x_max * self.slice_exponent
                        )

                    # print(f"{tup=}")

                    for i, d in enumerate(jnp.flip(jnp.array(tup))):
                        # print(f"vmap_decoratorping over {i=} {d=} {d-i=}")

                        n_before = len(tup) - i - 1

                        # print(f"{n_before=} {d=} {i=} {d-n_before}")

                        _f = vmap_decorator(_f, in_axes=int(d - n_before))

                    x_sum = _f(x)
                    # result is in reverse order

                    # x_sum = jnp.transpose(x_sum)  # reverses order of all axes

                    # x_sum = jnp.apply_over_axes(_f, x, dims)

                else:
                    x_sum = jnp.nansum(x**self.slice_exponent, axis=dims)

                # print(f"{x_sum=}")

                if self.log_exp_slice:
                    x_sum /= self.slice_exponent
                else:
                    x_sum = x_sum ** (1 / self.slice_exponent)

                # print(f"{x_sum=}")

                if self.log_exp_slice:
                    # output is log exp slice
                    values = x_sum * boltzmann * T

                else:
                    print(f"{x_sum=}")
                    values = x_sum * (boltzmann * T)

                # if inverted:
                #     values = -values

                if offset and self.log_exp_slice:
                    values -= jnp.nanmax(values)

                # print(f"{values=}")

                from IMLCV.implementations.bias import GridBias

                fes_nd = GridBias(
                    collective_variable=self.collective_variable[tup],
                    n=n_grid,
                    vals=values,
                    bounds=bounds[jnp.array(tup)],
                )

                free_energies[nd + 1][tup] = fes_nd

        return free_energies

    def apply(self, cvs: list[CV], shmap=False, macro_chunk_size=10000):
        from IMLCV.base.rounds import DataLoaderOutput

        return DataLoaderOutput._apply_bias(
            bias=self,
            x=cvs,
            shmap=False,
            macro_chunk=macro_chunk_size,
        )


class CompositeBias(Bias):
    """Class that combines several biases in one single bias."""

    biases: list[Bias]
    fun: Callable = field(pytree_node=False)

    @classmethod
    def create(cls, biases: list[Bias], fun=jnp.sum) -> CompositeBias:
        collective_variable: CollectiveVariable = None  # type: ignore

        biases_new = []

        for b in biases:
            if b is NoneBias:
                continue

            if collective_variable is None:
                collective_variable = b.collective_variable
            # else:
            #     assert (
            #         collective_variable == b.collective_variable
            #     ), f"encountered 2 different collective variables {collective_variable=}  {b.collective_variable=} "

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
            finalized=all([b.finalized for b in biases_new]),
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
    def create(cls, fun: Callable, bias: Bias, kwargs: dict = {}, static_kwargs: dict = {}) -> BiasModify:  # type: ignore[override]
        return BiasModify(
            collective_variable=bias.collective_variable,
            fun=fun,
            start=None,
            step=None,
            finalized=bias.finalized,
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
        cls,
        cvs: CollectiveVariable,
        g: Callable = _constant,
        kwargs: dict = {},
        static_kwargs: dict = {},
    ) -> BiasF:
        return BiasF(
            collective_variable=cvs,
            g=g,
            start=None,
            step=None,
            finalized=True,
            kwargs=kwargs,
            static_kwargs=static_kwargs,
        )

    def _compute(self, cvs):
        return self.g(cvs, **self.kwargs, **self.static_kwargs)


class NoneBias(BiasF):
    @classmethod
    def create(cls, collective_variable: CollectiveVariable) -> NoneBias:
        return NoneBias(
            collective_variable=collective_variable,
            g=_zero_fun,
        )
