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
from IMLCV.base.datastructures import (
    MyPyTreeNode,
    Partial_decorator,
    field,
    jit_decorator,
    vmap_decorator,
    # my_dataclass,
)
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

    def __add__(self: EnergyResult, other: EnergyResult) -> EnergyResult:
        assert isinstance(other, EnergyResult)

        gpos = self.gpos
        if gpos is None:
            assert other.gpos is None
        else:
            assert other.gpos is not None
            gpos += other.gpos

        vtens = self.vtens
        if vtens is None:
            assert other.vtens is None
        else:
            assert other.vtens is not None
            vtens += other.vtens

        return EnergyResult(
            energy=self.energy + other.energy,
            gpos=gpos,
            vtens=vtens,
        )


class EnergyError(Exception):
    pass


class Energy(MyPyTreeNode, ABC):
    external_callback: bool = field(pytree_node=False, default=True)
    manual_vtens: bool = field(pytree_node=False, default=False)

    # @property
    # def nl(self) -> NeighbourList | None:
    #     return None

    # @nl.setter
    # def nl(self, nl: NeighbourList):
    #     return

    # @property
    # @abstractmethod
    # def cell(self) -> jax.Array | None:
    #     pass

    # @cell.setter
    # @abstractmethod
    # def cell(self, cell: jax.Array | None):
    #     pass

    # @property
    # @abstractmethod
    # def coordinates(self) -> jax.Array | None:
    #     pass

    # @coordinates.setter
    # @abstractmethod
    # def coordinates(self, coordinates: jax.Array):
    #     pass

    # @property
    # def sp(self) -> SystemParams | None:
    #     c = self.coordinates
    #     if c is None:
    #         return None
    #     return SystemParams(coordinates=c, cell=self.cell)

    # @sp.setter
    # def sp(self, sp: SystemParams):
    #     self.cell = sp.cell
    #     self.coordinates = sp.coordinates

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

    @partial(
        jit_decorator,
        static_argnames=[
            "gpos",
            "vir",
            "manual_vir",
            "shmap",
            "shmap_kwarg",
        ],
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
        if sp.batched:
            _f = padded_vmap(
                Partial_decorator(
                    self.compute_from_system_params,
                    gpos=gpos,
                    vir=vir,
                    nl=nl,
                    manual_vir=manual_vir,
                    shmap=False,
                    shmap_kwarg=None,
                ),
                chunk_size=None,
            )

            if shmap:
                _f = padded_shard_map(_f, shmap_kwarg)

            return _f(sp)

        if manual_vir is None:
            manual_vir = self.manual_vtens

        if vir and manual_vir:
            return self.get_vtens_finite_difference(
                sp,
                nl,
                gpos=gpos,
            )

        # mock evaluation for pure callback
        def f(sp, nl):
            return self._compute_coor(
                sp,
                nl,
                gpos=gpos,
                vir=vir,
            )

        if self.external_callback:

            def _mock_f(sp):
                return EnergyResult(
                    energy=jnp.array(1.0),
                    gpos=None if not gpos else sp.coordinates,
                    vtens=None if not vir else sp.cell,
                )

            dtypes = jax.eval_shape(_mock_f, sp)

            out = jax.pure_callback(
                f,
                dtypes,
                sp,
                nl,
            )
        else:
            out = f(sp, nl)

        return out

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


class EnergyFn(Energy, MyPyTreeNode):
    external_callback: bool = field(pytree_node=False, default=False)

    f: Callable = field(pytree_node=False)

    kwargs: dict = field(pytree_node=True, default_factory=dict)
    static_kwargs: dict = field(pytree_node=False, default_factory=dict)

    @partial(jit_decorator, static_argnames=["gpos", "vir"])
    def _compute_coor(
        self,
        sp: SystemParams,
        nl: NeighbourList | None,
        gpos=False,
        vir=False,
    ) -> EnergyResult:
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


######################################
#       Biases                       #
######################################


class BiasError(Exception):
    pass


class Bias(ABC, MyPyTreeNode):
    """base class for biased MD runs."""

    collective_variable: CollectiveVariable | None = field(pytree_node=True)
    # collective_variable_path: Path | str | None = field(pytree_node=False, default=None)
    start: int | None = False
    step: int | None = 1
    finalized: bool = False

    @staticmethod
    def create(*args, **kwargs) -> Bias:
        raise NotImplementedError

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
            # "shmap_kwargs",
            # "return_cv",
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
                    use_jac=use_jac,
                    push_jac=push_jac,
                    rel=rel,
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

                assert self.collective_variable is not None

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
                c_inv: Array | None = None
                if rel:
                    sp_rel, c_inv = sp.to_relative()
                else:
                    sp_rel = sp

                (ener, cvs), de = value_and_grad(_compute, has_aux=True)(sp_rel, nl)

                # jax.debug.print("{de}", de=de)

                if gpos:
                    if rel:
                        assert c_inv is not None
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

            assert self.collective_variable is not None
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

    @partial(jit_decorator, static_argnames=["diff", "chunk_size", "shmap"])
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
    def _compute(self, cvs: CV) -> jax.Array:
        """function that calculates the bias potential."""
        raise NotImplementedError

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
        plot_FES=True,
        cv_title: str | bool = True,
        data_title: str | bool = True,
        **kwargs,
    ):
        from IMLCV.base.CVDiscovery import Transformer

        # option: if every weight is None, then we can use the bias to compute the weights
        # this might be a bad idea if there is not enough data

        if traj is None and dlo is not None:
            assert dlo_kwargs is not None
            traj = dlo.data_loader(**dlo_kwargs).cv

        bias = self

        if inverted:
            print("inverting bias")
            bias = BiasModify.create(fun=lambda x: -x, bias=bias)

        assert bias.collective_variable is not None

        Transformer.plot_app(
            collective_variables=[bias.collective_variable],
            cv_data=[[traj]] if traj is not None else None,
            duplicate_cv_data=False,
            name=name,
            color_trajectories=True,
            margin=margin,
            plot_FES=plot_FES,
            biases=[[bias]],
            vmax=vmax,
            dpi=dpi,
            T=T,
            indicate_plots=None,
            cv_titles=[cv_title] if isinstance(cv_title, str) else cv_title,
            data_titles=[data_title] if isinstance(data_title, str) else None,
            **kwargs,
        )

    def resample(self, cv_grid: CV | None = None, n=40, margin=0.3) -> Bias:
        # return same bias, but as gridded bias

        assert self.collective_variable is not None

        return GridBias.create(
            cvs=self.collective_variable,
            bias=self,
            n=n,
            bounds=None,
            margin=margin,
        )

    def save(self, filename: str | Path):
        # assert root is not None

        if isinstance(filename, str):
            filename = Path(filename)
        if not filename.parent.exists():
            filename.parent.mkdir(parents=True, exist_ok=True)

        # if self.collective_variable_path is not None and self.collective_variable is not None:
        #     cv_path = Path(root) / Path(self.collective_variable_path)

        #     if not cv_path.exists():
        #         self.collective_variable.save(Path(self.collective_variable_path))

        colvar = self.collective_variable

        self.collective_variable = None  # type:ignore

        if filename.suffix == ".json":
            with open(filename, "w") as f:
                f.writelines(jsonpickle.encode(self, indent=1, use_base85=True))  # type: ignore
        else:
            import cloudpickle

            with open(filename, "wb") as f:
                cloudpickle.dump(self, f)

        self.collective_variable = colvar

    @staticmethod
    def load(filename, collective_variable: CollectiveVariable) -> Bias:
        filename = Path(filename)
        if filename.suffix == ".json":
            with open(filename) as f:
                self = jsonpickle.decode(f.read(), context=unpickler)
        else:
            import cloudpickle

            with open(filename, "rb") as f:
                self = cloudpickle.load(f)

        assert isinstance(self, Bias)

        self.collective_variable = collective_variable

        # if self.collective_variable is None:
        #     assert self.collective_variable_path is not None
        #     cv_path = Path(root) / Path(self.collective_variable_path)

        #     assert cv_path.exists()

        #     self.collective_variable = CollectiveVariable.load(cv_path)

        return self

    def __getstate__(self):
        d = self.__dict__.copy()
        # d["collective_variable"] = None

        return d

    def __setstate__(self, statedict: dict):
        removed = []
        try:
            f_names = [f.name for f in fields(self.__class__)]

            for k in statedict.keys():
                if k not in f_names:
                    removed.append(k)

            for k in removed:
                del statedict[k]

            if "collective_variable" not in statedict:
                statedict["collective_variable"] = None

            self.__init__(**statedict)

        except Exception as e:
            print(
                f"tried to initialize {self.__class__} with from {statedict=} {f'{removed=}' if len(removed) == 0 else ''} but got exception",
            )
            raise e

    def bounds_from_bias(self, T, sign=1.0, margin=1e-10, n=50):
        margin = 1e-10

        colvar = self.collective_variable
        assert colvar is not None

        bins, _, grid, _, _ = colvar.metric.grid(n=50, margin=0.1)

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
        macro_chunk_size=1000,
    ):
        assert self.collective_variable is not None
        _, _, cvs, _, _ = self.collective_variable.metric.grid(n=n, margin=margin)

        u0 = (
            sign
            * self.apply(
                [cvs],
                macro_chunk_size=macro_chunk_size,
            )[0]
            / (T * boltzmann)
        )
        u0 -= jnp.max(u0)

        p_self = jnp.exp(u0)
        p_self /= jnp.sum(p_self)

        u1 = (
            sign
            * other.apply(
                [cvs],
                macro_chunk_size=macro_chunk_size,
            )[0]
            / (T * boltzmann)
        )
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

        # express in kjmol

        return kl * (boltzmann * T)  # type: ignore

    def slice(
        self,
        T,
        inverted=True,
        vmax=None,
        n_max_bias=1e5,
        margin=0.1,
        macro_chunk=10000,
        offset=True,
    ) -> dict[int, dict[tuple[int], Bias]]:
        free_energies = []

        assert self.collective_variable is not None

        n_grid = int(n_max_bias ** (1 / self.collective_variable.n))

        _, _, cv, _, bounds = self.collective_variable.metric.grid(
            n=n_grid,
            margin=margin,
        )

        # assert cv.shape[0] == n_grid

        x = self.apply([cv], macro_chunk_size=macro_chunk)[0]

        x /= boltzmann * T

        x = x.reshape((n_grid,) * self.collective_variable.n)

        cvi = self.collective_variable

        free_energies = dict()

        from itertools import combinations

        for nd in range(cvi.n):
            free_energies[nd + 1] = dict()

            for tup in combinations(range(cvi.n), nd + 1):
                dims = jnp.arange(cvi.n)
                dims = jnp.delete(dims, jnp.array(tup))
                dims = tuple([int(a) for a in dims])

                def _f(x):
                    x_max = jnp.nanmax(x)

                    return jnp.log(jnp.nansum(jnp.exp((x - x_max)))) + x_max

                for i, d in enumerate(jnp.flip(jnp.array(tup))):
                    n_before = len(tup) - i - 1

                    _f = vmap_decorator(_f, in_axes=int(d - n_before))

                x_sum = _f(x)

                values = x_sum * boltzmann * T

                if offset:
                    values -= jnp.nanmax(values)

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


# @my_dataclass
class CompositeBias(Bias):
    """Class that combines several biases in one single bias."""

    biases: list[Bias]
    fun: Callable = field(pytree_node=False)

    @classmethod
    def create(cls, biases: list[Bias], fun=jnp.sum) -> CompositeBias | Bias:
        assert len(biases) > 0
        collective_variable: CollectiveVariable | None = None  # type: ignore
        # colvar_path: str | Path | None = None

        biases_new = []

        for b in biases:
            if isinstance(b, NoneBias):
                continue

            if collective_variable is None:
                collective_variable = b.collective_variable

            # if colvar_path is None:
            #     colvar_path = b.collective_variable_path

            # b.collective_variable = None

            biases_new.append(b)

        if len(biases_new) == 0:
            # every bias must be none bias

            assert isinstance(biases[0], NoneBias)
            biases_new.append(biases[0])
            collective_variable = biases[0].collective_variable
            # colvar_path = biases[0].collective_variable_path

        if len(biases_new) == 1:
            # no need for composite bias
            new_bias = biases_new[0]
            new_bias.collective_variable = collective_variable
            # new_bias.collective_variable_path = colvar_path

            return new_bias

        assert collective_variable is not None

        return CompositeBias(
            collective_variable=collective_variable,
            # collective_variable_path=colvar_path,
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


class RoundBias(Bias):
    bias_r: Bias
    bias_i: Bias

    @classmethod
    def create(cls, bias_r: Bias, bias_i: Bias) -> RoundBias:
        assert bias_r.collective_variable is not None
        assert bias_i.collective_variable is not None
        # assert bias_r.collective_variable == bias_i.collective_variable
        return RoundBias(
            bias_r=bias_r,
            bias_i=bias_i,
            collective_variable=bias_r.collective_variable,
            start=0,
            step=1,
            finalized=bias_r.finalized and bias_i.finalized,
        )

    def _compute(self, cvs: CV):
        r = self.bias_r._compute(cvs)
        i = self.bias_i._compute(cvs)
        return r + i


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
        colvar = bias.collective_variable
        # bias.collective_variable = None

        return BiasModify(
            collective_variable=colvar,
            # collective_variable_path=bias.collective_variable_path,
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

    @staticmethod
    def create(
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
    @staticmethod
    def create(collective_variable: CollectiveVariable) -> NoneBias:  # type:ignore
        return NoneBias(
            collective_variable=collective_variable,
            g=_zero_fun,
        )


class GridBias(Bias):
    """Bias interpolated from lookup table on uniform grid.

    values are caluclated in bin centers
    """

    n: int
    bounds: jax.Array
    vals: jax.Array
    order: int = field(pytree_node=False, default=1)

    @staticmethod
    def adjust_bounds(bounds: Array, n) -> Array:
        print(f"new")
        diff = (bounds[:, 1] - bounds[:, 0]) / n  # space for each bin
        print(f"{diff=}")
        return jnp.array([[bounds[i, 0] + diff[i] / 2, bounds[i, 1] - diff[i] / 2] for i in range(bounds.shape[0])])

    @classmethod
    def create(
        cls,
        cvs: CollectiveVariable,
        bias: Bias,
        n=30,
        bounds: Array | None = None,
        margin=0.1,
        order=1,
    ) -> GridBias:
        grid, _, cv, _, bounds = cvs.metric.grid(
            n=n,
            bounds=bounds,
            margin=margin,
        )

        vals, _ = bias.compute_from_cv(cv)

        vals = vals.reshape((n,) * cvs.n)

        return GridBias(
            collective_variable=cvs,
            n=n,
            vals=vals,
            bounds=bounds,
            order=order,
        )

    def _compute(self, cvs: CV):
        # map between vals 0 and 1
        # if self.bounds is not None:
        coords = (cvs.cv - self.bounds[:, 0]) / (self.bounds[:, 1] - self.bounds[:, 0])

        import jax.scipy as jsp

        # def f(x):
        return jsp.ndimage.map_coordinates(
            self.vals,
            coords * (self.n - 1),  # type:ignore
            mode="constant",
            cval=jnp.nan,
            order=self.order,
        )

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
        # not only transform free energy, but also comopute std

        free_energies = []

        assert self.collective_variable is not None

        log_w = self.vals / (boltzmann * T)

        n_grid = self.n
        bounds = self.bounds
        order = self.order

        cvi = self.collective_variable

        free_energies = dict()

        from itertools import combinations

        for nd in range(cvi.n):
            free_energies[nd + 1] = dict()

            for tup in combinations(range(cvi.n), nd + 1):
                print(f" {nd=} {tup=}")

                dims = jnp.arange(cvi.n)
                dims = jnp.delete(dims, jnp.array(tup))
                dims = tuple([int(a) for a in dims])

                def _log_sum_exp(x, fac=1.0):
                    x_max = jnp.nanmax(x)

                    return jnp.log(jnp.nansum(jnp.exp(fac * (x - x_max)))) + fac * x_max

                _f_n = _log_sum_exp
                print(f"new f")

                for i, d in enumerate(jnp.flip(jnp.array(tup))):
                    n_before = len(tup) - i - 1

                    _f_n = vmap_decorator(_f_n, in_axes=int(d - n_before))

                log_w_sum = _f_n(log_w)

                if offset:
                    log_w_sum -= jnp.nanmax(log_w_sum)

                w_nd = GridBias(
                    collective_variable=self.collective_variable[tup],
                    n=n_grid,
                    vals=log_w_sum * (boltzmann * T),
                    bounds=bounds[jnp.array(tup)],
                    order=order,
                )

                free_energies[nd + 1][tup] = w_nd

        return free_energies


class StdBias(Bias):
    """Class that keeps track of variance of bias."""

    _bias: GridBias  # F
    _log_exp_sigma: GridBias  # sigma e^(-beta F)
    T: float = 300.0

    @staticmethod
    def create(bias: GridBias, log_exp_sigma: GridBias) -> StdBias:  # type:ignore
        colvar = bias.collective_variable

        assert bias.n == log_exp_sigma.n

        return StdBias(
            collective_variable=colvar,
            # collective_variable_path=bias.collective_variable_path,
            _bias=bias,
            _log_exp_sigma=log_exp_sigma,
            start=0,
            step=1,
            finalized=bias.finalized and log_exp_sigma.finalized,
        )

    def _compute(self, cvs):
        print(f"new test  {cvs=}")

        return -jnp.exp(self._log_exp_sigma._compute(cvs) - self._bias._compute(cvs) / (boltzmann * self.T))

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
        # not only transform free energy, but also comopute std

        free_energies = []

        assert self.collective_variable is not None

        log_w = self._bias.vals / (boltzmann * T)
        w_sigma = self._log_exp_sigma.vals

        n_grid = self._bias.n

        bounds = self._bias.bounds
        order = self._bias.order

        # print(f"{n_grid=} {bounds=} ")

        cvi = self.collective_variable

        free_energies = dict()

        from itertools import combinations

        for nd in range(cvi.n):
            free_energies[nd + 1] = dict()

            for tup in combinations(range(cvi.n), nd + 1):
                print(f" {nd=} {tup=}")

                dims = jnp.arange(cvi.n)
                dims = jnp.delete(dims, jnp.array(tup))
                dims = tuple([int(a) for a in dims])

                def _log_sum_exp(x, fac=1.0):
                    x_max = jnp.nanmax(x)

                    return jnp.log(jnp.nansum(jnp.exp(fac * (x - x_max)))) + fac * x_max

                _f_n = _log_sum_exp
                print(f"new f")

                for i, d in enumerate(jnp.flip(jnp.array(tup))):
                    n_before = len(tup) - i - 1

                    _f_n = vmap_decorator(_f_n, in_axes=(int(d - n_before), None))

                log_w_sum = _f_n(log_w, 1.0)
                log_sigma_sum = _f_n(w_sigma, 2.0) / 2.0

                # print(f"{n_grid=} {bounds=}")

                w_nd = GridBias(
                    collective_variable=self.collective_variable[tup],
                    n=n_grid,
                    vals=log_w_sum * (boltzmann * T),
                    bounds=bounds[jnp.array(tup)],
                    order=order,
                )

                w_sigma_sq_nd = GridBias(
                    collective_variable=self.collective_variable[tup],
                    n=n_grid,
                    vals=log_sigma_sum,
                    bounds=bounds[jnp.array(tup)],
                    order=order,
                )

                bias_std = StdBias.create(
                    bias=w_nd,
                    log_exp_sigma=w_sigma_sq_nd,
                )

                free_energies[nd + 1][tup] = bias_std

        return free_energies
