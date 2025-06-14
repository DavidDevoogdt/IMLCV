"""MD engine class peforms MD simulations in a given NVT/NPT ensemble."""

from __future__ import annotations

import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from pathlib import Path
from time import time

import cloudpickle
import h5py
import jax
import jax.numpy as jnp
import jsonpickle
from ase.data import atomic_masses
from jax import Array
from jax.lax import dynamic_slice_in_dim, dynamic_update_slice_in_dim
from typing_extensions import Self

from IMLCV import unpickler
from IMLCV.base.bias import Bias, Energy, EnergyResult
from IMLCV.base.CV import CV, NeighbourList, NeighbourListInfo, ShmapKwargs, SystemParams
from IMLCV.base.datastructures import MyPyTreeNode, field
from IMLCV.base.UnitsConstants import amu, angstrom, bar, kjmol

######################################
#             Trajectory             #
######################################


@dataclass
class StaticMdInfo:
    _attr = [
        "timestep",
        "r_cut",
        "timecon_thermo",
        "T",
        "P",
        "timecon_baro",
        "write_step",
        "equilibration",
        "screen_log",
        "max_grad",
        "invalid",
    ]

    _arr = [
        "atomic_numbers",
    ]

    timestep: float

    T: float
    timecon_thermo: float

    atomic_numbers: Array

    r_cut: float | None = None
    P: float | None = None
    timecon_baro: float | None = None

    write_step: int = 100
    equilibration: float | None = None
    screen_log: int = 1000

    invalid: bool = False

    max_grad: float | None = 200 * kjmol / angstrom

    @property
    def masses(self):
        return jnp.array([atomic_masses[int(n)] for n in self.atomic_numbers]) * amu

    @property
    def thermostat(self):
        return self.T is not None

    @property
    def barostat(self):
        return self.P is not None

    @property
    def neighbour_list_info(self) -> NeighbourListInfo:
        return NeighbourListInfo.create(
            r_cut=self.r_cut,
            z_array=self.atomic_numbers,
        )

    def __post_init__(self):
        if self.thermostat:
            assert self.timecon_thermo is not None

        if self.barostat:
            assert self.timecon_baro is not None

        # if self.equilibration is None:
        #     self.equilibration = 200 * self.timestep

    def _save(self, hf: h5py.File):
        for name in self._arr:
            prop = self.__getattribute__(name)
            if prop is not None:
                if name in hf:
                    del hf[name]

                hf[name] = prop

        for name in self._attr:
            prop = self.__getattribute__(name)
            if prop is not None:
                hf.attrs[name] = prop

    def save(self, filename: str | Path):
        if isinstance(filename, str):
            filename = Path(filename)

        if not filename.parent.exists():
            filename.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(str(filename), "r+" if filename.exists() else "w") as hf:
            self._save(hf=hf)

    @staticmethod
    def _load(hf: h5py.File) -> StaticMdInfo:
        props_static = {}
        attrs_static = {}

        for key, val in hf.items():
            props_static[key] = val[:]

        for key, val in hf.attrs.items():
            attrs_static[key] = val

        return StaticMdInfo(**attrs_static, **props_static)

    @staticmethod
    def load(filename) -> StaticMdInfo:
        with h5py.File(str(filename), "r") as hf:
            return StaticMdInfo._load(hf=hf)


# @partial(flax_dataclass, frozen=False, eq=False)
class TrajectoryInfo(MyPyTreeNode):
    _positions: Array
    _cell: Array | None = None
    _charges: Array | None = None

    _e_pot: Array | None = None
    _e_bias: Array | None = None

    _w: Array | None = None
    _rho: Array | None = None

    _cv: Array | None = None
    _cv_orig: Array | None = None  # usefull to reconstruct CV discovery

    _T: Array | None = None
    _P: Array | None = None
    _err: Array | None = None

    _t: Array | None = None

    _capacity: int = field(pytree_node=False, default=-1)
    _size: int = field(pytree_node=False, default=-1)

    _finished: int = field(pytree_node=False, default=False)
    _invalid: int = field(pytree_node=False, default=False)

    # static values
    _items_scal = [
        "_t",
        "_e_pot",
        "_e_bias",
        "_T",
        "_P",
        "_err",
        "_w",
        "_rho",
    ]
    _items_vec = [
        "_positions",
        "_cell",
        "_charges",
        "_cv",
        "_cv_orig",
    ]

    @staticmethod
    def create(
        positions: Array,
        cell: Array | None = None,
        charges: Array | None = None,
        e_pot: Array | None = None,
        e_bias: Array | None = None,
        cv: Array | None = None,
        cv_orig: Array | None = None,
        w: Array | None = None,
        rho: Array | None = None,
        T: Array | None = None,
        P: Array | None = None,
        err: Array | None = None,
        t: Array | None = None,
        capacity: int = -1,
        size: int = -1,
        finished=False,
        invalid=False,
    ) -> TrajectoryInfo:
        if capacity == -1:
            capacity = 1
        if size == -1:
            size = 1

        dict = {
            "_positions": positions,
            "_cell": cell,
            "_charges": charges,
            "_e_pot": e_pot,
            "_e_bias": e_bias,
            "_cv": cv,
            "_cv_orig": cv_orig,
            "_w": w,
            "_rho": rho,
            "_T": T,
            "_P": P,
            "_err": err,
            "_t": t,
            "_capacity": int(capacity),
            "_size": int(size),
            "_finished": finished,
            "_invalid": invalid,
        }

        # batch
        if len(positions.shape) == 2:
            for name in [*TrajectoryInfo._items_vec, *TrajectoryInfo._items_scal]:
                prop = dict[name]
                if prop is not None:
                    dict[name] = jnp.expand_dims(prop, 0)  # jnp.array([prop])

        # test wether cell is truly not None
        if dict["_cell"] is not None:
            if dict["_cell"].shape[-2] == 0:
                dict["_cell"] = None

        return TrajectoryInfo(**dict)

    def __getitem__(self, slices):
        "gets slice from indices. the output is truncated to the to include only items wihtin _size"

        slz = (jnp.ones(self._capacity, dtype=jnp.int64).cumsum() - 1)[slices]
        slz = slz[slz <= self._size]

        # dynamic_slice_in_dim

        return TrajectoryInfo(
            _positions=self._positions[slz, :],
            _cell=self._cell[slz, :] if self._cell is not None else None,
            _charges=(self._charges[slz, :] if self._cell is not None else None) if self._charges is not None else None,
            _e_pot=self._e_pot[slz,] if self._e_pot is not None else None,
            # _e_pot_gpos=self._e_pot_gpos[slz, :] if self._e_pot_gpos is not None else None,
            # _e_pot_vtens=self._e_pot_vtens[slz, :] if self._e_pot_vtens is not None else None,
            _e_bias=self._e_bias[slz,] if self._e_bias is not None else None,
            _w=self._w[slz,] if self._w is not None else None,
            _rho=self._rho[slz,] if self._rho is not None else None,
            # _e_bias_gpos=self._e_bias_gpos[slz, :] if self._e_bias_gpos is not None else None,
            # _e_bias_vtens=self._e_bias_vtens[slz, :] if self._e_bias_vtens is not None else None,
            _cv=self._cv[slz, :] if self._cv is not None else None,
            _T=self._T[slz,] if self._T is not None else None,
            _P=self._P[slz,] if self._P is not None else None,
            _err=self._err[slz,] if self._err is not None else None,
            _t=self._t[slz,] if self._t is not None else None,
            _capacity=int(jnp.size(slz)),
            _size=int(jnp.size(slz)),
        )

    # @jit
    def _stack(ti_out, *ti: TrajectoryInfo):
        tot_size = ti_out._size + sum([t._size for t in ti])

        # ti_out = ti[0]

        if ti_out._capacity <= tot_size:
            ti_out = ti_out._expand_capacity(nc=tot_size * 1.4)

        index = ti_out._size

        keys = ["_positions", "_cell", "_charges", "_e_pot", "_e_bias", "_w", "_rho", "_cv", "_T", "_P", "_err", "_t"]

        for tii in ti:
            _s = tii._size

            for key in keys:
                if ti_out.__dict__[key] is not None:
                    ti_out.__dict__[key] = dynamic_update_slice_in_dim(
                        ti_out.__dict__[key], dynamic_slice_in_dim(tii.__dict__[key], 0, _s), index, 0
                    )

            index += _s

        ti_out._size = index

        return ti_out

    def __add__(self, ti: TrajectoryInfo) -> TrajectoryInfo:
        return self._stack(ti)

    def _expand_capacity(self, nc=None) -> TrajectoryInfo:
        if nc is None:
            nc = min(self._capacity * 2, self._capacity + 1000)

        nc = int(nc)

        print(f"expanding capacity from {self._capacity} to {nc}")

        delta = nc - self._capacity

        dict = {
            "_capacity": int(nc),
            "_size": int(self._size),
        }

        for name in self._items_vec:
            prop = self.__dict__[name]

            if prop is not None:
                dict[name] = jnp.vstack((prop, jnp.zeros((delta, *prop.shape[1:]))))  # type:ignore

        for name in self._items_scal:
            prop = self.__dict__[name]
            if prop is not None:
                dict[name] = jnp.hstack([prop, jnp.zeros(delta)])  # type:ignore

        return TrajectoryInfo(**dict)  # type:ignore

    def _shrink_capacity(self) -> TrajectoryInfo:
        dict = {}

        print(f"shrinking capacity from {self._capacity} to {self._size} ")

        for name in self._items_vec:
            prop = self.__getattribute__(name)
            if prop is not None:
                dict[name] = prop[: self._size, :]

        for name in self._items_scal:
            prop = self.__getattribute__(name)
            if prop is not None:
                dict[name] = prop[: self._size]

        dict["_capacity"] = int(self._size)
        dict["_size"] = int(self._size)
        dict["_finished"] = int(self._finished)

        return TrajectoryInfo(**dict)

    def save(self, filename: str | Path):
        if isinstance(filename, str):
            filename = Path(filename)

        if not filename.parent.exists():
            filename.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(str(filename), "r+" if filename.exists() else "w") as hf:
            self._save(hf=hf)

    def _save(self, hf: h5py.File):
        for name in [*self._items_scal, *self._items_vec]:
            prop = self.__getattribute__(name)
            if prop is not None:
                if name in hf:
                    del hf[name]  # might be different size

                hf[name] = prop.__array__()  # save as numpy array

        hf.attrs.create("_capacity", self._capacity)
        hf.attrs.create("_size", self._size)
        hf.attrs.create("_finished", self._finished)
        hf.attrs.create("_invalid", self._invalid)

    @staticmethod
    def load(filename) -> TrajectoryInfo:
        with h5py.File(str(filename), "r") as hf:
            return TrajectoryInfo._load(hf=hf)

    @staticmethod
    def _load(hf: h5py.File):
        props = {}
        attrs = {}

        for key, val in hf.items():
            if (key not in TrajectoryInfo._items_scal) and (key not in TrajectoryInfo._items_vec):
                # print(f"{key=} deprecated, ignoring")
                continue

            props[key] = jnp.array(val[:])

        for key, val in hf.attrs.items():
            attrs[key] = val

        return TrajectoryInfo(
            **props,
            **attrs,
        )

    @property
    def sp(self) -> SystemParams:
        return SystemParams(
            coordinates=jnp.array(self._positions[0 : self._size, :]),
            cell=jnp.array(self._cell[0 : self._size, :]) if self._cell is not None else None,
        )

    @property
    def positions(self) -> Array | None:
        if self._positions is None:
            return None
        return self._positions[0 : self._size, :]

    @property
    def cell(self) -> Array | None:
        if self._cell is None:
            return None
        return self._cell[0 : self._size, :]

    @property
    def volume(self):
        if self.cell is not None:
            vol_unsigned = jnp.linalg.det(self._cell)
            if (vol_unsigned < 0).any():
                print("cell volume was negative")

            return jnp.abs(vol_unsigned)
        return None

    @property
    def charges(self) -> Array | None:
        if self._charges is None:
            return None
        return self._charges[0 : self._size, :]

    @property
    def e_pot(self) -> Array | None:
        if self._e_pot is None:
            return None
        return self._e_pot[0 : self._size]

    @property
    def w(self) -> Array | None:
        if self._w is None:
            return None
        return self._w[0 : self._size]

    @property
    def rho(self) -> Array | None:
        if self._rho is None:
            return None
        return self._rho[0 : self._size]

    @property
    def e_bias(self) -> Array | None:
        if self._e_bias is None:
            return None
        return self._e_bias[0 : self._size]

    @e_bias.setter
    def e_bias(self, val):
        assert val.shape[0] == self._size

        val = jnp.array(val)

        if self._e_bias is None:
            self._e_bias = jnp.zeros((self._capacity,))

        self._e_bias = self._e_bias.at[: self._size].set(val)

    @property
    def cv(self) -> Array | None:
        if self._cv is None:
            return None
        return self._cv[0 : self._size, :]

    @property
    def cv_orig(self) -> Array | None:
        if self._cv_orig is None:
            return None
        return self._cv_orig[0 : self._size, :]

    @property
    def T(self) -> Array | None:
        if self._T is None:
            return None
        return self._T[0 : self._size]

    @property
    def P(self) -> Array | None:
        if self._P is None:
            return None
        return self._P[0 : self._size]

    @property
    def err(self) -> Array | None:
        if self._err is None:
            return None
        return self._err[0 : self._size]

    @property
    def t(self) -> Array | None:
        if self._t is None:
            return None
        return self._t[0 : self._size]

    @t.setter
    def t(self, val):
        assert val.shape[0] == self._size

        if self._t is None:
            self._t = jnp.zeros((self._capacity,))

        self._t = self._t.at[: self._size].set(val)

    @property
    def shape(self):
        return self._size

    @property
    def CV(self) -> CV | None:
        if self._cv is not None:
            return CV(cv=self._cv[0 : self._size, :])
        return None

    @property
    def CV_orig(self) -> CV | None:
        if self._cv_orig is not None:
            return CV(cv=self._cv_orig[0 : self._size, :])
        return None


######################################
#             MDEngine               #
######################################


@dataclass
class MDEngine(ABC):
    """Base class for MD engine."""

    bias: Bias
    energy: Energy
    sp: SystemParams
    static_trajectory_info: StaticMdInfo
    trajectory_info: TrajectoryInfo | None = field(default=None)
    trajectory_file: Path | None = None
    time0: float = field(default_factory=time)

    step: int = 1

    nl: NeighbourList | None = None
    r_skin = 1.0 * angstrom

    @classmethod
    def create(
        cls,
        bias: Bias,
        energy: Energy,
        static_trajectory_info: StaticMdInfo,
        trajectory_info=None,
        trajectory_file=None,
        sp: SystemParams | None = None,
        **kwargs,
    ) -> Self:
        cont = False

        create_kwargs = {}

        if trajectory_file is not None:
            trajectory_file = Path(trajectory_file)
            # continue with existing file if it exists
            if Path(trajectory_file).exists():
                trajectory_info = TrajectoryInfo.load(trajectory_file)
                cont = True

        if not cont or trajectory_info is None:
            create_kwargs["step"] = 1
            if sp is not None:
                energy.sp = sp

        else:
            create_kwargs["step"] = trajectory_info._size
            energy.sp = trajectory_info.sp[-1]

        # self.update_nl()

        kwargs.update(create_kwargs)

        out = cls(
            bias=bias,
            energy=energy,
            static_trajectory_info=static_trajectory_info,
            trajectory_info=trajectory_info,
            trajectory_file=trajectory_file,
            **kwargs,
        )

        out.update_nl()

        return out

    def update_nl(self):
        if self.static_trajectory_info.r_cut is None:
            return None

        if self.nl is None:
            info = NeighbourListInfo.create(
                r_cut=self.static_trajectory_info.r_cut,
                z_array=self.static_trajectory_info.atomic_numbers,
                r_skin=self.r_skin,
            )

            nl = self.sp.get_neighbour_list(info)  # jitted update

            assert nl is not None
            b = True
        else:
            nl = self.nl
            b = not self.nl.needs_update(self.sp)

        info = nl.info

        if not b:
            print("nl - update")
            b, nl = nl.update_nl(self.sp)

        if not b:
            print("nl - slow update")
            nl = self.sp.get_neighbour_list(info)

        assert nl is not None

        nneigh = nl.nneighs()

        if jnp.mean(nneigh) <= 2.0:
            raise ValueError(f"Not all atoms have neighbour. Number neighbours = {nneigh - 1} {self.sp=}")

        if jnp.max(nneigh) > 100:
            raise ValueError(f"neighbour list is too large for at leat one  atom {nneigh=}")

        # nl = jax.device_put(nl, jax.devices("cpu")[0])

        self.nl = nl

    def save(self, file):
        filename = Path(file)
        if filename.suffix == ".json":
            with open(filename, "w") as f:
                f.writelines(jsonpickle.encode(self, indent=1, use_base85=True))  # type:ignore
        else:
            with open(filename, "wb") as f:
                cloudpickle.dump(self, f)

    @staticmethod
    def load(file, **kwargs) -> MDEngine:
        filename = Path(file)

        if filename.suffix == ".json":
            with open(filename) as f:
                self = jsonpickle.decode(f.read(), context=unpickler)
        else:
            with open(filename, "rb") as f:
                self = cloudpickle.load(f)

        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

            if key == "trajectory_file":
                continue

        assert isinstance(self, MDEngine)

        if (key := "trajectory_file") in kwargs.keys():
            print(f"loading ti  {self.trajectory_file} ")
            trajectory_file = kwargs[key]
            if Path(trajectory_file).exists():
                print("updating sp from trajectory file")

                self.trajectory_info = TrajectoryInfo.load(self.trajectory_file)
                self.step = self.trajectory_info._size
                self.sp = self.trajectory_info.sp[-1]

                print(f"loaded ti  {self.step=} ")

        self.time0 = time()

        self.update_nl()

        return self

    def new_bias(self, bias: Bias, **kwargs) -> MDEngine:
        with tempfile.NamedTemporaryFile() as tmp:
            self.save(tmp.name)
            kwargs["bias"] = bias
            mde = MDEngine.load(tmp.name, **kwargs)
        return mde

    def run(self, steps):
        """run the integrator for a given number of steps.

        Args:
            steps: number of MD steps
        """

        self.update_nl()

        if self.step != 1:
            steps = steps - self.step
            print(f"previous run had {self.step} steps, running for additional {int(steps)} steps!")
        else:
            print(f"running for {int(steps)} steps!")

        try:
            self._run(int(steps))

            if self.trajectory_info is not None:
                self.trajectory_info._finished = True

        except Exception as err:
            if self.step == 1:
                raise err
            print("The calculator finished early with error , marking as invalid")
            if self.trajectory_info is not None:
                self.trajectory_info._invalid = True

            raise err

        if self.trajectory_info is not None:
            self.trajectory_info = self.trajectory_info._shrink_capacity()
            if self.trajectory_file is not None:
                self.trajectory_info.save(self.trajectory_file)

    @abstractmethod
    def _run(self, steps):
        raise NotImplementedError

    def get_trajectory(self) -> TrajectoryInfo:
        assert self.trajectory_info is not None

        return self.trajectory_info._shrink_capacity()

    def save_step(
        self,
        T=None,
        P=None,
        t=None,
        err=None,
        cv=None,
        e_bias=None,
        e_pot=None,
        gpos_rmsd=None,
        gpos_bias_rmsd=None,
        canonicalize=False,
    ):
        if canonicalize and self.nl is not None:
            sp = self.nl.canonicalized_sp(self.sp)
        else:
            sp = self.sp

        ti = TrajectoryInfo.create(
            positions=sp.coordinates,
            cell=sp.cell,
            e_pot=e_pot,
            e_bias=e_bias,
            cv=cv,
            T=T,
            P=P,
            t=t,
            err=err,
        )

        if self.step == 1:
            str = f"{'step': ^10s}"
            str += f"|{'cons err': ^10s}"
            str += f"|{'e_pot[Kj/mol]': ^15s}"
            str += f"|{'e_bias[Kj/mol]': ^15s}"
            if ti._P is not None:
                str += f"|{'P[bar]': ^10s}"
            str += f"|{'T[K]': ^10s}|{'walltime[s]': ^11s}"
            if gpos_rmsd is not None:
                ss = "|\u2207\u2093U\u1d47|[Kj/\u212b]"
                str += f"|{ss: ^13s}"
            if gpos_bias_rmsd is not None:
                ss = "|\u2207\u2093U\u1d47|[Kj/\u212b]"
                str += f"|{ss: ^13s}"

            str += f"|{' CV': ^10s}"
            print(str, sep="")
            print(f"{'=' * len(str)}")

        if self.step % self.static_trajectory_info.screen_log == 0:
            str = f"{self.step: >10d}"
            assert ti._err is not None
            assert ti._T is not None
            assert ti._e_pot is not None
            assert ti._e_bias is not None

            str += f"|{ti._err[0]: >10.4f}"
            str += f"|{ti._e_pot[0] / kjmol: >15.8f}"
            str += f"|{ti._e_bias[0] / kjmol: >15.8f}"
            if ti._P is not None:
                str += f" {ti._P[0] / bar: >10.2f}"
            str += f" {ti._T[0]: >10.2f} {time() - self.time0: >11.2f}"
            if gpos_rmsd is not None:
                str += f"|{(gpos_rmsd / kjmol * angstrom): >13.2f}"
            if gpos_bias_rmsd is not None:
                str += f"|{(gpos_bias_rmsd / kjmol * angstrom): >13.2f}"
            if ti._cv is not None:
                str += f"| {ti._cv[0, :]}"
            print(str)

        # write step to trajectory
        if self.trajectory_info is None:
            self.trajectory_info = ti
        else:
            self.trajectory_info += ti

        if self.step % self.static_trajectory_info.write_step == 0:
            if self.trajectory_file is not None:
                self.trajectory_info.save(self.trajectory_file)  # type: ignore

        self.bias = self.bias.update_bias(self)

        self.step += 1

    def get_energy(
        self,
        sp: SystemParams,
        gpos: bool = False,
        vtens: bool = False,
        manual_vir=None,
    ) -> EnergyResult:
        def f(sp, nl):
            return self.energy.compute_from_system_params(
                gpos=gpos,
                vir=vtens,
                sp=sp,
                nl=nl,
                manual_vir=manual_vir,
            )

        if self.energy.external_callback:

            def _mock_f(sp):
                return EnergyResult(
                    energy=jnp.array(1.0),
                    gpos=None if not gpos else sp.coordinates,
                    vtens=None if not vtens else sp.cell,
                )

            dtypes = jax.eval_shape(_mock_f, sp)

            out = jax.pure_callback(
                f,
                dtypes,
                sp,
                self.nl,
            )
        else:
            out = f(sp, self.nl)

        return out

    def get_bias(
        self,
        sp: SystemParams,
        gpos: bool = False,
        vtens: bool = False,
        shmap: bool = False,
        use_jac=False,
        push_jac=False,
        rel=False,
        shmap_kwargs=ShmapKwargs.create(),
    ) -> tuple[CV, EnergyResult]:
        cv, ener = self.bias.compute_from_system_params(
            sp=sp,
            nl=self.nl,
            gpos=gpos,
            vir=vtens,
            shmap=shmap,
            use_jac=use_jac,
            push_jac=push_jac,
            rel=rel,
            shmap_kwargs=shmap_kwargs,
        )

        return cv, ener

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, statedict: dict):
        removed = []
        try:
            f_names = [f.name for f in fields(self.__class__)]

            for k in statedict.keys():
                if k not in f_names:
                    removed.append(k)

            for k in removed:
                del statedict[k]

            self.__class__.__init__(self, **statedict)

            if self.trajectory_file is not None:
                if Path(self.trajectory_file).exists():
                    assert self.trajectory_info is not None

                    self.step = self.trajectory_info._size
                    self.sp = self.trajectory_info.sp[-1]

            self.time0 = time()

        except Exception as e:
            print(
                f"tried to initialize {self.__class__} with from {statedict=} {f'{removed=}' if len(removed) == 0 else ''} but got exception",
            )
            raise e

        self.update_nl()
