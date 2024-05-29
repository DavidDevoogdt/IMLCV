"""MD engine class peforms MD simulations in a given NVT/NPT ensemble.

Currently, the MD is done with YAFF
"""
from __future__ import annotations

import tempfile
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import fields
from pathlib import Path
from time import time

import cloudpickle
import h5py
import jax.numpy as jnp
import jsonpickle
from IMLCV import unpickler
from IMLCV.base.bias import Bias
from IMLCV.base.bias import Energy
from IMLCV.base.bias import EnergyResult
from IMLCV.base.CV import CV
from IMLCV.base.CV import NeighbourList, NeighbourListInfo
from IMLCV.base.CV import SystemParams
from jax import Array
from molmod.periodic import periodic
from molmod.units import angstrom
from molmod.units import bar
from molmod.units import kjmol
from typing_extensions import Self

from flax.struct import PyTreeNode, field
from jax import jit

from jax.lax import dynamic_update_slice_in_dim

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

    max_grad: float | None = 200 * kjmol / angstrom

    @property
    def masses(self):
        return jnp.array([periodic[int(n)].mass for n in self.atomic_numbers])

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

        with h5py.File(str(filename), "w") as hf:
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


class TrajectoryInfo(PyTreeNode):
    _positions: Array
    _cell: Array | None = None
    _charges: Array | None = None

    _e_pot: Array | None = None
    _e_pot_gpos: Array | None = None
    _e_pot_vtens: Array | None = None

    _e_bias: Array | None = None
    _e_bias_gpos: Array | None = None
    _e_bias_vtens: Array | None = None

    _cv: Array | None = None

    _T: Array | None = None
    _P: Array | None = None
    _err: Array | None = None

    _t: Array | None = None

    _capacity: int = -1
    _size: int = -1

    # static values
    _items_scal = ["_t", "_e_pot", "_e_bias", "_T", "_P", "_err"]
    _items_vec = [
        "_positions",
        "_cell",
        "_e_pot_gpos",
        "_e_pot_vtens",
        "_e_bias_gpos",
        "_e_bias_vtens",
        "_charges",
        "_cv",
    ]

    def create(
        positions: Array,
        cell: Array | None = None,
        charges: Array | None = None,
        e_pot: Array | None = None,
        e_pot_gpos: Array | None = None,
        e_pot_vtens: Array | None = None,
        e_bias: Array | None = None,
        e_bias_gpos: Array | None = None,
        e_bias_vtens: Array | None = None,
        cv: Array | None = None,
        T: Array | None = None,
        P: Array | None = None,
        err: Array | None = None,
        t: Array | None = None,
        capacity: int = -1,
        size: int = -1,
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
            "_e_pot_gpos": e_pot_gpos,
            "_e_pot_vtens": e_pot_vtens,
            "_e_bias": e_bias,
            "_e_bias_gpos": e_bias_gpos,
            "_e_bias_vtens": e_bias_vtens,
            "_cv": cv,
            "_T": T,
            "_P": P,
            "_err": err,
            "_t": t,
            "_capacity": int(capacity),
            "_size": int(size),
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

        slz = (jnp.ones(self._capacity, dtype=jnp.int32).cumsum() - 1)[slices]
        slz = slz[slz <= self._size]

        # dynamic_slice_in_dim

        return TrajectoryInfo(
            _positions=self._positions[slz, :],
            _cell=self._cell[slz, :] if self._cell is not None else None,
            _charges=(self._charges[slz, :] if self._cell is not None else None) if self._charges is not None else None,
            _e_pot=self._e_pot[slz,] if self._e_pot is not None else None,
            _e_pot_gpos=self._e_pot_gpos[slz, :] if self._e_pot_gpos is not None else None,
            _e_pot_vtens=self._e_pot_vtens[slz, :] if self._e_pot_vtens is not None else None,
            _e_bias=self._e_bias[slz,] if self._e_bias is not None else None,
            _e_bias_gpos=self._e_bias_gpos[slz, :] if self._e_bias_gpos is not None else None,
            _e_bias_vtens=self._e_bias_vtens[slz, :] if self._e_bias_vtens is not None else None,
            _cv=self._cv[slz, :] if self._cv is not None else None,
            _T=self._T[slz,] if self._T is not None else None,
            _P=self._P[slz,] if self._P is not None else None,
            _err=self._err[slz,] if self._err is not None else None,
            _t=self._t[slz,] if self._t is not None else None,
            _capacity=jnp.size(slz),
            _size=jnp.size(slz),
        )

    @jit
    def _stack(ti_out, *ti: TrajectoryInfo):
        index = ti_out._size

        d = {
            "_positions": ti_out._positions,
            "_cell": ti_out._cell,
            "_charges": ti_out._charges,
            "_e_pot": ti_out._e_pot,
            "_e_pot_gpos": ti_out._e_pot_gpos,
            "_e_pot_vtens": ti_out._e_pot_vtens,
            "_e_bias": ti_out._e_bias,
            "_e_bias_gpos": ti_out._e_bias_gpos,
            "_e_bias_vtens": ti_out._e_bias_vtens,
            "_cv": ti_out._cv,
            "_T": ti_out._T,
            "_P": ti_out._P,
            "_err": ti_out._err,
            "_t": ti_out._t,
        }

        @jit
        def upd(arr_in, arr, index):
            if arr_in is not None and arr is not None:
                return dynamic_update_slice_in_dim(arr_in, arr, index, 0)
            return None

        for t in ti:
            # d = update_dict(d, index, t)

            d["_positions"] = upd(d["_positions"], t._positions, index)
            d["_cell"] = upd(d["_cell"], t._cell, index)
            d["_charges"] = upd(d["_charges"], t._charges, index)
            d["_e_pot"] = upd(d["_e_pot"], t._e_pot, index)
            d["_e_pot_gpos"] = upd(d["_e_pot_gpos"], t._e_pot_gpos, index)
            d["_e_pot_vtens"] = upd(d["_e_pot_vtens"], t._e_pot_vtens, index)
            d["_e_bias"] = upd(d["_e_bias"], t._e_bias, index)
            d["_e_bias_gpos"] = upd(d["_e_bias_gpos"], t._e_bias_gpos, index)
            d["_e_bias_vtens"] = upd(d["_e_bias_vtens"], t._e_bias_vtens, index)
            d["_cv"] = upd(d["_cv"], t._cv, index)
            d["_T"] = upd(d["_T"], t._T, index)
            d["_P"] = upd(d["_P"], t._P, index)
            d["_err"] = upd(d["_err"], t._err, index)
            d["_t"] = upd(d["_t"], t._t, index)

            index += t._size

        return TrajectoryInfo(**d, _capacity=ti_out._capacity, _size=index)

    @staticmethod
    def stack(*ti: TrajectoryInfo) -> TrajectoryInfo:
        if len(ti) == 1:
            return ti[0]

        tot_size = sum([t._size for t in ti])

        ti_out = ti[0]

        if ti_out._capacity <= tot_size:
            ti_out = ti_out._expand_capacity(nc=tot_size + 10)

        return ti_out._stack(*ti[1:])

    def __add__(self, ti: TrajectoryInfo) -> TrajectoryInfo:
        return TrajectoryInfo.stack(self, ti)

    def _expand_capacity(self, nc=None) -> TrajectoryInfo:
        if nc is None:
            nc = min(self._capacity * 2, self._capacity + 1000)

        delta = nc - self._capacity

        dict = {
            "_capacity": nc,
            "_size": self._size,
        }

        for name in self._items_vec:
            prop = self.__dict__[name]

            if prop is not None:
                dict[name] = jnp.vstack((prop, jnp.zeros((delta, *prop.shape[1:]))))

        for name in self._items_scal:
            prop = self.__dict__[name]
            if prop is not None:
                dict[name] = jnp.hstack([prop, jnp.zeros(delta)])

        return TrajectoryInfo(**dict)

    def _shrink_capacity(self) -> TrajectoryInfo:
        dict = {}

        for name in self._items_vec:
            prop = self.__getattribute__(name)
            if prop is not None:
                dict[name] = prop[: self._size, :]

        for name in self._items_scal:
            prop = self.__getattribute__(name)
            if prop is not None:
                dict[name] = prop[: self._size]

        dict["_capacity"] = self._size
        dict["_size"] = self._size

        return TrajectoryInfo(**dict)

    def save(self, filename: str | Path):
        # ti = self._shrink_capacity()
        self = self

        if isinstance(filename, str):
            filename = Path(filename)

        if not filename.parent.exists():
            filename.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(str(filename), "w") as hf:
            self._save(hf=hf)

    def _save(self, hf: h5py.File):
        for name in [*self._items_scal, *self._items_vec]:
            prop = self.__getattribute__(name)
            if prop is not None:
                hf[name] = prop

        hf.attrs.create("_capacity", self._capacity)
        hf.attrs.create("_size", self._size)

    @staticmethod
    def load(filename) -> TrajectoryInfo:
        with h5py.File(str(filename), "r") as hf:
            return TrajectoryInfo._load(hf=hf)

    @staticmethod
    def _load(hf: h5py.File):
        props = {}
        attrs = {}

        for key, val in hf.items():
            # if key == "static_info":
            #     tic = StaticTrajectoryInfo._load(hf[key])
            # continue
            props[key] = val[:]

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
    def e_pot_gpos(self) -> Array | None:
        if self._e_pot_gpos is None:
            return None
        return self._e_pot_gpos[0 : self._size, :]

    @property
    def e_pot_vtens(self) -> Array | None:
        if self._e_pot_vtens is None:
            return None
        return self._e_pot_vtens[0 : self._size, :]

    @property
    def e_bias(self) -> Array | None:
        if self._e_bias is None:
            return None
        return self._e_bias[0 : self._size]

    @property
    def e_bias_gpos(self) -> Array | None:
        if self._e_bias_gpos is None:
            return None
        return self._e_bias_gpos[0 : self._size, :]

    @property
    def e_bias_vtens(self) -> Array | None:
        if self._e_bias_vtens is None:
            return None
        return self._e_bias_vtens[0 : self._size, :]

    @property
    def cv(self) -> Array | None:
        if self._cv is None:
            return None
        return self._cv[0 : self._size, :]

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

    @property
    def shape(self):
        return self._size

    @property
    def CV(self) -> CV | None:
        if self._cv is not None:
            return CV(cv=self._cv[0 : self._size, :])
        return None


######################################
#             MDEngine               #
######################################


@dataclass
class MDEngine(ABC):
    """Base class for MD engine."""

    bias: Bias
    energy: Energy
    static_trajectory_info: StaticMdInfo
    trajectory_info: TrajectoryInfo | None = None
    trajectory_file: Path | None = None
    time0: float = field(default_factory=time)

    step: int = 1

    last_bias: EnergyResult = EnergyResult(0)
    last_ener: EnergyResult = EnergyResult(0)
    last_cv: CV | None = None
    _nl: NeighbourList | None = None

    @classmethod
    def create(
        self,
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

        if not cont:
            create_kwargs["step"] = 1
            if sp is not None:
                energy.sp = sp

        else:
            create_kwargs["step"] = trajectory_info._size
            energy.sp = trajectory_info.sp[-1]

        kwargs.update(create_kwargs)

        return self(
            bias=bias,
            energy=energy,
            static_trajectory_info=static_trajectory_info,
            trajectory_info=trajectory_info,
            trajectory_file=trajectory_file,
            **kwargs,
        )

    @property
    def sp(self) -> SystemParams:
        return self.energy.sp

    @sp.setter
    def sp(self, sp: SystemParams):
        self.energy.sp = sp

    @property
    def nl(self) -> NeighbourList | None:
        if self.static_trajectory_info.r_cut is None:
            return None

        info = NeighbourListInfo.create(
            r_cut=self.static_trajectory_info.r_cut,
            z_array=self.static_trajectory_info.atomic_numbers,
            r_skin=1.0 * angstrom,
        )

        if self._nl is None:
            nl = self.sp.get_neighbour_list(info)  # jitted update
            b = True
        else:
            nl = self._nl
            b = not self._nl.needs_update(self.sp)

        if not b:
            print("nl - update")
            b, nl = nl.update_nl(self.sp)

        if not b:
            print("nl - slow update")
            nl = self.sp.get_neighbour_list(info)

        self._nl = nl
        return nl

    def save(self, file):
        filename = Path(file)
        if filename.suffix == ".json":
            with open(filename, "w") as f:
                f.writelines(jsonpickle.encode(self, indent=1, use_base85=True))
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
        self._nl = None
        # print(f"{self.sp=}")

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

        if self.step != 1:
            steps = steps - self.step
            print(f"previous run had {self.step} steps, running for additional {int(steps)} steps!")
        else:
            print(f"running for {int(steps)} steps!")

        try:
            self._run(int(steps))

            if self.trajectory_file is not None:
                finished = Path(self.trajectory_file).parent / "finished"

                if not finished.exists():
                    with open(finished, "w+"):
                        pass

        except Exception as err:
            if self.step == 1:
                raise err
            print("The calculator finished early with error , marking as invalid")
            if self.trajectory_file is not None:
                inv = Path(self.trajectory_file).parent / "invalid"
                if not inv.exists():
                    with open(inv, "w+"):
                        pass

            raise err

        # self.trajectory_info =  self.trajectory_info._shrink_capacity()
        if self.trajectory_file is not None:
            self.trajectory_info.save(self.trajectory_file)

    @abstractmethod
    def _run(self, steps):
        raise NotImplementedError

    def get_trajectory(self) -> TrajectoryInfo:
        assert self.trajectory_info is not None

        return self.trajectory_info._shrink_capacity()

    def save_step(self, T=None, P=None, t=None, err=None, canonicalize=False):
        if canonicalize:
            sp = self.nl.canonicalized_sp(self.sp)
        else:
            sp = self.sp

        ti = TrajectoryInfo.create(
            positions=sp.coordinates,
            cell=sp.cell,
            e_pot=self.last_ener.energy,
            e_pot_gpos=self.last_ener.gpos,
            e_bias=self.last_bias.energy,
            e_bias_gpos=self.last_bias.gpos,
            e_pot_vtens=self.last_ener.vtens,
            e_bias_vtens=self.last_bias.vtens,
            cv=self.last_cv.cv,
            T=T,
            P=P,
            t=t,
            err=err,
        )

        if self.step == 1:
            str = f"{ 'step': ^10s}"
            str += f"|{ 'cons err': ^10s}"
            str += f"|{ 'e_pot[Kj/mol]': ^15s}"
            str += f"|{ 'e_bias[Kj/mol]': ^15s}"
            if ti._P is not None:
                str += f"|{'P[bar]': ^10s}"
            str += f"|{'T[K]': ^10s}|{'walltime[s]': ^11s}"
            ss = "|\u2207\u2093U\u1D47|[Kj/\u212B]"
            str += f"|{ ss  : ^13s}"
            str += f"|{' CV': ^10s}"
            print(str, sep="")
            print(f"{'='*len(str)}")

        if self.step % self.static_trajectory_info.screen_log == 0:
            str = f"{  self.step : >10d}"
            assert ti._err is not None
            assert ti._T is not None
            assert ti._e_pot is not None
            assert ti._e_bias is not None

            str += f"|{  ti._err[0] : >10.4f}"
            str += f"|{  ti._e_pot[0]  /kjmol : >15.8f}"
            str += f"|{  ti._e_bias[0] /kjmol : >15.8f}"
            if ti._P is not None:
                str += f" { ti._P[0]/bar : >10.2f}"
            str += f" { ti._T[0] : >10.2f} { time()-self.time0 : >11.2f}"
            str += f"|{  jnp.max(jnp.linalg.norm(ti._e_bias_gpos,axis=1) /kjmol*angstrom ) : >13.2f}"
            str += f"| {ti._cv[0,:]}"
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

    def get_energy(self, gpos: bool = False, vtens: bool = False) -> EnergyResult:
        return self.energy.compute_from_system_params(
            gpos=gpos,
            vir=vtens,
            sp=self.sp,
            nl=self.nl,
        )

    def get_bias(
        self,
        gpos: bool = False,
        vtens: bool = False,
    ) -> tuple[CV, EnergyResult]:
        cv, ener = self.bias.compute_from_system_params(
            sp=self.sp,
            nl=self.nl,
            gpos=gpos,
            vir=vtens,
        )

        return cv, ener

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

            self.__class__.__init__(self, **statedict)

            if self.trajectory_file is not None:
                if Path(self.trajectory_file).exists():
                    self.step = self.trajectory_info._size
                    self.sp = self.trajectory_info.sp[-1]

            self.time0 = time()

        except Exception as e:
            print(
                f"tried to initialize {self.__class__} with from {statedict=} {f'{removed=}' if len(removed) == 0  else ''} but got exception",
            )
            raise e

        self._nl = None
