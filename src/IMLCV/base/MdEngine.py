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

_static_attr = [
    "timestep",
    "save_step",
    "r_cut",
    "r_skin",
    "timecon_thermo",
    "T",
    "P",
    "timecon_baro",
    "write_step",
    "equilibration",
    "screen_log",
    "max_grad",
    "frac_full",
    "invalid",
]

_static_arr = [
    "atomic_numbers",
]


class StaticMdInfo(MyPyTreeNode):
    timestep: float

    T: float
    timecon_thermo: float

    atomic_numbers: Array

    r_cut: float | None = None
    r_skin: float | None = None
    P: float | None = None
    timecon_baro: float | None = None

    write_step: int = 100
    equilibration: float | None = None
    screen_log: int = 1000
    save_step: int = 10

    invalid: bool = False

    max_grad: float | None = 200 * kjmol / angstrom

    frac_full: float = 1.0

    @property
    def masses(self):
        return jnp.array([atomic_masses[int(n)] for n in self.atomic_numbers]) * amu

    @property
    def thermostat(self):
        return self.T is not None

    @property
    def barostat(self):
        return self.P is not None

    # @property
    def neighbour_list_info(self, r_cut=None, r_skin=None) -> NeighbourListInfo | None:
        if r_cut is None:
            r_cut = self.r_cut

        if r_cut is None:
            return None

        if r_skin is None:
            r_skin = self.r_skin

        return NeighbourListInfo.create(
            r_cut=r_cut,
            z_array=self.atomic_numbers,
            r_skin=r_skin,
        )

    def __post_init__(self):
        if self.thermostat:
            assert self.timecon_thermo is not None

        if self.barostat:
            assert self.timecon_baro is not None

    def _save(self, hf: h5py.File):
        for name in _static_arr:
            prop = self.__getattribute__(name)
            if prop is not None:
                if name in hf:
                    del hf[name]

                hf[name] = prop

        for name in _static_attr:
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
            try:
                props_static[key] = val[:]
            except Exception as e:
                print(f"could not load {key=}")

        for key, val in hf.attrs.items():
            try:
                attrs_static[key] = val
            except Exception as e:
                print(f"could not load {key=}")

        return StaticMdInfo(**attrs_static, **props_static)

    @staticmethod
    def load(filename) -> StaticMdInfo:
        with h5py.File(str(filename), "r") as hf:
            return StaticMdInfo._load(hf=hf)


# static values
_items_scal = [
    "_t",
    "_e_pot",
    "_e_bias",
    "_T",
    "_P",
    "_err",
    "_w",
    "_w_t",
    "_rho",
    "_rho_t",
    "_sigma",
]
_items_vec = [
    "_positions",
    "_positions_t",
    "_cell",
    "_cell_t",
    "_charges",
    "_cv",
    "_cv_t",
    "_cv_orig",
]
_items_attr = [
    "_finished",
    "_invalid",
    "_size",
    "_capacity",
    "_prev_save",
]


class TrajectoryInfo(MyPyTreeNode, ABC):
    # _size: int = field(pytree_node=False, default=-1)

    @abstractmethod
    def _get(self, prop_name: str):
        pass

    @abstractmethod
    def _set(self, prop_name: str, value):
        pass

    @abstractmethod
    def __getitem__(self, slices) -> TrajectoryInfo:
        pass

    @property
    def shape(self):
        return self.size

    @property
    def volume(self):
        if self.cell is not None:
            vol_unsigned = jnp.linalg.det(self.cell)
            if (vol_unsigned < 0).any():
                print("cell volume was negative")

            return jnp.abs(vol_unsigned)
        return None

    @property
    def sp(self) -> SystemParams | None:
        if self.positions is None:
            return None

        return SystemParams(
            coordinates=self.positions,
            cell=self.cell,
        )

    @sp.setter
    def sp(self, value: SystemParams):
        self.positions = value.coordinates
        self.cell = value.cell

    @property
    def sp_t(self) -> SystemParams | None:
        if self.positions_t is None:
            return None

        return SystemParams(
            coordinates=self.positions_t,
            cell=self.cell_t,
        )

    @sp_t.setter
    def sp_t(self, value: SystemParams):
        self.positions_t = value.coordinates
        self.cell_t = value.cell

    @property
    def CV(self) -> CV | None:
        if self.cv is not None:
            return CV(cv=self.cv)
        return None

    @CV.setter
    def CV(self, value: CV):
        self.cv = value.cv

    @property
    def CV_t(self) -> CV | None:
        if self.cv is not None:
            return CV(cv=self.cv)
        return None

    @CV_t.setter
    def CV_t(self, value: CV):
        self.cv_t = value.cv

    @property
    def CV_orig(self) -> CV | None:
        if self.cv_orig is not None:
            return CV(cv=self.cv_orig)
        return None

    @CV_orig.setter
    def CV_orig(self, value: CV):
        self.cv_orig = value.cv

    @property
    def positions(self) -> Array | None:
        return self._get("_positions")

    @positions.setter
    def positions(self, value: Array) -> Array | None:
        self._set("_positions", value)

    @property
    def positions_t(self) -> Array | None:
        return self._get("_positions_t")

    @positions_t.setter
    def positions_t(self, value: Array):
        self._set("_positions_t", value)

    @property
    def cell(self) -> Array | None:
        return self._get("_cell")

    @cell.setter
    def cell(self, value: Array):
        self._set("_cell", value)

    @property
    def cell_t(self) -> Array | None:
        return self._get("_cell_t")

    @cell_t.setter
    def cell_t(self, value: Array):
        self._set("_cell_t", value)

    @property
    def charges(self):
        return self._get("_charges")

    @charges.setter
    def charges(self, value: Array):
        self._set("_charges", value)

    @property
    def e_pot(self) -> Array | None:
        return self._get("_e_pot")

    @e_pot.setter
    def e_pot(self, value: Array):
        self._set("_e_pot", value)

    @property
    def e_bias(self) -> Array | None:
        return self._get("_e_bias")

    @e_bias.setter
    def e_bias(self, value: Array):
        self._set("_e_bias", value)

    @property
    def w(self) -> Array | None:
        return self._get("_w")

    @w.setter
    def w(self, value: Array):
        self._set("_w", value)

    @property
    def w_t(self) -> Array | None:
        return self._get("_w_t")

    @w_t.setter
    def w_t(self, value: Array):
        self._set("_w_t", value)

    @property
    def rho(self) -> Array | None:
        return self._get("_rho")

    @rho.setter
    def rho(self, value: Array):
        self._set("_rho", value)

    @property
    def rho_t(self) -> Array | None:
        return self._get("_rho_t")

    @rho_t.setter
    def rho_t(self, value: Array):
        self._set("_rho_t", value)

    @property
    def sigma(self) -> Array | None:
        return self._get("_sigma")

    @sigma.setter
    def sigma(self, value: Array):
        self._set("_sigma", value)

    @property
    def cv(self) -> Array | None:
        return self._get("_cv")

    @cv.setter
    def cv(self, value: Array):
        self._set("_cv", value)

    @property
    def cv_t(self) -> Array | None:
        return self._get("_cv_t")

    @cv_t.setter
    def cv_t(self, value: Array):
        self._set("_cv_t", value)

    @property
    def T(self) -> Array | None:
        return self._get("_T")

    @property
    def P(self) -> Array | None:
        return self._get("_P")

    @property
    def err(self) -> Array | None:
        return self._get("_err")

    @property
    def t(self) -> Array | None:
        return self._get("_t")

    @t.setter
    def t(self, value: Array):
        self._set("_t", value)

    @property
    def finished(self) -> bool:
        return self._get("_finished")

    @finished.setter
    def finished(self, value: bool):
        self._set("_finished", value)

    @property
    def invalid(self) -> bool:
        return self._get("_invalid")

    @invalid.setter
    def invalid(self, value: bool):
        self._set("_invalid", value)

    @property
    def size(self) -> int:
        return self._get("_size")

    @size.setter
    def size(self, value: int):
        self._set("_size", value)

    @property
    def capacity(self):
        return self._get("_capacity")

    @capacity.setter
    def capacity(self, value: int):
        self._set("_capacity", value)


class FullTrajectoryInfo(TrajectoryInfo):
    _positions: Array
    _positions_t: Array | None = None

    _cell: Array | None = None
    _cell_t: Array | None = None

    _charges: Array | None = None

    _e_pot: Array | None = None
    _e_bias: Array | None = None

    _w: Array | None = None
    _w_t: Array | None = None
    _rho: Array | None = None
    _rho_t: Array | None = None

    _sigma: Array | None = None

    _cv: Array | None = None
    _cv_t: Array | None = None

    _cv_orig: Array | None = None  # usefull to reconstruct CV discovery

    _T: Array | None = None
    _P: Array | None = None
    _err: Array | None = None

    _t: Array | None = None

    _capacity: int = field(pytree_node=False, default=-1)
    _size: int = field(pytree_node=False, default=-1)
    _prev_save: int = field(pytree_node=False, default=0)

    _finished: int = field(pytree_node=False, default=False)
    _invalid: int = field(pytree_node=False, default=False)

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
        w_t: Array | None = None,
        rho: Array | None = None,
        sigma: Array | None = None,
        T: Array | None = None,
        P: Array | None = None,
        err: Array | None = None,
        t: Array | None = None,
        capacity: int = -1,
        size: int = -1,
        finished=False,
        invalid=False,
    ) -> FullTrajectoryInfo:
        dict = {
            "_positions": positions,
            "_positions_t": None,
            "_cell": cell,
            "_cell_t": None,
            "_charges": charges,
            "_e_pot": e_pot,
            "_e_bias": e_bias,
            "_cv": cv,
            "_cv_t": None,
            "_cv_orig": cv_orig,
            "_w": w,
            "_w_t": w_t,
            "_rho": rho,
            "_rho_t": None,
            "_sigma": sigma,
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
            # print("adding batch dimension to trajectory info")
            for name in [*_items_vec, *_items_scal]:
                prop = dict[name]
                if prop is not None:
                    dict[name] = jnp.expand_dims(prop, 0)  # jnp.array([prop])

        # test wether cell is truly not None
        if dict["_cell"] is not None:
            if dict["_cell"].shape[-2] == 0:
                dict["_cell"] = None

        if capacity == -1:
            dict["_capacity"] = dict["_positions"].shape[0]

        if size == -1:
            dict["_size"] = dict["_positions"].shape[0]

        return FullTrajectoryInfo(**dict)

    def _get(self, prop_name: str):
        prop = self.__getattribute__(prop_name)
        if prop is None:
            return None

        if prop_name in _items_attr:
            return prop

        if prop_name in _items_scal:
            return prop[0 : self.size]  # type:ignore

        if prop_name in _items_vec:
            return prop[0 : self.size, :]  # type:ignore

        raise ValueError(f"property {prop_name} not found in trajectory info")

    def _set(self, prop_name: str, value: Array):
        # print(f"setting trajectory info property {prop_name} ")

        if prop_name in _items_attr:
            self.__dict__[prop_name] = value
            return

        if self._get(prop_name) is None:
            if prop_name in _items_scal:
                self.__dict__[prop_name] = jnp.zeros((self._capacity,))  # type:ignore
            elif prop_name in _items_vec:
                self.__dict__[prop_name] = jnp.zeros((self._capacity, *value.shape[1:]))  # type:ignore

        self.__dict__[prop_name] = dynamic_update_slice_in_dim(
            self.__dict__[prop_name],
            value,
            0,
            0,
        )

    def __getitem__(self, slices) -> FullTrajectoryInfo:
        "gets slice from indices. the output is truncated to the to include only items wihtin _size"

        slz = (jnp.ones(self._capacity, dtype=jnp.int64).cumsum() - 1)[slices]
        slz = slz[slz <= self._size]

        new_dict = {}

        for name in _items_scal:
            prop = self.__dict__[name]
            if prop is not None:
                new_dict[name] = prop[slz,]

        for name in _items_vec:
            prop = self.__dict__[name]
            if prop is not None:
                new_dict[name] = prop[slz, :]

        new_dict["_capacity"] = int(jnp.size(slz))
        new_dict["_size"] = int(jnp.size(slz))

        return FullTrajectoryInfo(**new_dict)

    def _stack(self, *ti: FullTrajectoryInfo):
        tot_size = self._size + sum([t._size for t in ti])

        if self._capacity <= tot_size:
            self = self._expand_capacity(nc=tot_size * 1.4)

        index = self._size

        keys = [*_items_vec, *_items_scal]

        for tii in ti:
            _s = tii._size

            for key in keys:
                if self.__dict__[key] is not None:
                    self.__dict__[key] = dynamic_update_slice_in_dim(
                        self.__dict__[key],
                        dynamic_slice_in_dim(tii.__dict__[key], 0, _s),
                        index,
                        0,
                    )

            index += _s

        self._size = index

        return self

    def __add__(self, ti: FullTrajectoryInfo) -> FullTrajectoryInfo:
        return self._stack(ti)

    def _expand_capacity(self, nc=None) -> FullTrajectoryInfo:
        if nc is None:
            nc = min(self._capacity * 2, self._capacity + 1000)

        nc = int(nc)

        print(f"expanding capacity from {self._capacity} to {nc}")

        delta = nc - self._capacity

        dict = {
            "_capacity": int(nc),
            "_size": int(self._size),
            "_prev_save": int(self._size),
        }

        for name in _items_vec:
            prop = self.__dict__[name]

            if prop is not None:
                dict[name] = jnp.vstack((prop, jnp.zeros((delta, *prop.shape[1:]))))  # type:ignore

        for name in _items_scal:
            prop = self.__dict__[name]
            if prop is not None:
                dict[name] = jnp.hstack([prop, jnp.zeros(delta)])  # type:ignore

        print(f"new capacity is {dict['_capacity']} ")

        return FullTrajectoryInfo(**dict)  # type:ignore

    def _shrink_capacity(self) -> FullTrajectoryInfo:
        dict = {}

        print(f"shrinking capacity from {self._capacity} to {self._size} ")

        for name in _items_vec:
            prop = self.__getattribute__(name)
            if prop is not None:
                dict[name] = prop[: self._size, :]

        for name in _items_scal:
            prop = self.__getattribute__(name)
            if prop is not None:
                dict[name] = prop[: self._size]

        dict["_capacity"] = int(self._size)
        dict["_prev_save"] = int(self._size)
        dict["_size"] = int(self._size)
        dict["_finished"] = int(self._finished)

        return FullTrajectoryInfo(**dict)

    def save(self, filename: str | Path):
        if isinstance(filename, str):
            filename = Path(filename)

        if not filename.parent.exists():
            filename.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(str(filename), "r+" if filename.exists() else "w") as hf:
            self._save(hf=hf)

    def _save(self, hf: h5py.File):
        print(f"saving trajectory info to {hf.filename}, size {self._size}/{self._capacity}")

        for name in [*_items_scal, *_items_vec]:
            prop = self.__getattribute__(name)
            if prop is not None:
                # if name in hf:
                #     del hf[name]  # might be different size

                if name not in hf:
                    # print(f"creating {name} with size {self._capacity} ")

                    hf[name] = prop.__array__()

                else:
                    # check if size changed
                    if hf[name].shape[0] != self._capacity:  # type:ignore
                        del hf[name]

                        hf[name] = prop.__array__()
                    else:
                        hf[name][self._prev_save : self._size] = prop[  # type:ignore
                            self._prev_save : self._size
                        ].__array__()  # save as numpy array, only the changed part

        self._prev_save = self._size

        hf.attrs.create("_capacity", self._capacity)
        hf.attrs.create("_size", self._size)
        hf.attrs.create("_finished", self._finished)
        hf.attrs.create("_invalid", self._invalid)
        hf.attrs.create("_prev_save", self._size)

    @staticmethod
    def load(filename) -> FullTrajectoryInfo:
        with h5py.File(str(filename), "r") as hf:
            return FullTrajectoryInfo._load(hf=hf)

    @staticmethod
    def _load(hf: h5py.File):
        props = {}
        attrs = {}

        for key, val in hf.items():
            if (key not in _items_scal) and (key not in _items_vec):
                # print(f"{key=} deprecated, ignoring")
                continue

            props[key] = jnp.array(val[:])

        for key, val in hf.attrs.items():
            attrs[key] = val

        return FullTrajectoryInfo(
            **props,
            **attrs,
        )


class EagerTrajectoryInfo(TrajectoryInfo):
    """Loads trajectory info from file on demand, only the requested properties."""

    path: Path = field(pytree_node=False)
    indices: Array
    overide_dict: dict = field(pytree_node=False, default_factory=dict)

    @staticmethod
    def create(file_path) -> EagerTrajectoryInfo:
        with h5py.File(str(file_path), "r") as hf:
            size = hf.attrs["_size"]

        return EagerTrajectoryInfo(
            path=Path(file_path),
            indices=jnp.arange(size),
        )

    def to_full(self) -> FullTrajectoryInfo:
        ti = {}
        for name in _items_scal:
            ti[name] = self._get(name)

        for name in _items_vec:
            ti[name] = self._get(name)

        for name in _items_attr:
            ti[name] = self._get(name)

        out = FullTrajectoryInfo(**ti)

        return out

    def _get(self, prop_name: str):
        if prop_name == "_size":
            return self.size

        if prop_name == "_capacity":
            return self.size

        if prop_name in self.overide_dict:
            thing = self.overide_dict[prop_name]
            return thing

        with h5py.File(str(self.path), "r") as hf:
            if prop_name in _items_attr:
                thing = hf.attrs[prop_name]
            elif prop_name in hf.keys():
                thing = jnp.array(hf[prop_name])
            else:
                return None

        if prop_name in _items_attr:
            self._set(prop_name, thing)  # cache attribute
            return thing

        if prop_name in _items_scal:
            return thing[self.indices]

        if prop_name in _items_vec:
            return thing[self.indices, :]

        raise ValueError(f"property {prop_name} not found in trajectory info")

    def _set(self, prop_name: str, value: Array):
        self.overide_dict[prop_name] = value

    def __getitem__(self, slices) -> EagerTrajectoryInfo:
        new_overide = {}

        for key in self.overide_dict.keys():
            if key in _items_scal:
                new_overide[key] = self.overide_dict[key][slices]
            elif key in _items_vec:
                new_overide[key] = self.overide_dict[key][slices, :]
            else:
                new_overide[key] = self.overide_dict[key]

        return EagerTrajectoryInfo(
            path=self.path,
            indices=self.indices[slices],
            overide_dict=new_overide,
        )

    @property
    def size(self):
        return self.indices.shape[0]


######################################
#             MDEngine               #
######################################


class MDEngine(MyPyTreeNode, ABC):
    """Base class for MD engine."""

    bias: Bias
    permanent_bias: Bias | None = field(default=None)
    energy: Energy
    sp: SystemParams
    static_trajectory_info: StaticMdInfo
    trajectory_info: FullTrajectoryInfo | None = field(default=None)
    trajectory_file: Path | None = field(pytree_node=False, default=None)
    time0: float = field(default_factory=time)

    step: int = 1

    nl: NeighbourList | None = None
    # r_skin = 1.0 * angstrom

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
                trajectory_info = FullTrajectoryInfo.load(trajectory_file)
                cont = True

        if not cont or trajectory_info is None:
            create_kwargs["step"] = 1
            if sp is not None:
                energy.sp = sp

        else:
            create_kwargs["step"] = trajectory_info.size
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
            info = self.static_trajectory_info.neighbour_list_info()
            assert info is not None

            self.nl = self.sp.get_neighbour_list(info)  # jitted update
            return

        self.nl = self.nl.slow_update_nl(self.sp)

    def save(self, file):
        filename = Path(file)

        b = self.bias
        pb = self.permanent_bias
        self.bias = None
        self.permanent_bias = None

        if filename.suffix == ".json":
            with open(filename, "w") as f:
                f.writelines(jsonpickle.encode(self, indent=1, use_base85=True))  # type:ignore
        else:
            with open(filename, "wb") as f:
                cloudpickle.dump(self, f)

        self.bias = b
        self.permanent_bias = pb

    @staticmethod
    def load(file, bias: Bias, permant_bias: Bias | None = None, **kwargs) -> MDEngine:
        filename = Path(file)

        if filename.suffix == ".json":
            with open(filename) as f:
                self = jsonpickle.decode(f.read(), context=unpickler)
        else:
            with open(filename, "rb") as f:
                self = cloudpickle.load(f)

        assert isinstance(self, MDEngine), f"{self=}"

        self.bias = bias
        self.permanent_bias = permant_bias

        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

            if key == "trajectory_file":
                continue

        assert isinstance(self, MDEngine), f"{self=}"

        if (key := "trajectory_file") in kwargs.keys():
            print(f"loading ti  {self.trajectory_file} ")
            trajectory_file = kwargs[key]
            if Path(trajectory_file).exists():
                print("updating sp from trajectory file")

                self.trajectory_info = FullTrajectoryInfo.load(self.trajectory_file)
                self.step = self.trajectory_info.size * self.static_trajectory_info.save_step
                self.sp = self.trajectory_info.sp[-1]

                print(f"loaded ti  {self.step=} ")

        # else:
        #     print(f"{self.sp=}")

        self.time0 = time()

        # self.update_nl()

        return self

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

        if steps <= 0:
            print("No steps to run, exiting")
            return

        try:
            for _ in range(int(steps)):
                self._step()

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
    def _step(self):
        raise NotImplementedError

    def get_trajectory(self) -> FullTrajectoryInfo:
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
        sp: SystemParams = None,
        canonicalize=False,
    ):
        screen_log = self.step % self.static_trajectory_info.screen_log == 0
        save_step = self.step % self.static_trajectory_info.save_step == 0
        write_step = self.step % self.static_trajectory_info.write_step == 0

        if screen_log or save_step or write_step:
            # print(f"{self.step=}  {screen_log=} or {save_step=} or {write_step=}")

            ti = FullTrajectoryInfo.create(
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
                # if gpos_rmsd is not None:
                #     ss = "|\u2207\u2093U\u1d47|[Kj/\u212b]"
                #     str += f"|{ss: ^13s}"
                # if gpos_bias_rmsd is not None:
                #     ss = "|\u2207\u2093U\u1d47|[Kj/\u212b]"
                #     str += f"|{ss: ^13s}"

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
                # if gpos_rmsd is not None:
                #     str += f"|{(gpos_rmsd / kjmol * angstrom): >13.2f}"
                # if gpos_bias_rmsd is not None:
                #     str += f"|{(gpos_bias_rmsd / kjmol * angstrom): >13.2f}"
                if ti._cv is not None:
                    str += f"| {ti._cv[0, :]}"
                print(str)

            if self.step % self.static_trajectory_info.save_step == 0:
                # write step to trajectory
                if self.trajectory_info is None:
                    self.trajectory_info = ti
                else:
                    self.trajectory_info += ti

            if self.step % self.static_trajectory_info.write_step == 0:
                if self.trajectory_file is not None:
                    self.trajectory_info.save(self.trajectory_file)  # type: ignore

            assert self.bias is not None

            # print(f"{jax.tree_util.tree_map(lambda x: x.shape,self.trajectory_info)=}")
            print(
                f"{self.trajectory_info.size=}  {self.trajectory_info.capacity=} {self.trajectory_info.positions.shape=}"
            )

            # print(f"done")

        self.bias = self.bias.update_bias(self)

        self.step += 1

    def get_energy(
        self,
        sp: SystemParams,
        nl: NeighbourList | None = None,
        gpos: bool = False,
        vtens: bool = False,
        manual_vir=None,
    ) -> EnergyResult:
        return self.energy.compute_from_system_params(
            gpos=gpos,
            vir=vtens,
            sp=sp,
            nl=nl,
            manual_vir=manual_vir,
        )

    def get_bias(
        self,
        sp: SystemParams,
        nl: NeighbourList | None = None,
        gpos: bool = False,
        vtens: bool = False,
        shmap: bool = False,
        use_jac=False,
        push_jac=False,
        rel=False,
        shmap_kwargs=ShmapKwargs.create(),
    ) -> tuple[CV, EnergyResult]:
        assert self.bias is not None

        cv, ener = self.bias.compute_from_system_params(
            sp=sp,
            nl=nl,
            gpos=gpos,
            vir=vtens,
            shmap=shmap,
            use_jac=use_jac,
            push_jac=push_jac,
            rel=rel,
            shmap_kwargs=shmap_kwargs,
        )

        if self.permanent_bias is not None:
            cv2, ener2 = self.permanent_bias.compute_from_system_params(
                sp=sp,
                nl=nl,
                gpos=gpos,
                vir=vtens,
                shmap=shmap,
                use_jac=use_jac,
                push_jac=push_jac,
                rel=rel,
                shmap_kwargs=shmap_kwargs,
            )

            ener = ener + ener2

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

                    self.step = self.trajectory_info.size * self.static_trajectory_info.save_step
                    self.sp = self.trajectory_info.sp[-1]

            self.time0 = time()

        except Exception as e:
            print(
                f"tried to initialize {self.__class__} with from {statedict=} {f'{removed=}' if len(removed) == 0 else ''} but got exception",
            )
            raise e

        # self.update_nl()
