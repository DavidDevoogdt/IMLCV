"""MD engine class peforms MD simulations in a given NVT/NPT ensemble.

Currently, the MD is done with YAFF/OpenMM
"""
from __future__ import annotations

import tempfile
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from time import time

import cloudpickle
import h5py
import jax.numpy as jnp
import numpy as np
import yaff.analysis.biased_sampling
import yaff.external
import yaff.log
import yaff.pes.bias
import yaff.pes.ext
import yaff.sampling.iterative
from IMLCV.base.bias import Bias
from IMLCV.base.bias import Energy
from IMLCV.base.bias import EnergyResult
from IMLCV.base.CV import CV
from IMLCV.base.CV import NeighbourList
from IMLCV.base.CV import SystemParams
from jax import Array
from molmod.periodic import periodic
from molmod.units import angstrom
from molmod.units import bar
from molmod.units import kjmol

yaff.log.set_level(yaff.log.silent)


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


@dataclass
class TrajectoryInfo:
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

    _capacity: int = -1
    _size: int = -1

    # https://stackoverflow.com/questions/7133885/fastest-way-to-grow-a-numpy-numeric-array
    def __post_init__(self):
        if self._capacity == -1:
            self._capacity = 1
        if self._size == -1:
            self._size = 1

        # batch
        if len(self._positions.shape) == 2:
            for name in [*self._items_vec, *self._items_scal]:
                prop = self.__getattribute__(name)
                if prop is not None:
                    self.__setattr__(name, np.array([prop]))

        # test wether cell is truly not None
        if self._cell is not None:
            if self._cell.shape[-2] == 0:
                self._cell = None

    def __getitem__(self, slices):
        "gets slice from indices. the output is truncated to the to include only items wihtin _size"

        slz = (jnp.ones(self._capacity).cumsum() - 1)[slices]
        ind = slz <= self._size
        # print(f"ind: {ind}, cap = {jnp.sum(ind)}, t : {self.t[slices][ind].shape}")

        return TrajectoryInfo(
            _positions=self._positions[slices, :][ind],
            _cell=self._cell[slices, :][ind] if self._cell is not None else None,
            _charges=(self._charges[slices, :][ind] if self._cell is not None else None)
            if self._charges is not None
            else None,
            _e_pot=self._e_pot[slices][ind] if self._e_pot is not None else None,
            _e_pot_gpos=self._e_pot_gpos[slices, :][ind] if self._e_pot_gpos is not None else None,
            _e_pot_vtens=self._e_pot_vtens[slices, :][ind] if self._e_pot_vtens is not None else None,
            _e_bias=self._e_bias[slices][ind] if self._e_bias is not None else None,
            _e_bias_gpos=self._e_bias_gpos[slices, :][ind] if self._e_bias_gpos is not None else None,
            _e_bias_vtens=self._e_bias_vtens[slices, :][ind] if self._e_bias_vtens is not None else None,
            _cv=self._cv[slices, :][ind] if self._cv is not None else None,
            _T=self._T[slices][ind] if self._T is not None else None,
            _P=self._P[slices][ind] if self._P is not None else None,
            _err=self._err[slices][ind] if self._err is not None else None,
            _t=self._t[slices][ind] if self._t is not None else None,
            _capacity=jnp.sum(ind),
            _size=jnp.sum(ind),
        )

    def __add__(self, ti: TrajectoryInfo):
        sz = ti._size

        while self._capacity <= self._size + ti._size:
            self._expand_capacity()

        for name in self._items_vec:
            prop_ti = ti.__getattribute__(name)
            prop_self = self.__getattribute__(name)
            if prop_ti is None:
                assert prop_self is None
            else:
                prop_self[self._size : self._size + sz, :] = prop_ti[0:sz, :]

        for name in self._items_scal:
            prop_ti = ti.__getattribute__(name)
            prop_self = self.__getattribute__(name)
            if prop_ti is None:
                assert prop_self is None
            else:
                prop_self[self._size : self._size + sz] = prop_ti[0:sz]

        self._size += sz

        return self

    def _expand_capacity(self):
        nc = min(self._capacity * 2, self._capacity + 10000)
        delta = nc - self._capacity
        self._capacity = nc

        for name in self._items_vec:
            prop = self.__getattribute__(name)
            if prop is not None:
                self.__setattr__(
                    name,
                    np.vstack((prop, np.zeros((delta, *prop.shape[1:])))),
                )
        for name in self._items_scal:
            prop = self.__getattribute__(name)
            if prop is not None:
                self.__setattr__(
                    name,
                    np.hstack([prop, np.zeros(delta)]),
                )

    def _shrink_capacity(self):
        for name in self._items_vec:
            prop = self.__getattribute__(name)
            if prop is not None:
                self.__setattr__(name, prop[: self._size, :])
        for name in self._items_scal:
            prop = self.__getattribute__(name)
            if prop is not None:
                self.__setattr__(name, prop[: self._size])
        self._capacity = self._size

    def save(self, filename: str | Path):
        self._shrink_capacity()

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
            # static_info=tic,
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
            return jnp.linalg.det(self._cell)
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


class MDEngine(ABC):
    """Base class for MD engine."""

    keys = [
        "bias",
        "energy",
        "static_trajectory_info",
        "trajectory_file",
        # "sp",
        # "sp",
    ]

    def __init__(
        self,
        bias: Bias,
        energy: Energy,
        static_trajectory_info: StaticMdInfo,
        trajectory_file=None,
        sp: SystemParams | None = None,
    ) -> None:
        self.static_trajectory_info = static_trajectory_info

        self.bias = bias
        self.energy = energy

        self.last_bias = EnergyResult(0)
        self.last_ener = EnergyResult(0)
        self.last_cv: CV | None = None

        # self._sp = sp
        self.trajectory_info: TrajectoryInfo | None = None

        self.step = 1
        if sp is not None:
            self.sp = sp
        self.trajectory_file = trajectory_file

        self.time0 = time()
        self._nl: NeighbourList | None = None

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

        def _nl():
            return self.sp.get_neighbour_list(
                r_cut=self.static_trajectory_info.r_cut,
                z_array=self.static_trajectory_info.atomic_numbers,
                r_skin=0.0,
            )

        if self._nl is None:
            nl = _nl()
        else:
            b, nl = self._nl.update(self.sp)  # jitted update

            if not b:
                nl = _nl()

        self._nl = nl
        return nl

    def save(self, file):
        with open(file, "wb") as f:
            cloudpickle.dump(self, f)

    def __getstate__(self):
        return {key: self.__getattribute__(key) for key in MDEngine.keys}

    def __setstate__(self, state):
        self.__init__(**state)
        return self

    @staticmethod
    def load(file, **kwargs) -> MDEngine:
        with open(file, "rb") as f:
            self = cloudpickle.load(f)

        print("Loading MD engine")
        for key in kwargs.keys():
            print(f"setting {key}={kwargs[key]}")

            self.__setattr__(key, kwargs[key])

        return self

    def new_bias(self, bias: Bias, **kwargs) -> MDEngine:
        with tempfile.NamedTemporaryFile() as tmp:
            self.save(tmp.name)
            mde = MDEngine.load(tmp.name, **{"bias": bias, **kwargs})
        return mde

    def run(self, steps):
        """run the integrator for a given number of steps.

        Args:
            steps: number of MD steps
        """

        print(f"running for {int(steps)} steps!")

        try:
            self._run(int(steps))
        except Exception as err:
            if self.step == 1:
                raise err
            print(f"The calculator finished early with error {err=},{type(err)=}")

        self.trajectory_info._shrink_capacity()
        if self.trajectory_file is not None:
            self.trajectory_info.save(self.trajectory_file)

    @abstractmethod
    def _run(self, steps):
        raise NotImplementedError

    def get_trajectory(self) -> TrajectoryInfo:
        assert self.trajectory_info is not None
        self.trajectory_info._shrink_capacity()
        return self.trajectory_info

    def save_step(self, T=None, P=None, t=None, err=None):
        ti = TrajectoryInfo(
            _positions=self.sp.coordinates,
            _cell=self.sp.cell,
            _e_pot=self.last_ener.energy,
            _e_pot_gpos=self.last_ener.gpos,
            _e_bias=self.last_bias.energy,
            _e_bias_gpos=self.last_bias.gpos,
            _e_pot_vtens=self.last_ener.vtens,
            _e_bias_vtens=self.last_bias.vtens,
            _cv=self.last_cv.cv,
            _T=T,
            _P=P,
            _t=t,
            _err=err,
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

        self.bias.update_bias(self)

        self.step += 1

    def get_energy(self, gpos: bool = False, vtens: bool = False) -> EnergyResult:
        return self.energy.compute_from_system_params(
            gpos,
            vtens,
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

    @property
    def yaff_system(self) -> MDEngine.YaffSys:
        return self.YaffSys(self.energy, self.static_trajectory_info)

    # definitons of different interfaces. These encode the state of the system in the format of a given md engine

    @dataclass
    class YaffSys:
        _ener: Energy
        _tic: StaticMdInfo

        @dataclass
        class YaffCell:
            _ener: Energy

            @property
            def rvecs(self):
                if self._ener.cell is None:
                    return np.zeros((0, 3))
                return np.array(self._ener.cell)

            @rvecs.setter
            def rvecs(self, rvecs):
                self._ener.cell = rvecs

            def update_rvecs(self, rvecs):
                self.rvecs = rvecs

            @property
            def nvec(self):
                return self.rvecs.shape[0]

            @property
            def volume(self):
                if self.nvec == 0:
                    return np.nan

                return np.linalg.det(self.rvecs)

        def __post_init__(self):
            self._cell = self.YaffCell(_ener=self._ener)

        @property
        def numbers(self):
            return self._tic.atomic_numbers

        @property
        def masses(self):
            return np.array(self._tic.masses)

        @property
        def charges(self):
            return None

        @property
        def cell(self):
            return self._cell

        @property
        def pos(self):
            return np.array(self._ener.coordinates)

        @pos.setter
        def pos(self, pos):
            self._ener.coordinates = pos

        @property
        def natom(self):
            return self.pos.shape[0]
