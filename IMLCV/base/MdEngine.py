"""MD engine class peforms MD simulations in a given NVT/NPT ensemble.

Currently, the MD is done with YAFF/OpenMM
"""
from __future__ import annotations

import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from time import time

import dill
import h5py
import jax.numpy as jnp
import numpy as np
from molmod.units import bar

import yaff.analysis.biased_sampling
import yaff.external
import yaff.log
import yaff.pes
import yaff.pes.bias
import yaff.pes.ext
import yaff.sampling
import yaff.sampling.iterative
from IMLCV.base.bias import Bias, Energy, EnergyError, EnergyResult
from IMLCV.base.CV import SystemParams
from yaff.log import log
from yaff.sampling.verlet import VerletIntegrator, VerletScreenLog


@dataclass
class StaticTrajectoryInfo:

    _attr = [
        "timestep",
        "timecon_thermo",
        "T",
        "P",
        "timecon_baro",
        "write_step",
        "equilibration",
        "screen_log",
    ]

    _arr = [
        "atomic_numbers",
        "masses",
    ]

    timestep: float

    T: float
    timecon_thermo: float

    atomic_numbers: np.ndarray
    masses: np.ndarray | None = None

    P: float | None = None
    timecon_baro: float | None = None

    write_step: int = 100
    equilibration: float | None = None
    screen_log: int = 1000

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

        if self.equilibration is None:
            self.equilibration = 200 * self.timestep

        if self.masses is None:
            from molmod.periodic import periodic

            self.masses = np.array([periodic[n].mass for n in self.atomic_numbers])

    def _save(self, hf: h5py.File):
        for name in self._arr:
            prop = self.__getattribute__(name)
            if prop is not None:
                hf[name] = prop

        for name in self._attr:
            prop = self.__getattribute__(name)
            if prop is not None:
                hf.attrs[name] = prop

    @staticmethod
    def _load(hf: h5py.File) -> StaticTrajectoryInfo:
        props_static = {}
        attrs_static = {}

        for key, val in hf.items():
            props_static[key] = val[:]

        for key, val in hf.attrs.items():
            attrs_static[key] = val

        return StaticTrajectoryInfo(**attrs_static, **props_static)


@dataclass
class TrajectoryInfo:

    positions: np.ndarray
    cell: np.ndarray | None = None
    charges: np.ndarray | None = None

    e_pot: np.ndarray | None = None
    gpos: np.ndarray | None = None
    vtens: np.ndarray | None = None

    T: np.ndarray | None = None
    P: np.ndarray | None = None
    err: np.ndarray | None = None

    t: np.ndarray | None = None

    # masses: np.ndarray | None = None

    # static_info: StaticTrajectoryInfo | None = None

    _items_scal = ["t", "e_pot", "T", "P", "err"]
    _items_vec = ["positions", "cell", "gpos", "vtens", "charges"]
    # _items_stat = ["static_info"]

    _capacity: int = -1
    _size: int = -1

    # https://stackoverflow.com/questions/7133885/fastest-way-to-grow-a-numpy-numeric-array
    def __post_init__(self):
        if self._capacity == -1:
            self._capacity = 1
        if self._size == -1:
            self._size = 1

        if self._size == 1:
            for name in [*self._items_vec, *self._items_scal]:
                prop = self.__getattribute__(name)
                if prop is not None:
                    self.__setattr__(name, np.array([prop]))

        # test wether cell is truly not None
        if self.cell is not None:
            if self.cell.shape[-2] == 0:
                self.cell = None

    def __add__(self, ti: TrajectoryInfo):

        assert ti._size == 1

        if self._size == self._capacity:
            self._expand_capacity()

        for name in self._items_vec:
            prop_ti = ti.__getattribute__(name)
            prop_self = self.__getattribute__(name)
            if prop_ti is None:
                assert prop_self is None
            else:
                prop_self[self._size - 1, :] = prop_ti[0]

        for name in self._items_scal:
            prop_ti = ti.__getattribute__(name)
            prop_self = self.__getattribute__(name)
            if prop_ti is None:
                assert prop_self is None
            else:
                prop_self[self._size - 1] = prop_ti[0]

        self._size += 1

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
                    np.hstack([prop, np.zeros((delta,))]),
                )

    def _shrink_capacity(self):
        for name in self._items_vec:
            prop = self.__getattribute__(name)
            if prop is not None:
                self.__setattr__(name, prop[: self._size - 1, :])
        for name in self._items_scal:
            prop = self.__getattribute__(name)
            if prop is not None:
                self.__setattr__(name, prop[: self._size - 1])
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

        # if self.static_info is not None:

        #     hf.create_group("static_info")
        #     self.static_info._save(hf=hf["static_info"])

    @staticmethod
    def load(filename) -> TrajectoryInfo:

        with h5py.File(str(filename), "r") as hf:
            return TrajectoryInfo._load(hf=hf)

    @staticmethod
    def _load(hf: h5py.File):
        props = {}
        attrs = {}

        tic = None

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
    def volume(self):
        if self.cell is not None:
            return jnp.linalg.det(self.cell)
        return None

    @property
    def sp(self) -> SystemParams:
        return SystemParams(
            coordinates=jnp.array(self.positions),
            cell=jnp.array(self.cell) if self.cell is not None else None,
        )


class MDEngine(ABC):
    """Base class for MD engine.

    Args:
        cvs: list of cvs
        T:  temp
        P:  press
        ES:  enhanced sampling method, choice = "None","MTD"
        timestep:  step verlet integrator
        timecon_thermo: thermostat time constant
        timecon_baro:  barostat time constant
    """

    keys = [
        "bias",
        "energy",
        "static_trajectory_info",
        "trajectory_file",
    ]

    def __init__(
        self,
        bias: Bias,
        energy: Energy,
        # sp: SystemParams,
        static_trajectory_info: StaticTrajectoryInfo,
        trajectory_file=None,
        sp: SystemParams | None = None,
    ) -> None:

        self.static_trajectory_info = static_trajectory_info

        self.bias = bias
        self.energy = energy

        # self._sp = sp
        self.trajectory_info: TrajectoryInfo | None = None

        self.step = 1
        if sp is not None:
            self.sp = sp
        self.trajectory_file = trajectory_file

        self.time0 = time()

    @property
    def sp(self) -> SystemParams:
        return self.energy.sp

    @sp.setter
    def sp(self, sp: SystemParams):
        self.energy.sp = sp

    def save(self, file):
        with open(file, "wb") as f:
            dill.dump(self, f)

    def __getstate__(self):
        return {key: self.__getattribute__(key) for key in MDEngine.keys}

    def __setstate__(self, state):
        self.__init__(**state)
        return self

    @staticmethod
    def load(file, **kwargs) -> MDEngine:

        with open(file, "rb") as f:
            self = dill.load(f)

        # replace and add kwargs
        for key in kwargs.keys():
            self.__dict__[key] = kwargs[key]

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
        except EnergyError as e:
            print(f"The calculator finished early with error {e}")

        print("saving the trajectory")

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

    def hook(self, ti: TrajectoryInfo):

        if self.step == 1:

            str = f"{ 'cons err': ^10s}"
            if ti.P is not None:
                str += f"|{'P[bar]': ^10s}"
            str += f"|{'T[K]': ^10s}|{'walltime[s]': ^10s}"
            print(str, sep="")
            print(f"{'='*len(str)}")
        else:
            str = f"{  ti.err[0] :>6.4f}"
            if ti.P is not None:
                str += f"|{ ti.P[0]/bar :>8.2f}"
            str += f"|{ ti.T[0] :>8.2f}|{ time()-self.time0 :>8.2f}"
            print(str)

        # write step to trajectory
        if self.trajectory_info is None:
            self.trajectory_info = ti
        else:
            self.trajectory_info += ti

        if self.step % self.static_trajectory_info.write_step == 0:
            if self.trajectory_file is not None:
                self.trajectory_info.save(self.trajectory_file)  # type: ignore

        self.bias.update_bias(self.sp)

        self.step += 1


class YaffEngine(MDEngine, yaff.sampling.iterative.Hook):
    """MD engine with YAFF as backend.

    Args:
        ff (yaff.pes.ForceField)
    """

    def __init__(
        self,
        bias: Bias,
        static_trajectory_info: StaticTrajectoryInfo,
        energy: Energy,
        # sp: SystemParams|None = None,
        # log_level=log.medium,
        trajectory_file=None,
        sp: SystemParams | None = None,
    ) -> None:

        yaff.log.set_level(log.warning)
        self.start = 0
        # self.step = 1
        self.name = "YaffEngineIMLCV"

        self._verlet: yaff.sampling.VerletIntegrator | None = None
        self._yaff_ener: YaffEngine._YaffFF | None = None

        super().__init__(
            energy=energy,
            bias=bias,
            static_trajectory_info=static_trajectory_info,
            trajectory_file=trajectory_file,
            sp=sp,
        )

    def __call__(self, iterative: VerletIntegrator):

        kwargs = dict(
            positions=iterative.pos,
            cell=iterative.rvecs,
            gpos=iterative.gpos,
            t=iterative.time,
            e_pot=iterative.epot,
            vtens=iterative.vtens,
            T=iterative.temp,
            err=iterative.cons_err,
        )
        if hasattr(iterative, "press"):
            kwargs["P"] = iterative.press

        self.hook(TrajectoryInfo(**kwargs))

    def _setup_verlet(self):

        hooks = [self, VerletScreenLog(step=self.static_trajectory_info.screen_log)]

        self._yaff_ener = YaffEngine._YaffFF(
            _energy=self.energy,
            _bias=self.bias,
            tic=self.static_trajectory_info,
        )

        if self.static_trajectory_info.thermostat:
            hooks.append(
                # yaff.sampling.NHCThermostat(
                #     self.static_trajectory_info.T,
                #     timecon=self.static_trajectory_info.timecon_thermo,
                # )
                yaff.sampling.LangevinThermostat(
                    self.static_trajectory_info.T,
                    timecon=self.static_trajectory_info.timecon_thermo,
                )
            )
        if self.static_trajectory_info.barostat:
            hooks.append(
                yaff.sampling.MTKBarostat(
                    self._yaff_ener,
                    self.static_trajectory_info.T,
                    self.static_trajectory_info.P,
                    timecon=self.static_trajectory_info.timecon_baro,
                    anisotropic=True,
                )
            )

        self._verlet = yaff.sampling.VerletIntegrator(
            self._yaff_ener,
            self.static_trajectory_info.timestep,
            temp0=self.static_trajectory_info.T,
            hooks=hooks,
        )

    @staticmethod
    def load(file, **kwargs) -> MDEngine:
        return super().load(file, **kwargs)

    def _run(self, steps):
        if self._verlet is None:
            self._setup_verlet()
        self._verlet.run(int(steps))

    @dataclass
    class _yaffCell:
        _ener: Energy

        @property
        def rvecs(self):
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

    @dataclass
    class _yaffSys:
        _ener: Energy
        _tic: StaticTrajectoryInfo

        # charges: np.ndarray | None = None

        def __post_init__(self):
            self._cell = YaffEngine._yaffCell(_ener=self._ener)

        @property
        def numbers(self):
            return self._tic.atomic_numbers

        @property
        def masses(self):
            return self._tic.masses

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

    class _YaffFF(yaff.pes.ForceField):
        def __init__(
            self,
            _energy: Energy,  # name clash with yaff.pes.ForceField
            _bias: Bias,
            tic: StaticTrajectoryInfo,
            name="IMLCV_YAFF_forcepart",
        ):

            self._sys = YaffEngine._yaffSys(_ener=_energy, _tic=tic)
            self._energy = _energy
            self.bias = _bias

            super().__init__(system=self.system, parts=[])

        @property
        def system(self):
            return self._sys

        @system.setter
        def system(self, sys):
            assert sys == self.system

        @property
        def sp(self):
            return self._energy.sp

        def update_rvecs(self, rvecs):
            self.clear()
            self.system.cell.rvecs = rvecs

        def update_pos(self, pos):
            self.clear()
            self.system.pos = pos

        def _internal_compute(self, gpos, vtens):

            ener = self._energy.compute_coor(
                gpos is not None,
                vtens is not None,
            )

            ener_bias: EnergyResult = self.bias.compute_coor(
                self.sp,
                gpos is not None,
                vtens is not None,
            )

            # arr = []
            # arr.append(["energy [angstrom]", ener.energy, ener_bias.energy])
            # if gpos is not None:
            #     arr.append(
            #         [
            #             "|gpos| [eV/angstrom]",
            #             jnp.linalg.norm(ener.gpos),
            #             jnp.linalg.norm(ener_bias.gpos),
            #         ]
            #     )

            # if vtens is not None:
            #     arr.append(
            #         [
            #             "P [bar]",
            #             ener.vtens / self.system.cell.volume / bar,
            #             ener_bias.vtens / self.system.cell.volume / bar,
            #         ]
            #     )

            # print(tabulate(arr, headers=["", "Energy", "Bias"]))

            total_energy = ener + ener_bias

            if gpos is not None:
                gpos[:] += np.array(total_energy.gpos)
            if vtens is not None:
                vtens[:] += np.array(total_energy.vtens)

            return total_energy.energy
