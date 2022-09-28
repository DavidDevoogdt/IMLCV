"""MD engine class peforms MD simulations in a given NVT/NPT ensemble.

Currently, the MD is done with YAFF/OpenMM
"""
from __future__ import annotations

import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import ase
import dill
import h5py
import jax.numpy as jnp
import numpy as np
from molmod.units import angstrom, electronvolt

import yaff.analysis.biased_sampling
import yaff.external
import yaff.log
import yaff.pes
import yaff.pes.bias
import yaff.pes.ext
import yaff.sampling
import yaff.sampling.iterative
from IMLCV.base.bias import Bias, Energy
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

    t: np.ndarray | None = None

    # masses: np.ndarray | None = None

    # static_info: StaticTrajectoryInfo | None = None

    _items_scal = ["t", "e_pot"]
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
        "screenlog",
        # "_sp",
        # "step",
    ]

    def __init__(
        self,
        bias: Bias,
        energy: Energy,
        # sp: SystemParams,
        static_trajectory_info: StaticTrajectoryInfo,
        trajectory_file=None,
        screenlog=1000,
    ) -> None:

        self.static_trajectory_info = static_trajectory_info

        self.bias = bias
        self.energy = energy

        self.screenlog = screenlog

        self._sp = energy.get_sp()
        self.trajectory_info: TrajectoryInfo | None = None

        self.step = 1

        self.trajectory_file = trajectory_file

    @property
    def sp(self):
        return self._sp

    @sp.setter
    def sp(self, sp):
        self._sp = sp

    def save(self, file):
        with open(file, "wb") as f:
            dill.dump(self, f)

    def __getstate__(self):
        return [self.__class__, {key: self.__dict__[key] for key in MDEngine.keys}]

    def __setstate__(self, state):
        cls, d = state
        return cls(**d)

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

    @abstractmethod
    def run(self, steps):
        """run the integrator for a given number of steps.

        Args:
            steps: number of MD steps
        """
        raise NotImplementedError

    def get_trajectory(self) -> TrajectoryInfo:
        assert self.trajectory_info is not None
        self.trajectory_info._shrink_capacity()
        return self.trajectory_info

    def hook(self, ti: TrajectoryInfo):

        # write step to trajectory
        if self.trajectory_info is None:
            self.trajectory_info = ti
        else:
            self.trajectory_info += ti

        if self.step % self.static_trajectory_info.write_step == 1:
            if self.trajectory_file is not None:
                self.trajectory_info.save(self.trajectory_file)  # type: ignore

        self.bias.update_bias(self.sp)

        self.step += 1

    def to_ASE_traj(self) -> ase.Atoms:
        traj = self.get_trajectory()

        # assert traj.static_info is not None

        pos_A = traj.positions / angstrom
        pbc = traj.cell is not None
        if pbc:
            cell_A = traj.cell / angstrom
            vol_A3 = traj.volume / angstrom**3
            vtens_eV = traj.vtens / electronvolt
            stresses_eVA3 = vtens_eV / vol_A3

            atoms = ase.Atoms(
                masses=self.static_trajectory_info.masses,
                positions=pos_A,
                pbc=pbc,
                cell=cell_A,
            )
            atoms.info["stress"] = stresses_eVA3
        else:
            atoms = ase.Atoms(
                masses=self.static_trajectory_info.masses,
                positions=pos_A,
            )

        if traj.gpos is not None:
            atoms.arrays["forces"] = -traj.gpos * angstrom / electronvolt
        if traj.e_pot is not None:
            atoms.info["energy"] = traj.e_pot / electronvolt

        return atoms

        # "bias",
        # "energy",
        # "static_trajectory_info",
        # "trajectory_file",
        # "screenlog",


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
        screenlog=1000,
    ) -> None:

        yaff.log.set_level(log.medium)
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
            screenlog=screenlog,
        )

    @property
    def sp(self):
        if self._yaff_ener is not None:
            return self._yaff_ener.sp
        return self._sp

    @sp.setter
    def sp(self, sp):
        assert self._yaff_ener is None
        self._sp = sp

    def __call__(self, iterative: VerletIntegrator):
        self.hook(
            TrajectoryInfo(
                positions=iterative.pos,
                cell=iterative.rvecs,
                gpos=iterative.gpos,
                t=iterative.time,
                e_pot=iterative.epot,
                vtens=iterative.vtens,
            )
        )

    def _setup_verlet(self):

        hooks = [self, VerletScreenLog(step=1)]

        self._yaff_ener = YaffEngine._YaffFF(
            self.sp,
            _energy=self.energy,
            bias=self.bias,
            tic=self.static_trajectory_info,
        )

        if self.static_trajectory_info.thermostat:
            hooks.append(
                yaff.sampling.NHCThermostat(
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

    def run(self, steps):
        if self._verlet is None:
            self._setup_verlet()
        self._verlet.run(int(steps))

    @dataclass
    class _yaffCell:
        rvecs: np.ndarray

        @property
        def nvec(self):
            return self.rvecs.shape[0]

        @property
        def volume(self):
            if self.nvec == 0:
                return np.nan

            return np.linalg.det(self.rvecs)

        def update_rvecs(self, rvecs):
            self.rvecs = rvecs

    # @dataclass
    # class _yaffSys:
    #     cell: YaffEngine._yaffCell
    #     pos: np.ndarray
    #     masses: np.ndarray
    #     charges: np.ndarray | None = None

    #     @property
    #     def natom(self):
    #         return self.pos.shape[0]

    class _YaffFF(yaff.pes.ForceField):
        def __init__(
            self,
            sp: SystemParams,
            _energy: Energy,  # name clash with yaff.pes.ForceField
            bias: Bias,
            tic: StaticTrajectoryInfo,
        ):

            assert tic.masses is not None

            from yaff.system import System

            super().__init__(
                system=System(
                    pos=np.array(sp.coordinates, dtype=np.double),
                    rvecs=np.array(sp.cell, dtype=np.double),
                    numbers=tic.atomic_numbers,
                ),
                parts=[],
            )

            # self.sp = sp
            self._energy = _energy
            self.bias = bias

        @property
        def sp(self) -> SystemParams:
            return SystemParams(
                coordinates=jnp.array(self.system.pos),
                cell=jnp.array(self.system.cell.rvecs),
            )

        # def update_rvecs(self, rvecs):
        #     super().update_rvecs(rvecs=rvecs)

        #     # with jax_dataclasses.copy_and_mutate(self.sp) as new_sp:
        #     #     new_sp.cell = jnp.array(rvecs)
        #     # self.sp = new_sp

        # def update_pos(self, pos):
        #     super().update_pos(pos=pos)

        #     # with jax_dataclasses.copy_and_mutate(self.sp) as new_sp:
        #     #     new_sp.coordinates = jnp.array(pos)
        #     # self.sp = new_sp

        def _internal_compute(self, gpos, vtens):

            ener, gpos_jax, vtens_jax = self._energy.compute_coor(
                self.sp,
                gpos is not None,
                vtens is not None,
            )

            ener_b, gpos_jax_b, vtens_jax_b = self.bias.compute_coor(
                self.sp,
                gpos is not None,
                vtens is not None,
            )

            # compute quantities
            ener += ener_b

            if gpos is not None:
                gpos_jax += gpos_jax_b
                gpos[:] = np.array(gpos_jax)
            if vtens is not None:
                vtens_jax += vtens_jax_b
                vtens[:] = np.array(vtens_jax)

            return float(ener)
