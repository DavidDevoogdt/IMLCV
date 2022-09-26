"""MD engine class peforms MD simulations in a given NVT/NPT ensemble.

Currently, the MD is done with YAFF/OpenMM
"""
from __future__ import annotations

import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import ase
import dill
import h5py
import jax.numpy as jnp
import jax_dataclasses
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
from IMLCV.base.bias import Bias, Energy, YaffEnergy
from IMLCV.base.CV import SystemParams
from yaff.log import log
from yaff.pes.ff import ForceField
from yaff.sampling.verlet import VerletScreenLog


@dataclass
class TrajectoryInfo:

    positions: np.ndarray
    cell: np.ndarray | None = None
    gpos: np.ndarray | None = None
    vtens: np.ndarray | None = None
    t: np.ndarray | None = None
    e_pot: np.ndarray | None = None
    masses: np.ndarray | None = None

    _items_scal = ["t", "e_pot"]
    _items_vec = ["positions", "cell", "gpos", "vtens"]
    _items_stat = ["masses"]

    _capacity: int = -1
    _size: int = -1

    # https://stackoverflow.com/questions/7133885/fastest-way-to-grow-a-numpy-numeric-array
    def __post_init__(self):
        if self._capacity == -1:
            self._capacity = 1
        if self._size == -1:
            self._size = 1

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
            self._expand_capacity

        for name in self._items_vec:
            prop_ti = ti.__getattribute__(name)
            prop_self = self.__getattribute__(name)
            if prop_ti is None:
                assert prop_self is None
            else:
                prop_self[self._size, :] = prop_ti[0]

        for name in self._items_scal:
            prop_ti = ti.__getattribute__(name)
            prop_self = self.__getattribute__(name)
            if prop_ti is None:
                assert prop_self is None
            else:
                prop_self[self._size] = prop_ti[0]

        for name in self._items_stat:
            prop_ti = ti.__getattribute__(name)
            prop_self = self.__getattribute__(name)

            assert (jnp.abs(prop_ti - prop_self) < 1e-6).all()

        self._size += 1

    def _expand_capacity(self):
        nc = max(self._capacity * 2, 10000)
        delta = self._capacity - nc
        self._capacity = nc

        for name in self._items_vec:
            prop = self.__getattribute__(name)
            if prop is not None:
                self.__setattr__(
                    name,
                    np.vstack([prop, np.zeros((delta, *prop.shape[1:]))]),
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

            for name in [*self._items_scal, *self._items_vec, *self._items_stat]:
                prop = self.__getattribute__(name)
                if prop is not None:
                    hf.create_dataset(name, prop)

            hf.attrs.create("_capacity", self._capacity)
            hf.attrs.create("_size", self._size)

    @classmethod
    def load(filename) -> TrajectoryInfo:

        props = {}
        attrs = {}

        with h5py.File(str(filename), "r") as hf:
            for key, val in hf.items():
                props[key] = val

            for key, val in hf.attrs.items():
                attrs[key] = val

        return TrajectoryInfo(**props, **attrs)

    @property
    def volume(self):
        if self.cell is not None:
            return jnp.linalg.det(self.cell)
        return None


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
        "ener",
        "T",
        "P",
        "timestep",
        "timecon_thermo",
        "timecon_baro",
        "write_step",
        "screenlog",
        "sp",
        "equilibration",
    ]

    def __init__(
        self,
        bias: Bias,
        energy: Energy,
        sp: SystemParams,
        T,
        P=None,
        timestep=None,
        timecon_thermo=None,
        timecon_baro=None,
        write_step=100,
        screenlog=1000,
        equilibration=None,
    ) -> None:

        self.bias = bias
        self.write_step = write_step

        self.timestep = timestep

        self.T = T
        self.thermostat = T is not None
        self.timecon_thermo = timecon_thermo
        if self.thermostat:
            assert timecon_thermo is not None

        self.P = P
        self.barostat = P is not None
        self.timecon_baro = timecon_baro
        if self.barostat:
            assert timecon_baro is not None

        self.screenlog = screenlog

        self.energy = energy

        self.equilibration = 100 * timestep if equilibration is None else equilibration

        self.sp = sp
        self.trajectory_info: TrajectoryInfo | None = None

        self._init_post()

    @abstractmethod
    def _init_post(self):
        pass

    def __getstate__(self):
        d = {}
        for key in self.keys:
            d[key] = self.__dict__[key]

        return [self.__class__, d]

    def __setstate__(self, arr):
        [cls, d] = arr

        return cls(**d)

    def save(self, file):
        with open(file, "wb") as f:
            dill.dump(self.__getstate__(), f)

    @staticmethod
    def load(file, **kwargs) -> MDEngine:

        with open(file, "rb") as f:
            [cls, d] = dill.load(f)

        # replace and add kwargs
        for key in kwargs.keys():
            d[key] = kwargs[key]

        return cls(**d)

    def new_bias(self, bias: Bias, **kwargs) -> MDEngine:
        with tempfile.NamedTemporaryFile() as tmp:
            self.save(tmp.name)
            mde = MDEngine.load(tmp.name, **{"bias": bias, **kwargs})
        return mde

    @abstractmethod
    def run(self, steps, filename=None):
        """run the integrator for a given number of steps.

        Args:
            steps: number of MD steps
        """
        raise NotImplementedError

    def get_trajectory(self) -> TrajectoryInfo:
        assert self.trajectory_info is not None
        # self.trajectory_info.finalize()
        return self.trajectory_info

    def hook(self, ti: TrajectoryInfo):

        # write step to trajectory
        if self.trajectory_info is None:
            self.trajectory_info = ti
            return
        self.trajectory_info += ti

        # update bias
        self.bias.update_bias(self.sp)

    def to_ASE_traj(self) -> ase.Atoms:
        traj = self.get_trajectory()

        pos_A = traj.positions / angstrom
        pbc = traj.cell is not None
        if pbc:
            cell_A = traj.cell / angstrom
            vol_A3 = traj.volume / angstrom**3
            vtens_eV = traj.vtens / electronvolt
            stresses_eVA3 = vtens_eV / vol_A3

            atoms = ase.Atoms(
                masses=traj.masses,
                positions=pos_A,
                pbc=pbc,
                cell=cell_A,
            )
            atoms.info["stress"] = stresses_eVA3
        else:
            atoms = ase.Atoms(
                masses=traj.masses,
                positions=pos_A,
            )

        if traj.gpos is not None:
            atoms.arrays["forces"] = -traj.gpos * angstrom / electronvolt
        if traj.e_pot is not None:
            atoms.info["energy"] = traj.e_pot / electronvolt

        return atoms


class YaffEngine(MDEngine, yaff.sampling.iterative.Hook):
    """MD engine with YAFF as backend.

    Args:
        ff (yaff.pes.ForceField)
    """

    def __init__(
        self,
        ener: Energy | Callable[[], ForceField],
        sp: SystemParams | None = None,
        log_level=log.medium,
        **kwargs,
    ) -> None:

        if not isinstance(ener, Energy):
            ener = YaffEnergy(ener)

        if sp is None:
            sp = ener.get_sp()

        # initialize yaff hook

        yaff.log.set_level(log_level)
        self.start = 0
        self.step = 1
        self.name = "YaffEngineIMLCV"

        super().__init__(energy=ener, sp=sp, **kwargs)

    def __call__(self, iterative):
        self.hook(
            TrajectoryInfo(
                positions=iterative.pos,
                cell=iterative.rvecs,
                gpos=iterative.gpos,
                t=iterative.time,
                masses=iterative.masses,
                e_pot=iterative.epot,
                vtens=iterative.vtens,
            )
        )

    def _init_post(self):
        self.verlet: yaff.sampling.VerletIntegrator | None = None

    def _setup_verlet(self):

        hooks = [self, VerletScreenLog(step=1)]

        if self.thermostat:
            hooks.append(
                yaff.sampling.NHCThermostat(
                    self.T,
                    timecon=self.timecon_thermo,
                )
            )
        if self.barostat:
            hooks.append(
                yaff.sampling.MTKBarostat(
                    self.energy,
                    self.T,
                    self.P,
                    timecon=self.timecon_baro,
                    anisotropic=True,
                )
            )

        self._yaff_ener = YaffEngine._YaffFF(
            self.sp,
            ener=self.energy,
            bias=self.bias,
        )

        self.verlet = yaff.sampling.VerletIntegrator(
            self._yaff_ener,
            self.timestep,
            temp0=self.T,
            hooks=hooks,
        )

    @staticmethod
    def load(file, **kwargs) -> MDEngine:
        return super().load(file, **kwargs)

    def run(self, steps, filename=None):
        if self.verlet is None:
            self._setup_verlet()
        self.verlet.run(int(steps))

    @dataclass
    class _yaffCell:
        rvecs: np.ndarray

        @property
        def nvec(self):
            return self.rvecs.shape[0]

        @property
        def volume(self):
            return jnp.abs(
                jnp.dot(self.rvecs[0], jnp.cross(self.rvecs[1], self.rvecs[2]))
            )

    @dataclass
    class _yaffSys:
        cell: YaffEngine._yaffCell
        pos: np.ndarray
        masses: np.ndarray
        charges: np.ndarray | None = None

        @property
        def natom(self):
            return self.pos.shape[0]

    class _YaffFF(yaff.pes.ForceField):
        def __init__(self, sp: SystemParams, ener: Energy, bias: Bias):

            super().__init__(
                system=YaffEngine._yaffSys(
                    pos=np.array(sp.coordinates),
                    cell=YaffEngine._yaffCell(rvecs=np.array(sp.cell)),
                    masses=np.array(sp.masses),
                ),
                parts=[],
            )

            self.sp = sp
            self.ener = ener
            self.bias = bias

        def update_rvecs(self, rvecs):
            with jax_dataclasses.copy_and_mutate(self.sp) as new_sp:
                new_sp.cell = jnp.array(rvecs)
            self.sp = new_sp

        def update_pos(self, pos):
            with jax_dataclasses.copy_and_mutate(self.sp) as new_sp:
                new_sp.coordinates = jnp.array(pos)
            self.sp = new_sp

        def _internal_compute(self, gpos, vtens):

            ener, gpos_jax, vtens_jax = self.ener.compute_coor(
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
            ener += ener_b[0]

            if gpos is not None:
                gpos_jax += gpos_jax_b
                gpos[:] = np.array(gpos_jax)
            if vtens is not None:
                vtens_jax += vtens_jax_b
                vtens[:] = np.array(vtens_jax)

            return ener
