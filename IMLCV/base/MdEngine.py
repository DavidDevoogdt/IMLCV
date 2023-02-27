"""MD engine class peforms MD simulations in a given NVT/NPT ensemble.

Currently, the MD is done with YAFF/OpenMM
"""
from __future__ import annotations

import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import TYPE_CHECKING

import dill
import h5py
import jax.numpy as jnp
import numpy as np
import pytest
from molmod.units import bar

import yaff.analysis.biased_sampling
import yaff.external
import yaff.log
import yaff.pes
import yaff.pes.bias
import yaff.pes.ext
import yaff.sampling
import yaff.sampling.iterative
from IMLCV.base.CV import SystemParams
from yaff.external import libplumed
from yaff.log import log
from yaff.sampling.verlet import VerletIntegrator, VerletScreenLog

yaff.log.set_level(yaff.log.silent)

# if TYPE_CHECKING:
from IMLCV.base.bias import Bias, Energy, EnergyResult

from molmod.periodic import periodic
from molmod.units import kjmol

######################################
#             Trajectory             #
######################################


@dataclass(frozen=True)
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
        # "masses",
    ]

    timestep: float

    T: float
    timecon_thermo: float

    atomic_numbers: np.ndarray
    # masses: np.ndarray | None = None

    P: float | None = None
    timecon_baro: float | None = None

    write_step: int = 100
    equilibration: float | None = None
    screen_log: int = 1000

    @property
    def masses(self):
        return np.array([periodic[n].mass for n in self.atomic_numbers])

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
    def _load(hf: h5py.File) -> StaticTrajectoryInfo:
        props_static = {}
        attrs_static = {}

        for key, val in hf.items():
            props_static[key] = val[:]

        for key, val in hf.attrs.items():
            attrs_static[key] = val

        return StaticTrajectoryInfo(**attrs_static, **props_static)

    @staticmethod
    def load(filename) -> StaticTrajectoryInfo:

        with h5py.File(str(filename), "r") as hf:
            return StaticTrajectoryInfo._load(hf=hf)


@dataclass
class TrajectoryInfo:

    positions: np.ndarray
    cell: np.ndarray | None = None
    charges: np.ndarray | None = None

    e_pot: np.ndarray | None = None
    e_pot_gpos: np.ndarray | None = None
    e_pot_vtens: np.ndarray | None = None

    e_bias: np.ndarray | None = None
    e_bias_gpos: np.ndarray | None = None
    e_bias_vtens: np.ndarray | None = None

    T: np.ndarray | None = None
    P: np.ndarray | None = None
    err: np.ndarray | None = None

    t: np.ndarray | None = None

    _items_scal = ["t", "e_pot", "e_bias", "T", "P", "err"]
    _items_vec = [
        "positions",
        "cell",
        "e_pot_gpos",
        "e_pot_vtens",
        "e_bias_gpos",
        "e_bias_vtens",
        "charges",
    ]

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
                    np.hstack([prop, np.zeros(delta)]),
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
            coordinates=jnp.array(self.positions[0 : self._size - 1, :]),
            cell=jnp.array(self.cell[0 : self._size - 1, :])
            if self.cell is not None
            else None,
        )

    @property
    def last_sp(self) -> SystemParams:
        return SystemParams(
            coordinates=jnp.array(self.positions[self._size - 2, :]),
            cell=jnp.array(self.cell[self._size - 2, :])
            if self.cell is not None
            else None,
        )


######################################
#             MDEngine               #
######################################


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
        static_trajectory_info: StaticTrajectoryInfo,
        trajectory_file=None,
        sp: SystemParams | None = None,
    ) -> None:

        self.static_trajectory_info = static_trajectory_info

        self.bias = bias
        self.energy = energy

        self.last_bias = EnergyResult(0)
        self.last_ener = EnergyResult(0)

        # self._sp = sp
        self.trajectory_info: TrajectoryInfo | None = None

        self.step = 1
        if sp is not None:
            self.sp = sp
        self.trajectory_file = trajectory_file

        self.time0 = time()
        self.last_ti: TrajectoryInfo | None = None

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

        from IMLCV.base.bias import EnergyError

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

    def save_step(self, T=None, P=None, t=None, err=None):

        ti = TrajectoryInfo(
            positions=self.sp.coordinates,
            cell=self.sp.cell,
            e_pot=self.last_ener.energy,
            e_pot_gpos=self.last_ener.gpos,
            e_bias=self.last_bias.energy,
            e_bias_gpos=self.last_bias.gpos,
            e_pot_vtens=self.last_ener.vtens,
            e_bias_vtens=self.last_bias.vtens,
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
            if ti.P is not None:
                str += f"|{'P[bar]': ^10s}"
            str += f"|{'T[K]': ^10s}|{'walltime[s]': ^10s}"
            print(str, sep="")
            print(f"{'='*len(str)}")

        if self.step % self.static_trajectory_info.screen_log == 0:
            str = f"{  self.step : >10d}"
            assert ti.err is not None
            assert ti.T is not None
            assert ti.e_pot is not None
            assert ti.e_bias is not None

            str += f"|{  ti.err[0] : >10.4f}"
            str += f"|{  ti.e_pot[0]  *kjmol : >15.4f}"
            str += f"|{  ti.e_bias[0] *kjmol : >15.4f}"
            if ti.P is not None:
                str += f" { ti.P[0]/bar : >10.2f}"
            str += f" { ti.T[0] : >10.2f} { time()-self.time0 : >10.2f}"
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

    def get_bias(self, gpos: bool = False, vtens: bool = False) -> EnergyResult:

        return self.bias.compute_from_system_params(
            self.sp,
            gpos,
            vtens,
        )

    @property
    def yaff_system(self) -> MDEngine.YaffSys:
        return self.YaffSys(self.energy, self.static_trajectory_info)

    # definitons of different interfaces. These encode the state of the system in the format of a given md engine

    @dataclass
    class YaffSys:
        _ener: Energy
        _tic: StaticTrajectoryInfo

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
        trajectory_file=None,
        sp: SystemParams | None = None,
        additional_parts=[],
    ) -> None:

        yaff.log.set_level(log.silent)
        self.start = 0
        self.name = "YaffEngineIMLCV"

        self._verlet: yaff.sampling.VerletIntegrator | None = None
        self._yaff_ener: YaffEngine.YaffFF | None = None

        self._verlet_initialized = False

        super().__init__(
            energy=energy,
            bias=bias,
            static_trajectory_info=static_trajectory_info,
            trajectory_file=trajectory_file,
            sp=sp,
        )
        self.initializing = False

        self.additional_parts = additional_parts

    def __call__(self, iterative: VerletIntegrator):

        if not self._verlet_initialized:
            return

        kwargs = dict(t=iterative.time, T=iterative.temp, err=iterative.cons_err)

        if hasattr(iterative, "press"):
            kwargs["P"] = iterative.press

        self.save_step(**kwargs)

    def _setup_verlet(self):

        # hooks = [self, VerletScreenLog(step=self.static_trajectory_info.screen_log)]

        hooks = [self]

        self._yaff_ener = YaffEngine.YaffFF(
            md_engine=self,
            additional_parts=self.additional_parts,
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

        # plumed hook
        for i in self.additional_parts:
            if isinstance(i, yaff.sampling.iterative.Hook):
                hooks.append(i)

        self._verlet = yaff.sampling.VerletIntegrator(
            self._yaff_ener,
            self.static_trajectory_info.timestep,
            temp0=self.static_trajectory_info.T,
            hooks=hooks,
        )

        self._verlet_initialized = True

    @staticmethod
    def load(file, **kwargs) -> MDEngine:
        return super().load(file, **kwargs)

    def _run(self, steps):
        if not self._verlet_initialized:
            self._setup_verlet()
        self._verlet.run(int(steps))

    class YaffFF(yaff.pes.ForceField):
        def __init__(
            self,
            md_engine: MDEngine,
            name="IMLCV_YAFF_forcepart",
            additional_parts=[],
        ):
            self.md_engine = md_engine

            super().__init__(system=self.system, parts=additional_parts)

        @property
        def system(self):
            return self.md_engine.yaff_system

        @system.setter
        def system(self, sys):
            assert sys == self.system

        @property
        def sp(self):
            return self.md_engine.energy.sp

        def update_rvecs(self, rvecs):
            self.clear()
            self.system.cell.rvecs = rvecs

        def update_pos(self, pos):
            self.clear()
            self.system.pos = pos

        def _internal_compute(self, gpos, vtens):

            energy = self.md_engine.get_energy(
                gpos is not None,
                vtens is not None,
            )

            bias = self.md_engine.get_bias(
                gpos is not None,
                vtens is not None,
            )

            self.md_engine.last_ener = energy
            self.md_engine.last_bias = bias

            if gpos is not None:
                gpos[:] += np.array(energy.gpos)
            if vtens is not None:
                vtens[:] += np.array(energy.vtens)

            return energy.energy


class PlumedEngine(YaffEngine):

    # Energy - kJ/mol
    # Length - nanometers
    # Time - picoseconds

    # https://github.com/giorginolab/plumed2-pycv/tree/v2.8-pycv/src/pycv
    # pybias  https://raimis.github.io/plumed-pybias/

    # pycv doesn't work with cell vectors?
    # this does ? https://github.com/giorginolab/plumed2-pycv/blob/v2.8-pycv/src/pycv/PythonFunction.cpp
    # see https://github.com/giorginolab/plumed2-pycv/blob/v2.6-pycv-devel/regtest/pycv/rt-f2/plumed.dat

    plumed_dat = """
    LOAD FILE=libpybias.so

    dist: DISTANCE ATOMS=1,2
    

    rc: PYTHONCV ATOMS=1,4,3 IMPORT=curvature FUNCTION=r

    # Creat a PyBias action, which executes "bias.py"
    PYBIAS ARG=rc


    RESTRAINT ARG=rc AT=0 KAPPA=0 SLOPE=1
    """

    def __init__(
        self,
        bias: Bias,
        static_trajectory_info: StaticTrajectoryInfo,
        energy: Energy,
        trajectory_file=None,
        sp: SystemParams | None = None,
    ) -> None:

        super().__init__(
            bias,
            static_trajectory_info,
            energy,
            trajectory_file,
            sp,
            additional_parts=[
                libplumed.ForcePartPlumed(timestep=static_trajectory_info.timestep)
            ],
        )


######################################
#              test                  #
######################################


def test_yaff_save_load_func(full_name):
    from IMLCV.examples.example_systems import alanine_dipeptide_yaff

    yaffmd = alanine_dipeptide_yaff()

    yaffmd.run(int(761))

    yaffmd.save("output/yaff_save.d")
    yeet = MDEngine.load("output/yaff_save.d")

    sp1 = yaffmd.sp
    sp2 = yeet.sp

    assert pytest.approx(sp1.coordinates) == sp2.coordinates
    assert pytest.approx(sp1.cell) == sp2.cell
    assert (
        pytest.approx(yaffmd.energy.compute_from_system_params(sp1).energy, abs=1e-6)
        == yeet.energy.compute_from_system_params(sp2).energy
    )


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        test_yaff_save_load_func(full_name=f"{tmp}/load_save.h5")
