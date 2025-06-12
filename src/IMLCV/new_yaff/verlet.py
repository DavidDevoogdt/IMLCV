# -*- coding: utf-8 -*-
# YAFF is yet another force-field code.
# Copyright (C) 2011 Toon Verstraelen <Toon.Verstraelen@UGent.be>,
# Louis Vanduyfhuys <Louis.Vanduyfhuys@UGent.be>, Center for Molecular Modeling
# (CMM), Ghent University, Ghent, Belgium; all rights reserved unless otherwise
# stated.
#
# This file is part of YAFF.
#
# YAFF is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# YAFF is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
# --
"""Generic Verlet integrator"""

from __future__ import annotations

from dataclasses import KW_ONLY
from typing import Self

import jax
import jax.numpy as jnp

from IMLCV.base.datastructures import MyPyTreeNode, field, jit_decorator
from IMLCV.base.UnitsConstants import boltzmann
from IMLCV.new_yaff.ff import YaffFF
from IMLCV.new_yaff.iterative import (
    AttributeStateItem,
    CellStateItem,
    Hook,
    PosStateItem,
    StateItem,
)
from IMLCV.new_yaff.utils import get_random_vel


# @partial(dataclass, frozen=False)
class TemperatureStateItem(StateItem):
    def get_value(self, iterative):
        return getattr(iterative, "temp", None)


# @partial(dataclass, frozen=False)
class ConsErrTracker(MyPyTreeNode):
    """
    A class that tracks the errors on the conserved quantity.
    Given its superior numerical accuracy, the algorithm below
    is used to calculate the running average. Its properties are discussed
    in Donald Knuth's Art of Computer Programming, vol. 2, p. 232, 3rd edition.
    """

    counter: jax.Array = field(default_factory=lambda: jnp.array(0))
    ekin_m: jax.Array = field(default_factory=lambda: jnp.array(0.0))
    ekin_s: jax.Array = field(default_factory=lambda: jnp.array(0.0))
    econs_m: jax.Array = field(default_factory=lambda: jnp.array(0.0))
    econs_s: jax.Array = field(default_factory=lambda: jnp.array(0.0))

    def update(self, ekin, econs):
        # if self.counter == 0:
        #     self.ekin_m = ekin
        #     self.econs_m = econs
        # else:
        ekin_tmp = ekin - self.ekin_m
        self.ekin_m += ekin_tmp / (self.counter + 1)
        self.ekin_s += ekin_tmp * (ekin - self.ekin_m)
        econs_tmp = econs - self.econs_m
        self.econs_m += econs_tmp / (self.counter + 1)
        self.econs_s += econs_tmp * (econs - self.econs_m)

        self.counter += 1

    def get(self):
        return jnp.where(self.counter > 1, jnp.sqrt(self.econs_s / self.ekin_s), 0.0)


# @partial(dataclass, frozen=False)
class VerletHook(Hook):
    """Specialized Verlet hook.

    This is mainly used for the implementation of thermostats and barostats.
    """

    econs_correction: jax.Array = field(default_factory=lambda: jnp.array(0.0))

    def __call__(self: Self, iterative: VerletIntegrator):
        pass

    def init(self: Self, iterative: VerletIntegrator) -> tuple[Self, VerletIntegrator]:
        raise NotImplementedError

    def pre(self: Self, iterative: VerletIntegrator, **kwargs) -> tuple[Self, VerletIntegrator]:
        raise NotImplementedError

    def post(self: Self, iterative: VerletIntegrator, **kwargs) -> tuple[Self, VerletIntegrator]:
        raise NotImplementedError


class ThermostatHook(VerletHook):
    temp: jax.Array

    def pre(  # type:ignore
        self: Self,
        iterative: VerletIntegrator,
        G1_add: jax.Array | None = None,
    ) -> tuple[Self, VerletIntegrator]:
        raise NotImplementedError

    def post(  # type:ignore
        self: Self,
        iterative: VerletIntegrator,
        G1_add: jax.Array | None = None,
    ) -> tuple[Self, VerletIntegrator]:
        raise NotImplementedError


class BarostatHook(VerletHook):
    temp: jax.Array
    press: jax.Array
    baro_ndof: int | None = None
    dim: int | None = None

    def pre(  # type:ignore
        self: Self,
        iterative: VerletIntegrator,
        chainvel0: jax.Array | None = None,
    ) -> tuple[Self, VerletIntegrator]:
        raise NotImplementedError

    def post(  # type:ignore
        self: Self,
        iterative: VerletIntegrator,
        chainvel0: jax.Array | None = None,
    ) -> tuple[Self, VerletIntegrator]:
        raise NotImplementedError


# @partial(dataclass, frozen=False)
class NVE(VerletHook):
    _: KW_ONLY
    temp: float = 0.0

    """Specialized Verlet hook.

    This is mainly used for the implementation of thermostats and barostats.
    """

    econs_correction: jax.Array = field(default_factory=lambda: jnp.array(0))

    def __call__(self, iterative):
        return

    def init(self: VerletHook, iterative: VerletIntegrator) -> tuple[VerletHook, VerletIntegrator]:
        return self, iterative

    def pre(  # type:ignore
        self: VerletHook,
        iterative: VerletIntegrator,
    ) -> tuple[VerletHook, VerletIntegrator]:
        return self, iterative

    def post(  # type:ignore
        self: VerletHook,
        iterative: VerletIntegrator,
    ) -> tuple[VerletHook, VerletIntegrator]:
        return self, iterative


# @partial(dataclass, frozen=False)
class VerletIntegrator(MyPyTreeNode):
    ff: YaffFF

    counter: jax.Array
    time: jax.Array
    ndof: int
    pos: jax.Array
    vel: jax.Array
    acc: jax.Array
    masses: jax.Array
    timestep: jax.Array

    # barostat quantities
    ptens: jax.Array
    rvecs: jax.Array
    vtens: jax.Array

    verlet_hook: VerletHook

    # instantaenous properties
    delta: jax.Array
    pos_old: jax.Array

    gpos: jax.Array
    gpos_bias: jax.Array

    temp: jax.Array
    press: jax.Array

    ekin: jax.Array
    ekin_new: jax.Array
    epot: jax.Array
    etot: jax.Array
    e_bias: jax.Array

    # conserved quantity
    _cons_err_tracker: ConsErrTracker = field(default_factory=ConsErrTracker)
    econs: jax.Array
    cons_err: jax.Array

    # deviation
    rmsd_delta: jax.Array
    rmsd_gpos: jax.Array
    rmsd_gpos_bias: jax.Array

    cv: jax.Array

    other_hooks: list[Hook] = field(default_factory=list)

    @staticmethod
    def create(
        ff: YaffFF,
        other_hooks: list[Hook],
        timestep: float,
        thermostat: ThermostatHook | None = None,
        barostat: BarostatHook | None = None,
        vel0: jax.Array | None = None,
        temp0: float = 300.0,
        scalevel0: bool = True,
        time0: float | None = None,
        ndof: int | None = None,
        counter0: int | None = None,
        key=42,
    ):
        """
        **Arguments:**

        ff
            A ForceField instance

        **Optional arguments:**

        timestep
            The integration time step (in atomic units)

        state
            A list with state items. State items are simple objects
            that take or derive a property from the current state of the
            iterative algorithm.

        hooks
            A function (or a list of functions) that is called after every
            iterative.

        vel0
            An array with initial velocities. If not given, random
            velocities are sampled from the Maxwell-Boltzmann distribution
            corresponding to the optional arguments temp0 and scalevel0

        temp0
            The (initial) temperature for the random initial velocities

        scalevel0
            If True (the default), the random velocities are rescaled such
            that the instantaneous temperature coincides with temp0.

        time0
            The time associated with the initial state.

        ndof
            When given, this option overrides the number of degrees of
            freedom determined from internal heuristics. When ndof is not
            given, its default value depends on the thermostat used. In most
            cases it is 3*natom, except for the NHC thermostat where the
            number if internal degrees of freedom is counted. The ndof
            attribute is used to derive the temperature from the kinetic
            energy.

        counter0
            The counter value associated with the initial state.

        """

        if thermostat is not None:
            assert thermostat.method == "thermostat"

        if barostat is not None:
            assert barostat.method == "barostat"

        if thermostat is None and barostat is None:
            vh = NVE()
            print("sampling NVE ensemble")
        elif thermostat is not None and barostat is None:
            print(f"sampling NVT ensemble {thermostat.name=}")
            vh = thermostat
        elif thermostat is not None and barostat is not None:
            from IMLCV.new_yaff.npt import TBCombination

            print(f"sampling NPT ensemble {thermostat.name=} {barostat.name=}")
            vh = TBCombination(thermostat=thermostat, barostat=barostat)
        else:
            print(f"sampling NPE ensemble  {barostat.name=}")  # type:ignore
            vh = barostat

        assert isinstance(vh, VerletHook)

        pos = ff.system.pos

        delta = jnp.zeros(pos.shape, float)

        res, (bias, cv) = ff.compute(gpos=True)
        epot, gpos = res.energy, res.gpos
        vtens = jnp.zeros((3, 3), float)
        ptens = jnp.zeros((3, 3), float)

        assert bias.gpos is not None

        cv, e_bias, gpos_bias = cv, bias.energy, bias.gpos

        masses = ff.system.masses

        assert gpos is not None
        acc = -gpos / masses.reshape(-1, 1)
        pos_old = pos

        vel = vel0 if vel0 is not None else get_random_vel(temp0, scalevel0, masses, key=jax.random.PRNGKey(key))

        # Configure the number of degrees of freedom if needed
        if ndof is None:
            ndof = pos.size

        self = VerletIntegrator(
            ff=ff,
            pos=pos,
            vel=vel,
            masses=masses,
            timestep=jnp.array(timestep),
            verlet_hook=vh,
            gpos=gpos,
            gpos_bias=gpos_bias,
            temp=jnp.array(temp0),
            ekin=jnp.array(0.0),
            ekin_new=jnp.array(0.0),
            epot=epot,
            etot=jnp.array(0.0),
            econs=jnp.array(0.0),
            cons_err=jnp.array(0.0),
            ptens=ptens,
            press=jnp.array(0.0),
            rmsd_delta=jnp.array(0.0),
            rmsd_gpos=jnp.array(0.0),
            rmsd_gpos_bias=jnp.array(0.0),
            counter=jnp.array(counter0 if counter0 is not None else 0),
            time=jnp.array(time0 if time0 is not None else 0.0),
            delta=delta,
            acc=acc,
            rvecs=ff.system.cell.rvecs,
            ndof=ndof,
            cv=cv,
            e_bias=e_bias,
            pos_old=pos_old,
            vtens=vtens,
            other_hooks=other_hooks,
            _cons_err_tracker=ConsErrTracker(),
        )

        # Allow for specialized initializations by the Verlet hooks.
        self.verlet_hook, self = self.verlet_hook.init(self)

        # Common post-processing of the initialization
        self = self.compute_properties()

        self.call_hooks()

        return self

    @jit_decorator
    def propagate(self: VerletIntegrator):
        # Allow specialized hooks to modify the state before the regular verlet
        # step.
        self.verlet_hook, self = self.verlet_hook.pre(self)

        # Regular verlet step
        assert self.gpos is not None
        self.acc = -self.gpos / self.masses.reshape(-1, 1)
        self.vel += 0.5 * self.acc * self.timestep
        self.pos += self.timestep * self.vel
        self.ff.system.pos = self.pos

        res, (bias, cv) = self.ff.compute(gpos=True)
        assert res.gpos is not None
        self.epot, self.gpos = res.energy, res.gpos
        assert bias.gpos is not None
        self.cv, self.e_bias, self.gpos_bias = cv, bias.energy, bias.gpos

        self.acc = -self.gpos / self.masses.reshape(-1, 1)

        self.vel += 0.5 * self.acc * self.timestep
        self.ekin = self._compute_ekin()

        # Allow specialized verlet hooks to modify the state after the step
        self.verlet_hook, self = self.verlet_hook.post(self)

        # Calculate the total position change
        pos_new = self.pos
        self.delta = pos_new - self.pos_old
        self.pos_old = pos_new

        # Common post-processing of a single step
        self.time += self.timestep
        self = self.compute_properties()

        self.counter += 1

        # print(f"inside {self.vel=}")

        return self

    @jit_decorator
    def _compute_ekin(self):
        """Auxiliary routine to compute the kinetic energy

        This is used internally and often also by the Verlet hooks.
        """
        return 0.5 * jnp.sum((self.vel**2 * self.masses.reshape(-1, 1)))

    @jit_decorator
    def compute_properties(self):
        # self.rmsd_gpos = jnp.sqrt((self.gpos**2).mean())
        self.rmsd_gpos = jnp.linalg.norm(self.gpos) / jnp.sqrt(self.gpos.shape[0])
        self.rmsd_gpos_bias = jnp.linalg.norm(self.gpos_bias) / jnp.sqrt(self.gpos.shape[0])

        assert self.delta is not None
        self.rmsd_delta = jnp.sqrt((self.delta**2).mean())
        self.ekin = self._compute_ekin()
        self.temp = self.ekin / self.ndof * 2.0 / boltzmann
        self.etot = self.ekin + self.epot
        self.econs = self.etot

        self.econs += self.verlet_hook.econs_correction

        self._cons_err_tracker.update(self.ekin, self.econs)
        self.cons_err = self._cons_err_tracker.get()
        if self.ff.system.cell.nvec > 0:
            self.ptens = (jnp.dot(self.vel.T * self.masses, self.vel) - self.vtens) / self.ff.system.cell.volume
            self.press = jnp.trace(self.ptens) / 3

        return self

    def finalize(self):
        pass

    def call_hooks(self):
        # state_updated = False

        if jnp.any(jnp.isnan(self.ff.system.sp.coordinates)):
            raise ValueError(f"sp containes nans: {self.ff.system.sp=}")

        if self.ff.system.sp.cell is not None:
            if jnp.any(jnp.isnan(self.ff.system.sp.cell)):
                raise ValueError(f"sp containes nans: {self.ff.system.sp=}")

        for hook in [self.verlet_hook, *self.other_hooks]:
            if hook.expects_call(self.counter):
                hook(self)

    def run(self: VerletIntegrator, nstep=None):
        if nstep is None:
            while True:
                self = self.propagate()
                self.call_hooks()
        else:
            for i in range(nstep):
                self = self.propagate()
                self.call_hooks()
        self.finalize()
