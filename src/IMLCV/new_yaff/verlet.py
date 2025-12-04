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
from IMLCV.base.CV import NeighbourList, SystemParams


class TemperatureStateItem(StateItem):
    def get_value(self, iterative):
        return getattr(iterative, "temp", None)


class ConsErrTracker(MyPyTreeNode):
    """
    A class that tracks the errors on the conserved quantity.
    Given its superior numerical accuracy, the algorithm below
    is used to calculate the running average. Its properties are discussed
    in Donald Knuth's Art of Computer Programming, vol. 2, p. 232, 3rd edition.
    """

    counter: jax.Array = field(default_factory=lambda: jnp.array(0))

    ekin_m: jax.Array
    econs_m: jax.Array

    ekin_s: jax.Array = field(default_factory=lambda: jnp.array(0.0))
    econs_s: jax.Array = field(default_factory=lambda: jnp.array(0.0))

    def update(self, ekin, econs):
        d_ekin = ekin - self.ekin_m
        self.ekin_m += d_ekin / (self.counter + 1)
        self.ekin_s += d_ekin * (ekin - self.ekin_m)

        econs_tmp = econs - self.econs_m
        self.econs_m += econs_tmp / (self.counter + 1)
        self.econs_s += econs_tmp * (econs - self.econs_m)

        self.counter += 1

        return self

    def get(self):
        # Returns the square root of the ratio of the
        # variance in Ekin to the variance in Econs

        return jnp.where(self.counter >= 1, jnp.sqrt(self.econs_s / self.ekin_s), 0.0)


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


class NVE(VerletHook):
    _: KW_ONLY
    temp: float = 0.0

    """Specialized Verlet hook.

    This is mainly used for the implementation of thermostats and barostats.
    """

    econs_correction: jax.Array = field(default_factory=lambda: jnp.array(0))

    def __call__(self, iterative):
        return

    def init(
        self: VerletHook,
        iterative: VerletIntegrator,
    ) -> tuple[VerletHook, VerletIntegrator]:
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


class NlUpdateHook(Hook):
    def __call__(self, iterative: VerletIntegrator):
        iterative.ff.system.update_nl()


class VerletIntegrator(MyPyTreeNode):
    ff: YaffFF

    counter: jax.Array
    time: jax.Array
    ndof: int
    pos: jax.Array
    vel: jax.Array
    # acc: jax.Array
    masses: jax.Array
    timestep: jax.Array

    # barostat quantities
    ptens: jax.Array
    rvecs: jax.Array
    vtens: jax.Array

    # verlet_hook: VerletHook

    # instantaenous properties
    gpos: jax.Array
    gpos_bias: jax.Array

    temp: jax.Array
    press: jax.Array

    ekin: jax.Array
    epot: jax.Array
    etot: jax.Array
    e_bias: jax.Array

    # conserved quantity
    _cons_err_tracker: ConsErrTracker | None = None
    econs: jax.Array
    cons_err: jax.Array

    cv: jax.Array

    other_hooks: list[Hook] = field(default_factory=list)

    @property
    def sp(self) -> SystemParams:
        return SystemParams(
            coordinates=self.pos,
            cell=self.rvecs,
        )

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

        res, (bias, cv) = ff.compute(gpos=True)
        epot, gpos = res.energy, res.gpos
        vtens = jnp.zeros((3, 3), float)
        ptens = jnp.zeros((3, 3), float)

        assert bias.gpos is not None

        cv, e_bias, gpos_bias = cv, bias.energy, bias.gpos

        masses = ff.system.masses

        assert gpos is not None

        vel = vel0 if vel0 is not None else get_random_vel(temp0, scalevel0, masses, key=jax.random.PRNGKey(key))

        # Configure the number of degrees of freedom if needed
        if ndof is None:
            ndof = pos.size

        if ff.system.nl is not None:
            other_hooks.append(
                NlUpdateHook(),
            )

        self = VerletIntegrator(
            ff=ff,
            pos=pos,
            vel=vel,
            masses=masses,
            timestep=jnp.array(timestep),
            gpos=gpos,
            gpos_bias=gpos_bias,
            temp=jnp.array(temp0),
            ekin=jnp.array(0.0),
            epot=epot,
            etot=jnp.array(0.0),
            econs=jnp.array(0.0),
            cons_err=jnp.array(0.0),
            ptens=ptens,
            press=jnp.array(0.0),
            counter=jnp.array(counter0 if counter0 is not None else 0),
            time=jnp.array(time0 if time0 is not None else 0.0),
            rvecs=ff.system.cell.rvecs,
            ndof=ndof,
            cv=cv,
            e_bias=e_bias,
            vtens=vtens,
            other_hooks=other_hooks,
            _cons_err_tracker=None,
        )

        verlet_hook = vh

        # Allow for specialized initializations by the Verlet hooks.
        verlet_hook, self = verlet_hook.init(self)

        # Common post-processing of the initialization
        self = self.compute_properties(verlet_hook)

        self.call_hooks(verlet_hook)

        return self, verlet_hook

    @jit_decorator
    def propagate(self: VerletIntegrator, verlet_hook: VerletHook) -> tuple[VerletIntegrator, VerletHook]:
        # Allow specialized hooks to modify the state before the regular verlet
        # step.

        verlet_hook, self = jax.lax.cond(
            verlet_hook.expects_call(self.counter),
            lambda self, verlet_hook: verlet_hook.pre(self),
            lambda self, verlet_hook: (verlet_hook, self),
            self,
            verlet_hook,
        )

        # Regular verlet step
        assert self.gpos is not None

        # time t
        acc = -self.gpos / self.masses.reshape(-1, 1)

        # time t + dt/2
        self.vel += 0.5 * acc * self.timestep
        # time t + dt
        self.pos += self.timestep * self.vel

        self.ff.system.pos = self.pos
        res, (bias, cv) = self.ff.compute(gpos=True)
        assert res.gpos is not None
        self.epot, self.gpos = res.energy, res.gpos
        assert bias.gpos is not None
        self.cv, self.e_bias, self.gpos_bias = cv, bias.energy, bias.gpos

        # time t + dt
        acc = -self.gpos / self.masses.reshape(-1, 1)
        self.vel += 0.5 * acc * self.timestep
        self.ekin = self._compute_ekin()

        # Allow specialized verlet hooks to modify the state after the step

        verlet_hook, self = jax.lax.cond(
            verlet_hook.expects_call(self.counter),
            lambda self, verlet_hook: verlet_hook.post(self),
            lambda self, verlet_hook: (verlet_hook, self),
            self,
            verlet_hook,
        )

        # Common post-processing of a single step
        self.time += self.timestep
        self = self.compute_properties(verlet_hook=verlet_hook)

        self.counter += 1

        return self, verlet_hook

    @jit_decorator
    def _compute_ekin(self):
        return 0.5 * jnp.sum((self.vel**2 * self.masses.reshape(-1, 1)))

    @jit_decorator
    def compute_properties(self, verlet_hook: VerletHook):
        # assert self.delta is not None

        self.ekin = self._compute_ekin()
        self.temp = self.ekin / self.ndof * 2.0 / boltzmann
        self.etot = self.ekin + self.epot
        self.econs = self.etot

        self.econs += verlet_hook.econs_correction

        if self._cons_err_tracker is None:
            self._cons_err_tracker = ConsErrTracker(counter=self.counter, ekin_m=self.ekin, econs_m=self.econs)
        else:
            self._cons_err_tracker = self._cons_err_tracker.update(self.ekin, self.econs)

        self.cons_err = self._cons_err_tracker.get()
        if self.ff.system.cell.nvec > 0:
            self.ptens = (jnp.dot(self.vel.T * self.masses, self.vel) - self.vtens) / self.ff.system.cell.volume
            self.press = jnp.trace(self.ptens) / 3

        return self

    def finalize(self):
        pass

    def call_hooks(self, verlet_hook):
        # state_updated = False

        if jnp.any(jnp.isnan(self.ff.system.sp.coordinates)):
            raise ValueError(f"sp containes nans: {self.ff.system.sp=}")

        if self.ff.system.sp.cell is not None:
            if jnp.any(jnp.isnan(self.ff.system.sp.cell)):
                raise ValueError(f"sp containes nans: {self.ff.system.sp=}")

        for hook in [verlet_hook, *self.other_hooks]:
            if hook.expects_call(self.counter):
                hook(self)

    def step(self: VerletIntegrator, verlet_hook: VerletHook):
        self, verlet_hook = self.propagate(verlet_hook)
        self.call_hooks(verlet_hook)

        if self.cons_err > 2.0:
            raise ValueError("Energy conservation error too large, stopping simulation.")

        return self, verlet_hook

    def run(self: VerletIntegrator, verlet_hook: VerletHook, nstep=None):
        if nstep is None:
            while True:
                self, verlet_hook = self.step(verlet_hook)
        else:
            for i in range(nstep):
                self, verlet_hook = self.step(verlet_hook)
        self.finalize()
