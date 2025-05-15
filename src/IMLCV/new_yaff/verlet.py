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
from functools import partial

import jax
import jax.numpy as jnp
from flax.struct import dataclass, field

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


@partial(dataclass, frozen=False)
class TemperatureStateItem(StateItem):
    _: KW_ONLY
    key: str = field(pytree_node=False, default="temp")

    def get_value(self, iterative):
        return getattr(iterative, "temp", None)

    def iter_attrs(self, iterative):
        yield "ndof", iterative.ndof


@partial(dataclass, frozen=False)
class ConsErrTracker:
    """
    A class that tracks the errors on the conserved quantity.
    Given its superior numerical accuracy, the algorithm below
    is used to calculate the running average. Its properties are discussed
    in Donald Knuth's Art of Computer Programming, vol. 2, p. 232, 3rd edition.
    """

    counter: int = 0
    ekin_m: float = 0.0
    ekin_s: float = 0.0
    econs_m: float = 0.0
    econs_s: float = 0.0

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


@partial(dataclass, frozen=False)
class VerletHook(Hook):
    _: KW_ONLY

    """Specialized Verlet hook.

    This is mainly used for the implementation of thermostats and barostats.
    """

    econs_correction: float = 0.0

    def __call__(self, iterative):
        pass

    def init(self, iterative) -> tuple[VerletHook, VerletIntegrator]:
        raise NotImplementedError

    def pre(self, iterative: VerletIntegrator) -> tuple[VerletHook, VerletIntegrator]:
        raise NotImplementedError

    def post(self, iterative: VerletIntegrator) -> tuple[VerletHook, VerletIntegrator]:
        raise NotImplementedError


@partial(dataclass, frozen=False)
class NVE(VerletHook):
    _: KW_ONLY

    """Specialized Verlet hook.

    This is mainly used for the implementation of thermostats and barostats.
    """

    econs_correction: float = 0.0

    def __call__(self, iterative):
        return

    def init(self, iterative) -> tuple[VerletHook, "VerletIntegrator"]:
        return

    def pre(self, iterative: "VerletIntegrator") -> tuple[VerletHook, "VerletIntegrator"]:
        return

    def post(self, iterative: "VerletIntegrator") -> tuple[VerletHook, "VerletIntegrator"]:
        return


@partial(dataclass, frozen=False)
class VerletIntegrator:
    ff: YaffFF

    pos: jax.Array
    vel: jax.Array
    masses: jax.Array
    timestep: float

    verlet_hook: VerletHook

    temp: float | None = 0.0
    ekin: float | None = 0.0
    ekin_new: float | None = 0.0
    epot: float = 0.0
    etot: float = 0.0
    econs: float = 0.0
    cons_err: float = 0.0
    ptens: jax.Array = field(default_factory=lambda: jnp.zeros((3, 3)))
    press: float = 0.0
    rmsd_delta: float = 0.0
    rmsd_gpos: float = 0.0
    counter: int = 0
    time: float = 0.0
    gpos: jax.Array | None = None
    delta: jax.Array | None = None
    acc: jax.Array | None = None
    vtens: jax.Array | None = None
    rvecs: jax.Array | None = None
    ndof: int | None = None
    cv: jax.Array | None = None
    e_bias: float | None = None

    pos_old: jax.Array | None = field(default=None)

    other_hooks: list[Hook] = field(pytree_node=False, default_factory=list)

    _cons_err_tracker: ConsErrTracker = field(default_factory=ConsErrTracker)

    state_list: list[StateItem] = field(
        pytree_node=False,
        default_factory=lambda: [
            AttributeStateItem(key="counter"),
            AttributeStateItem(key="time"),
            AttributeStateItem(key="epot"),
            PosStateItem(),
            CellStateItem(),
            AttributeStateItem(key="vel"),
            AttributeStateItem(key="rmsd_delta"),
            AttributeStateItem(key="rmsd_gpos"),
            AttributeStateItem(key="ekin"),
            TemperatureStateItem(),
            AttributeStateItem(key="etot"),
            AttributeStateItem(key="econs"),
            AttributeStateItem(key="cons_err"),
            AttributeStateItem(key="ptens"),
            AttributeStateItem(key="vtens"),
            AttributeStateItem(key="press"),
            AttributeStateItem(key="e_bias"),
            AttributeStateItem(key="cv"),
        ],
    )

    def create(
        # self,
        ff: YaffFF,
        other_hooks: list[Hook],
        timestep,
        thermostat: VerletHook | None = None,
        barostat: VerletHook | None = None,
        vel0=None,
        temp0=300,
        scalevel0=True,
        time0=None,
        ndof=None,
        counter0=None,
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
        # Assign init arguments

        # if ff.system.masses is None:
        #     ff.system.set_standard_masses()

        # # Look for the presence of a thermostat and/or barostat

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
            print(f"sampling NPE ensemble  {barostat.name=}")
            vh = barostat

        d = dict(
            ndof=ndof,
            time=time0 if time0 is not None else 0.0,
            counter=counter0 if counter0 is not None else 0,
            pos=ff.system.pos.copy(),
            rvecs=ff.system.cell.rvecs.copy(),
            timestep=timestep,
            masses=ff.system.masses,
            vel=vel0
            if vel0 is not None
            else get_random_vel(temp0, scalevel0, ff.system.masses, key=jax.random.PRNGKey(key)),
            gpos=jnp.zeros(ff.system.pos.shape, float),
            delta=jnp.zeros(ff.system.pos.shape, float),
            vtens=jnp.zeros((3, 3), float),
            ff=ff,
            other_hooks=other_hooks,
            verlet_hook=vh,
        )

        self = VerletIntegrator(**d)

        self.initialize()

        return self

    def initialize(self: VerletIntegrator):
        # Standard initialization of Verlet algorithm

        self.gpos = jnp.zeros(self.pos.shape, float)
        self.delta = jnp.zeros(self.pos.shape, float)
        self.vtens = jnp.zeros((3, 3), float)

        self.ff.system.pos = self.pos
        self.epot, self.gpos, _, (bias, cv) = self.ff.compute(self.gpos)

        self.cv, self.e_bias = cv, bias

        self.acc = -self.gpos / self.masses.reshape(-1, 1)
        self.pos_old = self.pos.copy()

        # Allow for specialized initializations by the Verlet hooks.
        self.verlet_hook, self = self.verlet_hook.init(self)

        # Configure the number of degrees of freedom if needed
        if self.ndof is None:
            self.ndof = self.pos.size

        # Common post-processing of the initialization
        self = self.compute_properties()

        self.call_hooks()

    @jax.jit
    def propagate(self):
        # Allow specialized hooks to modify the state before the regular verlet
        # step.
        self.verlet_hook, self = self.verlet_hook.pre(self)

        # Regular verlet step
        self.acc = -self.gpos / self.masses.reshape(-1, 1)
        self.vel += 0.5 * self.acc * self.timestep
        self.pos += self.timestep * self.vel
        self.ff.system.pos = self.pos

        self.gpos = self.gpos.at[:].set(0.0)
        self.vtens = self.vtens.at[:].set(0.0)
        self.epot, self.gpos, self.vtens, (bias, cv) = self.ff.compute(self.gpos, self.vtens)

        self.cv, self.e_bias = cv, bias

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
        self: VerletIntegrator = self.compute_properties()

        self.counter += 1

        # print(f"inside {self.vel=}")

        return self

    @jax.jit
    def _compute_ekin(self):
        """Auxiliary routine to compute the kinetic energy

        This is used internally and often also by the Verlet hooks.
        """
        return 0.5 * jnp.sum((self.vel**2 * self.masses.reshape(-1, 1)))

    @jax.jit
    def compute_properties(self):
        # self.rmsd_gpos = jnp.sqrt((self.gpos**2).mean())
        self.rmsd_gpos = jnp.linalg.norm(self.gpos)
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
        state_updated = False

        for hook in [self.verlet_hook, *self.other_hooks]:
            if hook.expects_call(self.counter):
                if not state_updated:
                    for item in self.state_list:
                        item.update(self)
                    state_updated = True

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
