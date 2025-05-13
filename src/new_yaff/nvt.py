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
"""Thermostats"""

from __future__ import division

from dataclasses import KW_ONLY
from functools import partial

import jax
import jax.numpy as jnp
from flax.struct import dataclass, field

from IMLCV.base.UnitsConstants import boltzmann, femtosecond
from new_yaff.iterative import StateItem
from new_yaff.utils import clean_momenta, get_ndof_internal_md, get_random_vel, stabilized_cholesky_decomp
from new_yaff.verlet import VerletHook, VerletIntegrator


@partial(dataclass, frozen=False)
class AndersenThermostat(VerletHook):
    name = "Andersen"
    kind = "stochastic"
    method = "thermostat"

    _: KW_ONLY

    temp: float
    select: jax.Array | None = None
    annealing: float = 1.0

    """
    This is an implementation of the Andersen thermostat. The method
    is described in:

            Andersen, H. C. J. Chem. Phys. 1980, 72, 2384-2393.

    **Arguments:**

    temp
            The average temperature of the NVT ensemble

    **Optional arguments:**

    start
            The first iteration at which this hook is called

    step
            The number of iterations between two subsequent calls to this
            hook.

    select
            An array of atom indexes to indicate which atoms controlled by
            the thermostat.

    annealing
            After every call to this hook, the temperature is multiplied
            with this annealing factor. This effectively cools down the
            system.
    """

    def init(self, iterative: VerletIntegrator):
        # It is mandatory to zero the external momenta.
        iterative.pos, iterative.vel = clean_momenta(
            iterative.pos, iterative.vel, iterative.masses, iterative.ff.system.cell
        )

    def pre(self, iterative: VerletIntegrator, G1_add=None):
        # Andersen thermostat step before usual Verlet hook, since it largely affects the velocities
        # Needed to correct the conserved quantity
        ekin_before = iterative._compute_ekin()
        # Change the (selected) velocities
        key, key0 = jax.random.split(self.key)
        self.key = key

        if self.select is None:
            iterative.vel = get_random_vel(self.temp, False, iterative.masses, key0)
        else:
            iterative.vel = iterative.vel.at[self.select].set(
                get_random_vel(self.temp, False, iterative.masses, self.select, key0)
            )
        # Zero any external momenta after choosing new velocities
        iterative.pos, iterative.vel = clean_momenta(
            iterative.pos, iterative.vel, iterative.masses, iterative.ff.system.cell
        )
        # Update the kinetic energy and the reference for the conserved quantity
        ekin_after = iterative._compute_ekin()
        self.econs_correction += ekin_before - ekin_after
        # Optional annealing
        self.temp *= self.annealing

    def post(self, iterative: VerletIntegrator, G1_add=None):
        pass


@partial(dataclass, frozen=False)
class BerendsenThermostat(VerletHook):
    name = "Berendsen"
    kind = "deterministic"
    method = "thermostat"

    _: KW_ONLY

    temp: float
    timecon: float = 100 * femtosecond

    """
    This is an implementation of the Berendsen thermostat. The algorithm
    is described in:

            Berendsen, H. J. C.; Postma, J. P. M.; van Gunsteren, W. F.;
            Dinola, A.; Haak, J. R. J. Chem. Phys. 1984, 81, 3684-3690

    **Arguments:**

    temp
            The temperature of thermostat.

    **Optional arguments:**

    start
            The step at which the thermostat becomes active.

    timecon
            The time constant of the Berendsen thermostat.

    """

    def init(self, iterative: VerletIntegrator):
        if not self.restart:
            # It is mandatory to zero the external momenta.
            iterative.pos, iterative.vel = clean_momenta(
                iterative.pos, iterative.vel, iterative.masses, iterative.ff.system.cell
            )
        if iterative.ndof is None:
            iterative.ndof = get_ndof_internal_md(iterative.pos.shape[0], iterative.ff.system.cell.nvec)

    def pre(self, iterative: VerletIntegrator, G1_add=None):
        ekin = iterative.ekin
        temp_inst = 2.0 * iterative.ekin / (boltzmann * iterative.ndof)
        c = jnp.sqrt(1 + iterative.timestep / self.timecon * (self.temp / temp_inst - 1))
        iterative.vel = c * iterative.vel
        iterative.ekin = iterative._compute_ekin()
        self.econs_correction += (1 - c**2) * ekin

    def post(self, iterative: VerletIntegrator, G1_add=None):
        pass


@partial(dataclass, frozen=False)
class LangevinThermostat(VerletHook):
    name = "Langevin"
    kind = "stochastic"
    method = "thermostat"

    _: KW_ONLY

    temp: float
    key: jax.Array = field(default_factory=lambda: jax.random.key(42))
    timecon: float = 100 * femtosecond

    def init(self, iterative: VerletIntegrator):
        # It is mandatory to zero the external momenta.
        iterative.pos, iterative.vel = clean_momenta(
            iterative.pos, iterative.vel, iterative.masses, iterative.ff.system.cell
        )

    def pre(self, iterative: VerletIntegrator, G1_add=None):
        ekin0 = iterative.ekin
        # Actual update
        self.thermo(iterative)
        ekin1 = iterative.ekin
        self.econs_correction += ekin0 - ekin1

    def post(self, iterative: VerletIntegrator, G1_add=None):
        ekin0 = iterative.ekin
        # Actual update
        self.thermo(iterative)
        ekin1 = iterative.ekin
        self.econs_correction += ekin0 - ekin1

    def thermo(self, iterative: VerletIntegrator):
        key, key1 = jax.random.split(self.key, 2)
        self.key = key

        c1 = jnp.exp(-iterative.timestep / self.timecon / 2)
        c2 = jnp.sqrt((1.0 - c1**2) * self.temp * boltzmann / iterative.masses).reshape(-1, 1)
        iterative.vel = c1 * iterative.vel + c2 * jax.random.normal(key1, shape=iterative.vel.shape)
        iterative.ekin = iterative._compute_ekin()


@partial(dataclass, frozen=False)
class CSVRThermostat(VerletHook):
    name = "CSVR"
    kind = "stochastic"
    method = "thermostat"

    _: KW_ONLY

    temp: float
    timecon: float = 100 * femtosecond
    kin: float | None = None

    """
        This is an implementation of the CSVR thermostat. The equations are
        derived in:

            Bussi, G.; Donadio, D.; Parrinello, M. J. Chem. Phys. 2007,
            126, 014101

        The implementation (used here) is derived in

            Bussi, G.; Parrinello, M. Comput. Phys. Commun. 2008, 179, 26-29

    **Arguments:**

    temp
            The temperature of thermostat.

    **Optional arguments:**

    start
            The step at which the thermostat becomes active.

    timecon
            The time constant of the CSVR thermostat.
    """

    def init(self, iterative: VerletIntegrator):
        # It is mandatory to zero the external momenta.
        iterative.pos, iterative.vel = clean_momenta(
            iterative.pos, iterative.vel, iterative.masses, iterative.ff.system.cell
        )
        if iterative.ndof is None:
            iterative.ndof = get_ndof_internal_md(iterative.pos.shape[0], iterative.ff.system.cell.nvec)
        self.kin = 0.5 * iterative.ndof * boltzmann * self.temp

    def pre(self, iterative: VerletIntegrator, G1_add=None):
        c = jnp.exp(-iterative.timestep / self.timecon)
        R = jnp.random.normal(0, 1)
        S = (jnp.random.normal(0, 1, iterative.ndof - 1) ** 2).sum()
        iterative.ekin = iterative._compute_ekin()
        fact = (1 - c) * self.kin / iterative.ndof / iterative.ekin
        alpha = jnp.sign(R + jnp.sqrt(c / fact)) * jnp.sqrt(c + (S + R**2) * fact + 2 * R * jnp.sqrt(c * fact))
        iterative.vel = alpha * iterative.vel
        iterative.ekin_new = alpha**2 * iterative.ekin
        self.econs_correction += (1 - alpha**2) * iterative.ekin
        iterative.ekin = iterative.ekin_new

    def post(self, iterative: VerletIntegrator, G1_add=None):
        pass


@partial(dataclass, frozen=False)
class GLEThermostat(VerletHook):
    name = "GLE"
    kind = "stochastic"
    method = "thermostat"

    _: KW_ONLY

    temp: float
    timecon: float = 100 * femtosecond
    kin: float | None = None
    a_p: jax.Array
    c_p: jax.Array | None = None
    ns: int | None = None
    key: jax.Array = field(default_factory=lambda: jax.random.key(42))

    """
    This hook implements the coloured noise thermostat. The equations
    are derived in:

        Ceriotti, M.; Bussi, G.; Parrinello, M J. Chem. Theory Comput.
        2010, 6, 1170-1180.

    **Arguments:**

    temp
        The temperature of thermostat.

    a_p
        Square drift matrix, with elements fitted to the specific problem.


    **Optional arguments:**

    c_p
        Square static covariance matrix. In equilibrium, its elements are fixed.
        For non-equilibrium dynamics, its elements should be fitted.

    start
        The step at which the thermostat becomes active.
    """

    def init(self, iterative: VerletIntegrator):
        self.ns = int(self.a_p.shape[0] - 1)
        if self.c_p is None:
            # Assume equilibrium dynamics if c_p is not provided
            self.c_p = boltzmann * self.temp * jnp.eye(self.ns + 1)

        # It is mandatory to zero the external momenta
        iterative.pos, iterative.vel = clean_momenta(
            iterative.pos, iterative.vel, iterative.masses, iterative.ff.system.cell
        )
        # Initialize the additional momenta
        self.s = 0.5 * boltzmann * self.temp * jnp.random.normal(size=(self.ns, iterative.pos.size))
        # Determine the update matrices
        eigval, eigvec = jnp.linalg.eig(-self.a_p * iterative.timestep / 2)
        self.t = jnp.dot(eigvec * jnp.exp(eigval), jnp.linalg.inv(eigvec)).real
        self.S = stabilized_cholesky_decomp(self.c_p - jnp.dot(jnp.dot(self.t, self.c_p), self.t.T)).real
        # Store the number of atoms for later use
        self.n_atoms = iterative.pos.shape[0]

    def pre(self, iterative: VerletIntegrator, G1_add=None):
        self.thermo(iterative)

    def post(self, iterative: VerletIntegrator, G1_add=None):
        self.thermo(iterative)

    def thermo(self, iterative: VerletIntegrator):
        ekin0 = iterative.ekin
        # define a (3N,) vector of rescaled momenta
        p = jnp.dot(jnp.diag(jnp.sqrt(iterative.masses)), iterative.vel).reshape(-1)
        # extend the s to include the real momenta
        s_extended_old = jnp.vstack([p, self.s])
        # update equation

        self.key, key0 = jax.random.split(self.key, 2)

        s_extended_new = jnp.dot(self.t, s_extended_old) + jnp.dot(
            self.S, jax.random.normal(key0, shape=(self.ns + 1, 3 * self.n_atoms))
        )
        # store the new variables in the correct place
        iterative.vel = jnp.dot(
            jnp.diag(jnp.sqrt(1.0 / iterative.masses)), s_extended_new[0, :].reshape((self.n_atoms, 3))
        )
        self.s[:] = s_extended_new[1 : s_extended_new.shape[0], :]
        # update the kinetic energy
        iterative.ekin = iterative._compute_ekin()
        # update the conserved quantity
        ekin1 = iterative.ekin
        self.econs_correction += ekin0 - ekin1


@partial(dataclass, frozen=False)
class NHChain(object):
    _: KW_ONLY

    lenth: int = 3
    timestep: float
    temp: float
    timecon: float = 100 * femtosecond

    restart_pos: bool = False
    restart_vel: bool = False

    pos0: jax.Array | None = None
    vel0: jax.Array | None = None

    masses: jax.Array | None = None

    ndof: int = 0

    key: jax.Array = field(default_factory=lambda: jax.random.key(42))

    def __post_init__(self):
        if self.pos0 is not None:
            self.restart_pos = True
        if self.vel0 is not None:
            self.restart_vel = True

        if self.ndof > 0:  # avoid setting self.masses with zero gaussian-width in set_ndof if ndof=0
            self.set_ndof(self.ndof)

        # allocate degrees of freedom
        if self.restart_pos:
            self.pos = self.pos0.copy()
        else:
            self.pos = jnp.zeros(self.length)
        if self.restart_vel:
            self.vel = self.vel0.copy()
        else:
            self.vel = jnp.zeros(self.length)

    def set_ndof(self, ndof):
        # set the masses according to the time constant
        self.ndof = ndof
        angfreq = 2 * jnp.pi / self.timecon
        self.masses = jnp.ones(self.length) * (boltzmann * self.temp / angfreq**2)
        self.masses[0] *= ndof
        if not self.restart_vel:
            self.vel = self.get_random_vel_therm()

    def get_random_vel_therm(self):
        # generate random velocities for the thermostat velocities using a Gaussian distribution
        shape = self.length

        key, key0 = jax.random.split(self.key)
        self.key = key

        return jax.random.normal(key0, shape=shape) * jnp.sqrt(self.masses * boltzmann * self.temp) / self.masses

    def __call__(self, ekin, vel, G1_add):
        def do_bead(k, ekin):
            # Compute g
            if k == 0:
                # coupling with atoms (and barostat)
                # L = ndof (+d(d+1)/2 (aniso NPT) / +1 (iso NPT)) because of equidistant time steps.
                g = 2 * ekin - self.ndof * self.temp * boltzmann
                if G1_add is not None:
                    # add pressure contribution to g1
                    g += G1_add
            else:
                # coupling between beads
                g = self.masses[k - 1] * self.vel[k - 1] ** 2 - self.temp * boltzmann
            g /= self.masses[k]

            # Lioville operators on relevant part of the chain
            if k == self.length - 1:
                # iL G_k h/4
                self.vel = self.vel.at[k].add(g * self.timestep / 4)
            else:
                # iL vxi_{k-1} h/8
                self.vel = self.vel.at[k].mul(jnp.exp(-self.vel[k + 1] * self.timestep / 8))
                # iL G_k h/4
                self.vel = self.vel.at[k].add(g * self.timestep / 4)
                # iL vxi_{k-1} h/8
                self.vel = self.vel.at[k].mul(jnp.exp(-self.vel[k + 1] * self.timestep / 8))

        # Loop over chain in reverse order
        for k in range(self.length - 1, -1, -1):
            do_bead(k, ekin)

        # iL xi (all) h/2
        self.pos += self.vel * self.timestep / 2
        # iL Cv (all) h/2
        factor = jnp.exp(-self.vel[0] * self.timestep / 2)
        vel *= factor
        ekin *= factor**2

        # Loop over chain in forward order
        for k in range(0, self.length):
            do_bead(k, ekin)
        return vel, ekin

    def get_econs_correction(self):
        kt = boltzmann * self.temp
        # correction due to the thermostat
        return 0.5 * jnp.sum(self.vel**2 * self.masses) + kt * (self.ndof * self.pos[0] + jnp.sum(self.pos[1:]))


@partial(dataclass, frozen=False)
class NHCThermostat(VerletHook):
    name = "NHC"
    kind = "deterministic"
    method = "thermostat"

    _: KW_ONLY

    temp: float
    chain: NHChain

    """
    This hook implements the Nose-Hoover chain thermostat. The equations
    are derived in:

        Martyna, G. J.; Klein, M. L.; Tuckerman, M. J. Chem. Phys. 1992,
        97, 2635-2643.

    The implementation (used here) of a symplectic integrator of the
    Nose-Hoover chain thermostat is discussed in:

        Martyna, G. J.;  Tuckerman, M. E.;  Tobias, D. J.;  Klein,
        M. L. Mol. Phys. 1996, 87, 1117-1157.

    **Arguments:**

    temp
        The temperature of thermostat.

    **Optional arguments:**

    start
        The step at which the thermostat becomes active.

    timecon
        The time constant of the Nose-Hoover thermostat.

    chainlength
        The number of beads in the Nose-Hoover chain.

    chain_pos0
        The initial thermostat chain positions

    chain_vel0
        The initial thermostat chain velocities

    restart
        Indicates whether the initalisation should be carried out
    """

    def init(self, iterative: VerletIntegrator):
        # It is mandatory to zero the external momenta
        iterative.pos, iterative.vel = clean_momenta(
            iterative.pos, iterative.vel, iterative.masses, iterative.ff.system.cell
        )
        # If needed, determine the number of _internal_ degrees of freedom
        if iterative.ndof is None:
            iterative.ndof = get_ndof_internal_md(iterative.pos.shape[0], iterative.ff.system.cell.nvec)
        # Configure the chain
        self.chain.timestep = iterative.timestep
        self.chain.set_ndof(iterative.ndof)

    def pre(self, iterative: VerletIntegrator, G1_add=None):
        vel_new, iterative.ekin = self.chain(iterative.ekin, iterative.vel, G1_add)
        # iterative.vel[:] = vel_new
        iterative.vel = vel_new

    def post(self, iterative: VerletIntegrator, G1_add=None):
        vel_new, iterative.ekin = self.chain(iterative.ekin, iterative.vel, G1_add)
        # iterative.vel[:] = vel_new
        iterative.vel = vel_new
        self.econs_correction = self.chain.get_econs_correction()


class NHCAttributeStateItem(StateItem):
    def __init__(self, attr):
        StateItem.__init__(self, "thermo_" + attr)
        self.attr = attr

    def get_value(self, iterative: VerletIntegrator):
        chain = None
        from new_yaff.npt import TBCombination

        for hook in iterative.hooks:
            if isinstance(hook, NHCThermostat):
                chain = hook.chain
                break
            elif isinstance(hook, TBCombination):
                if isinstance(hook.thermostat, NHCThermostat):
                    chain = hook.thermostat.chain
                break
        if chain is None:
            raise TypeError("Iterative does not contain a NHCThermostat hook.")
        return getattr(chain, self.attr)

    def copy(self):
        return self.__class__(self.attr)
