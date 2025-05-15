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
"""Barostats"""

from __future__ import division

from dataclasses import KW_ONLY
from functools import partial

import jax
import jax.numpy as jnp
from flax.struct import dataclass, field

from IMLCV.base.UnitsConstants import (
    bar,
    boltzmann,
    femtosecond,
)
from IMLCV.new_yaff.iterative import StateItem
from IMLCV.new_yaff.nvt import NHCThermostat
from IMLCV.new_yaff.utils import (
    cell_symmetrize,
    clean_momenta,
    get_ndof_baro,
    get_ndof_internal_md,
    get_random_vel_press,
)
from IMLCV.new_yaff.verlet import VerletHook, VerletIntegrator


@partial(dataclass, frozen=False)
class TBCombination(VerletHook):
    name = "TBCombination"

    _: KW_ONLY

    thermostat: VerletHook
    barostat: VerletHook

    step_thermo: int | None = None
    step_baro: int | None = None

    baro_expects_call: bool = field(pytree_node=False, default=True)
    thermo_expects_call: bool = field(pytree_node=False, default=True)

    chainvel0: float | None = None
    G1_add: float | None = None

    """
    VerletHook combining an arbitrary Thermostat and Barostat instance, which
    ensures these instances are called in the correct succession, and possible
    coupling between both is handled correctly.

    **Arguments:**

    thermostat
        A Thermostat instance

    barostat
        A Barostat instance

    """

    def init(self, iterative: VerletIntegrator):
        self.step_thermo = self.thermostat.step
        self.step_baro = self.barostat.step

        # different steps is curently disabled for jax

        assert self.step_baro == self.step_thermo, "different steps is curently disabled for jax"
        assert self.step_baro == 1, "different steps is curently disabled for jax"

        self.step = min(self.thermostat.step, self.barostat.step)

        if not self.verify():
            raise TypeError("The Thermostat or Barostat instance is not supported (yet)")

        # verify whether ndof is given as an argument
        set_ndof = iterative.ndof is not None
        # initialize the thermostat and barostat separately
        self.thermostat, iterative = self.thermostat.init(iterative)
        self.barostat, iterative, self.barostat.init(iterative)
        # ensure ndof = 3N if the center of mass movement is not suppressed and ndof is not determined by the user
        from IMLCV.new_yaff.nvt import GLEThermostat, LangevinThermostat

        p_cm_fluct = (
            isinstance(self.thermostat, LangevinThermostat)
            or isinstance(self.thermostat, GLEThermostat)
            or isinstance(self.barostat, LangevinBarostat)
        )
        if (not set_ndof) and p_cm_fluct:
            iterative.ndof = iterative.pos.size
        # variables which will determine the coupling between thermostat and barostat
        self.chainvel0 = None
        self.G1_add = None

        return self, iterative

    def pre(self, iterative: VerletIntegrator):
        # determine whether the barostat should be called
        if self.baro_expects_call:
            from IMLCV.new_yaff.nvt import NHCThermostat

            if isinstance(self.thermostat, NHCThermostat):
                # in case the barostat is coupled with a NHC thermostat:
                # v_{xi,1} is needed to update v_g
                self.chainvel0 = self.thermostat.chain.vel[0]
            # actual barostat update
            self.barostat, iterative = self.barostat.pre(iterative, self.chainvel0)
        # determine whether the thermostat should be called
        if self.thermo_expects_call:
            if isinstance(self.barostat, MTKBarostat):
                # in case the thermostat is coupled with a MTK barostat:
                # update equation of v_{xi,1} is altered via G_1
                self.G1_add = self.barostat.add_press_cont()
            # actual thermostat update
            self.thermostat, iterative = self.thermostat.pre(iterative, self.G1_add)

        return self, iterative

    def post(self, iterative: VerletIntegrator):
        # determine whether the thermostat should be called
        if self.thermo_expects_call:
            if isinstance(self.barostat, MTKBarostat):
                # in case the thermostat is coupled with a MTK barostat:
                # update equation of v_{xi,1} is altered via G_1
                self.G1_add = self.barostat.add_press_cont()
            # actual thermostat update
            self.thermostat, iterative = self.thermostat.post(iterative, self.G1_add)
        # determine whether the barostat should be called
        if self.baro_expects_call:
            from IMLCV.new_yaff.nvt import NHCThermostat

            if isinstance(self.thermostat, NHCThermostat):
                # in case the barostat is coupled with a NHC thermostat:
                # v_{xi,1} is needed to update v_g
                self.chainvel0 = self.thermostat.chain.vel[0]
            # actual barostat update
            self.barostat, iterative = self.barostat.post(iterative, self.chainvel0)
        # update the correction on E_cons due to thermostat and barostat
        self.econs_correction = self.thermostat.econs_correction + self.barostat.econs_correction
        if isinstance(self.thermostat, NHCThermostat):
            if isinstance(self.barostat, MTKBarostat) and self.barostat.baro_thermo is not None:
                pass
            else:
                # extra correction necessary if particle NHC thermostat is used to thermostat the barostat
                kt = boltzmann * self.thermostat.temp
                baro_ndof = self.barostat.baro_ndof
                self.econs_correction += baro_ndof * kt * self.thermostat.chain.pos[0]

        return self, iterative

    # def expectscall(self, iterative: VerletIntegrator, kind):
    #     return True

    def __call__(self, iterative: VerletIntegrator):
        self.thermo_expects_call = (
            iterative.counter >= self.start and (iterative.counter - self.start) % self.step_thermo == 0
        )

        self.baro_expects_call = (
            iterative.counter >= self.start and (iterative.counter - self.start) % self.step_thermo == 0
        )

    def verify(self):
        # returns whether the thermostat and barostat instances are currently supported by yaff
        from IMLCV.new_yaff.nvt import (
            AndersenThermostat,
            BerendsenThermostat,
            CSVRThermostat,
            GLEThermostat,
            LangevinThermostat,
            NHCThermostat,
        )

        thermo_correct = False
        baro_correct = False
        thermo_list = [
            AndersenThermostat,
            NHCThermostat,
            LangevinThermostat,
            BerendsenThermostat,
            CSVRThermostat,
            GLEThermostat,
        ]
        baro_list = [McDonaldBarostat, BerendsenBarostat, LangevinBarostat, MTKBarostat, PRBarostat, TadmorBarostat]
        if any(isinstance(self.thermostat, thermo) for thermo in thermo_list):
            thermo_correct = True
        if any(isinstance(self.barostat, baro) for baro in baro_list):
            baro_correct = True
        return thermo_correct and baro_correct


@partial(dataclass, frozen=False)
class McDonaldBarostat(VerletHook):
    name = "McDonald"
    kind = "stochastic"
    method = "barostat"

    _: KW_ONLY

    temp: float
    press: float
    amp: float = 1e-3
    dim: int = 3
    baro_ndof: int = 1

    key: jax.Array = field(default_factory=lambda: jax.random.key(42))

    """
    Warning: this code is not fully tested yet!

    **Arguments:**

    temp
            The average temperature of the jnp. ensemble

    press
            The external pressure of the jnp. ensemble

    **Optional arguments:**

    start
            The first iteration at which this hook is called

    step
            The number of iterations between two subsequent calls to this
            hook.

    amp
            The amplitude of the changes in the logarithm of the volume.
    """

    def init(self, iterative: VerletIntegrator):
        return self, iterative

    def pre(self, iterative: VerletIntegrator, chainvel0=None):
        return self, iterative

    def post(self, iterative: VerletIntegrator, chainvel0=None):
        def compute(pos, rvecs):
            iterative.pos = pos
            iterative.gpos = iterative.gpos.at[:].set(0.0)

            iterative.ff.system.cell.rvecs = rvecs
            iterative.ff.system.pos = pos

            iterative.epot, iterative.gpos, _, _ = iterative.ff.compute(iterative.gpos)
            iterative.acc = -iterative.gpos / iterative.masses.reshape(-1, 1)

            return iterative

        natom = iterative.ff.system.natom
        # Change the logarithm of the volume isotropically.

        key, key0, key1 = jax.random.split(self.key, 3)
        self.key = key

        scale = jnp.exp(jax.random.uniform(key0, minval=-self.amp, maxval=self.amp))
        # Keep track of old state
        vol0 = iterative.ff.system.cell.volume
        epot0 = iterative.epot
        rvecs0 = iterative.ff.system.cell.rvecs.copy()
        pos0 = iterative.pos.copy()
        # Scale the system and recompute the energy
        iterative = compute(pos0 * scale, rvecs0 * scale)
        epot1 = iterative.epot
        vol1 = iterative.ff.system.cell.volume
        # Compute the acceptance ratio
        beta = 1 / (boltzmann * self.temp)
        arg = epot1 - epot0 + self.press * (vol1 - vol0) - (natom + 1) / beta * jnp.log(vol1 / vol0)
        accepted = arg < 0 or jax.random.uniform(key1, minval=0.0, maxval=1.0) < jnp.exp(-beta * arg)
        if accepted:
            # add a correction to the conserved quantity
            self.econs_correction += epot0 - epot1
        else:
            # revert the cell and the positions in the original state
            iterative = compute(pos0, rvecs0)

        return self, iterative


@partial(dataclass, frozen=False)
class BerendsenBarostat(VerletHook):
    name = "Berendsen"
    kind = "deterministic"
    method = "barostat"

    _: KW_ONLY

    temp: float
    press: float
    timecon_press: float = 1000 * femtosecond
    beta: float = 4.57e-5 / bar

    anisotropic: bool = True
    vol_constraint = False
    mass_press: float | None = None
    dim: int = None
    baro_ndof: int | None = None
    cell: jax.Array | None = None

    """
    This hook implements the Berendsen barostat. The equations are derived in:

        Berendsen, H. J. C.; Postma, J. P. M.; van Gunsteren, W. F.;
        Dinola, A.; Haak, J. R. J. Chem. Phys. 1984, 81, 3684-3690

    **Arguments:**

    ff
        A ForceField instance.

    temp
        The temperature of thermostat.

    press
        The applied pressure for the barostat.

    **Optional arguments:**

    start
        The step at which the thermostat becomes active.

    timecon
        The time constant of the Berendsen barostat.

    beta
        The isothermal compressibility, conventionally the compressibility of liquid water

    anisotropic
        Defines whether anisotropic cell fluctuations are allowed.

    vol_constraint
        Defines whether the volume is allowed to fluctuate.
    """

    def init(self, iterative: VerletIntegrator):
        self.mass_press = 3.0 * self.timecon / self.beta
        self.dim = iterative.ff.system.cell.nvec
        self.baro_ndof = get_ndof_baro(self.dim, self.anisotropic, self.vol_constraint)

        self.cell = iterative.ff.system.cell.rvecs.copy()

        self.timestep_press = iterative.timestep
        if not self.restart:
            iterative.pos, iterative.vel = clean_momenta(
                iterative.pos, iterative.vel, iterative.masses, iterative.ff.system.cell
            )
        # compute gpos and vtens, since they differ
        # after symmetrising the cell tensor
        iterative.gpos = iterative.gpos.at[:].set(0.0)

        iterative.vtens = iterative.vtens.at[:].set(0.0)

        iterative.epot, iterative.gpos, iterative.vtens, _ = iterative.ff.compute(iterative.gpos, iterative.vtens)
        if iterative.ndof is None:
            iterative.ndof = get_ndof_internal_md(iterative.pos.shape[0], iterative.ff.system.cell.nvec)
        # rescaling of the barostat mass, to be in accordance with Langevin and MTTK
        self.mass_press *= jnp.sqrt(iterative.ndof)

        return self, iterative

    def pre(self, iterative: VerletIntegrator, chainvel0=None):
        return self, iterative

    def post(self, iterative: VerletIntegrator, chainvel0=None):
        # for bookkeeping purposes
        epot0 = iterative.epot
        # calculation of the internal pressure tensor
        ptens = (
            jnp.dot(iterative.vel.T * iterative.masses, iterative.vel) - iterative.vtens
        ) / iterative.ff.system.cell.volume
        # determination of mu
        dmu = self.timestep_press / self.mass_press * (self.press * jnp.eye(3) - ptens)
        if self.vol_constraint:
            dmu -= jnp.trace(dmu) / self.dim * jnp.eye(self.dim)
        mu = jnp.eye(3) - dmu
        mu = 0.5 * (mu + mu.T)
        if not self.anisotropic:
            mu = ((jnp.trace(mu) / 3.0) ** (1.0 / 3.0)) * jnp.eye(3)
        # updating the positions and cell vectors
        pos_new = jnp.dot(iterative.pos, mu)
        rvecs_new = jnp.dot(iterative.rvecs, mu)
        iterative.ff.system.pos = pos_new
        # iterative.pos[:] = pos_new
        iterative.pos = pos_new
        iterative.ff.system.cell.rvecs = rvecs_new
        # iterative.rvecs[:] = rvecs_new
        iterative.rvecs = rvecs_new
        # calculation of the virial tensor

        iterative.gpos = iterative.gpos.at[:].set(0.0)

        iterative.vtens = iterative.vtens.at[:].set(0.0)
        iterative.epot, iterative.gpos, iterative.vtens, _ = iterative.ff.compute(iterative.gpos, iterative.vtens)
        epot1 = iterative.epot
        self.econs_correction += epot0 - epot1

        return self, iterative


@partial(dataclass, frozen=False)
class LangevinBarostat(VerletHook):
    name = "Langevin"
    kind = "stochastic"
    method = "barostat"

    _: KW_ONLY

    temp: float
    press: float

    key: jax.Array = field(default_factory=lambda: jax.random.key(42))

    cell: jax.Array | None = None
    timecon: float = 1000 * femtosecond
    anisotropic: bool = field(pytree_node=False, default=True)
    vol_constraint: bool = field(pytree_node=False, default=False)

    vel_press: float | None = None
    timestep_press: float | None = None
    mass_press: float | None = None

    dim: int | None = None
    baro_ndof: int | None = None

    """
    This hook implements the Langevin barostat. The equations are derived in:

        Feller, S. E.; Zhang, Y.; Pastor, R. W.; Brooks, B. R.
        J. Chem. Phys. 1995, 103, 4613-4621

    **Arguments:**

    ff
        A ForceField instance.

    temp
        The temperature of thermostat.

    press
        The applied pressure for the barostat.

    **Optional arguments:**

    start
        The step at which the barostat becomes active.

    timecon
        The time constant of the Langevin barostat.

    anisotropic
        Defines whether anisotropic cell fluctuations are allowed.

    vol_constraint
        Defines whether the volume is allowed to fluctuate.
    """

    def init(self, iterative: VerletIntegrator):
        self.dim = iterative.ff.system.cell.nvec

        self.baro_ndof = get_ndof_baro(self.dim, self.anisotropic, self.vol_constraint)

        if self.anisotropic:
            # symmetrize the cell tensor
            iterative.ff.system.cell.rvecs, iterative.ff.system.pos, _, _ = cell_symmetrize(iterative.ff)

        self.cell = iterative.ff.system.cell.rvecs.copy()

        self.timestep_press = iterative.timestep
        iterative.pos, iterative.vel = clean_momenta(
            iterative.pos, iterative.vel, iterative.masses, iterative.ff.system.cell
        )
        # set the number of internal degrees of freedom (no restriction on p_cm)
        if iterative.ndof is None:
            iterative.ndof = iterative.pos.size
        # define the barostat 'mass'
        self.mass_press = (iterative.ndof + 3) / 3 * boltzmann * self.temp * (self.timecon / (2 * jnp.pi)) ** 2
        # define initial barostat velocity

        key, key0 = jax.random.split(self.key)
        self.key = key

        self.vel_press = get_random_vel_press(self.mass_press, self.temp, key=key0)
        # make sure the volume of the cell will not change if applicable
        if self.vol_constraint:
            self.vel_press -= jnp.trace(self.vel_press) / 3 * jnp.eye(3)
        if not self.anisotropic:
            self.vel_press = self.vel_press[0][0]

        # compute gpos and vtens, since they differ
        # after symmetrising the cell tensor
        iterative.gpos = iterative.gpos.at[:].set(0.0)
        iterative.vtens = iterative.vtens.at[:].set(0.0)
        iterative.epot, iterative.gpos, iterative.vtens, _ = iterative.ff.compute(iterative.gpos, iterative.vtens)

        # self.prng = jax.random.PRNGKey(rand_int)

        return self, iterative

    def pre(self, iterative: VerletIntegrator, chainvel0=None):
        # bookkeeping
        epot0 = iterative.epot
        ekin0 = iterative.ekin
        # the actual update
        self, iterative = self.baro(iterative, chainvel0)
        # some more bookkeeping
        epot1 = iterative.epot
        ekin1 = iterative.ekin
        self.econs_correction += epot0 - epot1 + ekin0 - ekin1

        return self, iterative

    def post(self, iterative: VerletIntegrator, chainvel0=None):
        # bookkeeping
        epot0 = iterative.epot
        ekin0 = iterative.ekin
        # the actual update
        self, iterative = self.baro(iterative, chainvel0)
        # some more bookkeeping
        epot1 = iterative.epot
        ekin1 = iterative.ekin
        self.econs_correction += epot0 - epot1 + ekin0 - ekin1

        return self, iterative

    def baro(self, iterative: VerletIntegrator, chainvel0):
        def update_baro_vel(self: LangevinBarostat):
            # updates the barostat velocity tensor
            # iL h/(8*tau)
            self.vel_press *= jnp.exp(-self.timestep_press / (8 * self.timecon))
            if chainvel0 is not None:
                # iL v_{xi} v_g h/8: extra contribution due to NHC thermostat
                self.vel_press *= jnp.exp(-self.timestep_press * chainvel0 / 8)
            # definition of P_intV and G
            ptens_vol = jnp.dot(iterative.vel.T * iterative.masses, iterative.vel) - iterative.vtens
            ptens_vol = 0.5 * (ptens_vol.T + ptens_vol)
            G = (
                ptens_vol
                + (2.0 * iterative.ekin / iterative.ndof - self.press * iterative.ff.system.cell.volume) * jnp.eye(3)
            ) / self.mass_press
            R, self = self.getR()
            if self.vol_constraint:
                G -= jnp.trace(G) / self.dim * jnp.eye(self.dim)
                R -= jnp.trace(R) / self.dim * jnp.eye(self.dim)
            if not self.anisotropic:
                G = jnp.trace(G)
                R = R[0][0]
            # iL (G_g-R_p/W) h/4
            self.vel_press += (G - R / self.mass_press) * self.timestep_press / 4
            # iL h/(8*tau)
            self.vel_press *= jnp.exp(-self.timestep_press / (8 * self.timecon))
            if chainvel0 is not None:
                # iL v_{xi} v_g h/8: extra contribution due to NHC thermostat
                self.vel_press *= jnp.exp(-self.timestep_press * chainvel0 / 8)

            return self

        # first part of the barostat velocity tensor update
        self = update_baro_vel(self)

        # iL v_g h/2
        if self.anisotropic:
            Dr, Qg = jnp.linalg.eigh(self.vel_press)
            Daccr = jnp.diagflat(jnp.exp(Dr * self.timestep_press / 2))
            rot_mat = jnp.dot(jnp.dot(Qg, Daccr), Qg.T)
            pos_new = jnp.dot(iterative.pos, rot_mat)
            rvecs_new = jnp.dot(iterative.rvecs, rot_mat)
        else:
            c = jnp.exp(self.vel_press * self.timestep_press / 2)
            pos_new = c * iterative.pos
            rvecs_new = c * iterative.rvecs

        # update the positions and cell vectors
        iterative.ff.system.cell.rvecs = rvecs_new
        iterative.ff.system.pos = pos_new

        iterative.pos = pos_new
        iterative.rvecs = rvecs_new

        # update the potential energy

        iterative.gpos = iterative.gpos.at[:].set(0.0)
        iterative.vtens = iterative.vtens.at[:].set(0.0)
        iterative.epot, iterative.gpos, iterative.vtens, _ = iterative.ff.compute(iterative.gpos, iterative.vtens)

        # -iL (v_g + Tr(v_g)/ndof) h/2
        if self.anisotropic:
            if self.vol_constraint:
                Dg, Eg = jnp.linalg.eigh(self.vel_press)
            else:
                Dg, Eg = jnp.linalg.eigh(self.vel_press + (jnp.trace(self.vel_press) / iterative.ndof) * jnp.eye(3))
            Daccg = jnp.diagflat(jnp.exp(-Dg * self.timestep_press / 2))
            rot_mat = jnp.dot(jnp.dot(Eg, Daccg), Eg.T)
            vel_new = jnp.dot(iterative.vel, rot_mat)
        else:
            vel_new = (
                jnp.exp(-((1.0 + 3.0 / iterative.ndof) * self.vel_press) * self.timestep_press / 2) * iterative.vel
            )
        # iterative.vel[:] = vel_new
        iterative.vel = vel_new

        # update kinetic energy
        iterative.ekin = iterative._compute_ekin()

        # second part of the barostat velocity tensor update
        self = update_baro_vel(self)

        return self, iterative

    def getR(self):
        shape = 3, 3
        # generate random 3x3 tensor

        key, key0 = jax.random.split(self.key)
        self.key = key

        rand = jax.random.normal(key0, shape=shape) * jnp.sqrt(
            2 * self.mass_press * boltzmann * self.temp / (self.timestep_press * self.timecon)
        )
        R = jnp.zeros(shape)
        # create initial symmetric pressure velocity tensor
        for i in range(3):
            for j in range(3):
                if i >= j:
                    R = R.at[i, j].set(rand[i, j])
                else:
                    R = R.at[i, j].set(rand[j, i])
        return R, self


@partial(dataclass, frozen=False)
class MTKBarostat(VerletHook):
    name = "MTTK"
    kind = "deterministic"
    method = "barostat"

    _: KW_ONLY

    temp: float
    press: float

    key: jax.Array = field(default_factory=lambda: jax.random.key(42))

    cell: jax.Array | None = None
    timecon: float = 1000 * femtosecond
    anisotropic: bool = True
    vol_constraint: bool = False

    vel_press: jax.Array

    dim: int | None = None
    baro_ndof: int | None = None

    baro_thermo: NHCThermostat | None = None

    timecon_press: float = 1000 * femtosecond

    """
    This hook implements the Martyna-Tobias-Klein barostat. The equations
    are derived in:

        Martyna, G. J.; Tobias, D. J.; Klein, M. L. J. Chem. Phys. 1994,
        101, 4177-4189.

    The implementation (used here) of a symplectic integrator of this
    barostat is discussed in

        Martyna, G. J.;  Tuckerman, M. E.;  Tobias, D. J.;  Klein,
        M. L. Mol. Phys. 1996, 87, 1117-1157.

    **Arguments:**

    ff
        A ForceField instance.

    temp
        The temperature of thermostat.

    press
        The applied pressure for the barostat.

    **Optional arguments:**

    start
        The step at which the barostat becomes active.

    timecon
        The time constant of the Martyna-Tobias-Klein barostat.

    anisotropic
        Defines whether anisotropic cell fluctuations are allowed.

    vol_constraint
        Defines whether the volume is allowed to fluctuate.

    baro_thermo
        NHCThermostat instance, coupled directly to the barostat

    vel_press0
        The initial barostat velocity tensor

    restart
        If true, the cell is not symmetrized initially
    """

    def init(self, iterative: VerletIntegrator):
        self.dim = iterative.ff.system.cell.nvec
        self.baro_ndof = get_ndof_baro(self.dim, self.anisotropic, self.vol_constraint)

        if self.anisotropic:
            # symmetrize the cell tensor
            iterative.ff.system.cell.rvecs, iterative.ff.system.pos, _, _ = cell_symmetrize(iterative.ff)

        self.cell = iterative.ff.system.cell.rvecs.copy()

        self.timestep_press = iterative.timestep

        iterative.pos, iterative.vel = clean_momenta(
            iterative.pos, iterative.vel, iterative.masses, iterative.ff.system.cell
        )
        # determine the internal degrees of freedom
        if iterative.ndof is None:
            iterative.ndof = get_ndof_internal_md(len(iterative.ff.system.numbers), iterative.ff.system.cell.nvec)
        # determine barostat 'mass'
        angfreq = 2 * jnp.pi / self.timecon_press
        self.mass_press = (iterative.ndof + self.dim**2) * boltzmann * self.temp / angfreq**2

        key, key0 = jax.random.split(self.key)
        self.key = key

        if self.vel_press is None:
            # define initial barostat velocity
            self.vel_press = get_random_vel_press(self.mass_press, self.temp, key0)
            if not self.anisotropic:
                self.vel_press = self.vel_press[0][0]
        # initialize the barostat thermostat if present
        if self.baro_thermo is not None:
            self.baro_thermo.chain.timestep = iterative.timestep
            self.baro_thermo.chain.set_ndof(self.baro_ndof)
        # make sure the volume of the cell will not change if applicable
        if self.vol_constraint:
            self.vel_press -= jnp.trace(self.vel_press) / 3 * jnp.eye(3)
        # compute gpos and vtens, since they differ
        # after symmetrising the cell tensor

        iterative.gpos = iterative.gpos.at[:].set(0.0)

        iterative.vtens = iterative.vtens.at[:].set(0.0)
        iterative.epot, iterative.gpos, iterative.vtens, _ = iterative.ff.compute(iterative.gpos, iterative.vtens)

        return self, iterative

    def pre(self, iterative: VerletIntegrator, chainvel0=None):
        if self.baro_thermo is not None:
            # overrule the chainvel0 argument in the TBC instance
            chainvel0 = self.baro_thermo.chain.vel[0]
        # propagate the barostat
        self, iterative = self.baro(iterative, chainvel0)
        # propagate the barostat thermostat if present
        if self.baro_thermo is not None:
            # determine the barostat kinetic energy
            ekin_baro = self._compute_ekin_baro()
            # update the barostat thermostat, without updating v_g (done in self.baro)
            vel_press_copy = jnp.zeros(self.vel_press.shape)
            vel_press_copy[:] = self.vel_press
            dummy_vel, dummy_ekin = self.baro_thermo.chain(ekin_baro, vel_press_copy, 0)

        return self, iterative

    def post(self, iterative: VerletIntegrator, chainvel0=None):
        # propagate the barostat thermostat if present
        if self.baro_thermo is not None:
            # determine the barostat kinetic energy
            ekin_baro = self._compute_ekin_baro()
            # update the barostat thermostat, without updating v_g (done in self.baro)
            # vel_press_copy = jnp.zeros(self.vel_press.shape)
            vel_press_copy = self.vel_press
            dummy_vel, dummy_ekin = self.baro_thermo.chain(ekin_baro, vel_press_copy, 0)
            # overrule the chainvel0 argument in the TBC instance
            chainvel0 = self.baro_thermo.chain.vel[0]
        # propagate the barostat
        self, iterative = self.baro(iterative, chainvel0)
        # calculate the correction due to the barostat alone
        self.econs_correction = self._compute_ekin_baro()
        # add the PV term if the volume is not constrained
        if not self.vol_constraint:
            self.econs_correction += self.press * iterative.ff.system.cell.volume
        if self.baro_thermo is not None:
            # add the correction due to the barostat thermostat
            self.econs_correction += self.baro_thermo.chain.get_econs_correction()

        return self, iterative

    def baro(self, iterative: VerletIntegrator, chainvel0):
        def update_baro_vel():
            # updates the barostat velocity tensor
            if chainvel0 is not None:
                # iL v_{xi} v_g h/8
                self.vel_press *= jnp.exp(-self.timestep_press * chainvel0 / 8)
            # definition of P_intV and G
            ptens_vol = jnp.dot(iterative.vel.T * iterative.masses, iterative.vel) - iterative.vtens
            ptens_vol = 0.5 * (ptens_vol.T + ptens_vol)
            G = (
                ptens_vol
                + (2.0 * iterative.ekin / iterative.ndof - self.press * iterative.ff.system.cell.volume) * jnp.eye(3)
            ) / self.mass_press
            if not self.anisotropic:
                G = jnp.trace(G)
            if self.vol_constraint:
                G -= jnp.trace(G) / self.dim * jnp.eye(self.dim)
            # iL G_g h/4
            self.vel_press += G * self.timestep_press / 4
            if chainvel0 is not None:
                # iL v_{xi} v_g h/8
                self.vel_press *= jnp.exp(-self.timestep_press * chainvel0 / 8)

        # first part of the barostat velocity tensor update
        update_baro_vel()

        # iL v_g h/2
        if self.anisotropic:
            Dr, Qg = jnp.linalg.eigh(self.vel_press)
            Daccr = jnp.diagflat(jnp.exp(Dr * self.timestep_press / 2))
            rot_mat = jnp.dot(jnp.dot(Qg, Daccr), Qg.T)
            pos_new = jnp.dot(iterative.pos, rot_mat)
            rvecs_new = jnp.dot(iterative.rvecs, rot_mat)
        else:
            c = jnp.exp(self.vel_press * self.timestep_press / 2)
            pos_new = c * iterative.pos
            rvecs_new = c * iterative.rvecs

        # update the positions and cell vectors
        iterative.ff.system.pos = pos_new
        # iterative.pos[:] = pos_new
        iterative.pos = pos_new
        iterative.ff.system.cell.rvecs = rvecs_new
        # iterative.rvecs[:] = rvecs_new
        iterative.rvecs = rvecs_new

        # update the potential energy

        iterative.gpos = iterative.gpos.at[:].set(0.0)

        iterative.vtens = iterative.vtens.at[:].set(0.0)
        iterative.epot, iterative.gpos, iterative.vtens, _ = iterative.ff.compute(iterative.gpos, iterative.vtens)

        # -iL (v_g + Tr(v_g)/ndof) h/2
        if self.anisotropic:
            if self.vol_constraint:
                Dg, Eg = jnp.linalg.eigh(self.vel_press)
            else:
                Dg, Eg = jnp.linalg.eigh(self.vel_press + (jnp.trace(self.vel_press) / iterative.ndof) * jnp.eye(3))
            Daccg = jnp.diagflat(jnp.exp(-Dg * self.timestep_press / 2))
            rot_mat = jnp.dot(jnp.dot(Eg, Daccg), Eg.T)
            vel_new = jnp.dot(iterative.vel, rot_mat)
        else:
            vel_new = (
                jnp.exp(-((1.0 + 3.0 / iterative.ndof) * self.vel_press) * self.timestep_press / 2) * iterative.vel
            )

        # update the velocities and the kinetic energy
        # iterative.vel[:] = vel_new
        iterative.vel = vel_new
        iterative.ekin = iterative._compute_ekin()

        # second part of the barostat velocity tensor update
        update_baro_vel()

        return self, iterative

    def add_press_cont(self):
        kt = self.temp * boltzmann
        # pressure contribution to thermostat: kinetic cell tensor energy
        # and extra degrees of freedom due to cell tensor
        if self.baro_thermo is None:
            return 2 * self._compute_ekin_baro() - self.baro_ndof * kt
        else:
            # if a barostat thermostat is present, the thermostat is decoupled
            return 0

    def _compute_ekin_baro(self):
        # returns the kinetic energy associated with the cell fluctuations
        if self.anisotropic:
            return 0.5 * self.mass_press * jnp.trace(jnp.dot(self.vel_press.T, self.vel_press))
        else:
            return 0.5 * self.mass_press * self.vel_press**2


@partial(dataclass, frozen=False)
class PRBarostat(VerletHook):
    name = "PR"
    kind = "deterministic"
    method = "barostat"

    _: KW_ONLY

    temp: float
    press: float | jax.Array
    timecon_press: float = 1000 * femtosecond

    anisotropic: bool = True
    vol_constraint = False
    mass_press: float | None = None
    dim: int = None
    baro_ndof: int | None = None
    cell: jax.Array | None = None
    baro_thermo: NHCThermostat | None = None

    P: jax.Array | None = None
    S_ani: jax.Array | None = None
    cellinv0: jax.Array | None = None
    vol0: jax.Array | None = None
    Sigma: jax.Array | None = None
    vel_press: jax.Array | None = None

    """
    This hook implements the Parrinello-Rahman barostat for finite strains.
    The equations are derived in:

        Parrinello, M.; Rahman, A. J. Appl. Phys. 1981, 52, 7182-7190.

    **Arguments:**

    ff
        A ForceField instance.

    temp
        The temperature of thermostat.

    press
        The applied pressure for the barostat.

    **Optional arguments:**

    start
        The step at which the barostat becomes active.

    timecon
        The time constant of the Tadmor-Miller barostat.

    anisotropic
        Defines whether anisotropic cell fluctuations are allowed.

    vol_constraint
        Defines whether the volume is allowed to fluctuate.

    baro_thermo
        NHCThermostat instance, coupled directly to the barostat

    vel_press0
        The initial barostat velocity tensor

    restart
        If true, the cell is not symmetrized initially
    """

    def init(self, iterative: VerletIntegrator):
        if not isinstance(self.press, jax.Array):
            self.press = self.press * jnp.eye(3)

        self.dim = iterative.ff.system.cell.nvec

        self.P = jnp.trace(self.press) / 3
        self.S_ani = self.press - self.P * jnp.eye(3)
        self.S_ani = 0.5 * (
            self.S_ani + self.S_ani.T
        )  # only symmetric part of stress tensor contributes to individual motion
        self.cellinv0 = jnp.linalg.inv(iterative.ff.system.cell.rvecs.copy())
        self.vol0 = iterative.ff.system.cell.volume
        # definition of Sigma = V_0*h_0^{-1} S_ani h_0^{-T}
        self.Sigma = self.vol0 * jnp.dot(jnp.dot(self.cellinv0.T, self.S_ani), self.cellinv0)

        self.baro_ndof = get_ndof_baro(self.dim, self.anisotropic, self.vol_constraint)
        if self.anisotropic and not self.restart:
            # symmetrize the cell and stress tensor
            iterative.ff.system.cell.rvecs, iterative.ff.system.pos, vc, tn = cell_symmetrize(
                iterative.ff, tensor_list=[self.press]
            )
            self.press = tn[0]

        self.timestep_press = iterative.timestep

        iterative.pos, iterative.vel = clean_momenta(
            iterative.pos, iterative.vel, iterative.masses, iterative.ff.system.cell
        )
        # determine the internal degrees of freedom
        if iterative.ndof is None:
            iterative.ndof = get_ndof_internal_md(len(iterative.ff.system.numbers), iterative.ff.system.cell.nvec)
        # determine barostat 'mass' following W = timecon*np.sqrt(n_part) * m_av
        # n_part = len(iterative.masses)
        # self.mass_press = self.timecon_press*np.sum(iterative.masses)/np.sqrt(len(iterative.ff.system.numbers))
        angfreq = 2 * jnp.pi / self.timecon_press
        self.mass_press = (iterative.ndof + self.dim**2) * boltzmann * self.temp / angfreq**2 / self.vol0 ** (2.0 / 3)

        key, key0 = jax.random.split(self.key)
        self.key = key

        if self.vel_press is None:
            # define initial barostat velocity
            self.vel_press = get_random_vel_press(self.mass_press, self.temp, key0)
            if not self.anisotropic:
                self.vel_press = self.vel_press[0][0]
        # initialize the barostat thermostat if present
        if self.baro_thermo is not None:
            self.baro_thermo.chain.timestep = iterative.timestep
            self.baro_thermo.chain.set_ndof(self.baro_ndof)
        # make sure the volume of the cell will not change if applicable
        if self.vol_constraint:
            self.vel_press -= jnp.trace(self.vel_press) / 3 * jnp.eye(3)
        # compute gpos and vtens, since they differ
        # after symmetrising the cell tensor

        iterative.gpos = iterative.gpos.at[:].set(0.0)

        iterative.vtens = iterative.vtens.at[:].set(0.0)
        iterative.epot, iterative.gpos, iterative.vtens, _ = iterative.ff.compute(iterative.gpos, iterative.vtens)

        return self, iterative

    def pre(self, iterative: VerletIntegrator, chainvel0=None):
        if self.baro_thermo is not None:
            # overrule the chainvel0 argument in the TBC instance
            chainvel0 = self.baro_thermo.chain.vel[0]
        # propagate the barostat
        self, iterative = self.baro(iterative, chainvel0)
        # propagate the barostat thermostat if present
        if self.baro_thermo is not None:
            # determine the barostat kinetic energy
            ekin_baro = self._compute_ekin_baro()
            # update the barostat thermostat, without updating v_g (done in self.baro)
            vel_press_copy = self.vel_press
            dummy_vel, dummy_ekin = self.baro_thermo.chain(ekin_baro, vel_press_copy, 0)

        return self, iterative

    def post(self, iterative: VerletIntegrator, chainvel0=None):
        # propagate the barostat thermostat if present
        if self.baro_thermo is not None:
            # determine the barostat kinetic energy
            ekin_baro = self._compute_ekin_baro()
            # update the barostat thermostat, without updating v_g (done in self.baro)
            vel_press_copy = jnp.zeros(self.vel_press.shape)
            vel_press_copy[:] = self.vel_press
            dummy_vel, dummy_ekin = self.baro_thermo.chain(ekin_baro, vel_press_copy, 0)
            # overrule the chainvel0 argument in the TBC instance
            chainvel0 = self.baro_thermo.chain.vel[0]
        # propagate the barostat
        self, iterative = self.baro(iterative, chainvel0)
        # calculate the correction due to the barostat alone
        h = iterative.rvecs.copy()
        self.econs_correction = (
            self._compute_ekin_baro()
            + self.P * iterative.ff.system.cell.volume
            + 0.5 * jnp.trace(jnp.dot(jnp.dot(self.Sigma, h), h.T))
        )

        if self.baro_thermo is not None:
            # add the correction due to the barostat thermostat
            self.econs_correction += self.baro_thermo.chain.get_econs_correction()

        return self, iterative

    def baro(self, iterative: VerletIntegrator, chainvel0):
        def update_baro_vel():
            # updates the barostat velocity tensor
            if chainvel0 is not None:
                # iL v_{xi} v_g h/8
                self.vel_press *= jnp.exp(-self.timestep_press * chainvel0 / 8)
            h = iterative.rvecs.copy()
            ptens_vol = jnp.dot(iterative.vel.T * iterative.masses, iterative.vel) - iterative.vtens
            S_PK2_vol = jnp.dot(jnp.dot(h.T, self.Sigma), h)
            # definition of G
            G = (
                jnp.dot(
                    ptens_vol - self.P * iterative.ff.system.cell.volume * jnp.eye(3) - S_PK2_vol, jnp.linalg.inv(h)
                )
                / self.mass_press
            )
            G = 0.5 * (G + G.T)
            # G = G+G.T-np.diagflat(jnp.diag(G))
            if not self.anisotropic:
                G = jnp.trace(G)
            if self.vol_constraint:
                G -= jnp.trace(G) / self.dim * jnp.eye(self.dim)
            # iL G_h h/4
            self.vel_press += G * self.timestep_press / 4
            if chainvel0 is not None:
                # iL v_{xi} v_g h/8
                self.vel_press *= jnp.exp(-self.timestep_press * chainvel0 / 8)

        # first part of the barostat velocity tensor update
        update_baro_vel()

        # update of the positions and cell vectors
        rvecs_new = iterative.rvecs + self.vel_press * self.timestep_press / 2

        if self.anisotropic:
            Dr, Qg = jnp.linalg.eigh(jnp.dot(self.vel_press, jnp.linalg.inv(iterative.rvecs.T)))
        else:
            Dr, Qg = jnp.linalg.eigh(self.vel_press * jnp.linalg.inv(iterative.rvecs.T))
        Daccr = jnp.diagflat(jnp.exp(Dr * self.timestep_press / 2))
        Daccv = jnp.diagflat(jnp.exp(-Dr * self.timestep_press / 2))
        rot_mat_r = jnp.dot(jnp.dot(Qg, Daccr), Qg.T)
        rot_mat_v = jnp.dot(jnp.dot(Qg, Daccv), Qg.T)
        pos_new = jnp.dot(iterative.pos, rot_mat_r)
        vel_new = jnp.dot(iterative.vel, rot_mat_v)

        iterative.ff.system.pos = pos_new
        # iterative.pos[:] = pos_new
        iterative.pos = pos_new
        iterative.ff.system.cell.rvecs = rvecs_new
        # iterative.rvecs[:] = rvecs_new
        iterative.rvecs = rvecs_new
        # iterative.vel[:] = vel_new
        iterative.vel = vel_new

        # update the potential and kinetic energy

        iterative.gpos = iterative.gpos.at[:].set(0.0)

        iterative.vtens = iterative.vtens.at[:].set(0.0)
        iterative.epot, iterative.gpos, iterative.vtens, _ = iterative.ff.compute(iterative.gpos, iterative.vtens)
        iterative.ekin = iterative._compute_ekin()

        # second part of the barostat velocity tensor update
        update_baro_vel()

        return self, iterative

    def add_press_cont(self):
        kt = self.temp * boltzmann
        # pressure contribution to thermostat: kinetic cell tensor energy
        # and extra degrees of freedom due to cell tensor
        if self.baro_thermo is None:
            return 2 * self._compute_ekin_baro() - self.baro_ndof * kt
        else:
            # if a barostat thermostat is present, the thermostat is decoupled
            return 0

    def _compute_ekin_baro(self):
        # returns the kinetic energy associated with the cell fluctuations
        if self.anisotropic:
            return 0.5 * self.mass_press * jnp.trace(jnp.dot(self.vel_press.T, self.vel_press))
        else:
            return 0.5 * self.mass_press * self.vel_press**2


@partial(dataclass, frozen=False)
class TadmorBarostat(VerletHook):
    name = "Tadmor"
    kind = "deterministic"
    method = "barostat"

    _: KW_ONLY

    temp: float
    press: float | jax.Array
    timecon_press: float = 1000 * femtosecond

    anisotropic: bool = True
    vol_constraint = False
    mass_press: float | None = None
    dim: int = None
    baro_ndof: int | None = None

    baro_thermo: NHCThermostat | None = None

    vel_press: jax.Array | None = None
    cell: jax.Array | None = None
    cellinv0: jax.Array | None = None
    Strans: jax.Array | None = None
    Vol0: jax.Array | None = None

    key: jax.Array = field(default_factory=lambda: jax.random.key(42))

    """
    This hook implements the Tadmor-Miller barostat for finite strains.
    The equations are derived in:

        Tadmor, E. B.; Miller, R. E. Modeling Materials: Continuum,
        Atomistic and Multiscale Techniques 2011, 520-527.

    **Arguments:**

    ff
        A ForceField instance.

    temp
        The temperature of thermostat.

    press
        The applied second Piola-Kirchhoff tensor for the barostat.

    **Optional arguments:**

    start
        The step at which the barostat becomes active.

    timecon
        The time constant of the Tadmor-Miller barostat.

    anisotropic
        Defines whether anisotropic cell fluctuations are allowed.

    vol_constraint
        Defines whether the volume is allowed to fluctuate.

    baro_thermo
        NHCThermostat instance, coupled directly to the barostat

    vel_press0
        The initial barostat velocity tensor

    restart
        If true, the cell is not symmetrized initially
    """

    def init(self, iterative: VerletIntegrator):
        self.dim = iterative.ff.system.cell.nvec
        self.baro_ndof = get_ndof_baro(self.dim, self.anisotropic, self.vol_constraint)
        if self.anisotropic and not self.restart:
            # symmetrize the cell tensor
            iterative.ff.system.cell.rvecs, iterative.ff.system.pos, _, _ = cell_symmetrize(iterative.ff)
        self.cell = iterative.ff.system.cell.rvecs.copy()
        self.cellinv0 = jnp.linalg.inv(self.cell)
        self.vol0 = iterative.ff.system.cell.volume
        # definition of h0^{-1} S h_0^{-T}
        self.Strans = jnp.dot(jnp.dot(self.cellinv0, self.press), self.cellinv0.T)

        self.timestep_press = iterative.timestep

        iterative.pos, iterative.vel = clean_momenta(
            iterative.pos, iterative.vel, iterative.masses, iterative.ff.system.cell
        )
        # determine the internal degrees of freedom
        if iterative.ndof is None:
            iterative.ndof = get_ndof_internal_md(len(iterative.ff.system.numbers), iterative.ff.system.cell.nvec)
        # determine barostat 'mass'
        angfreq = 2 * jnp.pi / self.timecon_press
        self.mass_press = (iterative.ndof + self.dim**2) * boltzmann * self.temp / angfreq**2

        key, key0 = jax.random.split(self.key)
        self.key = key

        if self.vel_press is None:
            # define initial barostat velocity
            self.vel_press = get_random_vel_press(self.mass_press, self.temp, key0)
            if not self.anisotropic:
                self.vel_press = self.vel_press[0][0]
        # initialize the barostat thermostat if present
        if self.baro_thermo is not None:
            self.baro_thermo.chain.timestep = iterative.timestep
            self.baro_thermo.chain.set_ndof(self.baro_ndof)
        # make sure the volume of the cell will not change if applicable
        if self.vol_constraint:
            self.vel_press -= jnp.trace(self.vel_press) / 3 * jnp.eye(3)
        # compute gpos and vtens, since they differ
        # after symmetrising the cell tensor

        iterative.gpos = iterative.gpos.at[:].set(0.0)

        iterative.vtens = iterative.vtens.at[:].set(0.0)
        iterative.epot, iterative.gpos, iterative.vtens, _ = iterative.ff.compute(iterative.gpos, iterative.vtens)

        return self, iterative

    def pre(self, iterative: VerletIntegrator, chainvel0=None):
        if self.baro_thermo is not None:
            # overrule the chainvel0 argument in the TBC instance
            chainvel0 = self.baro_thermo.chain.vel[0]
        # propagate the barostat
        self.baro(iterative, chainvel0)
        # propagate the barostat thermostat if present
        if self.baro_thermo is not None:
            # determine the barostat kinetic energy
            ekin_baro = self._compute_ekin_baro()
            # update the barostat thermostat, without updating v_g (done in self.baro)
            vel_press_copy = jnp.zeros(self.vel_press.shape)
            vel_press_copy[:] = self.vel_press
            dummy_vel, dummy_ekin = self.baro_thermo.chain(ekin_baro, vel_press_copy, 0)

        return self, iterative

    def post(self, iterative: VerletIntegrator, chainvel0=None):
        # propagate the barostat thermostat if present
        if self.baro_thermo is not None:
            # determine the barostat kinetic energy
            ekin_baro = self._compute_ekin_baro()
            # update the barostat thermostat, without updating v_g (done in self.baro)
            vel_press_copy = jnp.zeros(self.vel_press.shape)
            vel_press_copy[:] = self.vel_press
            dummy_vel, dummy_ekin = self.baro_thermo.chain(ekin_baro, vel_press_copy, 0)
            # overrule the chainvel0 argument in the TBC instance
            chainvel0 = self.baro_thermo.chain.vel[0]
        # propagate the barostat
        self.baro(iterative, chainvel0)
        # calculate the Lagrangian strain
        eta = 0.5 * (
            jnp.dot(self.cellinv0.T, jnp.dot(iterative.rvecs.T, jnp.dot(iterative.rvecs, self.cellinv0))) - jnp.eye(3)
        )
        # calculate the correction due to the barostat alone
        ang_ten = self._compute_angular_tensor(iterative.pos, iterative.vel, iterative.masses)
        self.econs_correction = (
            self.vol0 * jnp.trace(jnp.dot(self.press.T, eta))
            + self._compute_ekin_baro()
            - jnp.trace(jnp.dot(ang_ten, self.vel_press))
        )
        if self.baro_thermo is not None:
            # add the correction due to the barostat thermostat
            self.econs_correction += self.baro_thermo.chain.get_econs_correction()

        return self, iterative

    def baro(self, iterative: VerletIntegrator, chainvel0):
        def update_baro_vel():
            # updates the barostat velocity tensor
            if chainvel0 is not None:
                # iL v_{xi} v_g h/8
                self.vel_press *= jnp.exp(-self.timestep_press * chainvel0 / 8)
            # definition of sigma_{int} V
            ptens_vol = jnp.dot(iterative.vel.T * iterative.masses, iterative.vel) - iterative.vtens
            ptens_vol = 0.5 * (ptens_vol.T + ptens_vol)
            # definition of (Delta sigma) V
            ang_ten = self._compute_angular_tensor(iterative.pos, iterative.vel, iterative.masses)
            dptens_vol = jnp.dot(self.vel_press.T, ang_ten.T) - jnp.dot(ang_ten.T, self.vel_press.T)
            # definition of \tilde{sigma} V
            sigmaS_vol = self.vol0 * jnp.dot(jnp.dot(iterative.rvecs, self.Strans), iterative.rvecs.T)
            # definition of G
            G = (
                ptens_vol - sigmaS_vol + dptens_vol + 2.0 * iterative.ekin / iterative.ndof * jnp.eye(3)
            ) / self.mass_press
            if not self.anisotropic:
                G = jnp.trace(G)
            if self.vol_constraint:
                G -= jnp.trace(G) / self.dim * jnp.eye(self.dim)
            # iL G_g h/4
            self.vel_press += G * self.timestep_press / 4
            if chainvel0 is not None:
                # iL v_{xi} v_g h/8
                self.vel_press *= jnp.exp(-self.timestep_press * chainvel0 / 8)

        # first part of the barostat velocity tensor update
        update_baro_vel()

        # iL v_g h/2
        if self.anisotropic:
            Dr, Qg = jnp.linalg.eigh(self.vel_press)
            Daccr = jnp.diagflat(jnp.exp(Dr * self.timestep_press / 2))
            rot_mat = jnp.dot(jnp.dot(Qg, Daccr), Qg.T)
            rvecs_new = jnp.dot(iterative.rvecs, rot_mat)
        else:
            c = jnp.exp(self.vel_press * self.timestep_press / 2)
            rvecs_new = c * iterative.rvecs

        # update the positions and cell vectors
        iterative.ff.system.cell.rvecs = rvecs_new
        iterative.rvecs = rvecs_new

        # update the potential energy
        iterative.gpos = iterative.gpos.at[:].set(0.0)
        iterative.vtens = iterative.vtens.at[:].set(0.0)
        iterative.epot, iterative.gpos, iterative.vtens, _ = iterative.ff.compute(iterative.gpos, iterative.vtens)

        # -iL (Tr(v_g)/ndof) h/2
        if self.anisotropic:
            if not self.vol_constraint:
                vel_new = jnp.exp(-jnp.trace(self.vel_press) / iterative.ndof * self.timestep_press / 2)
        else:
            vel_new = jnp.exp(-3.0 / iterative.ndof * self.vel_press * self.timestep_press / 2) * iterative.vel

        # update the velocities and the kinetic energy
        # iterative.vel[:] = vel_new
        iterative.vel = vel_new
        iterative.ekin = iterative._compute_ekin()

        # second part of the barostat velocity tensor update
        update_baro_vel()

    def add_press_cont(self):
        kt = self.temp * boltzmann
        # pressure contribution to thermostat: kinetic cell tensor energy
        # and extra degrees of freedom due to cell tensor
        if self.baro_thermo is None:
            return 2 * self._compute_ekin_baro() - self.baro_ndof * kt
        else:
            # if a barostat thermostat is present, the thermostat is decoupled
            return 0

    def _compute_ekin_baro(self):
        # returns the kinetic energy associated with the cell fluctuations
        if self.anisotropic:
            return 0.5 * self.mass_press * jnp.trace(jnp.dot(self.vel_press.T, self.vel_press))
        else:
            return 0.5 * self.mass_press * self.vel_press**2

    def _compute_angular_tensor(self, pos, vel, masses):
        return jnp.dot(pos.T, masses.reshape(-1, 1) * vel)


class MTKAttributeStateItem(StateItem):
    def __init__(self, attr):
        StateItem.__init__(self, "baro_" + attr)
        self.attr = attr

    def get_value(self, iterative: VerletIntegrator):
        baro = None

        hook = iterative.verlet_hook
        if isinstance(hook, MTKBarostat):
            baro = hook

        elif isinstance(hook, TBCombination):
            if isinstance(hook.barostat, MTKBarostat):
                baro = hook.barostat

        if baro is None:
            raise TypeError("Iterative does not contain an MTKBarostat hook.")
        if self.key.startswith("baro_chain_"):
            if baro.baro_thermo is not None:
                key = self.key.split("_")[2]
                return getattr(baro.baro_thermo.chain, key)
            else:
                return 0
        else:
            return getattr(baro, self.attr)

    def copy(self):
        return self.__class__(self.attr)
