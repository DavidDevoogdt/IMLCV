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
"""Base class for iterative algorithms"""

from __future__ import division

# import numpy as np

from flax.struct import dataclass, field
import jax.numpy as jnp

from functools import partial

from IMLCV.base.MdEngine import MDEngine, StaticMdInfo


# @dataclass
# class YaffSys:
#     _md: MDEngine
#     _tic: StaticMdInfo

#     @dataclass
#     class YaffCell:
#         _md: MDEngine

#         @property
#         def rvecs(self):
#             if self._md.sp.cell is None:
#                 return jnp.zeros((0, 3))
#             return jnp.array(self._md.sp.cell)

#         @rvecs.setter
#         def rvecs(self, rvecs):
#             self._md.sp.cell = jnp.array(rvecs)

#         def update_rvecs(self, rvecs):
#             self.rvecs = rvecs

#         @property
#         def nvec(self):
#             return self.rvecs.shape[0]

#         @property
#         def volume(self):
#             if self.nvec == 0:
#                 return jnp.nan

#             vol_unsigned = jnp.linalg.det(self.rvecs)
#             if vol_unsigned < 0:
#                 print("cell volume was negative")

#             return jnp.abs(vol_unsigned)

#     def __post_init__(self):
#         self._cell = self.YaffCell(_md=self._md)

#     @property
#     def numbers(self):
#         return self._tic.atomic_numbers

#     @property
#     def masses(self):
#         return jnp.array(self._tic.masses)

#     @property
#     def charges(self):
#         return None

#     @property
#     def cell(self):
#         return self._cell

#     @property
#     def pos(self):
#         return jnp.array(self._md.sp.coordinates)

#     @pos.setter
#     def pos(self, pos):
#         self._md.sp.coordinates = jnp.array(pos)

#     @property
#     def natom(self):
#         return self.pos.shape[0]


# class YaffFF:
#     def __init__(
#         self,
#         md_engine: YaffEngine,
#         name="IMLCV_YAFF_forcepart",
#         additional_parts=[],
#     ):
#         self.md_engine = md_engine

#         self.energy = 0.0
#         self.gpos = jnp.zeros((self.system.natom, 3), float)
#         self.vtens = jnp.zeros((3, 3), float)
#         self.clear()

#     @property
#     def system(self):
#         return self.md_engine.yaff_system

#     @system.setter
#     def system(self, sys):
#         assert sys == self.system

#     @property
#     def sp(self):
#         return self.md_engine.sp

#     def update_rvecs(self, rvecs):
#         self.clear()
#         self.system.cell.rvecs = rvecs

#     def update_pos(self, pos):
#         self.clear()
#         self.system.pos = pos

#     def _internal_compute(self, gpos, vtens):
#         # print(f"inside _internal_compute {gpos=} {vtens=} {self.sp=}  ")

#         energy = self.md_engine.get_energy(
#             gpos is not None,
#             vtens is not None and self.md_engine.sp.cell is not None,
#         )

#         cv, bias = self.md_engine.get_bias(
#             gpos is not None,
#             vtens is not None and self.md_engine.sp.cell is not None,
#         )

#         res = energy + bias

#         self.md_engine.last_ener = energy
#         self.md_engine.last_bias = bias
#         self.md_engine.last_cv = cv

#         if res.gpos is not None:
#             gpos[:] += jnp.array(res.gpos, dtype=np.float64)
#         if res.vtens is not None:
#             vtens[:] += jnp.array(res.vtens, dtype=np.float64)

#         return res.energy

#     def clear(self):
#         self.energy = jnp.nan
#         self.gpos[:] = jnp.nan
#         self.vtens[:] = jnp.nan

#     def compute(self, gpos=None, vtens=None):
#         """Compute the energy and optionally some derivatives for this FF (part)

#         The only variable inputs for the compute routine are the atomic
#         positions and the cell vectors, which can be changed through the
#         ``update_rvecs`` and ``update_pos`` methods. All other aspects of
#         a force field are considered to be fixed between subsequent compute
#         calls. If changes other than positions or cell vectors are needed,
#         one must construct new ``ForceField`` and/or ``ForcePart`` objects.

#         **Optional arguments:**

#         gpos
#                 The derivatives of the energy towards the Cartesian coordinates
#                 of the atoms. ('g' stands for gradient and 'pos' for positions.)
#                 This must be a writeable numpy array with shape (N, 3) where N
#                 is the number of atoms.

#         vtens
#                 The force contribution to the pressure tensor. This is also
#                 known as the virial tensor. It represents the derivative of the
#                 energy towards uniform deformations, including changes in the
#                 shape of the unit cell. (v stands for virial and 'tens' stands
#                 for tensor.) This must be a writeable numpy array with shape (3,
#                 3). Note that the factor 1/V is not included.

#         The energy is returned. The optional arguments are Fortran-style
#         output arguments. When they are present, the corresponding results
#         are computed and **added** to the current contents of the array.
#         """
#         if gpos is None:
#             my_gpos = None
#         else:
#             my_gpos = self.gpos
#             my_gpos[:] = 0.0
#         if vtens is None:
#             my_vtens = None
#         else:
#             my_vtens = self.vtens
#             my_vtens[:] = 0.0
#         self.energy = self._internal_compute(my_gpos, my_vtens)
#         if jnp.isnan(self.energy):
#             raise ValueError("The energy is not-a-number (nan).")
#         if gpos is not None:
#             if jnp.isnan(my_gpos).any():
#                 raise ValueError("Some gpos element(s) is/are not-a-number (nan).")
#             gpos += my_gpos
#         if vtens is not None:
#             if jnp.isnan(my_vtens).any():
#                 raise ValueError("Some vtens element(s) is/are not-a-number (nan).")
#             vtens += my_vtens

#         # print(f"inside compute {self.energy=} {gpos=} {vtens=}")

#         return self.energy


class StateItem(object):
    def __init__(self, key):
        self.key = key
        self.shape = None
        self.dtype = None

    def update(self, iterative):
        self.value = self.get_value(iterative)
        if self.shape is None:
            if isinstance(self.value, jnp.ndarray):
                self.shape = self.value.shape
                self.dtype = self.value.dtype
            else:
                self.shape = tuple([])
                self.dtype = type(self.value)

    def get_value(self, iterative):
        raise NotImplementedError

    def iter_attrs(self, iterative):
        return []

    def copy(self):
        return self.__class__()


class AttributeStateItem(StateItem):
    def get_value(self, iterative):
        return getattr(iterative, self.key, None)

    def copy(self):
        return self.__class__(self.key)


class ConsErrStateItem(StateItem):
    def get_value(self, iterative):
        return getattr(iterative._cons_err_tracker, self.key, None)

    def copy(self):
        return self.__class__(self.key)


class PosStateItem(StateItem):
    def __init__(self):
        StateItem.__init__(self, "pos")

    def get_value(self, iterative):
        return iterative.ff.system.pos


class CellStateItem(StateItem):
    def __init__(self):
        StateItem.__init__(self, "cell")

    def get_value(self, iterative):
        return iterative.ff.system.cell.rvecs


class Hook(object):
    name = None
    kind = None
    method = None

    def __init__(self, start=0, step=1):
        """
        **Optional arguments:**

        start
             The first iteration at which this hook should be called.

        step
             The hook will be called every `step` iterations.
        """
        self.start = start
        self.step = step

    def expects_call(self, counter):
        return counter >= self.start and (counter - self.start) % self.step == 0

    def __call__(self, iterative):
        raise NotImplementedError
