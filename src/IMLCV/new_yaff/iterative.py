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

from dataclasses import KW_ONLY

import jax.numpy as jnp

from IMLCV.base.datastructures import MyPyTreeNode, field


class StateItem(MyPyTreeNode):
    _: KW_ONLY

    key: str = field(pytree_node=False)
    shape: tuple[int] | None = field(pytree_node=False, default=None)
    dtype: jnp.dtype | None = field(pytree_node=False, default=None)

    def update(self, iterative):
        self.value = self.get_value(iterative)
        if self.shape is None:
            if isinstance(self.value, jnp.ndarray):
                self.shape = self.value.shape  # type: ignore
                self.dtype = self.value.dtype
            else:
                self.shape = tuple([])
                self.dtype = type(self.value)  # type: ignore

    def get_value(self, iterative):
        raise NotImplementedError

    def iter_attrs(self, iterative):
        return []

    # def copy(self):
    #     return self.__class__()


class AttributeStateItem(StateItem):
    def get_value(self, iterative):
        return getattr(iterative, self.key, None)

    # def copy(self):
    #     return self.__class__(self.key)


class ConsErrStateItem(StateItem):
    def get_value(self, iterative):
        return getattr(iterative._cons_err_tracker, self.key, None)

    # def copy(self):
    #     return self.__class__(self.key)


class PosStateItem(StateItem):
    _: KW_ONLY
    key: str = field(pytree_node=False, default="pos")

    def get_value(self, iterative):
        return iterative.ff.system.pos


class CellStateItem(StateItem):
    _: KW_ONLY
    key: str = "cell"

    def get_value(self, iterative):
        return iterative.ff.system.cell.rvecs


class Hook(MyPyTreeNode):
    name = None
    kind = None
    method = None
    start: int = 0
    step: int = 1

    def expects_call(self, counter):
        return counter >= self.start and (counter - self.start) % self.step == 0

    def __call__(self, iterative):
        raise NotImplementedError
