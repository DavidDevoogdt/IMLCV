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
from typing import TYPE_CHECKING, Any, TypeVar

import jax
import jax.numpy as jnp

from IMLCV.base.datastructures import MyPyTreeNode, field


class StateItem(MyPyTreeNode):
    key: str = field(pytree_node=False)
    value: Any

    def update(self, iterative):
        self.value = self.get_value(iterative)

    def get_value(self, iterative) -> Any:
        raise NotImplementedError


class AttributeStateItem(StateItem):
    def get_value(self, iterative):
        return getattr(iterative, self.key, None)


class ConsErrStateItem(StateItem):
    def get_value(self, iterative):
        return getattr(iterative._cons_err_tracker, self.key, None)


class PosStateItem(StateItem):
    key = "pos"

    def get_value(self, iterative):
        return iterative.ff.system.pos


class CellStateItem(StateItem):
    key = "cell"

    def get_value(self, iterative):
        return iterative.ff.system.cell.rvecs


class Hook(MyPyTreeNode):
    name: str = field(pytree_node=False, default="hook")
    kind: str = field(pytree_node=False, default="base")
    method: str = field(pytree_node=False, default="none")
    start: jax.Array = field(default_factory=lambda: jnp.array(0))
    step: jax.Array = field(default_factory=lambda: jnp.array(1))

    def expects_call(self, counter):
        return jnp.logical_and(counter >= self.start, (counter - self.start) % self.step == 0)

    def __call__(self, iterative):
        raise NotImplementedError
