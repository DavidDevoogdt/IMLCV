from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp
# import numpy as np


from IMLCV.base.MdEngine import MDEngine, StaticMdInfo
from flax.struct import dataclass, field
from functools import partial
# from IMLCV.base.MdEngine import StaticMdInfo


@partial(dataclass, frozen=False)
class YaffSys:
    _md: MDEngine
    _tic: StaticMdInfo

    @partial(dataclass, frozen=False)
    class YaffCell:
        _md: MDEngine

        @property
        def rvecs(self):
            if self._md.sp.cell is None:
                return jnp.zeros((0, 3))
            return jnp.array(self._md.sp.cell)

        @rvecs.setter
        def rvecs(self, rvecs):
            self._md.sp.cell = jnp.array(rvecs)

        def update_rvecs(self, rvecs):
            self.rvecs = rvecs

        @property
        def nvec(self):
            return self.rvecs.shape[0]

        @property
        def volume(self):
            if self.nvec == 0:
                return jnp.nan

            vol_unsigned = jnp.linalg.det(self.rvecs)
            if vol_unsigned < 0:
                print("cell volume was negative")

            return jnp.abs(vol_unsigned)

    def __post_init__(self):
        self._cell = self.YaffCell(_md=self._md)

    @property
    def numbers(self):
        return self._tic.atomic_numbers

    @property
    def masses(self):
        return jnp.array(self._tic.masses)

    @property
    def charges(self):
        return None

    @property
    def cell(self):
        return self._cell

    @property
    def pos(self):
        return jnp.array(self._md.sp.coordinates)

    @pos.setter
    def pos(self, pos):
        self._md.sp.coordinates = jnp.array(pos)

    @property
    def natom(self):
        return self.pos.shape[0]
