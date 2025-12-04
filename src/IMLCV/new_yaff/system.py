from __future__ import annotations

import jax
import jax.numpy as jnp

from IMLCV.base.datastructures import MyPyTreeNode
from IMLCV.base.MdEngine import MDEngine, StaticMdInfo, SystemParams
from IMLCV.base.CV import NeighbourList


# @partial(dataclass, frozen=False)
class YaffCell(MyPyTreeNode):
    rvecs: jax.Array

    @staticmethod
    def create(sp: SystemParams):
        c = sp.cell

        if c is None:
            c = jnp.zeros((0, 3))

        return YaffCell(rvecs=c)

    @property
    def nvec(self):
        return self.rvecs.shape[0]

    @property
    def volume(self):
        if self.nvec == 0:
            return jnp.array(jnp.nan)

        vol_unsigned = jnp.linalg.det(self.rvecs)

        return jnp.abs(vol_unsigned)


# @partial(dataclass, frozen=False)
class YaffSys(MyPyTreeNode):
    numbers: jax.Array
    masses: jax.Array
    pos: jax.Array
    cell: YaffCell
    charges: jax.Array | None = None

    tic: StaticMdInfo

    nl: NeighbourList | None = None

    @staticmethod
    def create(sp: SystemParams, tic: StaticMdInfo, nl: NeighbourList | None = None):
        return YaffSys(
            numbers=tic.atomic_numbers,
            masses=jnp.array(tic.masses),
            pos=sp.coordinates,
            cell=YaffCell.create(sp),
            tic=tic,
        )

    @property
    def natom(self):
        return self.pos.shape[0]

    @property
    def sp(self):
        # print(f"getting sp {self.cell.rvecs}")

        return SystemParams(
            coordinates=self.pos,
            cell=self.cell.rvecs if self.cell.rvecs.shape[0] != 0 else None,
        )

    def update_nl(self):
        if self.nl is None:
            return

        self.nl = self.nl.slow_update_nl(self.sp)
