from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from flax.struct import dataclass

from IMLCV.base.MdEngine import MDEngine, StaticMdInfo, SystemParams


@partial(dataclass, frozen=False)
class YaffCell:
    rvecs: jax.Array

    def create(sp: SystemParams):
        c = sp.cell

        if sp.cell is None:
            c = jnp.zeros((0, 3))

        return YaffCell(rvecs=c)

    @property
    def nvec(self):
        return self.rvecs.shape[0]

    @property
    def volume(self):
        if self.nvec == 0:
            return jnp.nan

        vol_unsigned = jnp.linalg.det(self.rvecs)

        return jnp.abs(vol_unsigned)


@partial(dataclass, frozen=False)
class YaffSys:
    numbers: jax.Array
    masses: jax.Array
    pos: jax.Array
    cell: YaffCell
    charges: jax.Array | None = None

    def create(md: MDEngine, tic: StaticMdInfo):
        return YaffSys(
            numbers=tic.atomic_numbers,
            masses=jnp.array(tic.masses),
            pos=md.sp.coordinates,
            cell=YaffCell.create(md.sp),
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
