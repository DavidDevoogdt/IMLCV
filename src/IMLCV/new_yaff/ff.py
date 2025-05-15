from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from flax.struct import dataclass, field

from IMLCV.new_yaff.system import YaffSys

if TYPE_CHECKING:
    from IMLCV.implementations.MdEngine import NewYaffEngine


@partial(dataclass, frozen=False)
class YaffFF:
    energy: float
    gpos: jax.Array
    vtens: jax.Array

    system: YaffSys

    md_engine: NewYaffEngine = field(pytree_node=False)

    def create(
        md_engine: NewYaffEngine,
    ):
        yaff_ff = YaffFF(
            md_engine=md_engine,
            system=YaffSys.create(
                md=md_engine,
                tic=md_engine.static_trajectory_info,
            ),
            energy=0.0,
            gpos=jnp.zeros((md_engine.sp.shape[0], 3), float),
            vtens=jnp.zeros((3, 3), float),
        )

        yaff_ff.clear()

        return yaff_ff

    @property
    def sp(self):
        return self.md_engine.sp

    # def update_rvecs(self, rvecs):
    #     self.clear()
    #     self.system.cell.rvecs = rvecs

    # def update_pos(self, pos):
    #     self.clear()
    #     self.system.pos = pos

    def clear(self):
        self.energy = jnp.nan
        self.gpos = self.gpos.at[:].set(jnp.nan)
        self.vtens = self.vtens.at[:].set(jnp.nan)

    def compute(self, gpos=None, vtens=None):
        get_gpos = gpos is not None
        get_vtens = vtens is not None and self.md_engine.sp.cell is not None

        # update the position
        # self.md_engine.sp = self.system.sp

        # print(f"{self.system.sp=}")

        energy = self.md_engine.get_energy(self.system.sp, get_gpos, get_vtens)

        cv, bias = self.md_engine.get_bias(self.system.sp, get_gpos, get_vtens)

        res = energy + bias

        if get_gpos:
            gpos += jnp.array(res.gpos, dtype=jnp.float64)

        if get_vtens:
            vtens += jnp.array(res.vtens, dtype=jnp.float64)

        return res.energy, gpos, vtens, (bias.energy, cv.cv)
