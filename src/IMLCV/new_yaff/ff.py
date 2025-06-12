from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from IMLCV.base.datastructures import MyPyTreeNode, field
from IMLCV.new_yaff.system import YaffSys

if TYPE_CHECKING:
    from IMLCV.implementations.MdEngine import NewYaffEngine

from IMLCV.base.bias import EnergyResult


# @partial(dataclass, frozen=False)
class YaffFF(MyPyTreeNode):
    system: YaffSys
    md_engine: NewYaffEngine = field(pytree_node=False)

    @staticmethod
    def create(
        md_engine: NewYaffEngine,
    ):
        yaff_ff = YaffFF(
            md_engine=md_engine,
            system=YaffSys.create(
                md=md_engine,
                tic=md_engine.static_trajectory_info,
            ),
        )

        return yaff_ff

    def compute(self, gpos=False, vtens=False) -> tuple[EnergyResult, tuple[EnergyResult, jax.Array]]:
        sp = self.system.sp

        energy = self.md_engine.get_energy(self.system.sp, gpos, vtens and sp.cell is not None)
        cv, bias = self.md_engine.get_bias(self.system.sp, gpos, vtens and sp.cell is not None)

        res = energy + bias

        return res, (bias, cv.cv)
