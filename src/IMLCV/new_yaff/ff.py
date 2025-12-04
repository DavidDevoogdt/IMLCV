from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from IMLCV.base.datastructures import MyPyTreeNode, field
from IMLCV.new_yaff.system import YaffSys

if TYPE_CHECKING:
    from IMLCV.implementations.MdEngine import NewYaffEngine

from IMLCV.base.bias import EnergyResult
from IMLCV.base.bias import Energy, Bias
from IMLCV.base.MdEngine import MDEngine, StaticMdInfo
from IMLCV.base.CV import SystemParams, NeighbourList


# @partial(dataclass, frozen=False)
class YaffFF(MyPyTreeNode):
    system: YaffSys

    energy: Energy
    bias: Bias

    # md_engine: NewYaffEngine

    @staticmethod
    def create(
        energy: Energy,
        bias: Bias,
        sp: SystemParams,
        tic: StaticMdInfo,
    ):
        yaff_ff = YaffFF(
            energy=energy,
            bias=bias,
            system=YaffSys.create(
                sp=sp,
                tic=tic,
            ),
        )

        return yaff_ff

    def compute(self, gpos=False, vtens=False) -> tuple[EnergyResult, tuple[EnergyResult, jax.Array]]:
        sp = self.system.sp
        nl = self.system.nl

        def f(sp, nl):
            return self.energy.compute_from_system_params(
                gpos=gpos,
                vir=vtens,
                sp=sp,
                nl=nl,
            )

        if self.energy.external_callback:

            def _mock_f(sp):
                return EnergyResult(
                    energy=jnp.array(1.0),
                    gpos=None if not gpos else sp.coordinates,
                    vtens=None if not vtens else sp.cell,
                )

            dtypes = jax.eval_shape(_mock_f, sp)

            energy = jax.pure_callback(
                f,
                dtypes,
                sp,
                self,
            )
        else:
            energy = f(sp, self)

        cv, bias = self.bias.compute_from_system_params(
            sp=sp,
            nl=nl,
            gpos=gpos,
            vir=vtens,
        )

        res = energy + bias

        return res, (bias, cv.cv)

    def __setstate__(self, state: dict):
        system = state.pop("system")

        if "system" in state:
            print("Warning: 'system' found in state during unpickling YaffFF, overwriting.")

            mde: MDEngine = state.pop("md_engine")
            energy = mde.energy
            bias = mde.bias
        else:
            energy: Energy = state.pop("energy")
            bias: Bias = state.pop("bias")

        self.system = system
        self.energy = energy
        self.bias = bias
