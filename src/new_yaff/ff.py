from __future__ import annotations

from typing import TYPE_CHECKING

# import numpy as np
import jax.numpy as jnp

from IMLCV.base.MdEngine import MDEngine


# from IMLCV.base.MdEngine import StaticMdInfo
from flax.struct import dataclass, field
from new_yaff.system import YaffSys
from functools import partial


@partial(dataclass, frozen=False)
class YaffFF:
    energy: float
    gpos: jnp.ndarray
    vtens: jnp.ndarray

    md_engine: MDEngine

    def create(
        md_engine: MDEngine,
    ):
        yaff_ff = YaffFF(
            md_engine=md_engine,
            energy=0.0,
            gpos=jnp.zeros((md_engine.yaff_system.natom, 3), float),
            vtens=jnp.zeros((3, 3), float),
        )

        yaff_ff.clear()

        return yaff_ff

    @property
    def system(self) -> YaffSys:
        return self.md_engine.yaff_system

    @system.setter
    def system(self, sys: YaffSys):
        assert sys == self.system

    @property
    def sp(self):
        return self.md_engine.sp

    def update_rvecs(self, rvecs):
        self.clear()
        self.system.cell.rvecs = rvecs

    def update_pos(self, pos):
        self.clear()
        self.system.pos = pos

    def _internal_compute(self, gpos, vtens):
        # print(f"inside _internal_compute {gpos=} {vtens=} {self.sp=}  ")

        energy = self.md_engine.get_energy(
            gpos is not None,
            vtens is not None and self.md_engine.sp.cell is not None,
        )

        cv, bias = self.md_engine.get_bias(
            gpos is not None,
            vtens is not None and self.md_engine.sp.cell is not None,
        )

        res = energy + bias

        self.md_engine.last_ener = energy
        self.md_engine.last_bias = bias
        self.md_engine.last_cv = cv

        if res.gpos is not None:
            gpos += jnp.array(res.gpos, dtype=jnp.float64)
        if res.vtens is not None:
            vtens += jnp.array(res.vtens, dtype=jnp.float64)

        # print(f"internal {res=}")

        return res.energy, gpos, vtens

    def clear(self):
        self.energy = jnp.nan
        self.gpos = self.gpos.at[:].set(jnp.nan)
        self.vtens = self.vtens.at[:].set(jnp.nan)

    def compute(self, gpos=None, vtens=None):
        """Compute the energy and optionally some derivatives for this FF (part)

        The only variable inputs for the compute routine are the atomic
        positions and the cell vectors, which can be changed through the
        ``update_rvecs`` and ``update_pos`` methods. All other aspects of
        a force field are considered to be fixed between subsequent compute
        calls. If changes other than positions or cell vectors are needed,
        one must construct new ``ForceField`` and/or ``ForcePart`` objects.

        **Optional arguments:**

        gpos
                The derivatives of the energy towards the Cartesian coordinates
                of the atoms. ('g' stands for gradient and 'pos' for positions.)
                This must be a writeable numpy array with shape (N, 3) where N
                is the number of atoms.

        vtens
                The force contribution to the pressure tensor. This is also
                known as the virial tensor. It represents the derivative of the
                energy towards uniform deformations, including changes in the
                shape of the unit cell. (v stands for virial and 'tens' stands
                for tensor.) This must be a writeable numpy array with shape (3,
                3). Note that the factor 1/V is not included.

        The energy is returned. The optional arguments are Fortran-style
        output arguments. When they are present, the corresponding results
        are computed and **added** to the current contents of the array.
        """
        if gpos is None:
            my_gpos = None
        else:
            my_gpos = jnp.full_like(self.gpos, 0.0)
        if vtens is None:
            my_vtens = None
        else:
            my_vtens = jnp.full_like(self.vtens, 0.0)
        self.energy, my_gpos, my_vtens = self._internal_compute(my_gpos, my_vtens)
        if jnp.isnan(self.energy):
            raise ValueError("The energy is not-a-number (nan).")
        if gpos is not None:
            if jnp.isnan(my_gpos).any():
                raise ValueError("Some gpos element(s) is/are not-a-number (nan).")
            gpos += my_gpos
        if vtens is not None:
            if jnp.isnan(my_vtens).any():
                raise ValueError("Some vtens element(s) is/are not-a-number (nan).")
            vtens += my_vtens

        # print(f"inside compute {self.energy=} {gpos=} {vtens=}")

        return self.energy, gpos, vtens
