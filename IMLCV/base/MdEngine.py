"""MD engine class peforms MD simulations in a given NVT/NPT ensemble.

Currently, the MD is done with YAFF/OpenMM
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Union

# import ase
# import ase.geometry
# import ase.stress
import dill
import h5py
import jax.numpy as jnp
import numpy as np

import yaff.analysis.biased_sampling
import yaff.external
import yaff.pes
import yaff.pes.bias
import yaff.pes.ext
import yaff.sampling
import yaff.sampling.iterative
import yaff.system
from IMLCV.base.bias import Bias, Energy
from IMLCV.base.CV import SystemParams
from yaff.log import log
from yaff.sampling.io import XYZWriter


@dataclass
class TrajectoryInfo:
    positions: np.ndarray
    cell: np.ndarray | None = None
    forces: np.ndarray | None = None
    t: np.ndarray | None = None
    masses: np.ndarray | None = None
    e_pot: np.ndarray | None = None


class MDEngine(ABC):
    """Base class for MD engine.

    Args:
        cvs: list of cvs
        T:  temp
        P:  press
        ES:  enhanced sampling method, choice = "None","MTD"
        timestep:  step verlet integrator
        timecon_thermo: thermostat time constant
        timecon_baro:  barostat time constant
    """

    def __init__(
        self,
        bias: Bias,
        ener: Energy,
        T,
        P,
        timestep=None,
        timecon_thermo=None,
        timecon_baro=None,
        filename=None,
        write_step=100,
        screenlog=1000,
        equilibration=None,
    ) -> None:

        self.bias = bias
        self.filename = filename
        self.write_step = write_step

        self.timestep = timestep

        self.T = T
        self.thermostat = T is not None
        self.timecon_thermo = timecon_thermo
        if self.thermostat:
            assert timecon_thermo is not None

        self.P = P
        self.barostat = P is not None
        self.timecon_baro = timecon_baro
        if self.barostat:
            assert timecon_baro is not None

        self.screenlog = screenlog

        self.ener = ener

        self.equilibration = 100 * timestep if equilibration is None else equilibration

        self._init_post()

    @abstractmethod
    def _init_post(self):
        pass

    def __getstate__(self):
        keys = [
            "bias",
            "ener",
            "T",
            "P",
            "timestep",
            "timecon_thermo",
            "timecon_baro",
            "filename",
            "write_step",
            "screenlog",
        ]

        d = {}
        for key in keys:
            d[key] = self.__dict__[key]

        return [self.__class__, d]

    def __setstate__(self, arr):
        [cls, d] = arr
        d["filename"] = None

        return cls(**d)

    def save(self, file):
        with open(file, "wb") as f:
            dill.dump(self.__getstate__(), f)

    @staticmethod
    def load(file, **kwargs) -> MDEngine:

        with open(file, "rb") as f:
            [cls, d] = dill.load(f)
        d["filename"] = None

        # replace and add kwargs
        for key in kwargs.keys():
            d[key] = kwargs[key]

        return cls(**d)

    def new_bias(self, bias: Bias, filename, **kwargs) -> MDEngine:
        self.save(f"{filename}_temp")
        mde = MDEngine.load(
            f"{filename}_temp", **{"bias": bias, "filename": filename, **kwargs}
        )
        os.remove(f"{filename}_temp")
        return mde

    @abstractmethod
    def run(self, steps):
        """run the integrator for a given number of steps.

        Args:
            steps: number of MD steps
        """

    def get_trajectory(self) -> TrajectoryInfo:
        """returns numpy arrays with posititons, times, forces, energies,

        returns bias
        """
        traj = self._get_trajectory()

        index = np.argmax(traj.t > self.equilibration)

        return traj

    def _get_trajectory(self) -> TrajectoryInfo:
        """returns numpy arrays with posititons, times, forces, energies,

        returns bias
        """
        raise NotImplementedError

    @abstractmethod
    def get_state(self):
        """returns the coordinates and cell at current md step."""
        raise NotImplementedError


class YaffEngine(MDEngine):
    """MD engine with YAFF as backend.

    Args:
        ff (yaff.pes.ForceField)
    """

    def __init__(
        self,
        ener=Union[yaff.pes.ForceField, Energy, Callable],
        log_level=log.medium,
        **kwargs,
    ) -> None:

        if not isinstance(ener, _YaffFF):
            ener = _YaffFF(ener)

        self.log_level = log_level

        super().__init__(ener=ener, **kwargs)

    def _init_post(self):

        vhook = yaff.sampling.VerletScreenLog(step=self.screenlog)
        log.set_level(self.log_level)

        whook = self._whook()
        thook = self._thook()
        bhook = self._add_bias()

        hooks = [thook]
        if vhook is not None:
            hooks.append(vhook)
        if whook is not None:
            hooks.append(whook)
        if bhook.hook:
            hooks.append(bhook)

        self._verlet(hooks)

    def _add_bias(self):

        bhook = _YaffBias(self.ener, self.bias)
        part = yaff.pes.ForcePartBias(self.ener.system)
        part.add_term(bhook)
        self.ener.add_part(part)

        return bhook

    def _whook(self):
        # setup writer to collect results
        if self.filename is None:
            return None
        elif self.filename.endswith(".h5"):
            fh5 = h5py.File(self.filename, "w")
            h5writer = yaff.sampling.HDF5Writer(
                fh5, start=5 * self.write_step, step=self.write_step
            )
            whook = h5writer
        elif self.filename.endswith(".xyz"):
            xyzhook = XYZWriter(
                self.filename, start=5 * self.write_step, step=self.write_step
            )
            whook = xyzhook
        else:
            raise NotImplementedError("only h5 and xyz are supported as filename")

        return whook

    def _thook(self):
        """setup baro/thermostat."""
        if self.thermostat:
            nhc = yaff.sampling.NHCThermostat(self.T, timecon=self.timecon_thermo)
        if self.barostat:
            mtk = yaff.sampling.MTKBarostat(
                self.ener, self.T, self.P, timecon=self.timecon_baro, anisotropic=False
            )
        if self.thermostat and self.barostat:
            thook = yaff.sampling.TBCombination(nhc, mtk)
        elif self.thermostat and not self.barostat:
            thook = nhc
        elif not self.thermostat and self.barostat:
            thook = mtk
        else:
            raise NotImplementedError

        return thook

    def _verlet(self, hooks):
        self.verlet = yaff.sampling.VerletIntegrator(
            self.ener,
            self.timestep,
            temp0=self.T,
            hooks=hooks,
            # add forces as state item
            state=[self._GposContribStateItem()],
        )

    @staticmethod
    def load(file, **kwargs) -> MDEngine:
        return super().load(file, **kwargs)

    # @staticmethod
    # def create_forcefield_from_ASE(atoms, calculator) -> yaff.pes.ForceField:
    #     """Creates force field from ASE atoms instance."""

    #     class ForcePartASE(yaff.pes.ForcePart):
    #         """YAFF Wrapper around an ASE calculator.

    #         args:
    #                system (yaff.System): system object
    #                atoms (ase.Atoms): atoms object with calculator included.
    #         """

    #         def __init__(self, system: yaff.system.System, atoms, calculator):
    #             yaff.pes.ForcePart.__init__(self, 'ase', system)
    #             self.system = system
    #             self.atoms = atoms
    #             self.calculator = calculator

    #         def _internal_compute(self, gpos=None, vtens=None):
    #             self.atoms.set_positions(self.system.pos /
    #                                      molmod.units.angstrom)
    #             self.atoms.set_cell(
    #                 ase.geometry.Cell(self.system.cell._get_rvecs() /
    #                                   molmod.units.angstrom))
    #             energy = self.atoms.get_potential_energy(
    #             ) * molmod.units.electronvolt
    #             if gpos is not None:
    #                 forces = self.atoms.get_forces()
    #                 gpos[:] = -forces * molmod.units.electronvolt / \
    #                     molmod.units.angstrom
    #             if vtens is not None:
    #                 volume = np.linalg.det(self.atoms.get_cell())
    #                 stress = ase.stress.voigt_6_to_full_3x3_stress(
    #                     self.atoms.get_stress())
    #                 vtens[:] = volume * stress * molmod.units.electronvolt
    #             return energy

    #     system = yaff.System(
    #         numbers=atoms.get_atomic_numbers(),
    #         pos=atoms.get_positions() * molmod.units.angstrom,
    #         rvecs=atoms.get_cell() * molmod.units.angstrom,
    #     )
    #     system.set_standard_masses()
    #     part_ase = ForcePartASE(system, atoms, calculator)

    #     return yaff.pes.ForceField(system, [part_ase])

    def _get_trajectory(self) -> TrajectoryInfo:
        assert self.filename.endswith(".h5")
        with h5py.File(self.filename, "r") as f:
            energy = f["trajectory"]["epot"][:]
            positions = f["trajectory"]["pos"][:]
            forces = -f["trajectory"]["gpos_contribs"][:]
            if "cell" in f["trajectory"]:
                cell = f["trajectory"]["cell"][:]
            else:
                cell = None
            t = f["trajectory"]["time"][:]

        return TrajectoryInfo(
            positions=positions,
            cell=cell,
            forces=forces,
            t=t,
            masses=self.verlet.ff.system.masses,
            e_pot=energy,
        )

    # def to_ASE_traj(self):
    #     if self.filename.endswith(".h5"):
    #         with h5py.File(self.filename, 'r') as f:
    #             at_numb = f['system']['numbers']
    #             traj = []
    #             for frame, energy_au in enumerate(f['trajectory']['epot']):
    #                 pos_A = f['trajectory']['pos'][
    #                     frame, :, :] / molmod.units.angstrom
    #                 # vel_x = f['trajectory']['vel'][frame,:,:] / x
    #                 energy_eV = energy_au / molmod.units.electronvolt
    #                 forces_eVA = -f['trajectory']['gpos_contribs'][
    #                     frame, 0, :, :] * molmod.units.angstrom / \
    #                     molmod.units.electronvolt  # forces = -gpos

    #                 pbc = 'cell' in f['trajectory']
    #                 if pbc:
    #                     cell_A = f['trajectory']['cell'][
    #                         frame, :, :] / molmod.units.angstrom

    #                     vol_A3 = f['trajectory']['volume'][
    #                         frame] / molmod.units.angstrom**3
    #                     vtens_eV = f['trajectory']['vtens'][
    #                         frame, :, :] / molmod.units.electronvolt
    #                     stresses_eVA3 = vtens_eV / vol_A3

    #                     atoms = ase.Atoms(
    #                         numbers=at_numb,
    #                         positions=pos_A,
    #                         pbc=pbc,
    #                         cell=cell_A,
    #                     )
    #                     atoms.info['stress'] = stresses_eVA3
    #                 else:
    #                     atoms = ase.Atoms(numbers=at_numb, positions=pos_A)

    #                 # atoms.set_velocities(vel_x)
    #                 atoms.arrays['forces'] = forces_eVA

    #                 atoms.info['energy'] = energy_eV
    #                 traj.append(atoms)
    #         return traj
    #     else:
    #         raise NotImplementedError("only for h5, impl this")

    def run(self, steps):
        print(f"running for {steps} steps")
        self.verlet.run(int(steps))

    def get_state(self):
        """returns the coordinates and cell at current md step."""
        return [self.ener.system.pos[:], self.ener.system.cell.rvecs[:]]

    class _GposContribStateItem(yaff.sampling.iterative.StateItem):
        """Keeps track of all the contributions to the forces."""

        def __init__(self):
            yaff.sampling.iterative.StateItem.__init__(self, "gpos_contribs")

        def get_value(self, iterative):
            n = len(iterative.ff.parts)
            natom, _ = iterative.gpos.shape
            gpos_contribs = np.zeros((n, natom, 3))
            for i in range(n):
                gpos_contribs[i, :, :] = iterative.ff.parts[i].gpos
            return gpos_contribs

        def iter_attrs(self, iterative):
            yield "gpos_contrib_names", np.array(
                [part.name for part in iterative.ff.parts], dtype="S"
            )


class _YaffBias(yaff.sampling.iterative.Hook, yaff.pes.bias.BiasPotential):
    """placeholder for all classes which work with yaff."""

    def __init__(
        self,
        ff: yaff.pes.ForceField,
        bias: Bias,
    ) -> None:

        self.ff = ff
        self.bias = bias

        # not all biases have a hook
        self.hook = (self.bias.start is not None) and (self.bias.step is not None)
        if self.hook:
            self.init = True
            super().__init__(start=self.bias.start, step=self.bias.step)

        self.cvs = []

    def compute(self, gpos=None, vtens=None):

        sp = SystemParams(
            coordinates=jnp.array(self.ff.system.pos),
            cell=jnp.array(self.ff.system.cell.rvecs),
            z_array=jnp.array(self.ff.system.masses),
        )

        [ener, gpos_jax, vtens_jax] = self.bias.compute_coor(
            sp=sp, gpos=gpos is not None, vir=vtens is not None
        )

        if gpos is not None:
            gpos[:] = np.array(gpos_jax)
        if vtens is not None:
            vtens[:] = np.array(vtens_jax)

        return float(ener[:])

    def get_log(self):
        return "Yaff bias from IMLCV"

    def __call__(self, iterative):
        # skip initial hook called by verlet integrator
        if self.init:
            self.init = False
            return

        coordinates = self.ff.system.pos[:]
        cell = self.ff.system.cell.rvecs[:]

        sp = SystemParams(coordinates=coordinates, cell=cell)

        self.bias.update_bias(sp)


class _YaffFF(Energy, yaff.pes.ForceField):
    def __init__(self, ff: yaff.pes.ForceField | Callable):
        super().__init__()

        from_func = isinstance(ff, Callable)
        f = ff

        if from_func:
            ff = ff()
            assert isinstance(ff, yaff.pes.ForceField)

        # used for yaff logging

        self.__dict__ = ff.__dict__
        self.from_func = from_func
        if self.from_func:
            self.f = f

    def compute_coor(self, sp: SystemParams, gpos=None, vir=None):

        p_old = self.system.pos[:]
        c_old = self.system.cell.rvecs[:]

        self.system.pos[:] = sp.coordinates
        self.system.cell.update_rvecs(sp.cell)

        ener = self.compute(gpos=gpos, vtens=vir)

        self.system.pos[:] = p_old
        self.system.cell.update_rvecs(c_old)

        return ener

    def __getstate__(self):
        state_dict = {}

        state_dict["system"] = {
            "numbers": self.system.numbers,
            "pos": self.system.pos,
            "ffatypes": self.system.ffatypes,
            "ffatype_ids": self.system.ffatype_ids,
            "scopes": self.system.scopes,
            "scope_ids": self.system.scope_ids,
            "bonds": self.system.bonds,
            "rvecs": self.system.cell.rvecs,
            "charges": self.system.charges,
            "radii": self.system.radii,
            "valence_charges": self.system.valence_charges,
            "dipoles": self.system.dipoles,
            "radii2": self.system.radii2,
            "masses": self.system.masses,
        }

        state_dict["from_func"] = self.from_func

        state_dict["ff_dict"] = {
            "energy": self.energy,
            "gpos": self.gpos,
            "vtens": self.vtens,
        }

        if self.from_func:
            state_dict["func"] = self.f
            return state_dict

        raise NotImplementedError(
            """see https://github.com/cython/cython/issues/4713, generate
            from function instead"""
        )

        # import inspect

        # # fails:
        # # nlist <class 'yaff.pes.nlist.NeighborList'>
        # # pair_pot <class 'yaff.pes.ext.PairPotLJ'>
        # # dlist <class 'yaff.pes.dlist.DeltaList'>
        # # iclist <class 'yaff.pes.iclist.InternalCoordinateList'>
        # # vlist <class 'yaff.pes.vlist.ValenceList'>

        # def clean(k, d, p):
        #     if k == "system":
        #         d[k] = None
        #     elif k == "nlist":
        #         d[k] = None
        #     else:
        #         if k in p.__dict__.keys():
        #             d[k] = p.__dict__[k]

        #     return d

        # t = []
        # for p in self.parts:
        #     d = {}

        #     for k in inspect.signature(p.__init__).parameters.keys():
        #         d = clean(k, d, p)
        #     t.append([p.__class__, d])

        # nl = {}
        # for k in self.nlist.__dict__:
        #     nl = clean(k, nl, self.nlist)
        # nl = [p.__class__, nl]

        # return [sysdict, nl, t]

    def __setstate__(self, state_dict):

        # system = yaff.system.System(**state_dict["system"])

        if state_dict["from_func"]:
            self.__init__(state_dict["func"])
            self.system.__init__(**state_dict["system"])
            self.__dict__.update(state_dict["ff_dict"])

            self.needs_nlist_update = True

            return self

        raise NotImplementedError("see https://github.com/cython/cython/issues/4713")

        # [sysdict, nl, pp] = state_dict

        # nlist = yaff.pes.NeighborList.__new__()

        # ts = []
        # for state_dict in pp:
        #     [cls, kwargs] = state_dict
        #     if 'system' in kwargs.keys():
        #         kwargs['system'] = system
        #     if 'nlist' in kwargs.keys():
        #         kwargs['nlist'] = nlist()
        #     ts.append(cls(**kwargs))

        # ff = yaff.pes.ForceField(system, ts[1:], ts[0])

        # return self
