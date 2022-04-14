from __future__ import annotations
from abc import ABC
from functools import partial
from typing import Dict, Tuple
from ase import Atoms
from attr import attr

# import multiprocessing_on_dill as multiprocessing

import dill

from ase.io import write, read

from IMLCV.base.MdEngine import MDEngine
from IMLCV.base.bias import Bias, CompositeBias, NoneBias
from collections import Iterable
import os
import pathos

from math import floor

from molmod.constants import boltzmann
from molmod.units import nanosecond, kjmol
import numpy as np

import h5py


class Rounds(ABC):

    engine_keys = ["T", "P", "timecon_thermo", "timecon_baro"]

    def __init__(self, extension, folder="output") -> None:
        if extension != "extxyz":
            raise NotImplementedError("file type not known")

        self.round = -1
        self.extension = extension
        self.i = 0

        if not os.path.isdir(folder):
            os.makedirs(folder)

        self.folder = folder

        self.h5file = f"{folder}/rounds.h5"

        #overwrite if already exists
        h5py.File(self.h5file, 'w')

    def save(self):
        with open(f'{self.folder}/rounds', 'wb') as f:
            dill.dump(self, f)

    @staticmethod
    def load(folder) -> Rounds:
        with open(f"{folder}/rounds", 'rb') as f:
            self = dill.load(f)
        return self

    def add(self, i, dict, attrs=None):

        assert all(key in dict for key in ['energy', 'positions', 'forces', 'cell'])

        with h5py.File(self.h5file, 'r+') as f:
            f.create_group(f'{self.round}/{i}')

            for key in dict:
                if dict[key] is not None:
                    f[f'{self.round}/{i}'][key] = dict[key]

            if attrs is not None:
                for key in attrs:
                    if attrs[key] is not None:
                        f[f'{self.round}/{i}'].attrs[key] = attrs[key]

            f[f'{self.round}'].attrs['num'] += 1

    def new_round(self, attr):
        self.round += 1
        self.i = 0

        with h5py.File(self.h5file, 'r+') as f:
            f.create_group(f"{self.round}")

            for key in attr:
                if attr[key] is not None:
                    f[f'{self.round}'].attrs[key] = attr[key]

            f[f'{self.round}'].attrs['num'] = 0

    def iter_md_runs(self, round=None, num=3):
        if round == None:
            round = self.round
        with h5py.File(self.h5file, 'r') as f:
            for round in range(max(self.round - num, 0), self.round + 1):
                for i in f[f'{round}']:
                    yield f[f"{round}/{i}"]

    def _get_prop(self, name, round=None):
        with h5py.File(self.h5file, 'r') as f:
            if round is not None:
                f2 = f[f"{round}"]
            else:
                f2 = f

            if name in f2.attrs:
                return f2.attrs[name]

            return None

    @property
    def T(self):
        return self._get_prop('T')

    @property
    def P(self):
        return self._get_prop('P')

    def n(self, round=None):
        if round is None:
            round = self.round
        return self._get_prop('num', round=round)


class RoundsMd(Rounds):
    """helper class to save/load all data in a consistent way. Gets passed between modules"""

    trajectory_keys = [
        "filename",
        "write_step",
        "screenlog",
        "timestep",
    ]

    def __init__(self, extension, folder="output") -> None:
        super().__init__(extension=extension, folder=folder)

    def add(self, md: MDEngine, i=None):
        """adds all the saveble info of the md simulation. The resulting """

        if i == None:
            i = self.i
            self.i += 1

        self._validate(md)

        d, attr = RoundsMd._add(md, i, f'{self.folder}/round_{self.round}/bias_{i}')

        super().add(d, attrs=attr, i=i)

    @staticmethod
    def _add(md: MDEngine, i, name_bias):
        d = md.get_trajectory()
        md.bias.save(name_bias)

        attr = {k: md.__dict__[k] for k in RoundsMd.trajectory_keys}
        attr['name_bias'] = name_bias

        return [d, attr]

    def _validate(self, md: MDEngine):

        pass

        md0 = self._get_prop(self.round, 0, 'engine')

        #check equivalency of CVs
        md.bias.cvs == md0.bias.cvs

        #check equivalency of md engine params

        for k in self.engine_keys:
            assert md0.__dict__[k] == md.__dict__[k]

        #check equivalency of energy source
        dill.dumps(md.ener) == dill.dumps(md.ener)

    def new_round(self, md: MDEngine):

        r = self.round + 1

        dir = f'{self.folder}/round_{r}'
        if not os.path.isdir(dir):
            os.mkdir(dir)

        name_md = f'{self.folder}/round_{r}/engine'
        name_bias = f'{self.folder}/round_{r}/bias'
        md.save(name_md)
        md.bias.save(name_bias)

        attr = {key: md.__dict__[key] for key in self.engine_keys}
        attr['name_md'] = name_md
        attr['name_bias'] = name_bias

        super().new_round(attr=attr)

        if r == 0:
            with h5py.File(self.h5file, 'r+') as f:
                for key in self.engine_keys:
                    item = getattr(md, key)
                    if item is not None:
                        f.attrs[key] = item

    def get_bias(self, round=None, i=None):
        if round == None:
            round = self.round

        with h5py.File(self.h5file, 'r') as f:
            if i == None:
                bn = f[f'{round}'].attrs['name_bias']
            else:
                bn = f[f'{round}'][i].attrs['name_bias']
            return Bias.load(bn)

    def get_engine(self, round):
        with h5py.File(self.h5file, 'r') as f:
            MDEngine.load(f[f'{round}'].attrs['name_md'], filename=None)

    def run_par(self, biases: Iterable[Bias], steps):

        with h5py.File(self.h5file, 'r') as f:
            common_bias_name = f[f'{self.round}'].attrs['name_bias']
            common_md_name = f[f'{self.round}'].attrs['name_md']

        kwargs = []
        for i, b in enumerate(biases):
            kwargs.append({
                'bias': b,
                'new_name': f'{self.folder}/round_{self.round}/temp_{i}.h5',
                'steps': steps,
                'i': i
            })

        def _run_par(args):
            bias = args['bias']
            temp_name = args['new_name']
            steps = args['steps']
            i = args['i']

            b = CompositeBias([Bias.load(common_bias_name), bias])
            md = MDEngine.load(common_md_name, filename=temp_name, bias=b)

            md.run(steps=steps)

            d, attr = RoundsMd._add(md, i, f'{self.folder}/round_{self.round}/bias_{i}')

            return [d, attr, i]

        with pathos.pools.ProcessPool() as pool:
            for [d, attr, i] in pool.map(_run_par, kwargs):
                super().add(dict=d, attrs=attr, i=i)

        for kw in kwargs:
            os.remove(kw['new_name'])

    def unbias_rounds(self, cutoff=50 * kjmol, plot=False) -> Rounds:
        raise NotImplementedError

        ts = self.timestep
        beta = 1 / (self.T * boltzmann)

        positions = []
        cells = []
        energies = []
        forces = []
        biases = []
        kwargs = []

        for trajs, bias, t_kwargs in self.iter_md_runs(num=1):

            kwargs.append(t_kwargs)
            t_positions = []
            t_cells = []
            t_energies = []
            t_forces = []
            t_biases = []

            for atoms in trajs:
                t_positions.append(atoms.positions)
                t_cells.append(atoms.cell.array)

                forces = atoms.arrays['forces']
                gpos = np.zeros(forces.shape)
                b, gpos_jax, _ = bias.compute_coor(atoms.positions, atoms.cell.array, gpos=gpos)

                #compensate for bias
                t_energies.append(atoms.info['energy'] - b)
                t_forces.append(forces + np.array(gpos_jax))  #F=-gpos

                t_biases.append(b)

            positions.append(np.array(t_positions))
            cells.append(np.array(t_cells))
            energies.append(np.array(t_energies))
            forces.append(np.array(t_forces))
            biases.append(np.array(t_biases))

        n_positions = []
        n_cells = []

        rts = np.exp(-beta * cutoff)
        new_ts = rts * ts

        for (
                t_positions,
                t_cells,
                t_biases,
                # t_kwargs,
        ) in zip(
                positions,
                cells,
                biases,
                # kwargs,
        ):
            dts = np.zeros(t_biases.shape)
            dts[1:] = (np.exp(-beta * t_biases[1:]) + np.exp(-beta * t_biases[:-1])) / 2
            nts = np.cumsum(dts)  # * ts * t_kwargs['write_step']

            num = floor((nts.max() - nts.min()) / rts)
            nt = np.linspace(nts.min(), num * rts, num=num + 1)

            interp = lambda y: np.apply_along_axis(lambda x: np.interp(x=nt, xp=nts, fp=x), axis=0, arr=y)

            n_positions.append(interp(t_positions))
            n_cells.append(interp(t_cells))

        #reconstruct new self object
        md: MDEngine = self.engine
        cv = md.bias.cvs
        md = MDEngine.new_bias(md, bias=NoneBias(cvs=cv), filename=None, kwargs={'timestep': new_ts})

        new_rounds = RoundsMd(self.extension, f"{self.folder}_unbiased")
        new_rounds.new_round(md)

        for (t_positions, t_cells, t_biases) in zip(
                n_positions,
                n_cells,
                energies,
                forces,
        ):
            pass

        return n_positions, n_cells
