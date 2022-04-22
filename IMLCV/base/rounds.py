from __future__ import annotations
from abc import ABC
from functools import partial
from pickletools import optimize
from turtle import pos
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
from molmod.units import nanosecond, kjmol, picosecond
import numpy as np
from numpy import average, interp, linalg, linspace
import scipy as sp
from scipy import interpolate, optimize

import h5py


class Rounds(ABC):

    ENGINE_KEYS = ["T", "P", "timecon_thermo", "timecon_baro"]

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
            self: Rounds = dill.load(f)
        return self

    def add(self, i, dict, attrs=None):

        assert all(key in dict for key in ['energy', 'positions', 'forces', 'cell', 't'])

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

        self.save()

    def new_round(self, attr):
        self.round += 1
        self.i = 0

        dir = f'{self.folder}/round_{self.round}'
        if not os.path.isdir(dir):
            os.mkdir(dir)

        with h5py.File(self.h5file, 'r+') as f:
            f.create_group(f"{self.round}")

            for key in attr:
                if attr[key] is not None:
                    f[f'{self.round}'].attrs[key] = attr[key]

            f[f'{self.round}'].attrs['num'] = 0

        self.save()

    def iter(self, round=None, num=3):
        num = num - 1
        if round == None:
            round = self.round

        for round in range(max(self.round - num, 0), self.round + 1):
            round = self._get_round(round)
            for i in round['names']:
                i_dict = self._get_i(round['round'], i)
                yield {**i_dict, 'round': round}

    def _get_round(self, round):
        with h5py.File(self.h5file, 'r') as f:
            rounds = [k for k in f[f'{round}'].keys()]

            d = f[f"{round}"].attrs
            r_attr = {key: d[key] for key in d}
            r_attr['round'] = round
            r_attr['names'] = rounds
        return r_attr

    def _get_i(self, round, i):
        with h5py.File(self.h5file, 'r') as f:
            d = f[f"{round}/{i}"]
            y = {key: d[key][:] for key in d}
            attr = {key: d.attrs[key] for key in d.attrs}
        return {**y, 'attr': attr, 'i': i}

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
        return self._get_prop('T', round=self.round)

    @property
    def P(self):
        return self._get_prop('P', round=self.round)

    def n(self, round=None):
        if round is None:
            round = self.round
        return self._get_prop('num', round=round)


class RoundsCV(Rounds):
    """class for unbiased rounds"""
    pass


class RoundsMd(Rounds):
    """helper class to save/load all data in a consistent way. Gets passed between modules"""

    TRAJ_KEYS = [
        "filename",
        "write_step",
        "screenlog",
        "timestep",
    ]

    ENGINE_KEYS = ['timestep', *Rounds.ENGINE_KEYS]

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

        attr = {k: md.__dict__[k] for k in RoundsMd.TRAJ_KEYS}
        attr['name_bias'] = name_bias

        return [d, attr]

    def _validate(self, md: MDEngine):

        pass

        md0 = self._get_prop(self.round, 0, 'engine')

        #check equivalency of CVs
        md.bias.cvs == md0.bias.cvs

        #check equivalency of md engine params

        for k in self.ENGINE_KEYS:
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

        attr = {key: md.__dict__[key] for key in self.ENGINE_KEYS}
        attr['name_md'] = name_md
        attr['name_bias'] = name_bias

        super().new_round(attr=attr)

    def get_bias(self, round=None, i=None) -> Bias:
        if round == None:
            round = self.round

        with h5py.File(self.h5file, 'r') as f:
            if i == None:
                bn = f[f'{round}'].attrs['name_bias']
            else:
                bn = f[f'{round}'][i].attrs['name_bias']
            return Bias.load(bn)

    def get_engine(self, round=None) -> MDEngine:
        if round == None:
            round = self.round
        with h5py.File(self.h5file, 'r') as f:
            return MDEngine.load(f[f'{round}'].attrs['name_md'], filename=None)

    def run(self, bias, steps):
        self.run_par([bias], steps)

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
                'i': i + self.i
            })

        def _run_par(args):
            bias = args['bias']
            temp_name = args['new_name']
            steps = args['steps']
            i = args['i']

            if bias is None:
                b = Bias.load(common_bias_name)
            else:
                b = CompositeBias([Bias.load(common_bias_name), bias])
            md = MDEngine.load(common_md_name, filename=temp_name, bias=b)

            md.run(steps=steps)

            d, attr = RoundsMd._add(md, i, f'{self.folder}/round_{self.round}/bias_{i}')

            return [d, attr, i]

        if len(biases) != 1:
            with pathos.pools.ProcessPool() as pool:
                for [d, attr, i] in pool.map(_run_par, kwargs):
                    super().add(dict=d, attrs=attr, i=i)
        else:
            [d, attr, i] = _run_par(kwargs[0])
            super().add(dict=d, attrs=attr, i=i)

        self.i += len(kwargs)

        for kw in kwargs:
            os.remove(kw['new_name'])

    def unbias_rounds(self, steps=1e5, num=1e7, calc=False) -> RoundsCV:
        import jax.numpy as jnp

        md = self.get_engine()
        if self.n() > 1 or isinstance(self.get_bias(), NoneBias) or calc == True:
            from IMLCV.base.Observable import Observable
            obs = Observable(self)
            fesBias = obs.fes_Bias(plot=True)

            md = md.new_bias(fesBias, filename=None, write_step=5)
            self.new_round(md)

        if self.n() == 0:
            self.run(None, steps)

        #construct rounds object
        dict = self._get_i(self.round, 0)
        props = self._get_round(self.round)
        beta = 1 / (props['T'] * boltzmann)
        bias = Bias.load(dict['attr']['name_bias'])

        p = dict["positions"][:]
        c = dict.get('cell')
        t_orig = dict["t"][:]

        def _interp(x_new, x, y):
            if y is not None:
                return jnp.apply_along_axis(lambda yy: jnp.interp(x_new, x, yy), arr=y, axis=0)
            return None

        def bt(x, x_orig):
            pt = partial(_interp, x=x_orig, y=p)
            ct = partial(_interp, x=x_orig, y=c)
            return bias.compute_coor(coordinates=pt(x), cell=ct(x))[0]

        bt = jnp.vectorize(bt, excluded=[1])

        def integr(fx, x):
            return np.array([0, *np.cumsum((fx[1:] + fx[:-1]) * (x[1:] - x[:-1]) / 2)])

        t = t_orig[:]
        eb = np.exp(-beta * bt(t, t_orig))
        tau = integr(1 / eb, t)
        tau_new = np.linspace(start=0.0, stop=tau.max(), num=int(num))

        p_new = _interp(tau_new, tau, dict["positions"])
        c_new = _interp(tau_new, tau, dict.get("cell"))

        roundscv = RoundsCV(self.extension, f'{self.folder}_unbiased')
        roundscv.new_round(props)
        roundscv.add(0, {'energy': None, 'positions': p_new, 'forces': None, 'cell': c_new, 't': tau_new})

        return roundscv


if __name__ == '__main__':
    from IMLCV.test.test_scheme import test_cv_discovery
    test_cv_discovery()