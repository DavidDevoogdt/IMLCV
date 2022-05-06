from __future__ import annotations

import os
from abc import ABC
from collections import Iterable
from functools import partial

import dill
import h5py
import jax.numpy as jnp
import numpy as np
import pathos
from IMLCV.base.bias import Bias, BiasF, CompositeBias, NoneBias
from IMLCV.base.MdEngine import MDEngine
from molmod.constants import boltzmann

# import multiprocessing_on_dill as multiprocessing


class Rounds(ABC):

    ENGINE_KEYS = ["T", "P", "timecon_thermo", "timecon_baro"]

    def __init__(self, extension, folder="output", max_energy=None) -> None:
        if extension != "extxyz":
            raise NotImplementedError("file type not known")

        self.round = -1
        self.extension = extension
        self.i = 0

        self.max_energy = max_energy

        if not os.path.isdir(folder):
            os.makedirs(folder)

        self.folder = folder

        self.h5file = f"{folder}/rounds.h5"

        # overwrite if already exists
        h5py.File(self.h5file, 'w')

    def save(self):
        with open(f'{self.folder}/rounds', 'wb') as f:
            dill.dump(self, f)

    @staticmethod
    def load(folder) -> Rounds:
        with open(f"{folder}/rounds", 'rb') as f:
            self: Rounds = dill.load(f)
        return self

    def add(self, i, d, attrs=None):

        assert all(
            key in d for key in ['energy', 'positions', 'forces', 'cell', 't'])

        with h5py.File(self.h5file, 'r+') as f:
            f.create_group(f'{self.round}/{i}')

            for key in d:
                if d[key] is not None:
                    f[f'{self.round}/{i}'][key] = d[key]

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

    def iter(self, r=None, num=3):
        if r is None:
            r = self.round

        for r in range(max(r - (num - 1), 0), r + 1):
            r = self._get_round(r)
            for i in r['names']:
                i_dict = self._get_i(r['round'], i)
                yield {**i_dict, 'round': r}

    def _get_round(self, r):
        with h5py.File(self.h5file, 'r') as f:
            rounds = [k for k in f[f'{r}'].keys()]

            d = f[f"{r}"].attrs
            r_attr = {key: d[key] for key in d}
            r_attr['round'] = r
            r_attr['names'] = rounds
        return r_attr

    def _get_i(self, r, i):
        with h5py.File(self.h5file, 'r') as f:
            d = f[f"{r}/{i}"]
            y = {key: d[key][:] for key in d}
            attr = {key: d.attrs[key] for key in d.attrs}
        return {**y, 'attr': attr, 'i': i}

    def _get_prop(self, name, r=None):
        with h5py.File(self.h5file, 'r') as f:
            if r is not None:
                f2 = f[f"{r}"]
            else:
                f2 = f

            if name in f2.attrs:
                return f2.attrs[name]

            return None

    @property
    def T(self):
        return self._get_prop('T', r=self.round)

    @property
    def P(self):
        return self._get_prop('P', r=self.round)

    def n(self, r=None):
        if r is None:
            r = self.round
        return self._get_prop('num', r=r)


class RoundsCV(Rounds):
    """class for unbiased rounds."""
    pass


class RoundsMd(Rounds):
    """helper class to save/load all data in a consistent way.

    Gets passed between modules
    """

    TRAJ_KEYS = [
        "filename",
        "write_step",
        "screenlog",
        "timestep",
    ]

    ENGINE_KEYS = ['timestep', *Rounds.ENGINE_KEYS]

    def __init__(self, extension, folder="output", max_energy=None) -> None:
        super().__init__(extension=extension,
                         folder=folder,
                         max_energy=max_energy)

    def add(self, md: MDEngine, i=None):
        """adds all the saveble info of the md simulation.

        The resulting
        """

        if i is None:
            i = self.i
            self.i += 1

        self._validate(md)

        d, attr = RoundsMd._add(md, i,
                                f'{self.folder}/round_{self.round}/bias_{i}')

        super().add(d=d, attrs=attr, i=i)

    @staticmethod
    def _add(md: MDEngine, i, name_bias):
        d = md.get_trajectory()
        md.bias.save(name_bias)

        attr = {k: md.__dict__[k] for k in RoundsMd.TRAJ_KEYS}
        attr['name_bias'] = name_bias

        return [d, attr]

    def _validate(self, md: MDEngine):

        pass
        # md0 = self._get_prop(self.round, 0, 'engine')

        # #check equivalency of CVs
        # md.bias.cvs == md0.bias.cvs

        # #check equivalency of md engine params

        # for k in self.ENGINE_KEYS:
        #     assert md0.__dict__[k] == md.__dict__[k]

        # #check equivalency of energy source
        # dill.dumps(md.ener) == dill.dumps(md.ener)

    def new_round(self, md: MDEngine):

        r = self.round + 1

        directory = f'{self.folder}/round_{r}'
        if not os.path.isdir(directory):
            os.mkdir(directory)

        name_md = f'{self.folder}/round_{r}/engine'
        name_bias = f'{self.folder}/round_{r}/bias'
        md.save(name_md)
        md.bias.save(name_bias)

        attr = {key: md.__dict__[key] for key in self.ENGINE_KEYS}
        attr['name_md'] = name_md
        attr['name_bias'] = name_bias

        super().new_round(attr=attr)

    def get_bias(self, r=None, i=None) -> Bias:
        if r is None:
            r = self.round

        with h5py.File(self.h5file, 'r') as f:
            if i is None:
                bn = f[f'{r}'].attrs['name_bias']
            else:
                bn = f[f'{r}'][i].attrs['name_bias']
            return Bias.load(bn)

    def get_engine(self, r=None) -> MDEngine:
        if r is None:
            r = self.round
        with h5py.File(self.h5file, 'r') as f:
            return MDEngine.load(f[f'{r}'].attrs['name_md'], filename=None)

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

            if self.max_energy is not None:
                b = CompositeBias([
                    b,
                    BiasF(b.cvs, lambda _: jnp.ones(1,) * self.max_energy)
                ], jnp.min)

            md = MDEngine.load(common_md_name, filename=temp_name, bias=b)

            md.run(steps=steps)

            d, attr = RoundsMd._add(
                md, i, f'{self.folder}/round_{self.round}/bias_{i}')

            return [d, attr, i]

        if len(biases) != 1:

            with pathos.pools.ProcessPool() as pool:
                # with pathos.pools.SerialPool() as pool:
                for [d, attr, i] in pool.map(_run_par, kwargs):
                    super().add(d=d, attrs=attr, i=i)
        else:
            [d, attr, i] = _run_par(kwargs[0])
            super().add(d=d, attrs=attr, i=i)

        self.i += len(kwargs)

        for kw in kwargs:
            os.remove(kw['new_name'])

    def unbias_rounds(self, steps=1e5, num=1e7, calc=False) -> RoundsCV:

        md = self.get_engine()
        if self.n() > 1 or isinstance(self.get_bias(), NoneBias) or calc:
            from IMLCV.base.Observable import Observable
            obs = Observable(self)
            fesBias = obs.fes_bias(plot=True)

            md = md.new_bias(fesBias, filename=None, write_step=5)
            self.new_round(md)

        if self.n() == 0:
            self.run(None, steps)

        # construct rounds object
        r = self._get_i(self.round, 0)
        props = self._get_round(self.round)
        beta = 1 / (props['T'] * boltzmann)
        bias = Bias.load(r['attr']['name_bias'])

        p = r["positions"][:]
        c = r.get('cell')
        t_orig = r["t"][:]

        def _interp(x_new, x, y):
            if y is not None:
                return jnp.apply_along_axis(
                    lambda yy: jnp.interp(x_new, x, yy), arr=y, axis=0)
            return None

        def bt(x, x_orig):
            pt = partial(_interp, x=x_orig, y=p)
            ct = partial(_interp, x=x_orig, y=c)
            return bias.compute_coor(coordinates=pt(x), cell=ct(x))[0]

        bt = jnp.vectorize(bt, excluded=[1])

        def integr(fx, x):
            return np.array(
                [0, *np.cumsum((fx[1:] + fx[:-1]) * (x[1:] - x[:-1]) / 2)])

        t = t_orig[:]
        eb = np.exp(-beta * bt(t, t_orig))
        tau = integr(1 / eb, t)
        tau_new = np.linspace(start=0.0, stop=tau.max(), num=int(num))

        p_new = _interp(tau_new, tau, r["positions"])
        c_new = _interp(tau_new, tau, r.get("cell"))

        roundscv = RoundsCV(self.extension, f'{self.folder}_unbiased')
        roundscv.new_round(props)
        roundscv.add(
            0, {
                'energy': None,
                'positions': p_new,
                'forces': None,
                'cell': c_new,
                't': tau_new
            })

        return roundscv


if __name__ == '__main__':
    from IMLCV.test.test_scheme import test_cv_discovery
    test_cv_discovery()
