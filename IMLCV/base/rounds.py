from __future__ import annotations

import os
from abc import ABC
from collections.abc import Iterable
from functools import partial
from http import client
from logging import Logger
from threading import Lock, RLock
from typing import Optional

import dill
import h5py
import IMLCV
import jax.numpy as jnp
import numpy as np
from IMLCV import ROOT_DIR
from IMLCV.base.bias import Bias, BiasF, CompositeBias, NoneBias
from IMLCV.base.MdEngine import MDEngine
from IMLCV.launch.parsl_conf.bash_app_python import bash_app_python
from molmod.constants import boltzmann
from parsl.data_provider.files import File


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

        self.h5file_name = f"{folder}/rounds.h5"
        self.h5file = h5py.File(f"{folder}/rounds.h5", 'a')
        self.lock = Lock()

    def __del__(self):
        self.h5file.close()

    def save(self):
        with open(f'{self.folder}/rounds', 'wb') as f:

            d = self.__dict__.copy()
            del d['h5file']

            dill.dump(d, f)

    @staticmethod
    def load(folder) -> Rounds:
        with open(f"{folder}/rounds", 'rb') as f:

            self = object.__new__(RoundsMd)
            self.__dict__.update(dill.load(f))

        self.h5file = h5py.File(self.h5file_name, 'a')

        return self

    def add(self, i, d, attrs=None):

        assert all(
            key in d for key in ['energy', 'positions', 'forces', 'cell', 't'])

        # with self.lock:
        with self.lock:
            f = self.h5file
            f.create_group(f'{self.round}/{i}')

            for key in d:
                if d[key] is not None:
                    f[f'{self.round}/{i}'][key] = d[key]

            if attrs is not None:
                for key in attrs:
                    if attrs[key] is not None:
                        f[f'{self.round}/{i}'].attrs[key] = attrs[key]

                f[f'{self.round}/{i}'].attrs['valid'] = True

            f[f'{self.round}'].attrs['num'] += 1

        # self.save()

    def new_round(self, attr):
        self.round += 1
        self.i = 0

        dir = f'{self.folder}/round_{self.round}'
        if not os.path.isdir(dir):
            os.mkdir(dir)

        # with self.lock:
        with self.lock:
            f = self.h5file
            f.create_group(f"{self.round}")

            for key in attr:
                if attr[key] is not None:
                    f[f'{self.round}'].attrs[key] = attr[key]

            f[f'{self.round}'].attrs['num'] = 0
            f[f'{self.round}'].attrs['valid'] = True

        self.save()

    def iter(self, r=None, num=3):
        if r is None:
            r = self.round

        for r0 in range(max(r - (num - 1), 0), r + 1):

            r_data = self._get_round(r0)
            if not r_data['valid']:
                continue

            for i in r_data['names']:
                i_dict = self._get_i(r_data['round'], i)

                if i_dict['attr']['valid']:
                    yield {**i_dict, 'round': r_data}

    def _get_round(self, r):
        # with self.lock:
        with self.lock:
            f = self.h5file
            rounds = [k for k in f[f'{r}'].keys()]

            d = f[f"{r}"].attrs
            r_attr = {key: d[key] for key in d}
        r_attr['round'] = r
        r_attr['names'] = rounds
        return r_attr

    def _get_i(self, r, i):
        # with self.lock:
        with self.lock:
            f = self.h5file
            d = f[f"{r}/{i}"]
            y = {key: d[key][:] for key in d}
            attr = {key: d.attrs[key] for key in d.attrs}
        return {**y, 'attr': attr, 'i': i}

    def _get_attr(self, name, r=None):
        # with self.lock:
        with self.lock:
            f = self.h5file
            if r is not None:
                f2 = f[f"{r}"]
            else:
                f2 = f

            if name in f2.attrs:
                return f2.attrs[name]

        return None

    def _set_attr(self, name, value, r=None, i=None):

        # with self.lock:
        with self.lock:
            f = self.h5file
            if r is not None:
                f2 = f[f"{r}"]
            else:
                f2 = f

            if i is not None:
                assert r is not None, "also provide round"
                f2 = f[f"/{i}"]

            f2.attrs[name] = value

    @property
    def T(self):
        return self._get_attr('T', r=self.round)

    @property
    def P(self):
        return self._get_attr('P', r=self.round)

    def n(self, r=None):
        if r is None:
            r = self.round
        return self._get_attr('num', r=r)

    def invalidate_data(self, r=None, i=None):
        if r is None:
            r = self.round

        self._set_attr(name='valid', value=False, r=r, i=i)


class RoundsCV(Rounds):
    """class for unbiased rounds."""


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

    def __init__(self, extension, folder="output") -> None:
        super().__init__(extension=extension,
                         folder=folder)

    @staticmethod
    def load(folder) -> RoundsMd:
        return Rounds.load(folder=folder)

    def add(self, traj, md: MDEngine, bias: str, i: int):
        """adds all the saveble info of the md simulation.
        """

        if i is None:
            i = self.i
            self.i += 1

        self._validate(md)

        attr = {k: md.__dict__[k] for k in RoundsMd.TRAJ_KEYS}
        attr['name_bias'] = bias

        super().add(d=traj, attrs=attr, i=i)

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

        with self.lock:
            f = self.h5file
            # with h5py.File(self.h5file, 'r') as f:
            if i is None:
                bn = f[f'{r}'].attrs['name_bias']
            else:
                bn = f[f'{r}'][i].attrs['name_bias']
        return Bias.load(bn)

    def get_engine(self, r=None) -> MDEngine:
        if r is None:
            r = self.round
        with self.lock:
            f = self.h5file
        # with h5py.File(self.h5file, 'r') as f:
            name = f[f'{r}'].attrs['name_md']

        return MDEngine.load(name, filename=None)

    def run(self, bias, steps):
        self.run_par([bias], steps)

    def run_par(self, biases: Iterable[Optional[Bias]], steps):
        with self.lock:
            f = self.h5file
        # with h5py.File(self.h5file, 'r') as f:
            common_bias_name = f[f'{self.round}'].attrs['name_bias']
            common_md_name = f[f'{self.round}'].attrs['name_md']

        tasks = []

        md_engine = MDEngine.load(common_md_name)

        for i, bias in enumerate(biases):

            temp_name = f'{self.folder}/round_{self.round}/temp_{i}'
            if not os.path.exists(temp_name):
                os.mkdir(temp_name)

            # construct bias
            if bias is None:
                b = Bias.load(common_bias_name)
            else:
                b = CompositeBias([Bias.load(common_bias_name), bias])

            b_name = f"{temp_name}/bias"
            b_name_new = f"{temp_name}/bias_new"
            b.save(b_name)

            traj_file = f"{temp_name}/traj.h5"

            # create file
            # with open(traj_file, 'wb') as f:
            #     pass

            @bash_app_python()
            def run(steps: int, inputs=[], outputs=[]):

                bias = Bias.load(inputs[1].filepath)
                md = MDEngine.load(
                    inputs[0].filepath, bias=bias, filename=outputs[1].filepath)
                md.run(steps)

                # print(outputs[0])
                bias.save(outputs[0].filepath)
                d = md.get_trajectory()
                return d

            future = run(
                inputs=[File(common_md_name), File(b_name)],
                outputs=[File(b_name_new), File(traj_file)],
                steps=int(steps),
                stdout=f'{temp_name}/md.stdout',
                stderr=f'{temp_name}/md.stderr',
            )
            tasks.append((i, future))

        # wait for tasks to finish
        for i, future in tasks:
            d = future.result()
            self.add(traj=d, md=md_engine,
                     bias=future.task_def['kwargs']['outputs'][0].filepath, i=i)

        self.i += len(tasks)

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
