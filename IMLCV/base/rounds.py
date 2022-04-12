from __future__ import annotations
from functools import partial

# import multiprocessing_on_dill as multiprocessing

import dill

from ase.io import write, read

from IMLCV.base.MdEngine import MDEngine
from IMLCV.base.bias import Bias, CompositeBias
from collections import Iterable
import os
import pathos


class Rounds:
    """helper class to save/load all data in a consistent way. Gets passed between modules"""

    engine_keys = [
        "T",
        "P",
        "timestep",
        "timecon_thermo",
        "timecon_baro",
    ]
    trajectory_keys = [
        # "ener",
        "filename",
        "write_step",
        "screenlog",
    ]

    def __init__(self, extension, folder="output") -> None:
        if extension != "extxyz":
            raise NotImplementedError("file type not known")

        self.round = -1
        self.data = []
        self.extension = extension
        self.i = 0

        if not os.path.isdir(folder):
            os.makedirs(folder)

        self.folder = folder

    def add(self, md, i=None):
        """adds all the saveble info of the md simulation. The resulting """

        self._validate(md)

        if i == None:
            i = self.i
            self.i += 1

        #save trajectory
        name_t = Rounds._save_traj(md, i, self.folder, self.round, self.extension)
        self.data[self.round]['trajectories'].append(name_t)
        name_bias = self._save_bias(md, i, self.folder, self.round)
        self.data[self.round]['biases'].append(name_bias)
        self.data[self.round]['trajectory_kwargs'].append(self._save_traj_kwargs(self.md))

        self.data[self.round]['num'] += 1

    @staticmethod
    def _save_traj(md, i, folder, round, extension):
        name_t = f'{folder}/round_{round}/traj_{i}.{extension}'
        traj = md.to_ASE_traj()
        write(name_t, traj, format=extension, append=False),
        return name_t

    @staticmethod
    def _save_traj_kwargs(md):
        return {k: md.__dict__[k] for k in Rounds.trajectory_keys}

    @staticmethod
    def _save_bias(md, i, folder, round):
        name_bias = f'{folder}/round_{round}/bias_{i}'
        md.bias.save(name_bias)
        return name_bias

    def combine(self, rounds: Iterable[Rounds]):
        for r in rounds:
            for key in ['biases', 'trajectories', 'trajectory_kwargs']:
                for val in r.data[-1][key]:
                    self.data[-1][key].append(val)
            self.data[-1]['num'] += 1

    def save(self):
        with open(f'{self.folder}/rounds', 'wb') as f:
            dill.dump(self, f)

    @staticmethod
    def load(folder) -> Rounds:
        with open(f"{folder}/rounds", 'rb') as f:
            self = dill.load(f)
        return self

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)
        return self

    def _validate(self, md: MDEngine):

        md0 = self._get_prop(self.round, 0, 'engine')

        #check equivalency of CVs
        md.bias.cvs == md0.bias.cvs

        #check equivalency of md engine params

        for k in self.engine_keys:
            assert md0.__dict__[k] == md.__dict__[k]

        #check equivalency of energy source
        dill.dumps(md.ener) == dill.dumps(md.ener)

    def new_round(self, md):
        self.round += 1

        dir = f'{self.folder}/round_{self.round}'
        if not os.path.isdir(dir):
            os.mkdir(dir)

        name_md = f'{self.folder}/round_{self.round}/engine'
        md.save(name_md)

        self.data.append({
            'engine': name_md,
            'biases': [],
            'trajectories': [],
            'engine_kwargs': {key: md.__dict__[key] for key in self.engine_keys},
            'trajectory_kwargs': [],
            'num': 0,
        })

        self.i = 0

    def _get_prop(self, round, i, prop):
        """method to get new instance of desired properties"""

        dict = self.data[round]
        if prop == "bias":
            return Bias.load(dict['biases'][i])
        elif prop == "cv":
            bias = self._get_prop(round, i, "bias")
            return bias.cvs
        elif prop == "trajectory":
            return read(dict['trajectories'][i], index=':', format=self.extension)
        elif prop == "engine":
            return MDEngine.load(dict['engine'], filename=None)
        else:
            raise ValueError("unknown property")

    def commom_bias(self, round=-1):
        return self._get_prop(round, 0, 'engine').bias

    def get_trajectories_and_biases(self, round=-1, num=3):

        if round == -1:
            rounds = range(len(self.data))
        else:
            rounds = [round]

        if len(rounds) >= num:
            rounds = rounds[-num:]

        for round in rounds:
            data = self.data[round]
            for i in range(data['num']):
                yield [self._get_prop(round, i, 'trajectory'), self._get_prop(round, i, 'bias')]

    def __dir__(self):
        return dir(Rounds) + self.engine_keys + ['engine']

    def __getattr__(self, name: str):
        if name in self.engine_keys:
            return self.data[-1]['engine_kwargs'][name]
        if name == 'engine':
            return self._get_prop(-1, 0, 'engine')

    def run_par(self, biases: Iterable[Bias], steps):
        common_bias_name = f"{self.folder}/round_{self.round}/common_bias.t"
        common_md_name = f"{self.folder}/round_{self.round}/common_md.t"
        md = self._get_prop(self.round, 0, 'engine')

        md.save(common_md_name)
        md.bias.save(common_bias_name)

        kwargs = []
        for i, b in enumerate(biases):
            kwargs.append({
                'bias': b,
                'common_bias_name': common_bias_name,
                'engine_name': common_md_name,
                'new_name': f'{self.folder}/round_{self.round}/temp_{i}.h5',
                'steps': steps,
                'i': i
            })

        def _run_par(args):
            bias = args['bias']
            common_bias_name = args['common_bias_name']
            engine_name = args['engine_name']
            new_name = args['new_name']
            steps = args['steps']
            i = args['i']

            b = CompositeBias([Bias.load(common_bias_name), bias])
            md = MDEngine.load(engine_name, filename=new_name, bias=b)

            md.run(steps=steps)

            name_t = Rounds._save_traj(md, i, folder=self.folder, round=self.round, extension=self.extension)
            name_bias = Rounds._save_bias(md, i, folder=self.folder, round=self.round)
            name_kwargs = Rounds._save_traj_kwargs(md)

            return name_t, name_bias, name_kwargs

        with pathos.pools.ProcessPool() as pool:
            for [name_t, name_bias, traj_kwargs] in pool.map(_run_par, kwargs):
                self.data[self.round]['trajectories'].append(name_t)
                self.data[self.round]['biases'].append(name_bias)
                self.data[self.round]['trajectory_kwargs'].append(traj_kwargs)

                self.data[self.round]['num'] += 1

        for kw in kwargs:
            os.remove(kw['new_name'])
