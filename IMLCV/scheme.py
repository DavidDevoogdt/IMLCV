from __future__ import annotations
from fileinput import filename
from functools import partial

# import multiprocessing_on_dill as multiprocessing

import dill

from ase.io import write, read

from IMLCV.base.MdEngine import MDEngine, YaffEngine
from IMLCV.base.bias import BiasMTD, Bias, CompositeBias, HarmonicBias, NoneBias, GridBias
from IMLCV.base.CVDiscovery import CVDiscovery
from IMLCV.base.CV import CV

from thermolib.thermodynamics.fep import SimpleFreeEnergyProfile, FreeEnergySurface2D, plot_feps
from thermolib.thermodynamics.histogram import Histogram2D, plot_histograms
from thermolib.thermodynamics.bias import BiasPotential2D

from molmod.units import kjmol, femtosecond
from molmod.constants import boltzmann

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

import itertools
from collections import Iterable

# from multiprocessing import Manager, Pool, Process, Lock, Value

from pathos.multiprocessing import ProcessingPool, ThreadPool

from typing import Type


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
        name_t = '{}/traj_{}-{}.{}'.format(folder, round, i, extension)
        traj = md.to_ASE_traj()
        write(name_t, traj, format=extension, append=False),
        return name_t

    @staticmethod
    def _save_traj_kwargs(md):
        return {k: md.__dict__[k] for k in Rounds.trajectory_keys}

    @staticmethod
    def _save_bias(md, i, folder, round):
        name_bias = '{}/bias_{}-{}'.format(folder, round, i)
        md.bias.save(name_bias)
        return name_bias

    def combine(self, rounds: Iterable[Rounds]):
        for r in rounds:
            for key in ['biases', 'trajectories', 'trajectory_kwargs']:
                for val in r.data[-1][key]:
                    self.data[-1][key].append(val)
            self.data[-1]['num'] += 1

    def save(self, filename):
        with open(f'{self.folder}/{filename}', 'wb') as f:
            dill.dump(self, f)

    @staticmethod
    def load(filename) -> Rounds:
        with open(filename, 'rb') as f:
            self = dill.load(f)
        return self

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)
        return self

    def _validate(self, md):

        md0 = self._get_prop(self.round, 0, 'engine')

        #check equivalency of CVs
        pass

        #check equivalency of md engine params

        for k in self.engine_keys:
            assert md0.__dict__[k] == md.__dict__[k]

        #check equivalency of energy source
        pass

    def new_round(self, md):
        self.round += 1

        name_md = '{}/engine_{}'.format(self.folder, self.round)
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

    def get_trajectories_and_biases(self, round=-1):

        if round == -1:
            rounds = range(len(self.data))
        else:
            rounds = [round]

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
        common_bias_name = f"{self.folder}/common_bias.t"
        common_md_name = f"{self.folder}/common_md.t"
        md = self._get_prop(self.round, 0, 'engine')

        md.save(common_md_name)
        md.bias.save(common_bias_name)

        kwargs = []
        for i, b in enumerate(biases):
            kwargs.append({
                'bias': b,
                'common_bias_name': common_bias_name,
                'engine_name': common_md_name,
                'new_name': f'{self.folder}/temp_{i}.h5',
                'steps': steps,
                'save_traj': partial(Rounds._save_traj, folder=self.folder, round=self.round, extension=self.extension),
                'save_bias': partial(Rounds._save_bias, folder=self.folder, round=self.round),
                'i': i
            })

        def _run_par(args):
            bias = args['bias']
            common_bias_name = args['common_bias_name']
            engine_name = args['engine_name']
            new_name = args['new_name']
            steps = args['steps']
            save_traj = args['save_traj']
            save_bias = args['save_bias']
            i = args['i']

            b = CompositeBias([Bias.load(common_bias_name), bias])
            md = MDEngine.load(engine_name, filename=new_name, bias=b)

            md.run(steps=steps)

            name_t = save_traj(md, i)
            name_bias = save_bias(md, i)
            name_kwargs = Rounds._save_traj_kwargs(md)

            return name_t, name_bias, name_kwargs

        import pathos
        with pathos.pools.ProcessPool() as pool:
            for [name_t, name_bias, traj_kwargs] in pool.map(_run_par, kwargs):
                self.data[self.round]['trajectories'].append(name_t)
                self.data[self.round]['biases'].append(name_bias)
                self.data[self.round]['trajectory_kwargs'].append(traj_kwargs)

                self.data[self.round]['num'] += 1


class Scheme:
    """base class that implements iterative scheme.

    args:
        format (String): intermediate file type between rounds
        CVs: list of CV instances.
    """

    def __init__(self,
                 cvd: CVDiscovery,
                 cvs: CV,
                 Engine: Type[MDEngine],
                 ener,
                 T,
                 P=None,
                 timestep=None,
                 timecon_thermo=None,
                 timecon_baro=None,
                 extension="extxyz") -> None:

        filename = "output/init.h5"
        write_step = 100
        screenlog = 1000

        self.md = Engine(bias=NoneBias(cvs),
                         ener=ener,
                         T=T,
                         P=P,
                         timestep=timestep,
                         timecon_thermo=timecon_thermo,
                         timecon_baro=timecon_baro,
                         filename=filename,
                         write_step=write_step,
                         screenlog=screenlog)

        self.cvd = cvd

        self.rounds = Rounds(extension=extension)
        self.steps = 0

    def from_rounds(
        cvd: CVDiscovery,
        filename,
        steps=0,
    ) -> Scheme:

        self = Scheme.__new__(Scheme)

        rounds = Rounds.load(filename=filename)
        self.md = rounds.engine

        self.rounds = rounds
        self.cvd = cvd
        self.steps = 0

        return self

    def _MTDBias(self, steps, K=None, sigmas=None, start=50, step=500) -> Bias:
        """generate a metadynamics bias"""

        if sigmas is None:
            sigmas = (self.md.bias.cvs.periodicity[:, 1] - self.md.bias.cvs.periodicity[:, 0]) / 20

        if K is None:
            K = 0.1 * self.md.T * boltzmann

        biasmtd = BiasMTD(self.md.bias.cvs, K, sigmas, start=start, step=step)
        bias = CompositeBias([self.md.bias, biasmtd])
        self.md = self.md.new_bias(bias, filename="output/mtdbias2.h5")
        self.md.run(steps)
        self.md.bias.finalize()

    def _grid_umbrella(self, steps=1e4, US_grid=None, K=None, n=4):

        cvs = self.md.bias.cvs
        if np.isnan(cvs.periodicity).any():
            raise NotImplementedError("impl non periodic")

        if K == None:
            K = 1.0 * self.md.T * boltzmann * (n * 2 / (cvs.periodicity[:, 1] - cvs.periodicity[:, 0]))**2

        if US_grid is None:
            grid = [np.linspace(row[0], row[1], n, endpoint=False) for row in self.md.bias.cvs.periodicity]

        self.rounds.run_par(
            [HarmonicBias(self.md.bias.cvs, np.array(x), np.array(K)) for x in itertools.product(*grid)], steps=steps)

    def get_fes(self):
        obs = Observable(self.rounds)
        fes = obs.fes_2D(plot=True)
        fesBias = obs.fes_Bias()

        return fesBias

    def update_CV(self):
        pass

    def save(self, filename):
        raise NotImplementedError

    @classmethod
    def load(cls, filename):
        raise NotImplementedError


class Observable:
    """class to convert data and CVs to different thermodynamic/ kinetic observables."""

    def __init__(self, rounds: Rounds) -> None:
        self.rounds = rounds

        self.fes = None

    def fes_2D(self, plot=True):
        # fes = FreeEnergySurface2D.from_txt
        if self.fes is not None:
            return self.fes

        temp = self.rounds.T

        trajs, bss, bins = self._get_biasses(plot=plot)

        histo = Histogram2D.from_wham_c(
            bins=bins,
            traj_input=trajs,
            error_estimate='mle_f',
            biasses=bss,
            temp=temp,
        )

        fes = FreeEnergySurface2D.from_histogram(histo, temp)
        fes.set_ref()

        if plot:
            fes.plot('output/ala_fes_thermolib_{}.png'.format(0))

        self.fes = fes

        return fes

    def _get_biasses(self, plot=False):
        trajs = []
        tbss = []
        bss = []

        for (traj, bias) in self.rounds.get_trajectories_and_biases():
            arr = np.array([bias.cvs.compute(t.positions, cell=t.cell.array)[0] for t in traj], dtype=np.double)
            trajs.append(arr)
            bss.append(bias)
            tbss.append(Observable._thermo_bias2D(bias))

        temp = self.rounds.T

        bins = self._grid(n=51, endpoint=True)
        bin_centers = [0.5 * (row[:-1] + row[1:]) for row in bins]
        beta = 1 / (boltzmann * temp)
        cb = self.rounds.commom_bias()
        mg = np.meshgrid(*bin_centers)

        biases, _ = jnp.apply_along_axis(cb.compute, axis=0, arr=np.array(mg), diff=False)

        # pinit = np.exp(-biases * beta)

        def pl(b, name, trajs):
            plt.clf()
            contourf = plt.contourf(mg[0], mg[1], b, cmap=plt.get_cmap('rainbow'))
            contour = plt.contour(mg[0], mg[1], b)

            plt.xlabel('cv1', fontsize=16)
            plt.ylabel('cv2', fontsize=16)
            cbar = plt.colorbar(contourf, extend='both')
            cbar.set_label('Bias boltzmann factor [-]', fontsize=16)
            plt.clabel(contour, inline=1, fontsize=10)

            for traj in trajs:
                plt.scatter(traj[:, 0], traj[:, 1])

            plt.title(name)

            fig = plt.gcf()
            fig.set_size_inches([12, 8])
            plt.savefig(name)

        if plot:
            pl(biases, 'output/mtd', trajs)

            for i, b in enumerate(bss):
                bias, _ = jnp.apply_along_axis(b.compute, axis=0, arr=np.array(mg), diff=False)
                pl(bias, f'output/umbrella {i}', [trajs[i]])

        return trajs, tbss, bins

    class _thermo_bias2D(BiasPotential2D):

        def __init__(self, bias: Bias) -> None:
            self.bias = bias

            super().__init__("IMLCV_bias")

        def __call__(self, cv1, cv2):
            cvs = jnp.array([cv1, cv2])
            b, _ = jnp.apply_along_axis(self.bias.compute, axis=0, arr=cvs, diff=False)
            return np.array(b, dtype=np.double)

        def print_pars(self, *pars_units):
            pass

    def fes_Bias(self):
        fes = self.fes_2D(plot=False)

        #adapt bias, apparently tranpose is needed here
        bias = -np.transpose(fes.fs)
        bias -= bias[~np.isnan(bias)].min()
        bias[np.isnan(bias)] = 0.0

        return GridBias(cvs=self.rounds.commom_bias().cvs, vals=bias)

    def _plot_bias(self):
        """manual bias plot, convert to _thermo_bias2D and plot instead."""

        b = self.mdb[-1]

        x = self._grid(n=51, endpoint=False)
        mg = np.array(np.meshgrid(*x))  #(ncv,n,n) matrix
        biases, _ = jnp.apply_along_axis(b.compute, axis=0, arr=mg, diff=False)
        biases = -np.array(biases)

        if mg.shape[0] != 2:
            raise NotImplementedError("todo sum over extra dims to visualise")

        fes = biases - np.amin(biases)

        plt.clf()
        plt.contourf(mg[0, :, :], mg[1, :, :], fes / units.kjmol)
        plt.xlabel("CV1")
        plt.ylabel("CV2")
        plt.title("$F\,[\mathrm{kJ}\,\mathrm{mol}^{-1}]$")
        plt.colorbar()
        plt.savefig('output/ala_dipep.png')

    def _grid(self, n=51, endpoint=True):
        cvs = self.rounds._get_prop(0, 0, 'cv')
        if np.isnan(cvs.periodicity).any():
            raise NotImplementedError("add argument for range")

        return [np.linspace(p[0][0], p[0][1], n, endpoint=True) for p in np.split(cvs.periodicity, 2, axis=0)]
