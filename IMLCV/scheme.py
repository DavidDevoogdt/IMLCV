from unicodedata import name

import dill

from ase.io import write, read

from dataclasses import dataclass

from IMLCV.base.MdEngine import MDEngine
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
        "ener",
        "filename",
        "write_step",
        "screenlog",
    ]

    def __init__(self, extension, folder="output") -> None:
        if extension != "extxyz":
            raise NotImplementedError("file type not known")

        self.round = 0
        self.data = []
        self.extension = extension
        self.i = 0

        self.folder = folder

    def __getattr__(self, name: str):
        if name == 'T':
            return self.data[-1]['engine_kwargs']['T']
        elif name == 'P':
            return self.data[-1]['engine_kwargs']['P']
        elif name == 'timestep':
            return self.data[-1]['engine_kwargs']['timestep']

    def add(self, md):
        """adds all the saveble info of the md simulation. The resulting """

        if self.i == 0:
            #save whold md engine
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
        else:
            self._validate(md)

        #save trajectory
        name_t = '{}/traj_{}-{}.{}'.format(self.folder, self.round, self.i, self.extension)
        traj = md.to_ASE_traj()
        write(name_t, traj, format=self.extension, append=False)
        self.data[self.round]['trajectories'].append(name_t)

        #save the bias
        name_bias = '{}/bias_{}-{}'.format(self.folder, self.round, self.i)
        md.bias.save(name_bias)
        self.data[self.round]['biases'].append(name_bias)

        #engine kwargs

        self.data[self.round]['trajectory_kwargs'].append({k: md.__dict__[k] for k in self.trajectory_keys})

        self.data[self.round]['num'] += 1

        self.i += 1

    def save(self, filename):
        with open(f'{self.folder}/rounds.p', 'wb') as f:
            dill.dump(self, f)

    @staticmethod
    def load(filename):
        with open(f'{self.prefix}/rounds.p', 'rb') as f:
            self = dill.load(f)
        return self

    def _validate(self, md):
        if self.i == 0:
            return

        md0 = self._get_prop(self.round, 0, 'engine')

        #check equivalency of CVs
        pass

        #check equivalency of md engine params

        for k in self.engine_keys:
            assert md0.__dict__[k] == md.__dict__[k]

        #check equivalency of energy source
        pass

    def new_round(self):
        self.data.append([])
        self.round += 1
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
            return MDEngine.load(dict['engine'])
        else:
            raise ValueError("unknown property")

    def get_trajectories_and_biases(self, round=-1):

        if round == -1:
            rounds = range(len(self.data))
        else:
            rounds = [round]

        for round in rounds:
            data = self.data[round]
            for i in range(data['num']):
                yield [self._get_prop(round, i, 'trajectory'), self._get_prop(round, i, 'bias')]


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

    def _MTDBias(self, steps, K=None, sigmas=None, start=25, step=25) -> Bias:
        """generate a metadynamics bias"""

        if sigmas is None:
            sigmas = (self.md.bias.cvs.periodicity[:, 1] - self.md.bias.cvs.periodicity[:, 0]) / 20

        if K is None:
            K = 0.5 * self.md.T * boltzmann

        biasmtd = BiasMTD(self.md.bias.cvs, K, sigmas, start=start, step=step)
        bias = CompositeBias([self.md.bias, biasmtd])
        self.md = self.md.new_bias(bias, filename="mtdbias.h5")
        self.md.run(steps)
        self.md.bias.finalize()

    def _grid_umbrella(self, steps=1e4, US_grid=None, K=None):

        cvs = self.md.bias.cvs
        if np.isnan(cvs.periodicity).any():
            raise NotImplementedError("impl non periodic")

        n = 2
        if K == None:
            K = self.md.T * boltzmann * (cvs.periodicity[:, 1] - cvs.periodicity[:, 0]) / n

        if US_grid is None:
            grid = [np.linspace(row[0], row[1], n, endpoint=False) for row in self.md.bias.cvs.periodicity]

        for i, x in enumerate(itertools.product(*grid)):
            bias = CompositeBias([
                HarmonicBias(cvs, np.array(x), K),
                self.md.bias,
            ])
            mde = self.md.new_bias(bias, filename=f'temp_{i}.h5')
            mde.run(steps)
            self.rounds.add(mde)

    def get_fes(self, steps_mtd=1e5, steps_US=1e4, US_grid=None):
        self._MTDBias(steps=steps_mtd)
        self._grid_umbrella(steps=steps_US)

        obs = Observable(self.rounds)
        fes = obs.fes_2D(plot=True)
        fesBias = obs.fes_Bias()

        return fesBias

    def update_CV(self):
        pass

    def save(self, filename):
        pass

    @classmethod
    def load(cls, filename):
        pass


class Observable:
    """class to convert data and CVs to different thermodynamic/ kinetic observables."""

    def __init__(self, rounds: Rounds) -> None:
        self.rounds = rounds

        self.fes = None

    def fes_2D(self, plot=True):
        # fes = FreeEnergySurface2D.from_txt
        if self.fes is not None:
            return self.fes

        cv = []
        bss = []

        for (traj, bias) in self.rounds.get_trajectories_and_biases():
            arr = np.array([bias.cvs.compute(t.positions, cell=t.cell.array)[0] for t in traj])
            cv.append(arr)
            bss.append(Observable._thermo_bias2D(bias))

        grid = self._grid(n=50)

        histo = Histogram2D.from_wham(bins=grid,
                                      trajectories=cv,
                                      biasses=bss,
                                      temp=self.rounds.T,
                                      error_estimate='mle_f',
                                      plot_biases=True)

        fes = FreeEnergySurface2D.from_histogram(histo, self.rounds.T)
        fes.set_ref()

        if plot:
            fes.plot('output/ala_fes_thermolib_{}.png'.format(0))

        self.fes = fes

        return fes

    class _thermo_bias2D(BiasPotential2D):

        def __init__(self, bias: Bias) -> None:
            self.bias = bias

            super().__init__("IMLCV_bias")

        def __call__(self, cv1, cv2):
            cvs = jnp.array([cv1, cv2])
            b, _ = jnp.apply_along_axis(self.bias.compute, axis=0, arr=cvs, diff=False)
            return b

        def print_pars(self, *pars_units):
            pass

    def fes_Bias(self):
        fes = self.fes_2D(plot=False)

        #adapt bias, apparently tranpose is needed here
        bias = -np.transpose(fes.fs)
        bias -= bias[~np.isnan(bias)].min()
        bias[np.isnan(bias)] = 0.0

        return GridBias(cvs=self.mdb[-1].cvs, vals=bias)

    def _plot_bias(self):
        """manual bias plot, convert to _thermo_bias2D and plot instead."""

        b = self.mdb[-1]

        x = self._grid(n=51)
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

    def _grid(self, n=51):
        cvs = self.rounds._get_prop(0, 0, 'cv')
        if np.isnan(cvs.periodicity).any():
            raise NotImplementedError("add argument for range")

        return [np.linspace(p[0][0], p[0][1], n, endpoint=False) for p in np.split(cvs.periodicity, 2, axis=0)]