from IMLCV.base import *
import pickle
from ase.io import write, read

from IMLCV.base import MdEngine


class Scheme:
    """base class that implements iterative scheme.

    args:
        format (String): intermediate file type between rounds
        CVs: list of CV instances.
    """

    def __init__(self, md: MDEngine, cvd: CVDiscovery, extension="extxyz") -> None:
        self.md = md
        self.cvd = cvd
        self.cvs = md.bias.cvs

        self.md_rounds = 0
        self.cv_rounds = 0
        self.steps = 0

        self.bias_names = []
        self.traj_names = []

        self.trajs = []
        self.biases = []

        if extension != "extxyz":
            raise NotImplementedError("file type not known")

        self.extension = extension

    def run(self, rounds, bias_steps, sampling_steps):
        for _ in range(rounds):
            self.do_MD(bias_steps, sampling_steps)
            fes, fesBias = self.calc_obs()
            self.update_CV()

    def do_MD(self, bias_steps, sampling_steps):
        bias_steps = int(bias_steps)
        sampling_steps = int(sampling_steps)

        self.md.run(bias_steps)

        self._save_bias()

        # self.md.bias.finalize_bias()
        self.md.run(sampling_steps)

        self._save_traj(start=bias_steps)

        self.md_rounds += 1
        self.steps += bias_steps + sampling_steps

    def calc_obs(self):
        obs = Observable(self.biases, self.trajs, self.md.T)

        fes = obs.fes_2D(plot=True, round=self.md_rounds)
        fesBias = obs.fes_Bias()

        return fesBias

    def update_CV(self):
        pass

    def _save_bias(self):
        name = 'output/bias_{}'.format(self.md_rounds)

        self.md.bias.save_bias(name)
        self.bias_names.append(name)

        self.biases.append(self.md.bias)

    def _save_traj(self, start=1, stop=-1):
        traj = self.md.to_ASE_traj()

        if stop != -1:
            raise NotImplementedError

        cropped_traj = traj[int(start / self.md.write_step):stop]

        name = 'output/traj_{}.{}'.format(self.md_rounds, self.extension)
        write(name, cropped_traj, format=self.extension, append=False)

        self.trajs.append(cropped_traj)

    def load_round(self, round=0):
        bias_name = 'output/bias_{}'.format(round)
        traj_name = 'output/traj_{}.{}'.format(round, self.extension)

        self.biases.append(self.md.bias.load_bias(bias_name))
        self.bias_names.append(bias_name)

        self.trajs.append(read(traj_name, index=':', format=self.extension))
        self.traj_names.append(traj_name)
