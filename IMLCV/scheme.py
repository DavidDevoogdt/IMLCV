from IMLCV.base import *
import pickle

from IMLCV.base import MdEngine


class Scheme:
    """base class that implements iterative scheme.

    args:
        format (String): intermediate file type between rounds
        CVs: list of CV instances.
    """

    def __init__(self, md: MDEngine, cvd: CVDiscovery, format="extxyz") -> None:
        self.md = md
        self.cvd = cvd
        self.cvs = md.bias.cvs

        self.md_rounds = 0
        self.cv_rounds = 0
        self.steps = 0

        self.bias_names = []
        self.md_names = []

        if format != "extxyz":
            raise NotImplementedError("file type not known")

        self.format = format

    def run(self, rounds, bias_steps, sampling_steps):
        for _ in range(rounds):
            self.do_MD(bias_steps, sampling_steps)
            self.calc_obs()
            self.update_CV()

    def do_MD(self, bias_steps, sampling_steps):
        bias_steps = int(bias_steps)
        sampling_steps = int(sampling_steps)

        self.md.run(bias_steps)

        self._save_bias()

        self.md.bias.finalize_bias()
        self.md.run(sampling_steps)

        self._save_traj(start=bias_steps)

        self.md_rounds += 1
        self.steps += bias_steps + sampling_steps

    def calc_obs(self):
        obs = Observable(self.md.bias, self.bias_names[-1])
        obs.plot_bias()

    def update_CV(self):
        raise NotImplementedError

    def _save_bias(self):
        name = 'output/bias_{}'.format(self.md_rounds)

        self.md.bias.save_bias(name)
        self.bias_names.append(name)

    def _save_traj(self, start=1, stop=-1):
        traj = self.md.to_ASE_traj()

        name = 'output/md_{}.{}'.format(self.md_rounds, self.format)
        ase.io.write(name, traj[start:stop], format=self.format, append=False)
