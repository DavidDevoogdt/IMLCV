from IMLCV.base import *
import pickle

from IMLCV.base import MdEngine


class Scheme:
    """base class that implements iterative scheme.

    args:
        format (String): intermediate file type between rounds
        CVs: list of CV instances.
    """

    def __init__(self, md: MDEngine, cvs: CV, cvd: CVDiscovery, format="extxyz") -> None:
        self.md = md
        self.cvd = cvd
        self.cvs = cvs

        self.md_rounds = 0
        self.cv_rounds = 0
        self.steps = 0

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

        self.md.run(sampling_steps)

        sys = self.md.to_ASE_traj()

        n1 = int(bias_steps / self.md.write_step)

        # ase.io.write('output/md_otf_{}.xyz'.format(self.md_rounds), sys[:n1], format=self.format, append=False)
        # ase.io.write('output/md_{}.xyz'.format(self.md_rounds), sys[n1 + 1:], format=self.format, append=False)
        self.md.bias.save_bias('output/bias_{}.xyz'.format(self.md_rounds))

        bias = MdEngine.MdBias.load_bias('output/bias_{}.xyz'.format(self.md_rounds))

        cvs = np.array([0.0, 0.0])

        self.md.bias.compute(cvs=cvs)
        bias.compute(cvs=cvs)

        self.md_rounds += 1
        self.steps += bias_steps + sampling_steps

    def calc_obs(self):
        obs = Observable(self.md.bias)
        obs.plot_bias()

    def update_CV(self):
        raise NotImplementedError

        self.cv_rounds += 1
