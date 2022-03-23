from IMLCV.base import *


class scheme:
    """base class that implements iterative scheme.

    args:
        format (String): intermediate file type between rounds
        CVs: list of CV instances.
    """

    def __init__(self, md: MDEngine, CVs: CV, cvd: CVDiscovery, format="extxyz") -> None:
        self.md = MDEngine
        self.cvd = CVDiscovery

        self.md_rounds = 0
        self.cv_rounds = 0
        self.steps = 0

        if format != "extxyz":
            raise NotImplementedError("file type not known")

        self.format = format

    def scheme(self, rounds, steps):
        for _ in range(rounds):
            self.do_MD(steps)
            self.calc_obs()
            self.update_CV()

    def do_MD(self, steps):
        self.md.run(steps)

        sys = self.md.to_ASE_traj()
        ase.io.write('md_{}.xyz'.format(self.md_rounds), sys, format=self.format, append=False)

        self.md_rounds += 1
        self.steps += steps

    def calc_obs(self):
        raise NotImplementedError

    def update_CV(self):
        raise NotImplementedError

        self.cv_rounds += 1