from IMLCV.base.rounds import Rounds
from .CV import CV


class CVDiscovery:
    """convert set of coordinates to good collective variables."""

    def __init__(self) -> None:
        pass

    def _unbias_rounds(self, rounds: Rounds) -> Rounds:
        ts = rounds.timestep

        for traj, bias in rounds.get_trajectories_and_biases():
            pass

    def compute(self, data) -> CV:
        NotImplementedError
