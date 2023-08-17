from pathlib import Path

from IMLCV.base.rounds import Rounds
from IMLCV.scheme import Scheme

if __name__ == "__main__":
    folder = Path.cwd() / "notebooks" / "alanine_dipeptide_tica_017"
    rnds = Rounds(folder=folder, new_folder=False)
    scheme0 = Scheme.from_rounds(rnds)

    from IMLCV.configs.config_general import config

    config(local_ref_threads=4, initialize_logging=False)

    scheme0.FESBias(chunk_size=100, samples_per_bin=50, plot=True)
    scheme0.rounds.add_round_from_md(md=scheme0.md)
