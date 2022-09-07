import os
from datetime import datetime

import parsl
from IMLCV import ROOT_DIR
from IMLCV.launch.parsl_conf.config import config

if __name__ == "__main__":

    config(cluster="slaking", spawnjob=True)

    name = f"hpc_log_{datetime.now().strftime('%d-%m-%y_%H:%M:%S')}"
    print(
        f"starting on slacking cluster writing output to {name}.stdout and {name}.stderr in folder {ROOT_DIR}/logs"
    )

    if not os.path.exists(f"{ROOT_DIR}/logs"):
        os.mkdir(f"{ROOT_DIR}/logs")

    @parsl.bash_app
    def test_scheme(
        outputs=[],
        stdout=f"{ROOT_DIR}/logs/{name}.stdout",
        stderr=f"{ROOT_DIR}/logs/{name}.stderr",
    ):

        return f"python -u /user/gent/436/vsc43693/scratch_vo/projects/IMLCV/IMLCV/test/test_scheme.py"

    future = test_scheme(outputs=[parsl.File("./test_scheme.out")])
    future.outputs[0].result()
