# this is a helper function to perform md simulations. Executed by parsl on HPC infrastructure, but

import argparse
import tempfile
from dataclasses import dataclass
from sys import stderr
from typing import List, Optional

import dill
import numpy as np
from IMLCV import ROOT_DIR
from IMLCV.base.bias import Bias
from IMLCV.launch.parsl_conf.bash_app_python import bash_app_python
from IMLCV.launch.parsl_conf.config import config
from parsl import bash_app, python_app
from parsl.data_provider.files import File


@dataclass
class plotArgs:
    bias: Bias
    name: File
    n: int = 50
    traj = None
    vmin: float = 0
    vmax: float = 100
    map: bool = True
    traj: Optional[List[np.ndarray]] = None


@bash_app_python
def func(i, str):
    print(f"got i={i} str={str}")


if __name__ == "__main__":
    config()

    fut = func(i=7, str=20, stdout='test.out', stderr='test.stderr')

    fut.result()
