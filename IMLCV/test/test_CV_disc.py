import os
import shutil

import dill
import IMLCV
import matplotlib.pyplot as plt
import numpy as np
import parsl
import umap
from IMLCV.base.bias import Bias, NoneBias
from IMLCV.base.CVDiscovery import CVDiscovery, TranformerUMAP
from IMLCV.base.rounds import RoundsMd
from IMLCV.launch.parsl_conf.config import config
from molmod.units import kjmol
from py import test

# parsl.load()

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


def cleancopy(base):

    if not os.path.exists(f"{base}_orig"):
        assert os.path.exists(f"{base}"), "folder not found"
        shutil.copytree(f"{base}", f"{base}_orig")

    if os.path.exists(f"{base}"):
        shutil.rmtree(f"{base}")
    shutil.copytree(f"{base}_orig", f"{base}")


def test_cv_discovery():
    # make copy and restore orig
    name = "ala_cv3"
    base = f"output/{name}"

    cleancopy(base=base)

    rounds = RoundsMd.load(base)

    cvd = CVDiscovery()

    cvd.compute(rounds=rounds)


if __name__ == "__main__":
    test_cv_discovery()
