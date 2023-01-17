import parsl
import configs.vsc_hortense
import configs.local_threadpool

import re

import platform
from IMLCV import ROOT_DIR


py_env = f"source {ROOT_DIR}/Miniconda3/bin/activate; which python"


def config(env=None):

    if parsl.DataFlowKernelLoader._dfk is not None:
        print("parsl already configured, using previous setup")
        return

    if env is None:
        node = platform.node()
        if re.search("(node|login)[0-9]*.dodrio.os", node):
            env = "hortense"
        elif node == re.search("gligar[0-9]*.gastly.os", node):
            env = "stevin"
        elif node == "david-CMM":
            env = "local"
        else:
            raise ValueError("unknown pc, set env")

    print(env)

    path_internal = ROOT_DIR / "IMLCV" / ".runinfo"
    py_env = f"source {ROOT_DIR}/Miniconda3/bin/activate; which python"

    if env == "local":
        config = configs.local_threadpool.get_config(path_internal, py_env)
    elif env == "hortense":
        config = configs.vsc_hortense.get_config(
            path_internal, py_env, account="2022_069"
        )
    elif env == "stevin":
        raise NotImplementedError

    print(config)

    parsl.load(config=config)
