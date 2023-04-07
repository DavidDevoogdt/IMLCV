import os
import platform
import re
from pathlib import Path

import parsl
from parsl.config import Config

import configs.local_threadpool
from configs.hpc_ugent import config as config_ugent

ROOT_DIR = Path(os.path.dirname(__file__)).parent
py_env = f"source {ROOT_DIR}/Miniconda3/bin/activate; which python"


def config(
    env=None,
    singlepoint_nodes=16,
    walltime="48:00:00",
    bootstrap=False,
    memory_per_core=None,
    min_memery_per_node=None,
    path_internal: Path | None = None,
):
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

    if path_internal is None:
        path_internal = ROOT_DIR / "IMLCV" / ".runinfo"

    py_env = f"source {ROOT_DIR}/Miniconda3/bin/activate; which python"

    if env == "local":
        execs = configs.local_threadpool.get_config(path_internal, py_env)
    elif env == "hortense" | env == "stevin":
        execs = config_ugent(
            env=env,
            path_internal=path_internal,
            singlepoint_nodes=singlepoint_nodes,
            walltime=walltime,
            bootstrap=bootstrap,
            memory_per_core=memory_per_core,
            min_memery_per_node=min_memery_per_node,
        )

    config = Config(
        executors=execs,
        usage_tracking=True,
        run_dir=str(path_internal),
    )

    parsl.load(config=config)
