from __future__ import annotations

import os
import platform
import re
from pathlib import Path

import parsl
from IMLCV.configs.hpc_ugent import config as config_ugent
from IMLCV.configs.local_threadpool import get_config as get_config_local
from parsl.config import Config

ROOT_DIR = Path(os.path.dirname(__file__)).parent
py_env = f"source {ROOT_DIR}/Miniconda3/bin/activate; which python"


def get_platform():
    node = platform.node()
    print("node")
    if re.search("(node|login)[0-9]*.dodrio.os", node):
        env = "hortense"
    elif re.search(
        "(node|gligar)[0-9]*.(gastly|accelgor|delcatty|doduo|donphan|gallade|golett|joltik|kirlia|skitty|slaking|swalot|victini).os",
        node,
    ):
        env = "stevin"
    elif node == "david-CMM":
        env = "local"
    else:
        raise ValueError("unknown pc {node=}, set env")

    print(env)
    return env


def config(
    env=None,
    singlepoint_nodes=16,
    walltime="48:00:00",
    bootstrap=False,
    memory_per_core=None,
    min_memery_per_node=None,
    path_internal: Path | None = None,
    cpu_cluster=None,
    gpu_cluster=None,
):
    if parsl.DataFlowKernelLoader._dfk is not None:
        print("parsl already configured, using previous setup")
        return

    if env is None:
        env = get_platform()

    if path_internal is None:
        path_internal = ROOT_DIR / "IMLCV" / ".runinfo"

    if env == "local":
        execs = get_config_local(path_internal)
    elif env == "hortense" or env == "stevin":
        execs = config_ugent(
            env=env,
            path_internal=path_internal,
            singlepoint_nodes=singlepoint_nodes,
            walltime=walltime,
            bootstrap=bootstrap,
            memory_per_core=memory_per_core,
            min_memery_per_node=min_memery_per_node,
            cpu_cluster=cpu_cluster,
            gpu_cluster=gpu_cluster,
        )

    config = Config(
        executors=execs,
        usage_tracking=True,
        run_dir=str(path_internal),
    )

    parsl.load(config=config)


def get_cp2k():
    env = get_platform()
    if env == "hortense":
        return "export OMP_NUM_THREADS=1; mpirun  cp2k_shell.psmp"
    if env == "stevin":
        return "export OMP_NUM_THREADS=1; mpirun  cp2k_shell.popt "
    raise ValueError(f"unknow {env=} for cp2k ")