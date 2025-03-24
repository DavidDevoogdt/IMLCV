import platform
import re
from enum import Enum
from pathlib import Path

import parsl
from parsl.config import Config

from IMLCV.configs.hpc_ugent import config as config_ugent
from IMLCV.configs.local_threadpool import get_config as get_config_local

ROOT_DIR = Path(__file__).resolve().parent.parent


print(f"{ROOT_DIR=}")

py_env = " which python"


PARSL_DICT = {}
REFERENCE_COMMANDS = {
    "cp2k": "mpirun cp2k_shell.psmp",
}


class Executors(Enum):
    default = "default"
    training = "training"
    reference = "reference"
    threadpool = "threadpool"


class ReferenceCommands(Enum):
    cp2k = "cp2k"


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
    local_ref_threads=4,
    walltime="48:00:00",
    bootstrap=False,
    memory_per_core=None,
    min_memery_per_node=None,
    path_internal: Path | None = None,
    cpu_cluster=None,
    gpu_cluster=None,
    initialize_logging=False,
    account=None,
    executor="work_queue",
    default_on_threads=False,
    training_cores=32,
    training_on_threads=False,
    work_queue_local=True,
    max_threads_local=10,
):
    if parsl.DataFlowKernelLoader._dfk is not None:
        print("parsl already configured, using previous setup")
        return

    if env is None:
        env = get_platform()

    if path_internal is None:
        path_internal = ROOT_DIR / ".runinfo"

    if env == "local":
        execs, labels, precommands, ref_comm = get_config_local(
            path_internal,
            ref_threads=local_ref_threads,
            max_threads=max_threads_local,
            work_queue=work_queue_local,
        )
    elif env == "hortense" or env == "stevin":
        execs, labels, precommands, ref_comm = config_ugent(
            env=env,
            path_internal=path_internal,
            singlepoint_nodes=singlepoint_nodes,
            walltime=walltime,
            bootstrap=bootstrap,
            memory_per_core=memory_per_core,
            min_memery_per_node=min_memery_per_node,
            cpu_cluster=cpu_cluster,
            gpu_cluster=gpu_cluster,
            account=account,
            executor=executor,
            default_on_threads=default_on_threads,
            default_threads=local_ref_threads,
            training_cores=training_cores,
            training_on_threads=training_on_threads,
        )

    config = Config(
        executors=execs,
        usage_tracking=False,
        run_dir=str(path_internal),
        initialize_logging=initialize_logging,
    )

    global PARSL_DICT

    for k, l, p in zip(["default", "training", "reference", "threadpool"], labels, precommands):
        PARSL_DICT[k] = [l, p]

    global REFERENCE_COMMANDS

    REFERENCE_COMMANDS.update(ref_comm)

    parsl.load(config=config)
