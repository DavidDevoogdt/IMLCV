import platform
import re
from enum import Enum
from pathlib import Path

import parsl
from parsl.config import Config

from IMLCV.configs.cluster import config as config_cluster
from IMLCV.configs.local_threadpool import get_config as get_config_local

ROOT_DIR = Path(__file__).resolve().parent.parent


print(f"{ROOT_DIR=}")



PARSL_DICT: dict[str, tuple[list[str], str]] = {}

RESOURCES_DICT = {}

REFERENCE_COMMANDS: dict[str, str] = {
    "cp2k": "mpirun cp2k_shell.psmp",
}

class Executors(Enum):
    default = "default"
    training = "training"
    reference = "reference"
    threadpool = "threadpool"


class ReferenceCommands(Enum):
    cp2k = "cp2k"

class GpuKind(Enum):
    nvidia = "nvidia"
    rocm = "rocm"


def config(
    env=None,
    singlepoint_nodes=16,
    local_ref_threads=4,
    training_on_gpu=False,
    reference_on_gpu=False,
    reference_blocks=4,
    walltime_training="6:00:00",
    walltime_ref="1:00:00",
    bootstrap=False,
    memory_per_core=None,
    min_memery_per_node=None,
    path_internal: Path | None = None,
    cpu_cluster=None,
    gpu_cluster=None,
    cpu_part=None,
    gpu_part=None,
    initialize_logging=False,
    account=None,
    executor="work_queue",
    default_on_threads=False,
    training_cores=32,
    training_on_threads=False,
    work_queue_local=True,
    max_threads_local=10,
    gpu_kind:GpuKind=GpuKind.nvidia,

):
    print(f"{reference_blocks=}")

    if parsl.DataFlowKernelLoader._dfk is not None:  # type: ignore
        print("parsl already configured, using previous setup")
        return



    if path_internal is None:
        path_internal = "/tmp/.runinfo"

    if env == "local":
        execs, labels, precommands, ref_comm, resources = get_config_local(
            path_internal,
            ref_threads=local_ref_threads,
            max_threads=max_threads_local,
            work_queue=work_queue_local,
        )
    else:
        execs, labels, precommands, ref_comm, resources = config_cluster(
            env=env,
            path_internal=path_internal,
            singlepoint_nodes=singlepoint_nodes,
            walltime_training=walltime_training,
            walltime_ref=walltime_ref,
            bootstrap=bootstrap,
            memory_per_core=memory_per_core,
            min_memery_per_node=min_memery_per_node,
            cpu_cluster=cpu_cluster,
            gpu_cluster=gpu_cluster,
            cpu_part=cpu_part,
            gpu_part=gpu_part,
            account=account,
            executor=executor,
            default_on_threads=default_on_threads,
            default_threads=local_ref_threads,
            training_cores=training_cores,
            training_on_threads=training_on_threads,
            reference_blocks=reference_blocks,
            training_on_gpu=training_on_gpu,
            reference_on_gpu=reference_on_gpu,
            gpu_kind=gpu_kind,
        )

    config = Config(
        executors=execs,
        usage_tracking=False,
        run_dir=str(path_internal),
        initialize_logging=initialize_logging,
        app_cache=False,
        retries=0,
    )

    global PARSL_DICT

    for k, l, p in zip(["default", "training", "reference", "threadpool"], labels, precommands):
        PARSL_DICT[k] = (l, p)

    global REFERENCE_COMMANDS

    if ref_comm is not None:
        REFERENCE_COMMANDS.update(ref_comm)

    global RESOURCES_DICT

    RESOURCES_DICT.update(resources)

    parsl.load(config=config)
