import platform
import re
from pathlib import Path

import parsl
from IMLCV.configs.hpc_ugent import config as config_ugent
from IMLCV.configs.local_threadpool import get_config as get_config_local
from parsl.config import Config

ROOT_DIR = Path(__file__).resolve().parent.parent

print(f"{ROOT_DIR=}")

py_env = " which python"

DEFAULT_LABELS = []
REFERENCE_LABELS = []
TRAINING_LABELS = []


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
    executor="htex",
    default_on_threads=False,
    training_cores=32,
):
    if parsl.DataFlowKernelLoader._dfk is not None:
        print("parsl already configured, using previous setup")
        return

    if env is None:
        env = get_platform()

    if path_internal is None:
        path_internal = ROOT_DIR / ".runinfo"

    if env == "local":
        execs, [default_labels, trainig_labels, reference_labels] = get_config_local(
            path_internal, ref_threads=local_ref_threads
        )
    elif env == "hortense" or env == "stevin":
        execs, [default_labels, trainig_labels, reference_labels] = config_ugent(
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
            trainig_cores=training_cores,
        )

    config = Config(
        executors=execs,
        usage_tracking=False,
        run_dir=str(path_internal),
        initialize_logging=initialize_logging,
    )

    DEFAULT_LABELS.extend(default_labels)
    REFERENCE_LABELS.extend(reference_labels)
    TRAINING_LABELS.extend(trainig_labels)

    parsl.load(config=config)


def get_cp2k():
    env = get_platform()
    if env == "hortense":
        return "export OMP_NUM_THREADS=1; mpirun  cp2k_shell.popt"
    if env == "stevin":
        return "export OMP_NUM_THREADS=1; mpirun  cp2k_shell.popt"
    raise ValueError(f"unknow {env=} for cp2k ")


# -mca pml ucx -mca btl ^uct
# export OMPI_MCA_btl=^openib
# mca_base_component_show_load_errors 0
