import os
from pathlib import Path

from parsl import HighThroughputExecutor, WorkQueueExecutor
from parsl.channels import LocalChannel
from parsl.executors.base import ParslExecutor

from psiflow.psiflow.external import SlurmProviderVSC

ROOT_DIR = Path(os.path.dirname(__file__)).parent
py_env = f"source {ROOT_DIR}/Miniconda3/bin/activate; which python"


def get_slurm_provider(
    label,
    path_internal: Path | str,
    cpu_cluster,
    gpu_cluster=None,
    account=None,
    channel=LocalChannel(),
    gpu=False,
    cores=None,
    open_mp_threads_per_core: int | None = None,
    parsl_cores=False,
    mem=None,
    memory_per_core=None,
    walltime="48:00:00",
    init_blocks=1,
    min_blocks=1,
    max_blocks=1,
    parallelism=1,
    use_work_queue: bool = True,
    wq_timeout: int = 120,  # in seconds
    gpu_part="gpu_rome_a100",
    cpu_part="cpu_rome",
):
    if gpu_cluster is None:
        gpu_cluster = cpu_cluster

    assert (
        open_mp_threads_per_core is None
    ), "open_mp_threads_per_core is not tested yet"

    worker_init = f"{py_env}; \n"

    if not parsl_cores:
        if open_mp_threads_per_singlepoint is None:
            open_mp_threads_per_singlepoint = 1

        total_cores = cores * open_mp_threads_per_singlepoint

        worker_init += "unset SLURM_CPUS_PER_TASK\n"
        worker_init += f"export SLURM_CPUS_PER_TASK={open_mp_threads_per_singlepoint}\n"
        worker_init += f"export SLURM_NTASKS_PER_NODE={cores}\n"
        worker_init += f"export SLURM_TASKS_PER_NODE={cores}\n"
        worker_init += f"export SLURM_NTASKS={cores}\n"
        worker_init += f"export OMP_NUM_THREADS={open_mp_threads_per_singlepoint}\n"
    else:
        assert open_mp_threads_per_singlepoint is None, "parsl doens't use openmp cores"
        total_cores = cores

    if memory_per_core is not None:
        if mem is None:
            mem = total_cores * memory_per_core
        else:
            if mem < total_cores * memory_per_core:
                mem = total_cores * memory_per_core

    #  mem_per_nod = mem

    mem_per_node = mem

    vsc_kwargs = {
        "cluster": cpu_cluster if not gpu else gpu_cluster,
        "partition": cpu_part if not gpu else gpu_part,
        "account": account,
        "channel": channel,
        "exclusive": False,
        "cmd_timeout": 60,
        "worker_init": worker_init,
        "cores_per_node": total_cores,
        "mem_per_node": mem,
        "walltime": walltime,
        "init_blocks": init_blocks,
        "min_blocks": min_blocks,
        "max_blocks": max_blocks,
        "parallelism": parallelism,
        "label": label,
        "nodes_per_block": 1,
    }

    if gpu:
        vsc_kwargs["scheduler_options"] = (
            f"#SBATCH --gpus=1\n#SBATCH --cpus-per-gpu={cores}\n#SBATCH --export=None",
        )  # request gpu

    provider = SlurmProviderVSC(**vsc_kwargs)

    if use_work_queue:
        worker_options = [
            f"--cores={cores}",
            f"--gpus={0 if not gpu else 1}",
        ]
        if hasattr(provider, "walltime"):
            walltime_hhmmss = provider.walltime.split(":")
            assert len(walltime_hhmmss) == 3
            walltime = 0.0
            walltime += 3600 * float(walltime_hhmmss[0])
            walltime += 60 * float(walltime_hhmmss[1])
            walltime += float(walltime_hhmmss[2])
            walltime -= 60 * 4  # add 4 minutes of slack

            worker_options.append(f"--wall-time={walltime}")
            worker_options.append(f"--timeout={wq_timeout}")
            worker_options.append("--parent-death")
        executor: ParslExecutor = WorkQueueExecutor(
            label=label,
            working_dir=str(Path(path_internal) / label),
            provider=provider,
            shared_fs=True,
            autocategory=False,
            port=0,
            max_retries=0,
            worker_options=" ".join(worker_options),
        )
    else:
        executor: ParslExecutor = HighThroughputExecutor(
            label=label,
            working_dir=str(Path(path_internal) / label),
            cores_per_worker=cores,
            provider=provider,
        )
    return executor


def config(
    env=None,
    singlepoint_nodes=16,
    walltime="48:00:00",
    bootstrap=False,
    memory_per_core=None,
    min_memery_per_node=None,
    path_internal: Path | None = None,
):
    if env == "hortense":
        kw = {
            "cpu_cluster": "dodrio",
            "account": "2022_069",
            "cpu_part": "cpu_rome",
            "gpu_part": "gpu_rome_a100",
            "path_internal": path_internal,
        }

    elif env == "stevin":
        cpu = "doduo"
        gpu = "accelgor"
        kw = {
            "cpu_cluster": cpu,
            "cpu_part": cpu,
            "gpu_cluster": gpu,
            "gpu_part": gpu,
            "path_internal": path_internal,
        }

    if bootstrap:
        execs = [
            get_slurm_provider(
                **kw,
                label="default",
                init_blocks=1,
                min_blocks=1,
                max_blocks=1,
                parallelism=0,
                cores=1,
                parsl_cores=True,
                mem=10,
                walltime="72:00:00",
            )
        ]

    else:
        # general tasks
        default = get_slurm_provider(
            label="default",
            init_blocks=1,
            min_blocks=1,
            max_blocks=512,
            parallelism=1,
            cores=4,
            parsl_cores=True,
            walltime="02:00:00",
            **kw,
        )

        gpu_part = get_slurm_provider(
            gpu=True,
            label="training",
            init_blocks=0,
            min_blocks=0,
            max_blocks=4,
            parallelism=1,
            cores=12,
            parsl_cores=False,
            walltime="02:00:00",
            **kw,
        )

        reference = get_slurm_provider(
            gpu=True,
            label="reference",
            memory_per_core=memory_per_core,
            mem=min_memery_per_node,
            init_blocks=0,
            min_blocks=0,
            max_blocks=64,
            parallelism=1,
            cores=singlepoint_nodes,
            parsl_cores=False,
            walltime=walltime,
            **kw,
        )

        execs = [default, gpu_part, reference]

    return execs
