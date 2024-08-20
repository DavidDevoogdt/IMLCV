import logging

# import math
import os

# import time
from pathlib import Path

from parsl import HighThroughputExecutor, WorkQueueExecutor
from parsl.channels import LocalChannel
from parsl.executors.base import ParslExecutor

# from parsl.jobs.states import JobState
# from parsl.providers.base import JobStatus
# from parsl.providers.slurm.template import template_string
# from parsl.utils import wtime_to_minutes
from parsl.executors.taskvine import TaskVineExecutor, TaskVineFactoryConfig
from parsl.executors.threads import ThreadPoolExecutor
from parsl.launchers import SingleNodeLauncher

# import parsl.providers.slurm.slurm
from parsl.providers import LocalProvider, SlurmProvider

ROOT_DIR = Path(os.path.dirname(__file__)).parent.parent.parent


logger = logging.getLogger(__name__)


def get_slurm_provider(
    env,
    label,
    path_internal: Path | str,
    cpu_cluster,
    gpu_cluster=None,
    account=None,
    channel=LocalChannel(),
    gpu=False,
    parsl_tasks_per_block=None,
    threads_per_core: int | None = None,
    use_open_mp=False,
    parsl_cores=False,
    mem=None,
    memory_per_core=None,
    walltime="48:00:00",
    init_blocks=1,
    min_blocks=1,
    max_blocks=1,
    parallelism=1,
    executor="htex",
    wq_timeout: int = 60,  # in seconds
    gpu_part="gpu_rome_a100",
    cpu_part="cpu_rome",
    py_env=None,
    provider="slurm",
    launcher=SingleNodeLauncher(),
    load_cp2k=False,
):
    if py_env is None:
        if env == "hortense":
            print("setting python env for hortense")
            py_env = """
export MAMBA_EXE=$VSC_HOME/2024_026/IMLCV/bin/micromamba
export MAMBA_ROOT_PREFIX=$VSC_HOME/2024_026/IMLCV/micromamba
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2> /dev/null)"
micromamba activate py312
which python
            """
        elif env == "stevin":
            py_env = """
export MAMBA_EXE=$VSC_HOME/IMLCV/bin/micromamba
export MAMBA_ROOT_PREFIX=$VSC_HOME/IMLCV/micromamba
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2> /dev/null)"
micromamba activate py312
which python
"""

    if gpu_cluster is None:
        gpu_cluster = cpu_cluster

    # assert threads_per_core is None, "threads_per_core is not tested yet"

    worker_init = f"{py_env}\n"

    if env == "hortense":
        if load_cp2k:
            worker_init += "module load CP2K/2023.1-foss-2022b\n"
            worker_init += "module unload SciPy-bundle Python\n"
        else:
            worker_init += "module load OpenMPI\n"

    elif env == "stevin":
        if load_cp2k:
            worker_init += "module load CP2K/2023.1-foss-2022b\n"
            worker_init += "module unload SciPy-bundle Python\n"
        else:
            worker_init += "module load OpenMPI\n"

    if not parsl_cores:
        if threads_per_core is None:
            threads_per_core = 1

        total_cores = parsl_tasks_per_block * threads_per_core

        worker_init += f"export OMP_NUM_THREADS={threads_per_core if use_open_mp else 1   }\n"

        # give all cores to xla
        worker_init += f"export XLA_FLAGS='--xla_force_host_platform_device_count={threads_per_core}'\n"

    else:
        assert threads_per_core is None, "parsl doens't use openmp cores"
        total_cores = parsl_tasks_per_block

        worker_init += f"export XLA_FLAGS='--xla_force_host_platform_device_count={total_cores}'\n"

    worker_init += f"mpirun -report-bindings -np {total_cores} echo 'a' "

    common_kwargs = {
        "channel": channel,
        "init_blocks": init_blocks,
        "min_blocks": min_blocks,
        "max_blocks": max_blocks,
        "parallelism": parallelism,
        "nodes_per_block": 1,
        "worker_init": worker_init,
        "launcher": SingleNodeLauncher(),
    }

    if provider == "slurm":
        if memory_per_core is not None:
            if mem is None:
                mem = total_cores * memory_per_core
            else:
                if mem < total_cores * memory_per_core:
                    mem = total_cores * memory_per_core

        vsc_kwargs = {
            # "cluster": cpu_cluster if not gpu else gpu_cluster,
            "partition": cpu_part if not gpu else gpu_part,
            "account": account,
            "exclusive": False,
            "cores_per_node": total_cores,
            "mem_per_node": mem,
            "walltime": walltime,
            "cmd_timeout": 60,
        }

        sheduler_options = "#SBATCH --signal B:USR2"  # send signal to worker if job is cancelled/ time is up/ OOM

        if gpu:
            sheduler_options += (
                "\n#SBATCH --gpus=1\n#SBATCH --cpus-per-gpu={parsl_tasks_per_block}\n#SBATCH --export=None"
            )

        vsc_kwargs["scheduler_options"] = sheduler_options
        provider = SlurmProvider(**common_kwargs, **vsc_kwargs)
    elif provider == "local":
        provider = LocalProvider(
            **common_kwargs,
            cmd_timeout=1,
        )

    # let slurm use the cores as threads
    pre_command = f"srun  -N 1 -n 1 -c {threads_per_core} --cpu-bind=threads --export=ALL"

    ref_comm = {
        "cp2k": "export OMP_NUM_THREADS=1; mpirun -report-bindings  -mca pml ucx -mca btl ^uct,ofi -mca mtl ^ofi cp2k_shell.psmp ",
    }

    print(f"{executor=}")

    if executor == "work_queue":
        worker_options = [
            f"--cores={threads_per_core}",
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
            autolabel=False,
            autocategory=False,
            port=0,
            max_retries=1,  # do not retry task
            worker_options=" ".join(worker_options),
            coprocess=False,
            worker_executable="work_queue_worker",
        )

    elif executor == "task_vine":
        executor: ParslExecutor = TaskVineExecutor(
            label=label,
            factory_config=TaskVineFactoryConfig(
                cores=parsl_tasks_per_block,
                gpus=0 if not gpu else 1,
                scratch_dir=str(Path(path_internal) / label),
            ),
            provider=provider,
        )

    elif executor == "htex":
        executor: ParslExecutor = HighThroughputExecutor(
            label=label,
            working_dir=str(Path(path_internal) / label),
            cores_per_worker=threads_per_core,
            provider=provider,
        )
    else:
        raise ValueError(f"unknown executor {executor=}")
    return executor, pre_command, ref_comm


def config(
    env=None,
    singlepoint_nodes=16,
    walltime="48:00:00",
    bootstrap=False,
    memory_per_core=None,
    min_memery_per_node=None,
    path_internal: Path | None = None,
    cpu_cluster: str | list[str] | None = None,
    gpu_cluster: str | list[str] | None = None,
    py_env=None,
    account=None,
    executor="htex",
    default_on_threads=False,
    default_threads=4,
    training_on_threads=False,
    training_cores=12,
):
    def get_kwargs(cpu_cluster=None, gpu_cluster=None):
        if env == "hortense":
            if cpu_cluster is not None:
                assert cpu_cluster in [
                    "cpu_milan",
                    "cpu_rome",
                    "cpu_rome_512",
                    "cpu_rome_all",
                    "debug_rome",
                ]
            if gpu_cluster is not None:
                assert gpu_cluster in [
                    "gpu_rome_a100",
                    "gpu_rome_a100_401",
                    "gpu_rome_a100_80",
                ]

            kw = {
                "cpu_cluster": "dodrio",
                "account": account,
                "cpu_part": "cpu_rome" if cpu_cluster is None else cpu_cluster,
                "gpu_part": "gpu_rome_a100" if gpu_cluster is None else gpu_cluster,
                "path_internal": path_internal,
            }

        elif env == "stevin":
            if cpu_cluster is not None:
                assert cpu_cluster in [
                    "slaking",
                    "swalot",
                    "skitty",
                    "victini",
                    "kirlia",
                    "doduo",
                    "donphan",
                    "gallade",
                ]

            if gpu_cluster is not None:
                assert gpu_cluster in ["joltik", "accelgor"]

            cpu = "doduo" if cpu_cluster is None else cpu_cluster
            gpu = "accelgor" if gpu_cluster is None else gpu_cluster
            kw = {
                "cpu_cluster": cpu,
                "cpu_part": cpu,
                "gpu_cluster": gpu,
                "gpu_part": gpu,
                "path_internal": path_internal,
            }
        kw["env"] = env
        kw["py_env"] = py_env
        kw["executor"] = executor

        return kw

    if not isinstance(cpu_cluster, list) and cpu_cluster is not None:
        cpu_cluster = [cpu_cluster]

    if not isinstance(gpu_cluster, list) and gpu_cluster is not None:
        gpu_cluster = [gpu_cluster]

    if bootstrap:
        assert not isinstance(cpu_cluster, list), "bootstrap does not support multiple clusters"

        executor, pre_command, _ = get_slurm_provider(
            **get_kwargs(cpu_cluster),
            label="default",
            init_blocks=1,
            parsl_cores=True,
            mem=10,
            walltime="72:00:00",
        )

        execs = [executor]
        pre_commands = [pre_command]

    else:
        default_labels = []
        training_labels = []
        reference_labels = []

        default_pre_commands = []
        training_pre_commands = []
        reference_pre_commands = []

        reference_command = None

        execs = []

        if default_on_threads:
            label = "default"

            default = ThreadPoolExecutor(
                label=label,
                max_threads=default_threads,
                working_dir=str(Path(path_internal) / label),
            )

            execs.append(default)
            default_labels.append(label)
            default_pre_commands.append("")

        if not isinstance(cpu_cluster, list):
            cpu_cluster = [cpu_cluster]

        if training_on_threads:
            label = "training"

            training = ThreadPoolExecutor(
                label=label,
                max_threads=training_cores,
                working_dir=str(Path(path_internal) / label),
            )

            execs.append(training)
            training_labels.append(label)
            training_pre_commands.append("")

        else:
            if gpu_cluster is not None:
                for gpu in gpu_cluster:
                    kw = get_kwargs(gpu_cluster=gpu)

                    label = f"training_{gpu}"

                    gpu_part, pre_command, _ = get_slurm_provider(
                        gpu=True,
                        label=label,
                        init_blocks=0,
                        min_blocks=0,
                        max_blocks=4,
                        parallelism=1,
                        parsl_tasks_per_block=1,
                        threads_per_core=training_cores,
                        parsl_cores=False,
                        walltime="04:00:00",
                        **kw,
                    )
                    execs.append(gpu_part)
                    training_pre_commands.append(pre_command)
                    training_labels.append(label)
            else:
                for cpu in cpu_cluster:
                    kw = get_kwargs(cpu_cluster=cpu)

                    label = f"training_{cpu}"

                    cpu_part, pre_command, _ = get_slurm_provider(
                        label=label,
                        init_blocks=0,
                        min_blocks=0,
                        max_blocks=4,
                        parallelism=1,
                        parsl_tasks_per_block=1,
                        threads_per_core=training_cores,
                        parsl_cores=False,
                        walltime="04:00:00",
                        **kw,
                    )
                    execs.append(cpu_part)
                    training_labels.append(label)
                    training_pre_commands.append(pre_command)

        if cpu_cluster is not None:
            for cpu in cpu_cluster:
                kw = get_kwargs(cpu_cluster=cpu)

                label = f"reference_{cpu}"

                reference, pre_command, ref_com = get_slurm_provider(
                    label=label,
                    memory_per_core=memory_per_core,
                    mem=min_memery_per_node,
                    init_blocks=0,
                    min_blocks=0,
                    max_blocks=80,
                    parallelism=1,
                    parsl_tasks_per_block=1,
                    threads_per_core=singlepoint_nodes,
                    parsl_cores=False,
                    walltime=walltime,
                    load_cp2k=True,
                    **kw,
                )

                execs.append(reference)
                reference_labels.append(label)
                reference_pre_commands.append(pre_command)

                if reference_command is None:
                    reference_command = ref_com

                if not default_on_threads:
                    # general tasks
                    label = f"default_{cpu}"

                    default, pre_command, _ = get_slurm_provider(
                        label=label,
                        init_blocks=1,
                        min_blocks=1,
                        max_blocks=512,
                        parallelism=1,
                        parsl_tasks_per_block=1,
                        threads_per_core=default_threads,
                        parsl_cores=False,
                        walltime="04:00:00",
                        **kw,
                    )

                    execs.append(default)
                    default_labels.append(label)
                    default_pre_commands.append(pre_command)

        pre_commands = [
            default_pre_commands,
            training_pre_commands,
            reference_pre_commands,
        ]

    pre_commands_filtered = []

    for pc in pre_commands:
        pre_commands_filtered.append(pc[0])

        if len(pc) > 1:
            for pci in pc[1:]:
                assert pci == pc[0], "all pre_commands should be the same per category"

    return execs, [default_labels, training_labels, reference_labels], pre_commands_filtered, reference_command
