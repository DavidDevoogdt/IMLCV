import logging
import os
from pathlib import Path

from parsl import HighThroughputExecutor, WorkQueueExecutor

# from parsl.channels import LocalChannel
from parsl.executors.base import ParslExecutor
from parsl.executors.taskvine import TaskVineExecutor, TaskVineFactoryConfig
from parsl.executors.threads import ThreadPoolExecutor
from parsl.launchers import SingleNodeLauncher
from parsl.providers import LocalProvider, SlurmProvider

ROOT_DIR = Path(os.path.dirname(__file__)).parent.parent.parent

logger = logging.getLogger(__name__)


def get_slurm_provider(
    env,
    label,
    path_internal: Path | str,
    cpu_cluster,
    parsl_tasks_per_block: int,
    gpu_cluster=None,
    account=None,
    # channel=LocalChannel(),
    gpu=False,
    threads_per_core: int | None = None,
    use_open_mp=False,
    parsl_cores=False,
    mem=None,
    memory_per_core=None,
    wall_time="48:00:00",
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
echo "init pixi"
cd /dodrio/scratch/projects/2024_026/IMLCV
pwd
eval "$(pixi shell-hook)"
echo "post init pixi"
which python
            """
        elif env == "stevin":
            py_env = """
export MAMBA_EXE='/kyukon/scratch/gent/vo/000/gvo00003/vsc43693/IMLCV/IMLCV/bin/micromamba';
export MAMBA_ROOT_PREFIX='/kyukon/scratch/gent/vo/000/gvo00003/vsc43693/IMLCV/IMLCV/micromamba';
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

        worker_init += f"export OMP_NUM_THREADS={threads_per_core}\n"

        # give all cores to xla
        worker_init += f"export XLA_FLAGS='--xla_force_host_platform_device_count={threads_per_core}'\n"

    else:
        assert threads_per_core is None, "parsl doens't use openmp cores"
        total_cores = parsl_tasks_per_block

        worker_init += f"export XLA_FLAGS='--xla_force_host_platform_device_count={total_cores}'\n"

    worker_init += f"mpirun -report-bindings -np {total_cores} echo 'a' " if load_cp2k else ""

    common_kwargs = {
        # "channel": channel,
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
            # "cores_per_node": 1,#total_cores,
            "mem_per_node": mem,
            "walltime": wall_time,
            "cmd_timeout": 60,
        }

        sheduler_options = f"\n#SBATCH --ntasks-per-node={threads_per_core}"

        if gpu:
            sheduler_options += (
                "\n#SBATCH --gpus=1\n#SBATCH --cpus-per-gpu={parsl_tasks_per_block}\n#SBATCH --export=None"
            )

        vsc_kwargs["scheduler_options"] = sheduler_options
        _provider = SlurmProvider(**common_kwargs, **vsc_kwargs)
    elif provider == "local":
        _provider = LocalProvider(
            **common_kwargs,
            cmd_timeout=1,
        )
    else:
        raise ValueError

    # let slurm use the cores as threads
    # pre_command = f"srun  -N 1 -n 1 -c {threads_per_core} --cpu-bind=no --export=ALL"
    pre_command = f"export JAX_NUM_CPU_DEVICES={threads_per_core}; "

    ref_comm: dict[str, str] = {
        "cp2k": "export OMP_NUM_THREADS=1; mpirun -report-bindings  -mca pml ucx -mca btl ^uct,ofi -mca mtl ^ofi cp2k_shell.psmp ",
    }

    print(f"{executor=}")

    if hasattr(_provider, "walltime"):
        walltime_hhmmss = _provider.walltime.split(":")  # type:ignore
        assert len(walltime_hhmmss) == 3
        wall_time_s = 0.0
        wall_time_s += 3600 * float(walltime_hhmmss[0])
        wall_time_s += 60 * float(walltime_hhmmss[1])
        wall_time_s += float(walltime_hhmmss[2])
        wall_time_s -= 60 * 4  # add 4 minutes of slack

        print(f"{wall_time_s=}")

    if executor == "work_queue":
        worker_options = [
            f"--cores={threads_per_core}",
            f"--gpus={0 if not gpu else 1}",
        ]

        if wall_time_s is not None:
            worker_options.append(f"--wall-time={wall_time_s}")

        worker_options.append(f"--timeout={wq_timeout}")
        worker_options.append("--parent-death")

        _executor: ParslExecutor = WorkQueueExecutor(
            label=label,
            working_dir=str(Path(path_internal) / label),
            provider=_provider,
            shared_fs=True,
            autolabel=False,
            autocategory=False,
            port=0,
            max_retries=1,  # do not retry task
            worker_options=" ".join(worker_options),
            coprocess=False,
            worker_executable="work_queue_worker",
            # scaling_cores_per_worker=threads_per_core,
        )

    elif executor == "task_vine":
        _executor: ParslExecutor = TaskVineExecutor(
            label=label,
            factory_config=TaskVineFactoryConfig(
                cores=parsl_tasks_per_block,
                gpus=0 if not gpu else 1,
                scratch_dir=str(Path(path_internal) / label),
            ),
            provider=_provider,
        )

    elif executor == "htex":
        _executor: ParslExecutor = HighThroughputExecutor(
            label=label,
            working_dir=str(Path(path_internal) / label),
            cores_per_worker=1,
            provider=provider,  # type:ignore
            drain_period=int(wall_time_s),
        )
    else:
        raise ValueError(f"unknown executor {executor=}")
    return _executor, pre_command, ref_comm


def config(
    env=None,
    singlepoint_nodes=16,
    walltime_training="6:00:00",
    walltime_ref="1:00:00",
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
    load_cp2k=False,
):
    def get_kwargs(cpu_cluster=None, gpu_cluster=None):
        if env == "hortense":
            if cpu_cluster is not None:
                assert cpu_cluster in [
                    "cpu_milan",
                    "cpu_milan_rhel_9",
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
                    "shinx",
                ]

            if gpu_cluster is not None:
                assert gpu_cluster in ["joltik", "accelgor", "litleo"]

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
        raise NotImplementedError

    default_labels: list[str] = []
    training_labels: list[str] = []
    reference_labels: list[str] = []
    threadpool_labels: list[str] = []

    default_pre_commands: list[str] = []
    training_pre_commands: list[str] = []
    reference_pre_commands: list[str] = []
    threadpool_pre_commands: list[str] = []

    resources = {
        "default": {"cores": default_threads},
        "training": {"cores": training_cores},
        "reference": {"cores": singlepoint_nodes},
        "threadpool": {"cores": default_threads},
    }

    reference_command = None

    execs: list[ParslExecutor] = []

    if default_on_threads:
        label = "default"

        default = ThreadPoolExecutor(
            label=label,
            max_threads=default_threads,
            working_dir=str(Path(path_internal) / label),  # type:ignore
        )

        execs.append(default)
        default_labels.append(label)
        default_pre_commands.append("")

    if not isinstance(cpu_cluster, list):
        cpu_cluster = [cpu_cluster]  # type:ignore

    if training_on_threads:
        label = "training"

        training = ThreadPoolExecutor(
            label=label,
            max_threads=training_cores,
            working_dir=str(Path(path_internal) / label),  # type:ignore
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
                    wall_time=walltime_training,
                    **kw,
                )
                execs.append(gpu_part)
                training_pre_commands.append(pre_command)
                training_labels.append(label)
        else:
            for cpu in cpu_cluster:  # type:ignore
                kw = get_kwargs(cpu_cluster=cpu)

                label = f"training_{cpu}"

                cpu_part, pre_command, _ = get_slurm_provider(
                    label=label,
                    init_blocks=0,
                    min_blocks=0,
                    max_blocks=16,
                    parallelism=1,
                    parsl_tasks_per_block=1,
                    threads_per_core=training_cores,
                    parsl_cores=False,
                    wall_time=walltime_training,
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
                max_blocks=2048,
                parallelism=1,
                parsl_tasks_per_block=1,
                threads_per_core=singlepoint_nodes,
                parsl_cores=False,
                wall_time=walltime_ref,
                load_cp2k=load_cp2k,
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
                    min_blocks=0,
                    max_blocks=2048,
                    parallelism=1,
                    parsl_tasks_per_block=1,
                    threads_per_core=default_threads,
                    parsl_cores=False,
                    wall_time=walltime_ref,
                    **kw,
                )

                execs.append(default)
                default_labels.append(label)
                default_pre_commands.append(pre_command)

    label = "threadpool"

    tp = ThreadPoolExecutor(
        label=label,
        max_threads=default_threads,
        working_dir=str(Path(path_internal) / label),  # type:ignore
    )

    execs.append(tp)
    threadpool_labels.append(label)
    threadpool_pre_commands.append("")

    pre_commands = [default_pre_commands, training_pre_commands, reference_pre_commands, threadpool_pre_commands]

    pre_commands_filtered: list[str] = []

    for pc in pre_commands:
        pre_commands_filtered.append(pc[0])

        if len(pc) > 1:
            for pci in pc[1:]:
                assert pci == pc[0], "all pre_commands should be the same per category"

    return (
        execs,
        [default_labels, training_labels, reference_labels, threadpool_labels],
        pre_commands_filtered,
        reference_command,
        resources,
    )
