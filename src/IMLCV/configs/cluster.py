import logging
import os
from pathlib import Path

from typing import TYPE_CHECKING
from parsl import HighThroughputExecutor, WorkQueueExecutor

# from parsl.channels import LocalChannel
from parsl.executors.base import ParslExecutor
from parsl.executors.taskvine import TaskVineExecutor, TaskVineFactoryConfig, TaskVineManagerConfig
from parsl.executors.threads import ThreadPoolExecutor
from parsl.launchers import SimpleLauncher, SingleNodeLauncher, SrunLauncher
from parsl.launchers.launchers import Launcher
from parsl.providers import LocalProvider, SlurmProvider
from enum import Enum



if TYPE_CHECKING:
    from IMLCV.configs.cluster import GpuKind


ROOT_DIR = Path(os.path.dirname(__file__)).parent.parent.parent

logger = logging.getLogger(__name__)


class SlurmLauncher(Launcher):
    def __init__(self, debug: bool = True, overrides: str = ""):
        super().__init__(debug=debug)
        self.overrides = overrides

    def __call__(self, command: str, tasks_per_node: int, nodes_per_block: int) -> str:
        x = """set -e

NODELIST=$(scontrol show hostnames)
NODE_ARRAY=($NODELIST)
NODE_COUNT=${{#NODE_ARRAY[@]}}
EXPECTED_NODE_COUNT={nodes_per_block}

# Check if the length of NODELIST matches the expected number of nodes
if [ $NODE_COUNT -ne $EXPECTED_NODE_COUNT ]; then
  echo "Error: Expected $EXPECTED_NODE_COUNT nodes, but got $NODE_COUNT nodes."
  exit 1
fi


for NODE in $NODELIST; do
  srun --nodes=1  --export=ALL  -l {overrides} --nodelist=$NODE {command} &
  if [ $? -ne 0 ]; then
    echo "Command failed on node $NODE"
  fi
done

wait
""".format(
            nodes_per_block=nodes_per_block,
            command=command,
            overrides=self.overrides,
        )
        return x


class MyWorkQueueExecutor(WorkQueueExecutor):
    def __init__(self, *args, port: None | int | tuple[int, int] = None, **kwargs):
        super().__init__(*args, **kwargs)

        if port is not None:
            self.port = port

    def _get_launch_command(self, block_id):
        return self.worker_command


class MySlurmProvider(SlurmProvider):
    def __init__(self, *args, tasks_per_node=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.tasks_per_node = tasks_per_node

    def submit(self, command: str, tasks_per_node: int, job_name="parsl.slurm") -> str:
        return super().submit(command, self.tasks_per_node, job_name)



def get_slurm_provider(
    env,
    label,
    gpu_kind,
    path_internal: Path | str,
    parsl_tasks_per_block: int,
    cpu_cluster,
    gpu_cluster=None,
    cpu_part=None,
    gpu_part=None,
    
    account=None,
    # channel=LocalChannel(),
    gpu=False,
    threads_per_core: int | None = None,
    mem=None,
    memory_per_core=None,
    wall_time="48:00:00",
    init_blocks=1,
    min_blocks=1,
    max_blocks=1,
    # parallelism=1,
    executor="htex",
    wq_timeout: int = 60,  # in seconds

    py_env=None,
    provider="slurm",
    load_cp2k=False,
    
):
    if py_env is None:
        # if env == "hortense":

        environment = ("cuda13"  if gpu_kind.value=="nvidia" else "rocm")  if gpu else "cpu"

        print("setting python env for hortense")
        py_env = f"""
echo "init pixi"
cd {ROOT_DIR}
pwd
export PIXI_CACHE_DIR="./.pixi_cache"
export PATH="~/.pixi/bin:$PATH"
which pixi
eval "$(pixi shell-hook -e {environment} --as-is )"

which work_queue_worker
            """
    # else:
    #     raise ValueError

    if gpu_cluster is None:
        gpu_cluster = cpu_cluster

    # assert threads_per_core is None, "threads_per_core is not tested yet"

    worker_init = f"{py_env}\n"

    if load_cp2k:
        raise ValueError()

    # only submit 1/ parsl_tasks_per_block blocks at a time
    parallelism = 1 / parsl_tasks_per_block

    if threads_per_core is None:
        threads_per_core = 1

    total_cores = parsl_tasks_per_block * threads_per_core

    # give all cores to xla
    worker_init += f"export XLA_FLAGS='--xla_force_host_platform_device_count={threads_per_core}'\n"

    if gpu:
        # dynamic memory allocation, both for jax and pytorch, for all blocks.
        
        worker_init += "export XLA_PYTHON_CLIENT_PREALLOCATE=false\n"

        if gpu_kind.value == "nvidia":


            worker_init += "export TF_GPU_ALLOCATOR=cuda_malloc_async\n"
            worker_init += "export PYTORCH_ALLOC_CONF=backend:cudaMallocAsync\n"
            worker_init += """
# start MPS
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_$SLURM_JOB_ID
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY"
chmod 700 "$CUDA_MPS_PIPE_DIRECTORY"
export CUDA_MPS_LOG_DIRECTORY="$CUDA_MPS_PIPE_DIRECTORY"
nvidia-cuda-mps-control -d
echo "CUDA MPS server started at $CUDA_MPS_PIPE_DIRECTORY"

# ensure MPS is stopped and cleaned up on exit (normal or via signal)
cleanup() {
  printf "quit\\n" | nvidia-cuda-mps-control || true
  rm -rf "$CUDA_MPS_PIPE_DIRECTORY" || true
  echo "CUDA MPS server stopped"
}
trap cleanup EXIT

# give MPS a moment to initialize
sleep 1 
"""
            worker_init += "nvidia-smi\n"
        elif gpu_kind.value == "rocm":
            worker_init += "rocm-smi\n"

    overrides = f"--ntasks-per-node={parsl_tasks_per_block} --cpus-per-task={threads_per_core}"

    # if gpu:
    #     overrides += " --gres=gpu:1"

    common_kwargs = {
        # "channel": channel,
        "init_blocks": init_blocks,
        "min_blocks": min_blocks,
        "max_blocks": max_blocks,
        "parallelism": parallelism,
        "nodes_per_block": 1,
        "worker_init": worker_init,
        "launcher": SlurmLauncher(overrides=overrides),
    }

    if provider == "slurm":
        if memory_per_core is not None:
            if mem is None:
                mem = total_cores * memory_per_core
            else:
                if mem < total_cores * memory_per_core:
                    mem = total_cores * memory_per_core

        vsc_kwargs = {
            "clusters": cpu_cluster if not gpu else gpu_cluster,
            "partition": cpu_part if not gpu else gpu_part,
            "account": account,
            "exclusive": False,
            # "cores_per_node": 1,#total_cores,
            "mem_per_node": mem,
            "walltime": wall_time,
            "cmd_timeout": 60,
        }

        # no exports from submission env
        sheduler_options = f"""
#SBATCH --cpus-per-task={threads_per_core}
#SBATCH -v
#SBATCH --export=NONE 

 """

        if gpu:
            sheduler_options += f"\n#SBATCH --gres=gpu:1\n"

        vsc_kwargs["scheduler_options"] = sheduler_options
        _provider = MySlurmProvider(**common_kwargs, **vsc_kwargs, tasks_per_node=parsl_tasks_per_block)
    elif provider == "local":
        _provider = LocalProvider(
            **common_kwargs,
            cmd_timeout=1,
        )
    else:
        raise ValueError

    # let slurm use the cores as threads
    # pre_command = f"srun  --cpus-per-task {threads_per_core} "
    # pre_command = f"srun --cpus-per-task {threads_per_core} --ntasks=1 "
    pre_command = ""

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
            # f"--gpus={0 if not gpu else 1}", #gpu is not for workqueue, but command launched inside workqueue
        ]

        if wall_time_s is not None:
            worker_options.append(f"--wall-time={wall_time_s}")

        worker_options.append(f"--timeout={wq_timeout}")
        worker_options.append("--parent-death")

        _executor: ParslExecutor = MyWorkQueueExecutor(
            label=label,
            working_dir=str(Path(path_internal) / label),
            provider=_provider,
            shared_fs=True,
            autolabel=False,
            autocategory=False,
            port=(50000, 60000),
            max_retries=1,  # do not retry task max once
            worker_options=" ".join(worker_options),
            coprocess=False,
            worker_executable="work_queue_worker",
        )

    elif executor == "htex":
        _executor: ParslExecutor = HighThroughputExecutor(
            label=label,
            working_dir=str(Path(path_internal) / label),
            cores_per_worker=1,
            provider=_provider,  # type:ignore
            # drain_period=600,  # drain after 5 mins
        )
    else:
        raise ValueError(f"unknown executor {executor=}")

    return _executor, pre_command, ref_comm



def config(
    gpu_kind,
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
    cpu_part: str| list[str] | None = None,
    gpu_part: str| list[str] | None = None,
    reference_blocks: int = 1,
    py_env=None,
    account=None,
    executor="htex",
    default_on_threads=False,
    default_threads=4,
    training_on_threads=False,
    training_cores=12,
    load_cp2k=False,
    training_on_gpu=False,
    reference_on_gpu=False,
    
):


    kw={
        "cpu_cluster": cpu_cluster,
        "gpu_cluster": gpu_cluster,
        "cpu_part": cpu_part,
        "gpu_part": gpu_part,
        "path_internal": path_internal,
        "gpu_kind": gpu_kind,
        
    }

    kw["env"] = env
    kw["py_env"] = py_env
    kw["executor"] = executor
    kw["account"] = account



    if not isinstance(cpu_cluster, list) and cpu_cluster is not None:
        cpu_cluster = [cpu_cluster]

    if not isinstance(gpu_cluster, list) and gpu_cluster is not None:
        gpu_cluster = [gpu_cluster]

    if training_on_gpu:
        assert gpu_cluster is not None, "gpu_cluster must be provided when training_on_gpu or reference_on_gpu is True"
    if reference_on_gpu:
        assert gpu_cluster is not None, "gpu_cluster must be provided when training_on_gpu or reference_on_gpu is True"

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
        if training_on_gpu:
            
            for gpu in gpu_cluster:
               
                label = f"training_{gpu}"

                gpu_part, pre_command, _ = get_slurm_provider(
                    gpu=True,
                    label=label,
                    init_blocks=0,
                    min_blocks=0,
                    max_blocks=5,
                    # parallelism=1,
                    parsl_tasks_per_block=1,
                    threads_per_core=training_cores,
                    wall_time=walltime_training,
                    **kw,
                )
                execs.append(gpu_part)
                training_pre_commands.append(pre_command)
                training_labels.append(label)
        else:
            for cpu in cpu_cluster:  # type:ignore
             
                label = f"training_{cpu}"

                cpu_part, pre_command, _ = get_slurm_provider(
                    label=label,
                    init_blocks=0,
                    min_blocks=0,
                    max_blocks=50,
                    # parallelism=1,
                    parsl_tasks_per_block=1,
                    threads_per_core=training_cores,
                    wall_time=walltime_training,
                    **kw,
                )
                execs.append(cpu_part)
                training_labels.append(label)
                training_pre_commands.append(pre_command)

        if reference_on_gpu:
            for gpu in gpu_cluster:
              

                label = f"reference_{gpu}"

                reference, pre_command, ref_com = get_slurm_provider(
                    gpu=True,
                    label=label,
                    memory_per_core=memory_per_core,
                    mem=min_memery_per_node,
                    init_blocks=0,
                    min_blocks=0,
                    max_blocks=48,
                    # parallelism=1,
                    parsl_tasks_per_block=reference_blocks,
                    threads_per_core=singlepoint_nodes,
                    wall_time=walltime_ref,
                    load_cp2k=load_cp2k,
                    **kw,
                )

                execs.append(reference)
                reference_labels.append(label)
                reference_pre_commands.append(pre_command)

                if reference_command is None:
                    reference_command = ref_com

        else:
            for cpu in cpu_cluster:
               

                label = f"reference_{cpu}"

                reference, pre_command, ref_com = get_slurm_provider(
                    label=label,
                    memory_per_core=memory_per_core,
                    mem=min_memery_per_node,
                    init_blocks=0,
                    min_blocks=0,
                    max_blocks=2048,
                    # parallelism=1,
                    parsl_tasks_per_block=reference_blocks,
                    threads_per_core=singlepoint_nodes,
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
            for cpu in cpu_cluster:
                label = f"default_{cpu}"

                default, pre_command, _ = get_slurm_provider(
                    label=label,
                    init_blocks=1,
                    min_blocks=0,
                    max_blocks=2048,
                    # parallelism=1,
                    parsl_tasks_per_block=1,
                    threads_per_core=default_threads,
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
