import math
from pathlib import Path

from parsl.executors import ThreadPoolExecutor, WorkQueueExecutor
from parsl.executors.base import ParslExecutor
from parsl.launchers import SingleNodeLauncher
from parsl.providers import LocalProvider
from parsl.launchers.launchers import Launcher


# class SrunLauncher(Launcher):
#     def __init__(self, debug: bool = True, overrides: str = ""):
#         super().__init__(debug=debug)
#         self.overrides = overrides

#     def __call__(self, command: str, tasks_per_node: int, nodes_per_block: int, script_dir: str = "") -> str:
#         x = """set -e

# srun  --nodes=1  -l {overrides}  {command}


# """.format(
#             nodes_per_block=nodes_per_block,
#             command=command,
#             overrides=self.overrides,
#             app="",
#         )
#         return x


def get_config(
    path_internal,
    ref_threads=2,
    reference_blocks=4,
    training_cores=10,
    default_threads=2,
    work_queue=True,
):
    def _get_exec(label, num, threads, srun=False):
        print(f"providing {num=} executors for {label=} with {threads=}")

        if not work_queue:
            exec = ThreadPoolExecutor(
                label=label,
                max_threads=threads,
                working_dir=str(path_internal),
            )

        else:
            provider = LocalProvider(
                worker_init="",
                cmd_timeout=1,
                launcher=SingleNodeLauncher(),
                max_blocks=1,
            )

            exec = WorkQueueExecutor(
                label=label,
                working_dir=str(Path(path_internal) / label),
                worker_options=f"--cores={threads} --gpus 0 --timeout=60 --parent-death ",
                provider=provider,
                port=0,
                shared_fs=True,
                autolabel=False,
                autocategory=False,
                max_retries=1,
                worker_executable=f"srun --cpus-per-task={threads} --ntasks={num} --threads-per-core=1 --nodes=1 work_queue_worker"
                if srun
                else "work_queue_worker",
            )

        return f"export JAX_NUM_CPU_DEVICES={threads}; ", exec

    com_executors: list[tuple[str, ParslExecutor]] = [
        _get_exec("training", num=1, threads=training_cores),
        _get_exec("default", num=1, threads=default_threads),
        _get_exec("reference", num=reference_blocks, threads=ref_threads, srun=True),
        (
            "threadpool",
            ThreadPoolExecutor(
                label="threadpool",
                max_threads=1,
                working_dir=str(path_internal),
            ),
        ),
    ]

    pre_commands, executors = list(zip(*com_executors))  # type:ignore
    pre_commands: list[str]
    executors: list[ParslExecutor]

    resources = {
        "default": {"cores": default_threads},
        "training": {"cores": training_cores},
        "reference": {"cores": ref_threads},
        "threadpool": {"cores": 1},
    }

    ref_comm: dict[str, str] = {}
    # resources
    # default on reference
    return (
        executors,
        [
            ["training"],
            ["default"],
            ["reference"],
            ["threadpool"],
        ],
        pre_commands,
        ref_comm,
        resources,
    )
