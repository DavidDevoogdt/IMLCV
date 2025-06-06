import math
from pathlib import Path

from parsl.executors import ThreadPoolExecutor, WorkQueueExecutor
from parsl.launchers import SingleNodeLauncher
from parsl.providers import LocalProvider


def get_config(path_internal, ref_threads=2, default_threads=2, max_threads=10, work_queue=True):
    py_env = """
export MAMBA_EXE=~/projects/IMLCV/bin/micromamba
export MAMBA_ROOT_PREFIX=~/projects/IMLCV/micromamba
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2> /dev/null)"
micromamba activate py312
which python """

    def _get_exec(label, num, threads):
        print(f"providing {num=} executors for {label=} with {threads=}")

        if not work_queue:
            exec = ThreadPoolExecutor(
                label=label,
                max_threads=threads,
                working_dir=str(path_internal),
            )

        else:
            provider = LocalProvider(
                worker_init=f"""
    {py_env}
    micromamba activate py312
    export XLA_FLAGS='--xla_force_host_platform_device_count={threads}'
    """,
                cmd_timeout=1.0,
                launcher=SingleNodeLauncher(),
                max_blocks=num,
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
            )

        return f"export JAX_NUM_CPU_DEVICES={threads}; ", exec

    com_executors = [
        _get_exec("training", num=1, threads=max_threads),
        _get_exec("default", num=1, threads=default_threads),
        # _get_exec("default", num=math.floor(max_threads / 2), threads=2),
        _get_exec("reference", num=math.floor(max_threads / ref_threads), threads=ref_threads),
        [
            "",
            ThreadPoolExecutor(
                label="threadpool",
                max_threads=ref_threads,
                working_dir=str(path_internal),
            ),
        ],
    ]

    pre_commands, executors = list(zip(*com_executors))

    resources = {
        "default": {"cores": default_threads},
        "training": {"cores": max_threads},
        "reference": {"cores": ref_threads},
        "threadpool": {"cores": default_threads},
    }

    # resources
    # default on reference
    return executors, [["training"], ["default"], ["reference"], ["threadpool"]], pre_commands, {}, resources
