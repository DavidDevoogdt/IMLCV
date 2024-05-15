from parsl.executors import ThreadPoolExecutor, WorkQueueExecutor
from parsl.providers import LocalProvider
from parsl.launchers import SimpleLauncher
from pathlib import Path
import math


def get_config(path_internal, ref_threads=2, max_threads=10, work_queue=True):
    if not work_queue:
        executors = [
            ThreadPoolExecutor(
                label="training",
                max_threads=1,
                working_dir=str(path_internal),
            ),
            ThreadPoolExecutor(
                label="default",
                max_threads=1,
                working_dir=str(path_internal),
            ),
            ThreadPoolExecutor(
                label="reference",
                max_threads=ref_threads,
                working_dir=str(path_internal),
            ),
        ]
    else:
        py_env = """
export MAMBA_EXE=~/projects/IMLCV/bin/micromamba
export MAMBA_ROOT_PREFIX=~/projects/IMLCV/micromamba
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2> /dev/null)"
micromamba activate py311
which python """

        def _get_exec(label, num, threads):
            provider = LocalProvider(
                worker_init=f"""
{py_env}
micromamba activate py311
export XLA_FLAGS='--xla_force_host_platform_device_count={threads}'
""",
                cmd_timeout=1.0,
                launcher=SimpleLauncher(),
                max_blocks=num,
            )

            return WorkQueueExecutor(
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

        executors = [
            _get_exec("training", num=1, threads=max_threads),
            # _get_exec("default", num=math.floor(max_threads / 2), threads=2),
            _get_exec("reference", num=math.floor(max_threads / ref_threads), threads=ref_threads),
        ]
    # default on reference
    return executors, [["reference"], ["training"], ["reference"]]
