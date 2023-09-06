import logging
import math
import os
import time
from pathlib import Path

import parsl.providers.slurm.slurm
from parsl import HighThroughputExecutor
from parsl import WorkQueueExecutor
from parsl.channels import LocalChannel
from parsl.executors.base import ParslExecutor
from parsl.providers.base import JobState
from parsl.providers.base import JobStatus
from parsl.providers.slurm.template import template_string
from parsl.utils import wtime_to_minutes

ROOT_DIR = Path(os.path.dirname(__file__)).parent.parent.parent


logger = logging.getLogger(__name__)


translate_table = {
    "PD": JobState.PENDING,
    "R": JobState.RUNNING,
    "CA": JobState.CANCELLED,
    "CF": JobState.PENDING,  # (configuring),
    "CG": JobState.RUNNING,  # (completing),
    "CD": JobState.COMPLETED,
    "F": JobState.FAILED,  # (failed),
    "TO": JobState.TIMEOUT,  # (timeout),
    "NF": JobState.FAILED,  # (node failure),
    "RV": JobState.FAILED,  # (revoked) and
    "SE": JobState.FAILED,  # (special exit state)
}


# taken from psiflow
class SlurmProviderVSC(parsl.providers.slurm.slurm.SlurmProvider):
    """Specifies cluster and partition for sbatch, scancel, and squeue"""

    def __init__(self, cluster=None, **kwargs):
        super().__init__(**kwargs)
        self.cluster = cluster
        self.scheduler_options += "#SBATCH --export=NONE\n"

        # both cluster and partition need to be specified
        assert self.cluster is not None
        assert self.partition is not None

    def submit(self, command, tasks_per_node, job_name="parsl.slurm"):
        """Submit the command as a slurm job.

        This function differs in its parent in the self.execute_wait()
        call, in which the slurm partition is explicitly passed as a command
        line argument as this is necessary for some SLURM-configered systems
        (notably, Belgium's HPC infrastructure).
        In addition, the way in which the job_id is extracted from the returned
        log after submission is slightly modified, again to account for
        the specific cluster configuration of HPCs in Belgium.

        Parameters
        ----------
        command : str
            Command to be made on the remote side.
        tasks_per_node : int
            Command invocations to be launched per node
        job_name : str
            Name for the job
        Returns
        -------
        None or str
            If at capacity, returns None; otherwise, a string identifier for the job
        """

        scheduler_options = self.scheduler_options
        worker_init = self.worker_init
        if self.mem_per_node is not None:
            scheduler_options += f"#SBATCH --mem={self.mem_per_node}g\n"
            worker_init += f"export PARSL_MEMORY_GB={self.mem_per_node}\n"
        if self.cores_per_node is not None:
            cpus_per_task = math.floor(self.cores_per_node / tasks_per_node)
            scheduler_options += f"#SBATCH --cpus-per-task={cpus_per_task}"
            worker_init += f"export PARSL_CORES={cpus_per_task}\n"

        job_name = f"{job_name}.{time.time()}"

        script_path = f"{self.script_dir}/{job_name}.submit"
        script_path = os.path.abspath(script_path)

        job_config = {}
        job_config["submit_script_dir"] = self.channel.script_dir
        job_config["nodes"] = self.nodes_per_block
        job_config["tasks_per_node"] = tasks_per_node
        job_config["walltime"] = wtime_to_minutes(self.walltime)
        job_config["scheduler_options"] = scheduler_options
        job_config["worker_init"] = worker_init
        job_config["user_script"] = command

        # Wrap the command
        job_config["user_script"] = self.launcher(
            command,
            tasks_per_node,
            self.nodes_per_block,
        )

        self._write_submit_script(template_string, script_path, job_name, job_config)

        if self.move_files:
            channel_script_path = self.channel.push_file(
                script_path,
                self.channel.script_dir,
            )
        else:
            channel_script_path = script_path

        submit_cmd = "sbatch --clusters={2} --partition={1} {0}".format(
            channel_script_path,
            self.partition,
            self.cluster,
        )
        retcode, stdout, stderr = self.execute_wait(submit_cmd)

        job_id = None
        if retcode == 0:
            for line in stdout.split("\n"):
                if line.startswith("Submitted batch job"):
                    # job_id = line.split("Submitted batch job")[1].strip()
                    job_id = line.split("Submitted batch job")[1].strip().split()[0]
                    self.resources[job_id] = {
                        "job_id": job_id,
                        "status": JobStatus(JobState.PENDING),
                    }
        else:
            logger.error("Submit command failed")
            logger.error(
                "Retcode:%s STDOUT:%s STDERR:%s",
                retcode,
                stdout.strip(),
                stderr.strip(),
            )
        return job_id

    def _status(self):
        """Returns the status list for a list of job_ids
        Args:
              self
        Returns:
              [status...] : Status list of all jobs
        """
        job_id_list = ",".join(
            [jid for jid, job in self.resources.items() if not job["status"].terminal],
        )
        if not job_id_list:
            logger.debug("No active jobs, skipping status update")
            return

        cmd = "squeue --clusters={1} --noheader --format='%i %t' --job '{0}'".format(
            job_id_list,
            self.cluster,
        )
        logger.debug("Executing %s", cmd)
        retcode, stdout, stderr = self.execute_wait(cmd)
        logger.debug("squeue returned %s %s", stdout, stderr)

        # Execute_wait failed. Do no update
        if retcode != 0:
            logger.warning(f"squeue failed with non-zero exit code {retcode}:")
            logger.warning(stdout)
            logger.warning(stderr)
            return

        jobs_missing = set(self.resources.keys())
        for line in stdout.split("\n"):
            if not line:
                # Blank line
                continue
            job_id, slurm_state = line.split()
            if slurm_state not in translate_table:
                logger.warning(f"Slurm status {slurm_state} is not recognized")
            status = translate_table.get(slurm_state, JobState.UNKNOWN)
            logger.debug(
                "Updating job {} with slurm status {} to parsl state {!s}".format(
                    job_id,
                    slurm_state,
                    status,
                ),
            )
            self.resources[job_id]["status"] = JobStatus(status)
            jobs_missing.remove(job_id)

        # squeue does not report on jobs that are not running. So we are filling in the
        # blanks for missing jobs, we might lose some information about why the jobs failed.
        for missing_job in jobs_missing:
            logger.debug(f"Updating missing job {missing_job} to completed status")
            self.resources[missing_job]["status"] = JobStatus(JobState.COMPLETED)

    def cancel(self, job_ids):
        """Cancels the jobs specified by a list of job ids
        Args:
        job_ids : [<job_id> ...]
        Returns :
        [True/False...] : If the cancel operation fails the entire list will be False.
        """

        job_id_list = " ".join(job_ids)
        retcode, stdout, stderr = self.execute_wait(
            f"scancel --clusters={self.cluster} {job_id_list}",
        )
        rets = None
        if retcode == 0:
            for jid in job_ids:
                self.resources[jid]["status"] = JobStatus(
                    JobState.CANCELLED,
                )  # Setting state to cancelled
            rets = [True for i in job_ids]
        else:
            rets = [False for i in job_ids]

        return rets


def get_slurm_provider(
    env,
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
    py_env=None,
):
    if py_env is None:
        if env == "hortense":
            print("setting python env for hortense")
            py_env = f"source {ROOT_DIR}/micromamba/bin/activate; which python"
        else:
            py_env = f"source {ROOT_DIR}/Miniconda3/bin/activate; which python"

    if gpu_cluster is None:
        gpu_cluster = cpu_cluster

    assert open_mp_threads_per_core is None, "open_mp_threads_per_core is not tested yet"

    worker_init = f"{py_env}; \n"
    if env == "hortense":
        worker_init += "module load CP2K/8.2-foss-2021a \n"

    elif env == "stevin":
        worker_init += "module load CP2K/7.1-foss-2020a \n"
    worker_init += "module unload SciPy-bundle Python \n"
    # worker_init += "module load texlive \n"

    if not parsl_cores:
        if open_mp_threads_per_core is None:
            open_mp_threads_per_core = 1

        total_cores = cores * open_mp_threads_per_core

        worker_init += "unset SLURM_CPUS_PER_TASK\n"
        worker_init += f"export SLURM_CPUS_PER_TASK={open_mp_threads_per_core}\n"
        worker_init += f"export SLURM_NTASKS_PER_NODE={cores}\n"
        worker_init += f"export SLURM_TASKS_PER_NODE={cores}\n"
        worker_init += f"export SLURM_NTASKS={cores}\n"
        worker_init += f"export OMP_NUM_THREADS={open_mp_threads_per_core}\n"

        worker_init += f"export XLA_FLAGS='--xla_force_host_platform_device_count={open_mp_threads_per_core}'\n"

    else:
        assert open_mp_threads_per_core is None, "parsl doens't use openmp cores"
        total_cores = cores

        worker_init += f"export XLA_FLAGS='--xla_force_host_platform_device_count={total_cores}'\n"

    if memory_per_core is not None:
        if mem is None:
            mem = total_cores * memory_per_core
        else:
            if mem < total_cores * memory_per_core:
                mem = total_cores * memory_per_core

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
        # "label": label,
        "nodes_per_block": 1,
    }

    if gpu:
        vsc_kwargs[
            "scheduler_options"
        ] = f"#SBATCH --gpus=1\n#SBATCH --cpus-per-gpu={cores}\n#SBATCH --export=None"  # request gpu

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
            # env={"OMP_NUM_THREADS":f"open_mp_threads_per_core",},
            working_dir=str(Path(path_internal) / label),
            provider=provider,
            shared_fs=True,
            autocategory=False,
            port=0,
            max_retries=1,  # do not retry task
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
    cpu_cluster=None,
    gpu_cluster=None,
    py_env=None,
    account=None,
    use_work_queue=False,
):
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
    kw["use_work_queue"] = use_work_queue

    if bootstrap:
        execs = [
            get_slurm_provider(
                **kw,
                label="default",
                init_blocks=1,
                min_blocks=1,
                max_blocks=1,
                parallelism=0,
                cores=4,
                parsl_cores=True,
                mem=10,
                walltime="72:00:00",
            ),
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
