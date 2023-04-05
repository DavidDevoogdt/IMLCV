import logging
import math
import os
import time

import parsl.providers.slurm.slurm
from parsl.channels import LocalChannel
from parsl.providers.provider_base import JobState, JobStatus
from parsl.providers.slurm.template import template_string
from parsl.utils import wtime_to_minutes

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
            command, tasks_per_node, self.nodes_per_block
        )

        self._write_submit_script(template_string, script_path, job_name, job_config)

        if self.move_files:
            channel_script_path = self.channel.push_file(
                script_path, self.channel.script_dir
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
            [jid for jid, job in self.resources.items() if not job["status"].terminal]
        )
        if not job_id_list:
            logger.debug("No active jobs, skipping status update")
            return

        cmd = "squeue --clusters={1} --noheader --format='%i %t' --job '{0}'".format(
            job_id_list, self.cluster
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
                    job_id, slurm_state, status
                )
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
            f"scancel --clusters={self.cluster} {job_id_list}"
        )
        rets = None
        if retcode == 0:
            for jid in job_ids:
                self.resources[jid]["status"] = JobStatus(
                    JobState.CANCELLED
                )  # Setting state to cancelled
            rets = [True for i in job_ids]
        else:
            rets = [False for i in job_ids]

        return rets


cluster = "dodrio"  # all partitions reside on a single cluster


def get_config(
    path_internal,
    py_env,
    account="2022_069",
    channel=LocalChannel(),
    singlepoint_cores=16,
    walltime="48:00:00",
    bootstrap=False,
    memory_per_core=None,
    min_memory=None,
):

    from parsl.config import Config
    from parsl.executors import HighThroughputExecutor

    if bootstrap:
        worker_init = f"{py_env};\n"
        provider = SlurmProviderVSC(
            cluster=cluster,
            partition="cpu_rome",
            account=account,
            channel=channel,
            nodes_per_block=1,
            cores_per_node=8,
            init_blocks=1,
            min_blocks=1,
            max_blocks=1,
            parallelism=0,
            walltime=walltime,
            worker_init=worker_init,
            exclusive=False,
            mem_per_node=10,
            cmd_timeout=60,
        )
        bootstrap = HighThroughputExecutor(
            label="default",
            provider=provider,
            cores_per_worker=4,
            # address=os.environ["HOSTNAME"],
            working_dir=str(path_internal / "bootstrap"),
        )
        return Config(
            executors=[bootstrap],
            usage_tracking=True,
            run_dir=str(path_internal),
        )

    worker_init = f"{py_env};\n"
    provider = SlurmProviderVSC(
        cluster=cluster,
        partition="cpu_rome",
        account=account,
        channel=channel,
        nodes_per_block=1,
        cores_per_node=4,
        init_blocks=1,
        min_blocks=1,
        max_blocks=512,
        parallelism=1,
        walltime="02:00:00",
        worker_init=worker_init,
        exclusive=False,
        cmd_timeout=60,
    )
    default = HighThroughputExecutor(
        label="default",
        provider=provider,
        # address=os.environ["HOSTNAME"],
        working_dir=str(path_internal / "default_executor"),
        cores_per_worker=1,
    )

    cores_per_model = 4
    worker_init = f"{py_env}; \n"
    worker_init += f"set OMP_NUM_THREADS={cores_per_model}\n"
    provider = SlurmProviderVSC(
        cluster=cluster,
        partition="cpu_rome",
        account=account,
        channel=channel,
        nodes_per_block=1,
        cores_per_node=4,
        init_blocks=0,
        min_blocks=0,
        max_blocks=512,
        parallelism=1,
        walltime=walltime,
        worker_init=worker_init,
        exclusive=False,
        cmd_timeout=60,
        mem_per_node=10,
    )
    model = HighThroughputExecutor(
        label="model",
        provider=provider,
        # address=os.environ["HOSTNAME"],
        working_dir=str(path_internal / "model_executor"),
        cores_per_worker=cores_per_model,
    )
    cores_per_gpu = 12
    worker_init = f"{py_env}; \n"
    worker_init += "unset SLURM_CPUS_PER_TASK\n"
    worker_init += f"export SLURM_NTASKS_PER_NODE={cores_per_gpu}\n"
    worker_init += f"export SLURM_TASKS_PER_NODE={cores_per_gpu}\n"
    worker_init += f"export SLURM_NTASKS={cores_per_gpu}\n"
    worker_init += f"export SLURM_NPROCS={cores_per_gpu}\n"
    worker_init += f"export OMP_NUM_THREADS={cores_per_gpu}\n"
    provider = SlurmProviderVSC(
        cluster=cluster,
        partition="gpu_rome_a100",
        account=account,
        channel=channel,
        nodes_per_block=1,
        cores_per_node=cores_per_gpu,  # must be equal to cpus-per-gpu
        init_blocks=0,
        min_blocks=0,
        max_blocks=4,
        parallelism=1.0,
        walltime="01:05:00",  # slightly larger than walltime in train app
        worker_init=worker_init,
        exclusive=False,
        scheduler_options="#SBATCH --gpus=1\n#SBATCH --cpus-per-gpu=12\n#SBATCH --export=None",  # request gpu
        cmd_timeout=60,
    )
    training = HighThroughputExecutor(
        label="training",
        provider=provider,
        # address=os.environ["HOSTNAME"],
        working_dir="gpu_working_dir",
        cores_per_worker=cores_per_gpu,
    )
    # to get MPI to recognize the available slots correctly, it's necessary
    # to override the slurm variables as set by the jobscript, as these are
    # based on the number of parsl tasks, NOT on the number of MPI tasks for
    # cp2k. Essentially, this means we have to reproduce the environment as
    # if we launched a job using 'qsub -l nodes=1:ppn=cores_per_singlepoint'
    # singlepoint_nodes = 16
    open_mp_threads_per_singlepoint = 1
    total_cores = singlepoint_cores * open_mp_threads_per_singlepoint

    worker_init = f"{py_env}; \n"
    worker_init += f"export SLURM_CPUS_PER_TASK={open_mp_threads_per_singlepoint}\n"
    worker_init += f"export SLURM_NTASKS_PER_NODE={singlepoint_cores}\n"
    worker_init += f"export SLURM_TASKS_PER_NODE={singlepoint_cores}\n"
    worker_init += f"export SLURM_NTASKS={singlepoint_cores}\n"
    worker_init += f"export OMP_NUM_THREADS={open_mp_threads_per_singlepoint}\n"

    mem = min_memory
    if memory_per_core is not None:

        if mem is None:

            mem = total_cores * memory_per_core
        else:
            if mem < total_cores * memory_per_core:
                mem = total_cores * memory_per_core

    provider = SlurmProviderVSC(
        cluster=cluster,
        partition="cpu_rome",
        account=account,
        channel=channel,
        nodes_per_block=1,
        cores_per_node=total_cores,
        init_blocks=0,
        min_blocks=0,
        max_blocks=256,
        parallelism=1,
        walltime=walltime,
        worker_init=worker_init,
        exclusive=False,
        cmd_timeout=60,
        mem_per_node=mem,
    )
    reference = HighThroughputExecutor(
        label="reference",
        provider=provider,
        # address=os.environ["HOSTNAME"],
        working_dir=str(path_internal / "reference_executor"),
        cores_per_worker=total_cores,
        cpu_affinity="alternating",
    )
    return Config(
        executors=[default, model, reference, training],
        usage_tracking=True,
        run_dir=str(path_internal),
    )
