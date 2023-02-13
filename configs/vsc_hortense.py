import logging
import math
import os
import time

import typeguard
from parsl.channels import LocalChannel
from parsl.channels.base import Channel
from parsl.launchers import SingleNodeLauncher
from parsl.launchers.launchers import Launcher
from parsl.providers.cluster_provider import ClusterProvider
from parsl.providers.provider_base import JobState, JobStatus
from parsl.providers.slurm.template import template_string
from parsl.utils import RepresentationMixin, wtime_to_minutes

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


class SlurmProvider(ClusterProvider, RepresentationMixin):
    """Slurm Execution Provider

    This provider uses sbatch to submit, squeue for status and scancel to cancel
    jobs. The sbatch script to be used is created from a template file in this
    same module.

    Parameters
    ----------
    partition : str
        Slurm partition to request blocks from. If unspecified or ``None``, no partition slurm directive will be specified.
    account : str
        Slurm account to which to charge resources used by the job. If unspecified or ``None``, the job will use the
        user's default account.
    channel : Channel
        Channel for accessing this provider. Possible channels include
        :class:`~parsl.channels.LocalChannel` (the default),
        :class:`~parsl.channels.SSHChannel`, or
        :class:`~parsl.channels.SSHInteractiveLoginChannel`.
    nodes_per_block : int
        Nodes to provision per block.
    cores_per_node : int
        Specify the number of cores to provision per node. If set to None, executors
        will assume all cores on the node are available for computation. Default is None.
    mem_per_node : int
        Specify the real memory to provision per node in GB. If set to None, no
        explicit request to the scheduler will be made. Default is None.
    min_blocks : int
        Minimum number of blocks to maintain.
    max_blocks : int
        Maximum number of blocks to maintain.
    parallelism : float
        Ratio of provisioned task slots to active tasks. A parallelism value of 1 represents aggressive
        scaling where as many resources as possible are used; parallelism close to 0 represents
        the opposite situation in which as few resources as possible (i.e., min_blocks) are used.
    walltime : str
        Walltime requested per block in HH:MM:SS.
    scheduler_options : str
        String to prepend to the #SBATCH blocks in the submit script to the scheduler.
    worker_init : str
        Command to be run before starting a worker, such as 'module load Anaconda; source activate env'.
    exclusive : bool (Default = True)
        Requests nodes which are not shared with other running jobs.
    launcher : Launcher
        Launcher for this provider. Possible launchers include
        :class:`~parsl.launchers.SingleNodeLauncher` (the default),
        :class:`~parsl.launchers.SrunLauncher`, or
        :class:`~parsl.launchers.AprunLauncher`
    move_files : Optional[Bool]: should files be moved? by default, Parsl will try to move files.
    """

    @typeguard.typechecked
    def __init__(
        self,
        partition: str | None = None,
        account: str | None = None,
        channel: Channel = LocalChannel(),
        nodes_per_block: int = 1,
        cores_per_node: int | None = None,
        mem_per_node: int | None = None,
        init_blocks: int = 1,
        min_blocks: int = 0,
        max_blocks: int = 1,
        parallelism: float = 1,
        walltime: str = "00:10:00",
        scheduler_options: str = "",
        worker_init: str = "",
        cmd_timeout: int = 10,
        exclusive: bool = True,
        move_files: bool = True,
        launcher: Launcher = SingleNodeLauncher(),
    ):
        label = "slurm"
        super().__init__(
            label,
            channel,
            nodes_per_block,
            init_blocks,
            min_blocks,
            max_blocks,
            parallelism,
            walltime,
            cmd_timeout=cmd_timeout,
            launcher=launcher,
        )

        self.partition = partition
        self.cores_per_node = cores_per_node
        self.mem_per_node = mem_per_node
        self.exclusive = exclusive
        self.move_files = move_files
        self.account = account
        self.scheduler_options = scheduler_options + "\n"
        if exclusive:
            self.scheduler_options += "#SBATCH --exclusive\n"
        if partition:
            self.scheduler_options += f"#SBATCH --partition={partition}\n"
        if account:
            self.scheduler_options += f"#SBATCH --account={account}\n"
        self.worker_init = worker_init + "\n"

    def _status(self):
        """Internal: Do not call. Returns the status list for a list of job_ids

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

        cmd = f"squeue --noheader --format='%i %t' --job '{job_id_list}'"
        logger.debug("Executing %s", cmd)
        retcode, stdout, stderr = self.execute_wait(cmd)
        logger.debug("squeue returned %s %s", stdout, stderr)

        # Execute_wait failed. Do no update
        if retcode != 0:
            logger.warning(f"squeue failed with non-zero exit code {retcode}")
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

    def submit(self, command, tasks_per_node, job_name="parsl.slurm"):
        """Submit the command as a slurm job.

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

        logger.debug(f"Requesting one block with {self.nodes_per_block} nodes")

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

        logger.debug("Writing submit script")
        self._write_submit_script(template_string, script_path, job_name, job_config)

        if self.move_files:
            logger.debug("moving files")
            channel_script_path = self.channel.push_file(
                script_path, self.channel.script_dir
            )
        else:
            logger.debug("not moving files")
            channel_script_path = script_path

        retcode, stdout, stderr = self.execute_wait(
            "sbatch {1} {0}".format(
                channel_script_path, f"--partition={self.partition}"
            )
        )

        job_id = None
        if retcode == 0:
            for line in stdout.split("\n"):
                if line.startswith("Submitted batch job"):
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

    def cancel(self, job_ids):
        """Cancels the jobs specified by a list of job ids

        Args:
        job_ids : [<job_id> ...]

        Returns :
        [True/False...] : If the cancel operation fails the entire list will be False.
        """

        job_id_list = " ".join(job_ids)
        retcode, stdout, stderr = self.execute_wait(f"scancel {job_id_list}")
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

    @property
    def status_polling_interval(self):
        return 60


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
        provider = SlurmProvider(
            partition="cpu_rome",
            account=account,
            channel=channel,
            nodes_per_block=1,
            cores_per_node=4,
            init_blocks=1,
            min_blocks=1,
            max_blocks=1,
            parallelism=0,
            walltime=walltime,
            worker_init=worker_init,
            exclusive=False,
            mem_per_node=5,
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
    provider = SlurmProvider(
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
    provider = SlurmProvider(
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
    provider = SlurmProvider(
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

    provider = SlurmProvider(
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
