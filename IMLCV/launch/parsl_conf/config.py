#####
# This provides a minimal setup configuration for parsl on UGent HPC infrastructure (https://www.ugent.be/hpc/en)
#
#####

import logging
import math
import os
import time
from typing import Optional

import parsl
import typeguard
from IMLCV import ROOT_DIR
from parsl.addresses import address_by_hostname
from parsl.channels import LocalChannel
from parsl.channels.base import Channel
from parsl.config import Config
from parsl.executors import HighThroughputExecutor, ThreadPoolExecutor
from parsl.launchers import AprunLauncher, MpiRunLauncher, SingleNodeLauncher
from parsl.launchers.launchers import Launcher
from parsl.providers.cluster_provider import ClusterProvider
from parsl.providers.provider_base import JobState, JobStatus
from parsl.providers.slurm.slurm import logger, translate_table
from parsl.providers.slurm.template import template_string
from parsl.providers.torque.torque import TorqueProvider
from parsl.utils import RepresentationMixin, wtime_to_minutes

# This provides custom scripts to load the right cluster
parsl.providers.torque.torque.translate_table["F"] = JobState.FAILED


class ClusterChannel(Channel):
    """channel that swaps the cluster module before submitting commands"""

    def __init__(self, channel: Channel, cluster="victini"):
        self.channel = channel
        self.cluster = cluster
        assert cluster in ["victini", "doduo"]

    def execute_wait(self, cmd, walltime=None, envs={}):
        cmd = f"""module swap cluster/{self.cluster}\n{cmd}"""
        return self.channel.execute_wait(cmd, walltime, envs)

    @property
    def script_dir(self):
        return self.channel.script_dir

    @script_dir.setter
    def script_dir(self, value):
        self.channel.script_dir = value

    def push_file(self, source, dest_dir):
        return self.channel.pull_file(source, dest_dir)

    def pull_file(self, remote_source, local_dir):
        return self.channel.pull_file(remote_source, local_dir)

    def close(self):
        return self.channel.close()

    def makedirs(self, path, mode=511, exist_ok=False):
        return self.channel.makedirs(path, mode, exist_ok)

    def isdir(self, path):
        return self.channel.isdir(path)

    def abspath(self, path):
        return self.channel.abspath(path)


class VSCTorqueProvider(TorqueProvider):
    def __init__(
        self,
        channel=LocalChannel(),
        account=None,
        queue=None,
        scheduler_options="",
        worker_init="",
        nodes_per_block=2,
        init_blocks=1,
        min_blocks=0,
        max_blocks=1,
        parallelism=1,
        launcher=AprunLauncher(),
        walltime="00:20:00",
        cmd_timeout=120,
        cluster="doduo",
    ):

        # augment channel
        channel = ClusterChannel(channel, cluster)

        super().__init__(
            channel,
            account,
            queue,
            scheduler_options,
            worker_init,
            nodes_per_block,
            init_blocks,
            min_blocks,
            max_blocks,
            parallelism,
            launcher,
            walltime,
            cmd_timeout,
        )
        # fix template string (-S flag is outdate and whitespace not allowed)
        self.template_string = """#!/bin/bash
#PBS -N ${jobname}
#PBS -m n
#PBS -l walltime=$walltime
#PBS -l nodes=${nodes_per_block}:ppn=${tasks_per_node}
#PBS -o ${submit_script_dir}/${jobname}.submit.stdout
#PBS -e ${submit_script_dir}/${jobname}.submit.stderr
${scheduler_options}

${worker_init}

export JOBNAME="${jobname}"

${user_script}
"""
# PBS -l nodes=${nodes_per_block}:ppn=${tasks_per_node}

        self.cluster = cluster


def config(python_env="source /user/gent/436/vsc43693/scratch_vo/projects/IMLCV/Miniconda3/bin/activate base"):
    exec_dir = os.getcwd()

    print(exec_dir)

    def provider_init(cluster="victini", provider="slurm", mpi=True):
        ssh_chan = LocalChannel(envs={"PYTHONPATH": exec_dir})
        mpi_string = "module load impi" if mpi else ""
        worker_init = f"""
cd {exec_dir}
{mpi_string}
{python_env}
    """
        if provider == "PBS":
            provider = VSCTorqueProvider(
                channel=ssh_chan,
                worker_init=worker_init,
                launcher=MpiRunLauncher() if mpi else SingleNodeLauncher(),
                min_blocks=0,
                max_blocks=2,
                init_blocks=1,
                nodes_per_block=1,
                walltime="00:20:00",
                parallelism=0.5,
                cluster=cluster,
            )

        elif provider == "slurm":

            provider = VSCProviderSlurm(
                cluster=cluster,
                channel=ssh_chan,
                worker_init=worker_init,
                launcher=MpiRunLauncher() if mpi else SingleNodeLauncher(),
                exclusive=False,
                min_blocks=0,
                max_blocks=4,
                init_blocks=1,
                nodes_per_block=3,
                # cores_per_node=18,
                walltime="00:20:00",
                parallelism=0.5,
                # mem_per_node=8,  # inn GB
            )
        else:
            raise ValueError("unknonw provider")

        return provider

    config = Config(
        executors=[
            HighThroughputExecutor(
                label="hpc_doduo",
                max_workers=4,
                address=address_by_hostname(),
                # cores_per_worker=0.1,
                provider=provider_init(cluster="doduo", mpi=False),
                worker_logdir_root=f"{exec_dir}/hpc_log",
                worker_debug=True,
            ),
            HighThroughputExecutor(
                label="hpc_victini",
                max_workers=4,
                address=address_by_hostname(),
                # cores_per_worker=0.1,
                provider=provider_init(cluster="victini", mpi=False),
                worker_logdir_root=f"{exec_dir}/hpc_log",
                worker_debug=True,
            ),
            # ThreadPoolExecutor(
            #     label="local",
            #     max_threads=16,
            # ),
        ],
        retries=0,
        internal_tasks_max_threads=2,
    )

    parsl.load(config=config)


class VSCProviderSlurm(ClusterProvider, RepresentationMixin):
    """Slurm Execution Provider

    This provider uses sbatch to submit, squeue for status and scancel to cancel
    jobs. The sbatch script to be used is created from a template file in this
    same module.

    Parameters
    ----------
    cluster : str
        Slurm cluster to request blocks from. If unspecified or ``None``, no cluster slurm directive will be specified.
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
        #SBATCH blocks in the submit script to the scheduler.
        String to prepend to the
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
        cluster: str,
        account: Optional[str] = None,
        channel: Channel = LocalChannel(),
        nodes_per_block: int = 1,
        cores_per_node: Optional[int] = None,
        mem_per_node: Optional[int] = None,
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
        channel = ClusterChannel(channel, cluster=cluster)

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

        self.cluster = cluster
        self.cores_per_node = cores_per_node
        self.mem_per_node = mem_per_node
        self.exclusive = exclusive
        self.move_files = move_files
        self.account = account
        self.scheduler_options = scheduler_options + "\n"
        if exclusive:
            self.scheduler_options += "#SBATCH --exclusive\n"
        # if cluster:
        #     self.scheduler_options += "#SBATCH --clusters={}\n".format(cluster)
        if account:
            self.scheduler_options += "#SBATCH --account={}\n".format(account)
        self.scheduler_options += "#SBATCH --export=NONE\n"
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

        cmd = "squeue --job {0}".format(job_id_list)
        logger.debug("Executing %s", cmd)
        retcode, stdout, stderr = self.execute_wait(cmd)
        logger.debug("sqeueue returned %s %s", stdout, stderr)

        # Execute_wait failed. Do no update
        if retcode != 0:
            logger.warning(
                "squeue failed with non-zero exit code {}".format(retcode))
            return

        jobs_missing = list(self.resources.keys())
        for line in stdout.split("\n"):
            parts = line.split()
            if parts and parts[0] not in ["JOBID", "CLUSTER:"]:
                job_id = parts[0]
                status = translate_table.get(parts[4], JobState.UNKNOWN)
                logger.debug(
                    "Updating job {} with slurm status {} to parsl status {}".format(
                        job_id, parts[4], status
                    )
                )
                self.resources[job_id]["status"] = JobStatus(status)
                jobs_missing.remove(job_id)

        # squeue does not report on jobs that are not running. So we are filling in the
        # blanks for missing jobs, we might lose some information about why the jobs failed.
        for missing_job in jobs_missing:
            logger.debug(
                "Updating missing job {} to completed status".format(
                    missing_job)
            )
            self.resources[missing_job]["status"] = JobStatus(
                JobState.COMPLETED)

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
            scheduler_options += "#SBATCH --mem={}g\n".format(
                self.mem_per_node)
            worker_init += "export PARSL_MEMORY_GB={}\n".format(
                self.mem_per_node)
        if self.cores_per_node is not None:
            cpus_per_task = math.floor(self.cores_per_node / tasks_per_node)
            scheduler_options += "#SBATCH --cpus-per-task={}".format(
                cpus_per_task)
            worker_init += "export PARSL_CORES={}\n".format(cpus_per_task)

        job_name = "{0}.{1}".format(job_name, time.time())

        script_path = "{0}/{1}.submit".format(self.script_dir, job_name)
        script_path = os.path.abspath(script_path)

        logger.debug("Requesting one block with {} nodes".format(
            self.nodes_per_block))

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
        self._write_submit_script(
            template_string, script_path, job_name, job_config)

        if self.move_files:
            logger.debug("moving files")
            channel_script_path = self.channel.push_file(
                script_path, self.channel.script_dir
            )
        else:
            logger.debug("not moving files")
            channel_script_path = script_path

        retcode, stdout, stderr = self.execute_wait(
            "sbatch {0}".format(channel_script_path)
        )

        job_id = None
        if retcode == 0:
            for line in stdout.split("\n"):
                if line.startswith("Submitted batch job"):
                    job_id = line.split("Submitted batch job")[
                        1].strip().split()[0]
                    self.resources[job_id] = {
                        "job_id": job_id,
                        "status": JobStatus(JobState.PENDING),
                    }
        else:
            print("Submission of command to scale_out failed")
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
        retcode, stdout, stderr = self.execute_wait(
            "scancel {0}".format(job_id_list))
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
