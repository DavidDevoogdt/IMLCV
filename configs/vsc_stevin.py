#####
# This provides a minimal setup configuration for parsl on UGent HPC infrastructure (https://www.ugent.be/hpc/en)
#
#####

from typing import Optional

import parsl
import typeguard
from parsl.addresses import address_by_hostname
from parsl.channels import LocalChannel
from parsl.channels.base import Channel
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.launchers import  SimpleLauncher, SingleNodeLauncher
from parsl.launchers.launchers import Launcher
from parsl.providers.cluster_provider import ClusterProvider
from parsl.providers.local.local import LocalProvider
from parsl.providers.provider_base import JobState

from IMLCV import CP2K_THREADS, HPC_WORKER_INIT, LOCAL, PY_ENV, ROOT_DIR


def config(
    cluster="doduo",
    provider="slurm",
    max_blocks=10,
    spawnjob=False,
    time="12:00:00",
    mem_per_node=10,
):

    if parsl.DataFlowKernelLoader._dfk is not None:
        print("parsl already configured, using previous setup")
        return

    if LOCAL:
        choice = 2

        if choice == 0:
            exec = parsl.HighThroughputExecutor(
                working_dir=f"{ROOT_DIR}/.workdir",
                address=address_by_hostname(),
                max_workers=6,
                provider=LocalProvider(
                    launcher=SimpleLauncher(),
                    worker_init=PY_ENV,
                ),
            )
        elif choice == 1:
            exec = parsl.WorkQueueExecutor(
                address=address_by_hostname(),
                provider=LocalProvider(
                    worker_init=PY_ENV,
                    # max_blocks=max_blocks,
                ),
                autolabel=True,
                autocategory=True,
                # worker_options="--memory  5000 --cores 12 ",
                # shared_fs=True,
            )
        elif choice == 2:

            exec = parsl.ThreadPoolExecutor(
                max_threads=min(15, max_blocks),
            )
        else:
            raise NotImplementedError

    else:

        kwargs = {
            "channel": LocalChannel(),
            "cluster": cluster,
            "walltime": time,
            "worker_init": HPC_WORKER_INIT,
        }

        if spawnjob is True:
            kwargs["cores_per_node"] = 1
            kwargs["max_blocks"] = 1
            kwargs["nodes_per_block"] = 1
            plabel = f"bootstrap_{cluster}"
            max_workers = 4
            kwargs["launcher"] = SingleNodeLauncher()
            cores_per_worker = 1

        else:
            kwargs["cores_per_node"] = CP2K_THREADS
            kwargs["launcher"] = SingleNodeLauncher()
            kwargs["max_blocks"] = 100
            kwargs["nodes_per_block"] = 1
            kwargs["mem_per_node"] = mem_per_node
            plabel = f"hpc_{cluster}"
            max_workers = 100
            cores_per_worker = CP2K_THREADS

        if provider == "PBS":
            provider = VSCTorqueProvider(**kwargs)
        elif provider == "slurm":
            provider = VSCProviderSlurm(**kwargs)
        else:
            raise ValueError("unknonw provider")

        exec = HighThroughputExecutor(
            label=plabel,
            provider=provider,
            address=address_by_hostname(),
            mem_per_worker=3,
            max_workers=max_workers,
            cores_per_worker=cores_per_worker,
        )

    config = Config(
        executors=[exec],
        retries=0,
        run_dir=str(ROOT_DIR / "IMLCV" / ".runinfo"),
        max_idletime=60 * 10,
    )
    parsl.load(config=config)


class ClusterChannel(Channel):
    """channel that swaps the cluster module before submitting commands"""

    def __init__(self, channel: Channel, cluster="victini"):
        self.channel = channel
        self.cluster = cluster
        assert cluster in ["victini", "doduo", "slaking"]

    def execute_wait(self, cmd, walltime=None, envs={}):

        cmd = f""" if [ "$(module list | grep -o "cluster/\\w*" |  cut -c 9-)"  != "{self.cluster}" ]; then  module swap cluster/{self.cluster}; fi;\n{cmd}"""
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



# This provides custom scripts to load the right cluster
parsl.providers.torque.torque.translate_table["F"] = JobState.FAILED


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

    template_string = """#!/bin/bash

#SBATCH --job-name=${jobname}
#SBATCH --output=${submit_script_dir}/${jobname}.submit.stdout
#SBATCH --error=${submit_script_dir}/${jobname}.submit.stderr
#SBATCH --nodes=${nodes}
#SBATCH --time=${walltime}
#SBATCH --ntasks-per-node=${5}
${scheduler_options}

${worker_init}

export JOBNAME="${jobname}"

$user_script
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
        exclusive: bool = False,
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
            self.scheduler_options += f"#SBATCH --account={account}\n"
        # self.scheduler_options += "#SBATCH --export=NONE\n"
        self.worker_init = worker_init + "\n"

    
def get_config(folder):
    pass