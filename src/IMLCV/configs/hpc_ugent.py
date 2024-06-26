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
from parsl.jobs.states import JobState
from parsl.providers.base import JobStatus
from parsl.providers.slurm.template import template_string
from parsl.utils import wtime_to_minutes
from parsl.executors.taskvine import TaskVineExecutor, TaskVineFactoryConfig
from parsl.executors.threads import ThreadPoolExecutor
from parsl.providers import LocalProvider
from parsl.launchers import SingleNodeLauncher

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
        else:
            cpus_per_task = 1

        #     scheduler_options += f"#SBATCH --cpus-per-task={cpus_per_task}"
        #     worker_init += f"export PARSL_CORES={cpus_per_task}\n"

        job_name = f"{job_name}.{time.time()}"

        script_path = f"{self.script_dir}/{job_name}.submit"
        script_path = os.path.abspath(script_path)

        job_config = {}
        job_config["submit_script_dir"] = self.channel.script_dir
        job_config["nodes"] = self.nodes_per_block
        job_config["tasks_per_node"] = tasks_per_node * cpus_per_task
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
    parsl_tasks_per_block=None,
    threads_per_core: int | None = None,
    use_open_mp=False,
    parsl_cores=False,
    mem=None,
    memory_per_core=None,
    walltime="48:00:00",
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
export MAMBA_EXE=$VSC_HOME/IMLCV_scratch/bin/micromamba
export MAMBA_ROOT_PREFIX=$VSC_HOME/IMLCV_scratch/micromamba
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2> /dev/null)"
micromamba activate py311
which python
            """
        elif env == "stevin":
            py_env = """
export MAMBA_EXE=$VSC_HOME/IMLCV/bin/micromamba
export MAMBA_ROOT_PREFIX=$VSC_HOME/IMLCV/micromamba
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2> /dev/null)"
micromamba activate py311
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

        worker_init += f"export OMP_NUM_THREADS={threads_per_core if use_open_mp else 1   }\n"

        # give all cores to xla
        worker_init += f"export XLA_FLAGS='--xla_force_host_platform_device_count={threads_per_core}'\n"

    else:
        assert threads_per_core is None, "parsl doens't use openmp cores"
        total_cores = parsl_tasks_per_block

        worker_init += f"export XLA_FLAGS='--xla_force_host_platform_device_count={total_cores}'\n"

    worker_init += f"mpirun -report-bindings -np {total_cores} echo 'a' "

    common_kwargs = {
        "channel": channel,
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
            "cluster": cpu_cluster if not gpu else gpu_cluster,
            "partition": cpu_part if not gpu else gpu_part,
            "account": account,
            "exclusive": False,
            "cores_per_node": total_cores,
            "mem_per_node": mem,
            "walltime": walltime,
            "cmd_timeout": 60,
        }

        sheduler_options = "#SBATCH --signal B:USR2"  # send signal to worker if job is cancelled/ time is up/ OOM

        if gpu:
            sheduler_options += (
                "\n#SBATCH --gpus=1\n#SBATCH --cpus-per-gpu={parsl_tasks_per_block}\n#SBATCH --export=None"
            )

        vsc_kwargs["scheduler_options"] = sheduler_options
        provider = SlurmProviderVSC(**common_kwargs, **vsc_kwargs)
    elif provider == "local":
        provider = LocalProvider(
            **common_kwargs,
            cmd_timeout=1,
        )

    # let slurm use the cores as threads
    pre_command = f"srun  -N 1 -n 1 -c {threads_per_core} --cpu-bind=threads --export=ALL"

    ref_comm = {
        "cp2k": "export OMP_NUM_THREADS=1; mpirun -report-bindings  -mca pml ucx -mca btl ^uct,ofi -mca mtl ^ofi cp2k_shell.psmp ",
    }

    print(f"{executor=}")

    if executor == "work_queue":
        worker_options = [
            f"--cores={threads_per_core}",
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
            working_dir=str(Path(path_internal) / label),
            provider=provider,
            shared_fs=True,
            autolabel=False,
            autocategory=False,
            port=0,
            max_retries=1,  # do not retry task
            worker_options=" ".join(worker_options),
            coprocess=False,
            worker_executable="work_queue_worker",
        )

    elif executor == "task_vine":
        executor: ParslExecutor = TaskVineExecutor(
            label=label,
            factory_config=TaskVineFactoryConfig(
                cores=parsl_tasks_per_block,
                gpus=0 if not gpu else 1,
                scratch_dir=str(Path(path_internal) / label),
            ),
            provider=provider,
        )

    elif executor == "htex":
        executor: ParslExecutor = HighThroughputExecutor(
            label=label,
            working_dir=str(Path(path_internal) / label),
            cores_per_worker=threads_per_core,
            provider=provider,
        )
    else:
        raise ValueError(f"unknown executor {executor=}")
    return executor, pre_command, ref_comm


def config(
    env=None,
    singlepoint_nodes=16,
    walltime="48:00:00",
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
):
    def get_kwargs(cpu_cluster=None, gpu_cluster=None):
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
        kw["executor"] = executor

        return kw

    if not isinstance(cpu_cluster, list) and cpu_cluster is not None:
        cpu_cluster = [cpu_cluster]

    if not isinstance(gpu_cluster, list) and gpu_cluster is not None:
        gpu_cluster = [gpu_cluster]

    if bootstrap:
        assert not isinstance(cpu_cluster, list), "bootstrap does not support multiple clusters"

        executor, pre_command, _ = get_slurm_provider(
            **get_kwargs(cpu_cluster),
            label="default",
            init_blocks=1,
            parsl_cores=True,
            mem=10,
            walltime="72:00:00",
        )

        execs = [executor]
        pre_commands = [pre_command]

    else:
        default_labels = []
        training_labels = []
        reference_labels = []

        default_pre_commands = []
        training_pre_commands = []
        reference_pre_commands = []

        reference_command = None

        execs = []

        if default_on_threads:
            label = "default"

            default = ThreadPoolExecutor(
                label=label,
                max_threads=default_threads,
                working_dir=str(Path(path_internal) / label),
            )

            execs.append(default)
            default_labels.append(label)
            default_pre_commands.append("")

        if not isinstance(cpu_cluster, list):
            cpu_cluster = [cpu_cluster]

        if training_on_threads:
            label = "training"

            training = ThreadPoolExecutor(
                label=label,
                max_threads=training_cores,
                working_dir=str(Path(path_internal) / label),
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
                        walltime="04:00:00",
                        **kw,
                    )
                    execs.append(gpu_part)
                    training_pre_commands.append(pre_command)
                    training_labels.append(label)
            else:
                for cpu in cpu_cluster:
                    kw = get_kwargs(cpu_cluster=cpu)

                    label = f"training_{cpu}"

                    cpu_part, pre_command, _ = get_slurm_provider(
                        label=label,
                        init_blocks=0,
                        min_blocks=0,
                        max_blocks=4,
                        parallelism=1,
                        parsl_tasks_per_block=1,
                        threads_per_core=training_cores,
                        parsl_cores=False,
                        walltime="04:00:00",
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
                    max_blocks=80,
                    parallelism=1,
                    parsl_tasks_per_block=1,
                    threads_per_core=singlepoint_nodes,
                    parsl_cores=False,
                    walltime=walltime,
                    load_cp2k=True,
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
                        min_blocks=1,
                        max_blocks=512,
                        parallelism=1,
                        parsl_tasks_per_block=1,
                        threads_per_core=default_threads,
                        parsl_cores=False,
                        walltime="04:00:00",
                        **kw,
                    )

                    execs.append(default)
                    default_labels.append(label)
                    default_pre_commands.append(pre_command)

        pre_commands = [
            default_pre_commands,
            training_pre_commands,
            reference_pre_commands,
        ]

    pre_commands_filtered = []

    for pc in pre_commands:
        pre_commands_filtered.append(pc[0])

        if len(pc) > 1:
            for pci in pc[1:]:
                assert pci == pc[0], "all pre_commands should be the same per category"

    return execs, [default_labels, training_labels, reference_labels], pre_commands_filtered, reference_command
