"""summary IMLCV is still underdevelopement."""
import os

import jax
import parsl
from parsl import bash_app, python_app
from parsl.addresses import (address_by_hostname, address_by_interface,
                             address_by_query, address_by_route)
from parsl.channels import LocalChannel, SSHChannel, SSHInteractiveLoginChannel
from parsl.config import Config
from parsl.data_provider import rsync
from parsl.data_provider.data_manager import default_staging
from parsl.data_provider.files import File
from parsl.data_provider.ftp import FTPInTaskStaging
from parsl.data_provider.http import HTTPInTaskStaging
from parsl.executors import HighThroughputExecutor, ThreadPoolExecutor
from parsl.launchers import MpiRunLauncher, SimpleLauncher, SrunLauncher
from parsl.monitoring import MonitoringHub
from parsl.providers import LocalProvider, PBSProProvider, SlurmProvider
from yaff.log import log

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# SETUP Jax

jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)
# jax.config.update('jax_disable_jit', True)


# SETUP Parsl


log.set_level(log.silent)


ssh_chan = LocalChannel(script_dir=".parsl", userhome=".parsl")


def provider_init(cluster="victini", mpi=True):

    mpi_string = "module load impi" if mpi else ""

    worker_init = f"""
module swap cluster/{cluster}
{mpi_string}
source ../Miniconda3/bin/activate
conda activate parsl_env 
"""
    # provider = PBSProProvider(
    #     channel=ssh_chan,
    #     worker_init=worker_init,
    #     launcher=MpiRunLauncher() if mpi else SimpleLauncher(),
    # )

    provider = SlurmProvider(
        channel=ssh_chan,
        worker_init=worker_init,
        launcher=MpiRunLauncher() if mpi else SimpleLauncher(),
        exclusive=False,
    )

    return provider


config = Config(
    executors=[
        HighThroughputExecutor(
            label="hpc_doduo",
            max_workers=4,
            provider=provider_init(cluster="doduo", mpi=False),
        ),
        HighThroughputExecutor(
            label="hpc_victini",
            max_workers=4,
            provider=provider_init(cluster="victini", mpi=True),
        ),
        # HighThroughputExecutor(
        #     provider=LocalProvider(),


        # )
    ],
    monitoring=MonitoringHub(
        hub_address=address_by_hostname(),
        hub_port=55055,
        monitoring_debug=True,
        resource_monitoring_interval=5,
    ),
)

parsl.load(config=config)
