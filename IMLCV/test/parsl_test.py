import os

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
from parsl.providers import PBSProProvider, SlurmProvider

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
        # HighThroughputExecutor(
        #     label="hpc_victini",
        #     max_workers=4,
        #     provider=provider_init(cluster="victini", mpi=True),
        # ),
    ],
    monitoring=MonitoringHub(
        hub_address=address_by_hostname(),
        hub_port=55055,
        monitoring_debug=True,
        resource_monitoring_interval=5,
    ),
)

parsl.load(config=config)


@bash_app(executors=["hpc_doduo"])
def generate(outputs=[]):
    import datetime

    now = datetime.datetime.now()
    print(f"generate {now:%Y-%m-%d %H:%M}")

    return "module list | grep 'cluster' \n  echo $(( RANDOM ))   &> {}".format(
        outputs[0]
    )


@bash_app(executors=["hpc_doduo"])
def concat(inputs=[], outputs=[]):
    import datetime

    now = datetime.datetime.now()
    print(f"generate {now:%Y-%m-%d %H:%M}")

    return "module list | grep 'cluster' \n cat {0} > {1}".format(
        " ".join([i.filepath for i in inputs]), outputs[0]
    )


@python_app(executors=["hpc_doduo"])
def total(inputs=[]):
    total = 0
    with open(inputs[0], "r") as f:
        for l in f:
            total += int(l)
    return total


output_files = []
for i in range(5):
    output_files.append(
        generate(
            outputs=[File(os.path.join(os.getcwd(), "random-{}.txt".format(i)))])
    )

cc = concat(
    inputs=[i.outputs[0] for i in output_files],
    outputs=[File(os.path.join(os.getcwd(), "all.txt"))],
)

total = total(inputs=[cc.outputs[0]])
print(total.result())
