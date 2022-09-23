"""summary IMLCV is still underdevelopement."""

import functools
import getpass
import os
import sys
from logging import warning
from pathlib import Path

import jax
import tensorflow as tf
from jax.interpreters import batching

from IMLCV.external.tf2jax import call_tf_p, loop_batcher

# from yaff.log import log

ROOT_DIR = Path(os.path.dirname(__file__)).parent

if "mpi4py" in sys.modules:
    warning("mpi 4 py doens't work wel lwith cp2k calc atm")


name = getpass.getuser()
if name == "vsc43693":
    LOCAL = False
elif name == "david":
    LOCAL = True
else:
    raise ValueError("unknow pc")


# For Linux 64, Open MPI is built with CUDA awareness but this support is disabled by default.
# To enable it, please set the environment variable OMPI_MCA_opal_cuda_support=true before
# launching your MPI processes. Equivalently, you can set the MCA parameter in the command line:
# mpiexec --mca opal_cuda_support 1 ...

# In addition, the UCX support is also built but disabled by default.
# To enable it, first install UCX (conda install -c conda-forge ucx). Then, set the environment
# variables OMPI_MCA_pml="ucx" OMPI_MCA_osc="ucx" before launching your MPI processes.
# Equivalently, you can set the MCA parameters in the command line:
# mpiexec --mca pml ucx --mca osc ucx ...
# Note that you might also need to set UCX_MEMTYPE_CACHE=n for CUDA awareness via UCX.
# Please consult UCX's documentation for detail.


#      mpirun --mca ^pml ucx --mca btl ^uct --mca orte_keep_fqdn_hostnames 1 --map-by ${SLURM_CPUS_ON_NODE}:node:PE=1:SPAN:NOOVERSUBSCRIBE  cp2k_shell.psmp   '


print(f"AVAILABLE CPUS {len(os.sched_getaffinity(0))}")


CP2K_MPI_SLOTS = 16

# mpirun --map-by ppr:1:socket:PE=5 --display-allocation --display-map

# mpirun --map-by ppr:1:node:PE=6 --display-allocation --display-map  echo "True"

# setup HPC stuff
# print diagnostics to the error stream
CP2K_COMMAND = f"mpirun --map-by ppr:1:node:PE={CP2K_MPI_SLOTS} --display-allocation --display-map  true 1>&2; mpirun --map-by ppr:1:node:PE={CP2K_MPI_SLOTS}  cp2k_shell.psmp"  # print diagnostics to stderr. --map-by ppr:1:socket:PE=N:  1 processes per resource ,  CP2K_MPI_SLOTS cpus per process
PY_EMV = f"source {ROOT_DIR}/Miniconda3/bin/activate"
HPC_WORKER_INIT = f"""
export OMP_NUM_THREADS=1
lscpu
{PY_EMV}"""


DEBUG = True
GPU = False


# SETUP Jax
if not GPU:
    jax.config.update("jax_platform_name", "cpu")
    tf.config.experimental.set_visible_devices([], "GPU")
# jax.config.update('jax_disable_jit', True)


batching.primitive_batchers[call_tf_p] = functools.partial(loop_batcher, call_tf_p)

# parsl.set_stream_logger(level=logging.ERROR)
# os.environ["NUMBA_DISABLE_JIT"] = "1"
# tf.data.experimental.enable_debug_mode()
# tf.config.run_functions_eagerly(True)
