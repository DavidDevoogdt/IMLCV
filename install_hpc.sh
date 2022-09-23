doduo
ml purge
ml gompi/2021a
ml CP2K/8.2-foss-2021a

unset PYTHONPATH


[ -d "Miniconda3" ] && rm -rf Miniconda3
[ -d "src" ] && rm -rf src

[ -d ".pip_cache" ] && rm -rf .pip_cache
mkdir .pip_cache
export PIP_CACHE_DIR=.pip_cache

curl -sL \
  "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" > \
  "Miniconda3.sh"
bash Miniconda3.sh -b -p ./Miniconda3
source Miniconda3/bin/activate
# eval "$(conda shell.bash hook)"



conda update -y -n base -c defaults conda
conda install -y python=3.10 
conda install -y mamba
conda install pip

# python -m pip install --no-cache-dir mpi4py

mamba install -y pip git tensorflow-cpu jax jaxlib cython ndcctools numpy tensorflow numpy pytest


pip install  -e  git+https://github.com/molmod/yaff.git#egg=yaff
pip install  -e git+https://github.ugent.be/lvduyfhu/ThermoLIB#egg=thermolib
pip install  -e ./

mamba install -y mypy


# OMPI_MCA_opal_cuda_support=true




#doduo

# ml purge
# ml CP2K/8.2-foss-2021a
# ml jax/0.3.9-foss-2021a
# ml TensorFlow/2.6.0-foss-2021a
# module unload  SciPy-bundle

# python -m venv .venv
# source .venv/bin/activate 

# ml gompi/2022a
# ml Python/3.10

# mkdir python_lib/lib/python3.10/site-packages/

# export PYTHONPATH="/user/gent/436/vsc43693/scratch_vo/projects/IMLCV/python_lib/lib/python3.10/site-packages/:${PYTHONPATH}"

# pip install --prefix="$(pwd)/python_lib"  --upgrade -I pip setuptools

# pip install Cython
# pip install --prefix="$(pwd)/python_lib"  --upgrade pip
# pip install  --prefix="$(pwd)/python_lib"  cython
# pip install  --prefix="$(pwd)/python_lib" numpy
# pip  install --prefix="$(pwd)/python_lib" -e  git+https://github.com/molmod/yaff.git#egg=yaff
# pip  install --prefix="$(pwd)/python_lib" -e git+https://github.ugent.be/lvduyfhu/ThermoLIB#egg=thermolib
# pip  install --prefix="$(pwd)/python_lib" -e ./
# pip install -e git+https://github.com/google/jax@jaxlib-v0.3.9#egg=jax #corresponding jax vesion
