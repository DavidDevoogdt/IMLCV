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
conda config --add channels conda-forgen
conda config --set channel_priority strict
conda install -y  mamba
conda install -y mamba pip

mamba install -y pip git tensorflow-cpu jax jaxlib cython ndcctools numpy tensorflow numpy pytest

mamba install -y mypy


pip install  -e  git+https://github.com/molmod/yaff.git#egg=yaff
pip install  -e git+https://github.ugent.be/lvduyfhu/ThermoLIB.git#egg=thermolib
pip install  -e ./

pip install install cryptography interface  pyopenssl --upgrade

mamba install -y tensorflow numpy scipy

mamba -y install cp2k

pip install setuptools==59.5.0 #AttributeError: module 'distutils' has no attribute 'version'