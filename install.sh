curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba

mkdir micromamba
mkdir .pip_cache
export PIP_CACHE_DIR="$(pwd)/.pip_cache"

# Linux/bash:
./bin/micromamba shell init -s bash -p ./micromamba  #
source ~/.bashrc

micromamba create  -n py311 -c conda-forge python=3.11
micromamba activate py311
micromamba install -c conda-forge ndcctools  texlive-core

#cython 3.0.0 has breaking changes
pip install cython==0.29.36 numpy


#requires password
pip install thermolib@git+https://github.ugent.be/lvduyfhu/ThermoLIB@david
# pip install yaff@git+https://github.com/molmod/yaff.git molmod


pip install -e .
