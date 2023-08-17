

# curl micro.mamba.pm/install.sh | bash
# micromamba create -n .venv
# eval "$(micromamba shell hook --shell=bash)"
# micromamba activate .venv
# micromamba  install -y  python=3.10 tox



curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba

mkdir micromamaba


mkdir .pip_cache

export PIP_CACHE_DIR="$(pwd)/.pip_cache"

# Linux/bash:
./bin/micromamba shell init -s bash -p ./micromamba  #
source ~/.bashrc
micromamba activate
micromamba install python=3.10 jupyter -c conda-forge
micromamba install -c conda-forge ndcctools

#cython 3.0.0 has breaking changes
pip install cython==0.29 numpy

pip install -e .