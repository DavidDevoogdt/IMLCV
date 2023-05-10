

curl micro.mamba.pm/install.sh | bash
micromamba -n .venv  install
eval "$(micromamba shell hook --shell=bash)"
micromamba activate .venv
micromamba  install -y  python=3.10
pip install tox setuptools setuptools_scm wheel Cython Numpy


pip install .
