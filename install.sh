

curl micro.mamba.pm/install.sh | bash
micromamba -n .venv python=3.10
eval "$(micromamba shell hook --shell=bash)"
micromamba activate .venv
pip install tox setuptools setuptools_scm wheel
pip  install -e .
