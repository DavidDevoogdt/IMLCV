

curl micro.mamba.pm/install.sh | bash
micromamba create -n .venv
eval "$(micromamba shell hook --shell=bash)"
micromamba activate .venv
micromamba  install -y  python=3.10 tox
