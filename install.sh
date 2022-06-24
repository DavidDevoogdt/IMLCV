curl -sL \
  "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" > \
  "Miniconda3.sh"
bash Miniconda3.sh -b -p ./Miniconda3
source Miniconda3/bin/activate
eval "$(conda shell.bash hook)"
# conda update -n base -c defaults conda -y
conda create  python=3.9 -y 
conda install mamba  -c conda-forge

mamba install -y umap-learn
mamba install -y pandas sqlalchemy_utils flask_sqlalchemy plotly networkx
mamba install -y autopep8 pylint

mamba install -y ndcctools #parsl extreme scale executor


pip install -e git+https://github.com/molmod/molmod.git#egg=molmod
pip install --upgrade "jax[cpu]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
pip install -e  git+https://github.com/molmod/yaff.git#egg=yaff
pip install -e git+https://github.ugent.be/lvduyfhu/ThermoLIB.git@63d7d63c7cfa5bdfa712c198e00308cabf01c92f#egg=thermolib
pip install -e git+https://github.com/Parsl/parsl.git#egg=parsl
