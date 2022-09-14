


# ml purge
# ml CP2K
# ml CP2K/8.2-foss-2021a
# ml jax/0.3.9-foss-2021a
# ml TensorFlow/2.6.0-foss-2021a

[ -d "Miniconda3" ] && rm -rf Miniconda3
[ -d "src" ] && rm -rf Miniconda3

[ -d ".pip_cache" ] && rm -rf .pip_cache
mkdir .pip_cache
export PIP_CACHE_DIR=.pip_cache

curl -sL \
  "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" > \
  "Miniconda3.sh"
bash Miniconda3.sh -b -p ./Miniconda3
source Miniconda3/bin/activate
eval "$(conda shell.bash hook)"


conda update -y -n base -c defaults conda
conda install -y python=3.10 
conda install -y mamba
mamba install -y pip git tensorflow-cpu jax jaxlib cython ndcctools numpy cp2k tensorflow numpy pytest



# pip install Cython
pip  install -e  git+https://github.com/molmod/yaff.git#egg=yaff
pip  install -e git+https://github.ugent.be/lvduyfhu/ThermoLIB#egg=thermolib
pip  install -e ./


