[ -d "Miniconda3" ] && rm -rf Miniconda3
[ -d ".pip_cache" ] && rm -rf .pip_cache
mkdir .pip_cache
export PIP_CACHE_DIR=.pip_cache


[ -d "src" ] && rm -rf src

curl -sL \
  "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" > \
  "Miniconda3.sh"
bash Miniconda3.sh -b -p ./Miniconda3
source Miniconda3/bin/activate
eval "$(conda shell.bash hook)"


conda create  

conda install -y python=3.10 
conda install -y mamba
mamba install -y pip git

# mamba install -y -c conda-forge tensorflow-gpu 
pip install --upgrade jax[uda11.cudnn82] jaxlib -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install --upgrade tensorflow-cpu
pip install --upgrade tensorrt

pip install pymanopt@git+https://github.com/pymanopt/pymanopt.git

mamba update -y  cython
pip  install -e  git+https://github.com/molmod/yaff.git#egg=yaff
pip  install -e git+https://github.ugent.be/lvduyfhu/ThermoLIB#egg=thermolib
pip  install -e  git+https://github.com/svandenhaute/psiflow.git#egg=psiflow
pip  install -e ./

mamba update  -y  ndcctools




# pip install --upgrade tensorflow-cpu
# pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# pip install --upgrade tensorrt

# conda install -c nvidia cuda-nvcc #install separately