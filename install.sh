[ -d "Miniconda3" ] && rm -rf Miniconda3

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

mamba install -y -c conda-forge tensorflow-gpu 
pip install --upgrade jax==0.3.15 jaxlib==0.3.15+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html



mamba -y  install cython
pip  install -e  git+https://github.com/molmod/yaff.git#egg=yaff
pip  install -e git+https://github.ugent.be/lvduyfhu/ThermoLIB#egg=thermolib
pip  install -e ./

mamba -y install ndcctools