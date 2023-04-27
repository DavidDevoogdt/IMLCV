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
mamba install ndcctools

pip  install -e ./




