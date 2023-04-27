# Installing IMLCV


## Local install
Make sure you are in a suitable conda/ pip environment.

```
git clone git@github.com:DavidDevoogdt/IMLCV.git
cd IMLCV
pip install .
```


## HPC install 
IMLCV is build around the [parsl](https://parsl.readthedocs.io/en/stable/index.html) framework. You will need to adapt the code in `configs/config_general` in order to work with a different HPC cluster. More info is available [here](https://parsl.readthedocs.io/en/stable/userguide/configuring.html). 

You'll need to create a conda environment, an example is given in `install.sh`. This environment should be accessible on the different worker nodes.


## docs and tests

coming soon