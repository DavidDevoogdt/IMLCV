#!/bin/sh

# nohup python IMLCV/external/parsl_conf/bootstrap_hpc.py $@  >/dev/null 2>&1 &

cd /dodrio/scratch/projects/2022_069/david/IMLCV/
source /dodrio/scratch/projects/2022_069/david/IMLCV/Miniconda3/bin/activate base
python IMLCV/examples/CsPbI3.py
