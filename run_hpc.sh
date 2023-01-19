#!/bin/sh

# nohup python IMLCV/external/parsl_conf/bootstrap_hpc.py $@  >/dev/null 2>&1 &

#  qsub -I -A "2022_069" -l walltime="48:00:00" -l nodes=1:ppn=5

cd /dodrio/scratch/projects/2022_069/david/IMLCV/
source /dodrio/scratch/projects/2022_069/david/IMLCV/Miniconda3/bin/activate base



python IMLCV/examples/CsPbI3.py
