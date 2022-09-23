#!/bin/bash

#PBS -l nodes=1:ppn=4

# ml gompi/2021a
cd $PBS_O_WORKDIR
source Miniconda3/bin/activate

pwd

python IMLCV/test/common.py