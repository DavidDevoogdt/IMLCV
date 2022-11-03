#!/bin/sh

nohup python IMLCV/external/parsl_conf/bootstrap_hpc.py $@  >/dev/null 2>&1 &