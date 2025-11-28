#!/bin/bash

set -e

pwd

export PIXI_CACHE_DIR="./.pixi_cache"
export PATH="~/.pixi/bin:$PATH"
which pixi
eval "$(pixi shell-hook -e cpu --as-is )"

# Unset all SLURM-related environment variables
while IFS='=' read -r name _; do
    unset "$name"
done < <(env | awk -F= '/^SLURM_/ {print $1}')



python "${1}"

echo "Script finished"

exit 0
