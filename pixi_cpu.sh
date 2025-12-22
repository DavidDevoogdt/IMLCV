#!/bin/bash

# set -e

# pwd

set -Eeuox pipefail

# srun

# echo "Setting up pixi environment"
# export PIXI_CACHE_DIR="./.pixi_cache"
# export PATH="~/.pixi/bin:$PATH"
# which pixi

# echo "Loading pixi shell-hook for cpu environment"
# echo  "$(pixi shell-hook -e cpu --as-is )"

# printf "Evaluating pixi shell-hook for cpu environment\n"
# eval "$(pixi shell-hook -e cpu --as-is )"

echo "Starting script"


~/.pixi/bin/pixi run -e cpu --as-is "$@"


# echo "Script finished"

# exit 0
