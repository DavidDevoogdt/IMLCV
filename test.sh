#!/bin/bash

set -e

pwd

export PIXI_CACHE_DIR="./.pixi_cache"
export PATH="~/.pixi/bin:$PATH"
which pixi
eval "$(pixi shell-hook -e cpu --as-is )"


"$@"

echo "Script finished"

exit 0
