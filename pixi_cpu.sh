#!/bin/bash


set -Eeuox pipefail

echo "Starting script"


~/.pixi/bin/pixi run -e cpu --as-is "$@"
