#!/bin/bash

set -e

set -Eeuox pipefail

echo "Starting script"


~/.pixi/bin/pixi run -e rocm6 --as-is  "$@"
