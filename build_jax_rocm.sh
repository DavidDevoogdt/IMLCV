#!/bin/bash
set -e

# Configuration
WORK_DIR="/pfs/lustrep3/scratch/project_465002430/david"
CONTAINER_NAME="rocm-dev-6.3.4.sif"
JAX_DIR="jax-build"

echo "=== JAX Build Setup for ROCm 6.3.4 on LUMI ==="

export PIXI_CACHE_DIR="/users/devoogdt/project_scratch_2/.pixi_cache"

mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

echo "Working directory: $WORK_DIR"

# Step 1: Ensure container exists
if [ ! -f "$CONTAINER_NAME" ]; then
    echo "Pulling ROCm dev container..."
    singularity pull "$CONTAINER_NAME" docker://rocm/dev-ubuntu-24.04:6.3.4-complete
fi

# Step 2: Clone JAX if needed
if [ ! -d "$JAX_DIR" ]; then
    echo "Cloning JAX from ROCm fork..."
    git clone https://github.com/ROCm/jax.git "$JAX_DIR"
    cd "$JAX_DIR"
    git checkout v0.4.35
    cd "$WORK_DIR"
else
    echo "JAX directory already present"
fi

# Step 3: Create SLURM job script
SUBMIT_SCRIPT="$WORK_DIR/submit_build.sh"
cat > "$SUBMIT_SCRIPT" <<'SBATCH_EOF'
#!/bin/bash
#SBATCH --partition=dev-g
#SBATCH -N 1 -n 1 -c 8 --gpus 1
#SBATCH --mem=32GB --time=3:00:00
#SBATCH --account=project_465002430

set -e

WORK_DIR="/pfs/lustrep3/scratch/project_465002430/david"
JAX_DIR="jax-build"
CONTAINER_PATH="$WORK_DIR/rocm-dev-6.3.4.sif"

echo "=== Starting JAX build ==="
echo "JAX source: $WORK_DIR/$JAX_DIR"
echo "Container: $CONTAINER_PATH"
echo "Time limit: 3 hours"
echo ""

# Clean cache
export PIXI_CACHE_DIR="/users/devoogdt/project_scratch_2/.pixi_cache"
rm -rf "$PIXI_CACHE_DIR"
mkdir -p "$PIXI_CACHE_DIR"

# Change to IMLCV so pixi finds pyproject.toml
cd "$WORK_DIR/IMLCV"

echo "=== Running build with pixi CPU env ==="

cd "$WORK_DIR/IMLCV"

# Get the Python path from pixi environment
PIXI_PYTHON=$(pixi run -e cpu which python)
echo "Pixi Python: $PIXI_PYTHON"

# Run singularity with pixi's Python explicitly
singularity exec \
    --bind "$WORK_DIR:/workspace" \
    "$CONTAINER_PATH" \
    bash -c '
export ROCM_PATH=/opt/rocm
export HIP_PATH=$ROCM_PATH
export HIP_PLATFORM=amd
export HSA_PATH=$ROCM_PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH
export PATH=$ROCM_PATH/bin:$ROCM_PATH/llvm/bin:$PATH

cd /workspace/jax-build

echo "=== Build Environment ==="
echo "Working dir: $(pwd)"
echo "ROCM_PATH: $ROCM_PATH"
echo "HIP_PATH: $HIP_PATH"
echo ""

# Clean any previous build artifacts and kill stale Bazel processes
rm -rf .bazel .bazelrc .jax_configure.bazelrc bazel-* 2>/dev/null || true
pkill -9 bazel || true
sleep 2

echo "=== Starting JAX build (this takes 1-2 hours) ==="
'"$PIXI_PYTHON"' build/build.py build \
    --bazel_options=--nodaemon \
    --wheels=jaxlib,jax-rocm-plugin,jax-rocm-pjrt \
    --rocm_version=60 \
    --rocm_path=$ROCM_PATH

echo ""
echo "=== Build Complete ==="
ls -lh dist/
'

echo ""
echo "Build finished!"
echo ""
echo "Wheels location:"
ls -1 "$WORK_DIR/$JAX_DIR/dist/" 2>/dev/null | head -5 || echo "  (check after job completes)"
SBATCH_EOF

chmod +x "$SUBMIT_SCRIPT"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next: submit the build job"
echo "  sbatch $SUBMIT_SCRIPT"
echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  tail -f slurm-*.out"