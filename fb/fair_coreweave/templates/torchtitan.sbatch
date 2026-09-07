#!/usr/bin/env bash
#SBATCH --job-name=@JOB_NAME@
#SBATCH --account=@ACCOUNT@
#SBATCH --qos=@QOS@
#SBATCH --partition=@PARTITION@
#SBATCH --nodes=@NODES@
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=@GPUS_PER_NODE@
#SBATCH --cpus-per-task=@CPUS_PER_TASK@
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=@TIME@
#SBATCH --segment=@SEGMENT@
#SBATCH --output=@LOG_DIR@/%x-%j.out
#SBATCH --error=@LOG_DIR@/%x-%j.err

set -euo pipefail

workdir="@WORKDIR@"
venv_python="@VENV@/bin/python"

export PYTHONPATH="@PYTHONPATH@"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export PYTORCH_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH="@VENV@/lib:${LD_LIBRARY_PATH:-}"

cuda_compat="@VENV@/lib/cuda-compat-13-1"
if [ -d "$cuda_compat" ]; then
  export LD_PRELOAD="$cuda_compat/libcuda.so.1:$cuda_compat/libnvidia-ptxjitcompiler.so.1"
  export TRITON_LIBCUDA_PATH="$cuda_compat"
fi

# Ranks must not write .pyc back to the shared home. Concurrent writers there
# hand out stale NFS file handles; it starts biting around 16 ranks.
export PYTHONDONTWRITEBYTECODE=1

# Same reason, for every cache a rank touches. XDG_CACHE_HOME does not cover
# Triton or the CUDA JIT cache, so both are set explicitly.
job_tmp="/tmp/cw-$SLURM_JOB_ID"
export HF_HOME="$job_tmp/hf"
export HF_DATASETS_CACHE="$job_tmp/hf/datasets"
export XDG_CACHE_HOME="$job_tmp/xdg"
export TRITON_CACHE_DIR="$job_tmp/triton"
export CUDA_CACHE_PATH="$job_tmp/nv"

srun --overlap --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" \
  --ntasks-per-node=1 \
  mkdir -p "$HF_DATASETS_CACHE" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" \
           "$CUDA_CACHE_PATH"

head_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
head_ip=$(srun --overlap --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
head_ip=${head_ip%% *}

echo "job $SLURM_JOB_ID: $SLURM_NNODES nodes, head $head_node ($head_ip)"
"$venv_python" -c 'import torch; print("torch", torch.__version__)'

# --export=ALL is required: without it the exports above never reach the ranks.
# --chdir is required: relative paths such as ./tests/assets/tokenizer resolve
# against the rank's cwd, which is not the submit directory on other nodes.
srun \
  --export=ALL \
  --chdir="$workdir" \
  --nodes="$SLURM_NNODES" \
  --ntasks="$SLURM_NNODES" \
  --ntasks-per-node=1 \
  --gpus-per-task=@GPUS_PER_NODE@ \
  "$venv_python" -m torch.distributed.run \
    --nnodes="$SLURM_NNODES" \
    --nproc-per-node=@GPUS_PER_NODE@ \
    --rdzv-id="$SLURM_JOB_ID" \
    --rdzv-backend=c10d \
    --rdzv-endpoint="$head_ip:29500" \
    -m torchtitan.train \
    --dump-folder "@OUTPUT_DIR@/$SLURM_JOB_ID" \
    @TRAIN_ARGS@
