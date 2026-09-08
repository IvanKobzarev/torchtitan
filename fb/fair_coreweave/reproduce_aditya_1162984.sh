#!/usr/bin/env bash
# Reproduce FAIR_CW_USE2_1 job 1162984 (~6,000 steady-state tokens/s).

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$script_dir/../.." && pwd)

runtime_id="${CW_RUNTIME_ID:-}"
sync_source=1
dry_run=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --runtime-id)
      runtime_id="$2"
      shift 2
      ;;
    --no-sync)
      sync_source=0
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    *)
      echo "unknown option: $1" >&2
      exit 2
      ;;
  esac
done

if [ -z "$runtime_id" ]; then
  echo "pass --runtime-id ID or set CW_RUNTIME_ID" >&2
  exit 2
fi

remote_venv="\$HOME/cw/runtimes/$runtime_id/conda"
common_env=(
  CW_TORCHTITAN_SRC="$repo_dir"
  CW_VENV="$remote_venv"
)

if [ "$sync_source" -eq 1 ] && [ "$dry_run" -eq 0 ]; then
  env "${common_env[@]}" "$script_dir/cw.sh" sync
  env "${common_env[@]}" "$script_dir/cw.sh" setup-overlay
fi
if [ "$dry_run" -eq 0 ]; then
  env "${common_env[@]}" "$script_dir/cw.sh" verify-runtime
fi

submit_options=(
  --nodes 64
  --segment 16
  --time 00:30:00
  --name gt-dsv3-671b-dp256-ep64-lbs1-ga16-noac-wgrad-v11-repro
)
if [ "$dry_run" -eq 1 ]; then
  submit_options+=(--dry-run)
fi

env "${common_env[@]}" "$script_dir/cw.sh" submit "${submit_options[@]}" -- \
  --module graph_trainer.deepseek_v3 \
  --config graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_mlperf_64gpu \
  --training.num-tokens-per-microbatch-per-dp-rank 4096 \
  --training.num-tokens-per-train-step 16777216 \
  --training.max-context-length 4096 \
  --training.steps 22 \
  --parallelism.data-parallel-replicate-degree 1 \
  --parallelism.data-parallel-shard-degree 256 \
  --parallelism.tensor-parallel-degree 1 \
  --parallelism.context-parallel-degree 1 \
  --parallelism.pipeline-parallel-degree 1 \
  --parallelism.expert-parallel-degree 64 \
  --parallelism.fsdp-reshard-after-forward never \
  --parallelism.enable-fsdp-symm-mem \
  --parallelism.fsdp-symm-mem-policy widest \
  --compile.mode aot_fx_trace \
  --compile.inductor-compilation none \
  --compile.memory-policy none \
  --compile.enable-graph-gradient-accumulation \
  --compile.enable-deferred-fsdp-gradient-sync \
  --compile.enable-fsdp-ag-rs-overlap \
  --compile.enable-fsdp-dense-region-overlap \
  --training.no-disable-cuda-graphs \
  --compile.require-cudagraph \
  --optimizer.implementation fused_opt_states_bf16 \
  --hf-assets-path ./tests/assets/tokenizer \
  --metrics.log-freq 1 \
  --metrics.no-enable-tensorboard \
  --checkpoint.no-enable \
  --comm.trace-buf-size 0 \
  --debug.seed 42 \
  --debug.no-print-config \
  --profiler.enable-profiling \
  --profiler.profile-freq 20 \
  --profiler.profiler-warmup 3 \
  --profiler.profiler-active 1 \
  --profiler.profiler-repeat 1 \
  activation-checkpoint:none
