#!/usr/bin/env bash
# Compare the optimized DistMoE graph against the same graph with CODA.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$script_dir/../.." && pwd)

arm="${1:-}"
runtime_id="${CW_RUNTIME_ID:-}"
sync_source=1
steps=30
dry_run=0

shift || true
while [ "$#" -gt 0 ]; do
  case "$1" in
    --runtime-id)
      runtime_id="$2"
      shift 2
      ;;
    --steps)
      steps="$2"
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

case "$arm" in
  baseline)
    config=graph_trainer_deepseek_v3_16b_dist_moe_mxfp8_mlperf_64gpu
    ;;
  coda)
    config=graph_trainer_deepseek_v3_16b_dist_moe_mxfp8_mlperf_64gpu_coda
    ;;
  *)
    echo "usage: $0 baseline|coda --runtime-id ID [--steps N] [--no-sync] [--dry-run]" >&2
    exit 2
    ;;
esac

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

timestamp=$(date +%m%d-%H%M%S)
submit_options=(
  --nodes 16
  --segment 16
  --time 01:00:00
  --name "gt-dsv3-16b-dp64-ep64-${arm}-${timestamp}"
)
if [ "$dry_run" -eq 1 ]; then
  submit_options+=(--dry-run)
fi

env "${common_env[@]}" "$script_dir/cw.sh" submit "${submit_options[@]}" -- \
  --module graph_trainer.deepseek_v3 \
  --config "$config" \
  --training.steps "$steps" \
  --parallelism.fsdp-reshard-after-forward never \
  --parallelism.enable-fsdp-symm-mem \
  --parallelism.fsdp-symm-mem-policy widest \
  --compile.mode aot_fx_trace \
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
  --profiler.no-enable-profiling \
  activation-checkpoint:none
