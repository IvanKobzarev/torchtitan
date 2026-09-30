# DeepSeek V3 671B DistMoE 256-GPU Run

## Revision

```bash
git clone https://github.com/pytorch/torchtitan.git
cd torchtitan
git fetch origin fix-gt-dryrun-memory-stacked-a87
git switch --detach 6487c7915b8993a6050aff43eacbd3afcbf4bbe7
test "$(git rev-parse HEAD)" = 6487c7915b8993a6050aff43eacbd3afcbf4bbe7
test -z "$(git status --porcelain)"
```

Commit `6487c7915b8993a6050aff43eacbd3afcbf4bbe7` is the validated code
revision. The final command must run from that exact, clean checkout.

## Run request

- 256 NVIDIA GB300 GPUs, one process per GPU.
- Reference physical packing: 64 hosts with four GPUs per host. Report any
  different packing because it can change performance.
- Keep every EP64 group within a supported high-bandwidth fabric domain.
- DP-shard 256, EP64, expert FSDP degree 4, TP1, CP1, PP1.
- Sequence length 4096, local batch size 1, gradient accumulation 16.
- FSDP reshard-after-forward `never`.
- Unshard in the first microbatch and reduce-grad in the last microbatch.
- Optimization ladder rung R4: in-place WGrad accumulation through WGrad
  producer fusion.
- FP32 persistent parameters and gradients; BF16 compute, reductions, and Adam
  moments; MXFP8 dense and routed-expert GEMMs.
- Forced-balanced routing and a maximum-useful DistMoE device budget of
  58.324 GiB.
- `c4_test` with cuDNN SDPA.
- CUDA graphs enabled and activation checkpointing disabled.
- Benchmark for 60 steps and log metrics every 10 steps.
- Compute benchmark statistics from steps 20, 30, 40, 50, and 60.

Use this entry point:

```text
module: graph_trainer.deepseek_v3
config: graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_chien_chin_256gpu
```

Before launch, verify that the runtime provides the required operators:

```bash
pytest -q tests/unit_tests/cpu/test_dist_moe.py \
  -k chien_chin_256gpu_recipe_matches_ladder1_r4

python - <<'PY'
import torch
import grain.python
import torchao.prototype.mx_formats.kernels
import dist_moe
import dist_moe._blockscaled
from dist_moe import BlockScaledFormat, DistMoeBlockScaledConfig
from torch.utils.checkpoint import _is_cacheable_effect

assert torch.cuda.get_device_capability() == (10, 3)
assert torch.cuda.get_device_properties(0).total_memory >= 250 * 1024**3
assert hasattr(torch.ops.aten, "_scaled_addmm_")
assert hasattr(torch.ops.dist_moe, "block_scaled_backward_accumulate")
assert hasattr(torch.ops.dist_moe, "bf16_backward_accumulate")
assert torch._C._dispatch_has_kernel_for_dispatch_key(
    "torchao::mxfp8_quantize", "CUDA"
)
print("torch", torch.__version__, torch.version.git_version)
print("grain", grain.python.__file__)
print("torchao", torchao.prototype.mx_formats.kernels.__file__)
print("dist_moe", dist_moe.__file__)
print("dist_moe symbols", BlockScaledFormat, DistMoeBlockScaledConfig)
PY
```

Stop if any assertion fails. These checks do not validate the 256-rank
communication path; only the complete distributed run does that.

Launch exactly 256 workers, one per GPU. The launch environment must supply
valid `RANK`, `WORLD_SIZE=256`, `LOCAL_RANK`, `MASTER_ADDR`, and `MASTER_PORT`
to every worker. All workers must use the same clean checkout and runtime and
start from the repository root.

Provide a unique shared `RUN_OUTPUT` for the experiment and a unique
node-local `LOCAL_CACHE_ROOT` on each host. Point `TRITON_CACHE_DIR`,
`TORCHINDUCTOR_CACHE_DIR`, `CUTE_DSL_CACHE_DIR`, and `CUDA_CACHE_PATH` to
subdirectories of `LOCAL_CACHE_ROOT`.

For the benchmark, disable `TORCH_TRACE` and the profiler:

```bash
unset TORCH_TRACE
python -u -m torchtitan.train \
  --dump-folder "$RUN_OUTPUT/benchmark" \
  --module graph_trainer.deepseek_v3 \
  --config graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_chien_chin_256gpu \
  --hf-assets-path ./tests/assets/tokenizer \
  --training.steps 60 \
  --metrics.log-freq 10 \
  --metrics.no-enable-tensorboard \
  --comm.trace-buf-size 0 \
  --debug.seed 42 \
  --debug.no-print-config \
  --profiler.no-enable-profiling
```

Run profiling separately. Before starting TorchTitan, set the per-rank
environment as follows so only rank zero writes `TORCH_TRACE` data:

```bash
if [ "$RANK" -eq 0 ]; then
  export TORCH_TRACE="$RUN_OUTPUT/profile/tlparse"
else
  unset TORCH_TRACE
fi
```

Launch the same 256-GPU configuration for 41 steps, with only step 41 active in
the profiler:

```bash
python -u -m torchtitan.train \
  --dump-folder "$RUN_OUTPUT/profile" \
  --module graph_trainer.deepseek_v3 \
  --config graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_chien_chin_256gpu \
  --hf-assets-path ./tests/assets/tokenizer \
  --training.steps 41 \
  --metrics.log-freq 10 \
  --metrics.no-enable-tensorboard \
  --comm.trace-buf-size 0 \
  --debug.seed 42 \
  --debug.no-print-config \
  --profiler.enable-profiling \
  --profiler.profile-freq 41 \
  --profiler.profiler-warmup 0 \
  --profiler.profiler-active 1 \
  --profiler.profiler-repeat 1
```

Do not override the recipe's parallelism, batch size, FSDP placement,
precision, or memory settings. Never include the profiling run in the
benchmark statistics. The PyTorch profiler is enabled on every rank and may
write 256 trace files; reserve enough output capacity for all of them.

## Verify

The startup log must show:

- 671,026,404,352 total and 36,625,603,584 active parameters.
- Mesh `dp_shard=256` and `ep=64`.
- DistMoE rows `32768/33152/33152` and device budget 58.324 GiB.
- WGrad fusion `BF16=0, MXFP8=306, DistMoE=58`.
- First microbatch with unshard, last microbatch with reduce-grad, then reshard.

All 60 benchmark steps and all 41 profiling steps must finish without OOM,
NaN, graph recapture, or distributed errors. Compute performance only from
benchmark steps 20, 30, 40, 50, and 60. At every logged step, loss and gradient
norm must be finite and gradient norm must be nonzero. Do not require an exact
gradient-norm value.

## Report

Provide:

1. Full source commit, both complete commands, GPU model/count, host/GPU
   packing, and resolved package versions and locations.
2. Tokens/s/GPU at steps 20, 30, 40, 50, and 60, with mean and standard
   deviation. Also report aggregate tokens/s as the per-GPU mean multiplied by
   256.
3. MFU at steps 20, 30, 40, 50, and 60, with mean and standard deviation. If
   TorchTitan reports MFU as `N/A`, report TFLOP/s with the same statistics.
4. Peak allocated and peak reserved GPU memory, preferably min/median/max
   across all ranks.
5. Complete logged loss, gradient norm, and any emitted gradient diagnostics.
6. Complete stdout and stderr logs from all ranks for both runs, including
   unabridged error logs if either run fails.
7. The raw rank-zero `TORCH_TRACE` data and rendered `tlparse` output.
8. The rank-zero step-41 trace from
   `$RUN_OUTPUT/profile/profiling/traces/iteration_41/rank0_trace.json.gz`.
   From a machine with an fbsource checkout and internal `fbpython` and
   `manifold` access, upload it and include the printed Perfetto URL:

   ```bash
   FBSOURCE_ROOT=/path/to/fbsource
   TRACE_PATH="$RUN_OUTPUT/profile/profiling/traces/iteration_41/rank0_trace.json.gz"
   test -x "$FBSOURCE_ROOT/arvr/scripts/perfetto/share_trace.py"
   test -f "$TRACE_PATH"
   "$FBSOURCE_ROOT/arvr/scripts/perfetto/share_trace.py" "$TRACE_PATH" \
     | tee "$RUN_OUTPUT/profile/rank0_trace_share.txt"
   ```

   `share_trace.py` uploads the trace with a 28-day TTL by default. It is an
   internal helper and is not supplied by TorchTitan, so trace sharing can be
   done after copying the artifact off the training cluster.

The historical comparison point is 5,880.6 +/- 13.8 tokens/s/GPU,
approximately 1.51 million aggregate tokens/s. It was computed from steps 20,
30, 40, 50, and 60 in the original performant-run report,
[P2483940224](https://www.internalfb.com/phabricator/paste/view/P2483940224).
