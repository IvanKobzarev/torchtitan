# Graph-owned gradient accumulation

Status: locally validated and adversarially approved; CoreWeave validation pending

## Summary

GraphTrainer should accumulate reduced parameter gradients into stable,
trainer-owned buffers inside the final minimal-FX forward/backward graph. The
optimizer continues to run outside the graph and consumes those buffers through
`parameter.grad`.

This removes the current post-replay loop over every parameter. Once sinks are
placed at grad-ready points, it also shortens temporary-gradient lifetimes and
lets each accumulation run as soon as its FSDP reduction and dtype conversion
finish. It preserves GraphTrainer's functional `torch.autograd.grad` tracing
model while making gradient state an explicit mutable graph input.

The initial scope is AOT GraphTrainer in SPMD mode. GraphPP has two distinct
accumulation levels and needs a separate extension after this contract is
validated.

## Motivation

The 16-GPU DeepSeek-V3 16B DP16/EP8, LBS4/GA8 trace has eight CUDA graph
replays per optimizer step. Each replay returns 324 sharded parameter gradients.
`accumulate_param_grads_` then adds them to live `parameter.grad` buffers after
the replay:

```text
8 replays x 324 gradients = 2,592 uncaptured additions per step
```

Across ranks, GraphTrainer spends about 17.8 ms in those additions. The total
inter-replay gap is about 58.9 ms, versus 14.5 ms in MainTrainer. The roughly
44 ms difference includes the additions, layout copies, Python dispatch, and
GPU idle time; it must not all be attributed to addition kernel time.

MainTrainer avoids this boundary because native autograd and FSDP update stable
`parameter.grad` buffers during captured backward. GraphTrainer instead traces
`torch.autograd.grad`, returns gradients as graph outputs, and installs them on
parameters after replay.

## Goals

- Accumulate every reduced gradient inside the minimal FX graph, whether that
  graph runs directly or is wrapped by CUDA graph capture.
- Use one stable optimizer-visible gradient buffer per unique parameter.
- Place each accumulation immediately after its gradient becomes ready, rather
  than serializing all accumulation at the end of backward.
- Preserve current numerical ordering and gradient dtypes.
- Support GA1 and GA>1 without a data-dependent first-microbatch branch.
- Preserve FSDP, HSDP, EP, TP, chunked loss, activation checkpointing, and
  CUDA-graph replay semantics for models in the supported v1 subset.
- Remove parameter gradients from graph outputs once all consumers migrate.
- Avoid adding gradient buffers to checkpoints.

## Non-goals

- Capturing the optimizer, gradient clipping, scheduler, or checkpointing.
- Accumulating unsharded gradients across outer microbatches to reduce the
  number of reduce-scatters. This changes memory and communication behavior.
- Changing reduction precision. The chunked LM-head BF16 fix is separate.
- Fusing gradient accumulation into every producer kernel. In-place MXFP8
  WGrad can be evaluated independently after graph-owned buffers work.
- GraphPP. It must account separately for unsharded pipeline-microbatch
  accumulation and final sharded-gradient accumulation.
- Sparse gradients, data-dependent unused parameters, parameter freezing or
  unfreezing after trace, tied parameters, and distinct parameter views sharing
  storage in v1.
- Full-Inductor or precompiled-artifact execution in v1. Both remain gated
  until mutable-input and serialization contracts are tested.

## Current execution

The traced callable computes:

```text
loss, reduced_grad_0, ..., reduced_grad_N = traced_forward_backward(inputs)
```

After CUDA graph replay, Python runs:

```text
for parameter, reduced_grad in zip(parameters, reduced_grads):
    repair_layout_if_needed(reduced_grad)
    parameter.grad += reduced_grad
```

This creates three problems:

1. The CUDA graph owns its output storage, so the first returned gradient must
   be cloned before another replay can overwrite it.
2. Accumulation is serialized after the complete backward instead of running
   as individual FSDP buckets become ready.
3. A returned gradient with a different stride requires an additional
   `empty_like(parameter)` and copy before accumulation.

## Proposed design

### 1. Gradient-buffer state

Introduce a `GraphGradientState` owned by `GraphTrainer`. It contains one
buffer for each unique optimizer parameter. Runtime identity comes from the
optimizer parameter object; artifact identity uses a canonical FQN plus every
alias FQN.

V1 rejects tied parameters and distinct parameter views sharing storage. It
also requires every optimizer-owned trainable parameter to produce a gradient
at trace time. Prebinding a zero gradient for a statically unused parameter is
not equivalent to preserving `parameter.grad is None`: AdamW could advance its
state or apply weight decay. Support for unused parameters therefore requires
an explicit optimizer mask and is deferred.

Each buffer must have:

- the same global shape, local shape, global and local stride, device, dtype,
  device mesh, and DTensor placements as the optimizer-visible gradient;
- a stable address for the lifetime of the traced artifact and its CUDA graph;
- no checkpoint persistence.

The trainer creates and binds the state after checkpoint loading and any lazy
optimizer-state initialization, but before the first `train_step` zeroing.
While this state is active, a GraphTrainer-owned zeroing path always clears the
buffers in place at the start of an optimizer step. Every microbatch follows
the same path, with or without CUDA graph capture: always add into an existing
zero-or-partially-accumulated buffer.

Before every step, validate parameter identity, `parameter.grad` identity, and
local data pointers. Layout and dtype checks may be debug-only after the state
is established. An optimizer, hook, or checkpoint operation that replaces a
parameter or gradient is unsupported in v1 and fails loudly rather than
letting replay write an orphan buffer.

Do not create a first-microbatch `None` versus later-microbatch branch inside
the graph.

### 2. Explicit static graph inputs

Gradient buffers must be explicit graph inputs, not tensors captured as hidden
Python constants. Extend the tracing runtime with a separate named,
non-checkpointed static-state collection. Do not append them to `state_fqns`,
whose ordering describes model state and is consumed by FSDP graph passes.

`TracedResult` records gradient-state FQNs and their tensor-subclass flattening
metadata separately. `GraphGradientState` owns the live buffers in that order,
binds them to `parameter.grad`, and validates their object and storage
identities before execution.

At trace time, `TracedResult` also records an explicit mapping from each unique
parameter's canonical FQN to its logical gradient-output index and flattened
output leaves. Every graph pass that changes outputs updates this mapping. The
terminal sink consumes the mapping directly; it never rediscovers gradients by
shape, order, or node-name heuristics.

The canonical table starts from model parameter identities in trace order.
Initialization separately flattens optimizer parameter groups and verifies
exact one-to-one membership. Omitted, extra, or duplicate optimizer entries
cannot silently change graph-output mapping; optimizer group order does not
affect the mapping because each buffer is bound to its parameter object.

Flattened inputs use one order: model state, gradient state, optimizer state,
then user inputs. Gradient state is inserted before the user-input boundary, so
all currently supported static state remains a leading prefix. `TracedResult`
stores gradient-state input indices separately; callers must not infer the
model-state mapping from the combined prefix.

The CUDA-graph pass marks every flattened gradient-buffer tensor as static.
Replay must never copy these buffers into graph-private input storage. Debug
mode verifies their addresses before every replay.

### 3. Accumulation in the joint graph

The raw AOT trace remains the current pure `[loss, *grads]` graph because SAC,
EP chunking, FSDP bucketing, and scheduling consume its gradient outputs. After
those transformations, a mandatory terminal pass changes the final minimal-FX
graph's semantic boundary. CUDA graph capture is a later, optional wrapper
around that already stateful minimal FX graph; capture being disabled or
skipped does not change gradient semantics.

After all existing gradient and communication passes, a terminal gradient-sink
pass uses the static buffer placeholders and inserts an
alias-annotated `aten.add_.Tensor` for each final optimizer-visible gradient:

```text
reduced_grad = backward(...)
grad_buffer.add_(reduced_grad)
```

The mutation is inserted after AOT/autograd functionalization, not before it.
The pass verifies that DCE and each enabled backend preserve the input mutation.
V1 supports regional or no Inductor compilation and rejects full-Inductor until
mutable-input execution and artifact round trips are proven.

Every call to the wrapped graph must contribute to the buffers exactly once,
including the warmup-to-capture transition. `CUDAGraphWrapper` currently uses
the first call for eager warmup, and the second call to record and immediately
replay the CUDA graph. CUDA stream capture records GPU work without executing
it, so the recording itself contributes zero and that immediate replay
contributes one. The mutation-aware wrapper must preserve this contract and
must not add another eager execution around capture. If a future capture
backend executes mutations while recording, it must instead restore the
pre-capture buffer contents before one replay, or return that execution without
replaying. A backend whose behavior cannot be established is unsupported.

The terminal pass removes gradient outputs and updates `TracedResult`'s output
specification and tensor-subclass metadata; SPMD execution returns loss and
metrics only. Keeping hundreds of buffer aliases as outputs would retain
rewrapping overhead and could keep temporary gradients live. Trainer requires
`TracedResult.grad_sink_active` to confirm that the sink pass and new output
contract are both active. A missing sink is a correctness error, not a reason
to fall back to external accumulation.

For the first correctness milestone, the additions may remain at the backward
tail. This removes the Python boundary and replay-output clones. It does not by
itself guarantee lower peak memory because all temporary gradients may remain
live until that tail, and it does not yet recover all overlap.

### 4. Grad-ready placement

A follow-up version of the sink pass moves each independent accumulation after
the final node that establishes the optimizer-visible gradient:

- after the collective wait for FSDP/HSDP gradients;
- after the post-reduction cast to the persistent parameter dtype;
- after any required placement and dtype normalization, which must remain after
  reduction;
- before unrelated later backward work whenever dependencies permit.

The destination buffer itself establishes the optimizer-required stride, so v1
does not materialize a separate source merely to match strides. The pass
operates on the final gradient-output mapping, rather than inferring parameters
from shapes, and rejects ambiguous mappings.

Placement is stream-aware, not merely topological. It integrates with the
existing FSDP gradient scheduler so the add runs on the post-reduce stream, or
uses explicit events that preserve the same dependency, without forcing an
early wait on the main compute stream. The scheduler emits one final join before
the optimizer can observe the buffers. The pass also verifies that the
gradient has no remaining users before moving its sink.

This matches MainTrainer's FSDP post-reduce behavior: accumulation can overlap
the remaining backward and communication instead of extending the gap between
graph replays.

### 5. Runtime and optimizer boundary

The optimizer remains unchanged. It sees the same `parameter.grad` object on
every step. Gradient clipping, optimizer execution, and scheduler execution
remain outside the forward/backward graph.

At the start of each optimizer step, `zero_grad(set_to_none=False)` clears
the buffers in place. Optimizer parameter-group membership is fixed while a
trace is active. Grad hooks or custom optimizers that replace `.grad`, sparse
gradients, and parameter freezing or unfreezing require fallback or retracing.
V1 permits checkpoint load only before tracing; replacement or
re-parallelization after tracing fails fast and requires starting a new Trainer.
Buffers are constructed as zeros after the load. Mid-accumulation checkpoints
are unsupported.

### 6. Precompiled artifacts

V1 rejects precompiled artifacts. Before enabling them, serialization must
include `GradBufferSpec`, explicit static-input indices, subclass layouts, and
the parameter alias table. The artifact fingerprint must include every field
that changes accumulation code, including reduction dtype. Loading validates
the live parameter mapping before binding buffers or replaying the graph.

### 7. Future GraphPP design

GraphPP currently has two levels: unsharded per-pipeline-microbatch
accumulation followed by scheduled reduction, and final sharded-gradient
accumulation into parameters. Replacing only its final
`accumulate_param_grads_` call is insufficient.

A future design needs stage-local persistent reduction-dtype unsharded
accumulators, one scheduled reduction, persistent optimizer-facing sharded
buffers, and explicit events for `OVERLAP_F_B` graphs. Cross-stage tied
parameters remain unsupported until their ownership and ordering are defined.

## Correctness invariants

1. V1 fails at trace time for tied parameters, storage aliases, sparse
   gradients, or optimizer-owned trainable parameters without gradients.
2. One buffer exists per unique optimizer parameter identity.
3. Buffer dtype, shape, stride, device, mesh, and DTensor placements match the
   optimizer-visible gradient.
4. Reduction uses `training.mixed_precision_reduce`; conversion to persistent
   dtype occurs after reduction.
5. Every contribution is added exactly once in the same microbatch order as
   the current implementation.
6. The optimizer never observes a replay-owned temporary.
7. Buffer addresses remain stable across replay.
8. Buffers are excluded from model and optimizer checkpoints.
9. Graph teardown releases its references. Checkpoint load, model
   re-parallelization, or parameter replacement after tracing is rejected in
   v1 rather than attempting an in-place retrace.
10. Grad scaling, clipping, and optimizer semantics are unchanged.
11. Chunked loss sinks only its final reduced LM-head gradient; it does not add
    another per-chunk accumulation path.

## Alternatives considered

### Capture a separate post-replay accumulation graph

This is a useful diagnostic milestone but not the final design. It removes
Python launch overhead while retaining a serialized tail after every replay,
retains graph-output gradient storage, and cannot overlap accumulation with
backward.

### Trace `loss.backward()` and native `.grad` mutation

This resembles MainTrainer but gives up GraphTrainer's functional gradient
outputs, complicating graph partitioning, rematerialization, and GraphPP. It
also relies on autograd hook side effects that the current AOT pipeline was
designed to avoid.

### Accumulate before reduce-scatter

This could reduce communication frequency but retains full unsharded gradients
across outer microbatches and changes rounding and memory behavior. It is a
separate optimization, especially for large models.

### Foreach accumulation after replay

`foreach_add_` may reduce launch overhead but still runs outside the graph and
at the backward tail. It is a fallback if graph mutation proves unsafe, not the
target architecture.

## Validation plan

### Unit and fake-process-group tests

- GA1 and GA8 match the current implementation bitwise with seed 42 and
  deterministic mode.
- BF16 and FP32 reduction policies produce the expected collective input dtype
  and FP32 optimizer-visible gradients.
- FSDP, HSDP, EP+eFSDP, and degree-one FSDP placements are preserved.
- Contiguous, transposed, and DTensor-local parameter strides are preserved.
- V1 rejects shared/storage-aliased and unused-gradient models with actionable
  errors; later support receives dedicated optimizer-semantics tests.
- CUDA graph replay produces the same loss, gradients, and optimizer update for
  multiple steps.
- Direct minimal-FX execution with CUDA graphs disabled accumulates multiple
  microbatches into the same stable buffers and matches eager optimizer updates.
- Starting from a nonzero buffer, assert its exact contents after each of the
  first three wrapper calls: eager warmup contributes once, capture recording
  plus its immediate replay contributes once total, and ordinary replay
  contributes once. Run this test against every supported capture backend.
- Gradient buffers are excluded from non-static input copying.
- No-Inductor and regional-Inductor execution both pass; full-Inductor fails
  configuration validation in v1.
- CUDA graphs, direct FX execution, and capture fallback all use the same
  graph-owned buffers and terminal mutation.
- Standard and fused optimizers preserve `parameter.grad` identity across
  zeroing, clipping, and stepping.
- Per-parameter gradient and parameter-update hashes match the control path.
- Checkpoint state dictionaries do not contain graph gradient buffers.
- Post-trace checkpoint load, parameter replacement, and re-parallelization are
  rejected without leaving stale buffer aliases.

### Local GPU validation

- DeepSeek-V3 debug and 16B fake-process-group configurations complete at
  least ten steps with GA1 and GA8.
- Compare loss and grad norm against the current path with seed 42 and
  deterministic mode.
- Confirm that no parameter-gradient add or layout-repair kernel runs between
  CUDA graph replays.
- At the tail-sink milestone, confirm that CUDA graph outputs retain no
  model-sized gradient storage. Only after grad-ready placement, measure whether
  shortened temporary lifetimes reduce peak memory.

### CoreWeave validation

Run the established 16-GPU DP16/EP8, LBS4/GA8, seq4096, no-AC, MXFP8 workload.
Require:

- successful end-to-end execution;
- matching loss and grad norm against an identically configured control;
- no new collective or allocator failures;
- all 2,592 additions moved inside capture, with no additions between replays;
- inter-replay gaps materially reduced from the current approximately 58.9 ms;
- no regression in communication overlap or peak memory;
- throughput improvement reproduced across at least two measured runs before
  enabling by default.

Then validate GA1 and full activation checkpointing before GraphPP integration.

## Rollout

1. Land the BF16 chunked LM-head correction independently.
2. Add SPMD graph-owned buffers behind an initially disabled AOT FX option;
   reject unsupported aliases, unused/sparse gradients, full-Inductor,
   precompiled artifacts, and custom pass pipelines.
3. Insert the terminal sink after all existing gradient/FSDP passes, remove
   gradient outputs while updating `TracedResult` metadata, and validate tail
   accumulation.
4. Add stream-aware grad-ready placement and validate the trace.
5. Run the 16-GPU acceptance workload and retain the old path as a kill switch.
6. Add precompiled-artifact support and enable for supported SPMD AOT execution
   after numerical and performance gates pass.
7. Write and review the separate GraphPP extension.

## Open questions

- Whether bucket-level or foreach accumulation can reduce the remaining 2,592
  in-graph kernel launches without delaying grad-ready execution.

## Performance interpretation

Moving accumulation inside the FX graph removes the Python accumulation
boundary for every execution backend. When CUDA graph capture is active, this
may remove much of the roughly 44 ms inter-replay gap difference, but the
measured add kernels themselves are only about 17.8 ms. A tail-only sink removes
host launch and replay-boundary work but still executes the additions serially.
Grad-ready placement targets overlap; bucket-level or fused accumulation
targets kernel-count reduction. No end-to-end gain is claimed until measured on
the 16-GPU workload.

## Review record

Two independent adversarial design reviews were completed before
implementation. Their findings shaped unused-gradient semantics,
tied/storage-aliased parameter handling, explicit trace-to-runtime gradient
mapping, pass ordering, CUDA-independent execution, static-input metadata,
precompile safety, zeroing and checkpoint lifecycle, stream ordering, GraphPP
scope, and performance attribution. A fresh adversarial code review after local
validation approved the SPMD implementation for CoreWeave testing. It also
required a fail-closed guard for MXFP8 and DistMoE in-place WGrad modes, whose
first-contribution semantics are incompatible with prebound gradient buffers.
