# TorchTitan on the FAIR CoreWeave GB300 clusters

Scripts and notes for running TorchTitan on `fair-cw-use2-1` / `fair-cw-use2-3`
from a devserver. The point of this directory is that a local edit to TorchTitan,
PyTorch, or torchao ends up in the next cluster run without anyone retyping the
commands.

Sources: [P2468526779](https://www.internalfb.com/intern/paste/P2468526779/)
(submission guide) and
[P2469701911](https://www.internalfb.com/intern/paste/P2469701911/) (what
actually broke while using it). The
[GB300 onboarding doc](https://docs.google.com/document/d/1g2GuajX6ek-Y4O34DDdi6TCv4zXpauY5l4ok2fttHSE/edit)
is the source of truth for cluster behavior; this file only records what the
scripts encode.

## Cluster facts

| | |
|---|---|
| Nodes | 1872 x 4 GB300, 955 GB RAM, 144 cores, 28T node-local `/tmp` |
| GPU | **sm_103**, 284 GB, CUDA 13.0 |
| CPU | **aarch64** on login and GPU nodes; the `cpu_x86` partition is x86 |
| Topology | 18 nodes / 72 GPUs per NVL72 rack, RoCEv2 between racks |
| Filesystem | `$HOME` is shared NFS across all nodes; `/tmp` is node-local |
| Network | direct internet egress, so `pip` works on compute nodes |
| Slurm | partition `g3`, account `faircw-pytorch-access`, qos `g3_lowest` |

Confirm your own entitlement rather than copying the account above:

```bash
sacctmgr show assoc where user="$USER" format=Cluster,Account,Partition,QOS%40
```

## Connecting, and the Duo problem

The bastion enforces `AuthenticationMethods publickey,keyboard-interactive`. The
certificate satisfies only the first half:

```
Server accepts key: ... RSA-CERT ...
Authenticated using "publickey" with partial success.
Authentications that can continue: keyboard-interactive
```

The second half is a Duo prompt, which needs a real TTY. No script and no coding
agent can answer it. So a human opens one multiplexed master from a terminal, and
everything afterwards rides that socket without authenticating again.

Two prerequisites, both easy to miss:

- **An ssh-agent must be running.** `cloud hpc login` stores the minted
  certificate in the agent and nowhere else, so with no agent the login fails and
  leaves no credential behind. On a devserver the agent ships as an inactive
  systemd user unit; `cw.sh connect` starts it and exports `SSH_AUTH_SOCK`.
- **The master must be opened from a terminal**, not from an agent's shell tool
  and not from Claude Code's `!` prefix. Neither has a TTY.

```bash
./cw.sh connect     # prints the exact commands to paste into a terminal
```

Run what it prints, answer Duo, and re-run `./cw.sh connect`. It leaves a master
at `/tmp/cw-<cluster>-<user>.ctl` with `ControlPersist=yes`; later `ssh`, `scp`
and `rsync` reuse it and never contact the bastion. The socket outlives the
certificate, so redo this only when `cw.sh connect` says so.

## Getting local changes onto the cluster

Three layers, three different answers. The split is entirely about whether the
layer contains compiled code.

| Layer | Compiled code | How it ships | Command |
|---|---|---|---|
| TorchTitan | no | rsync, then `PYTHONPATH` | `cw.sh sync` |
| torchao, dist_moe | yes | rsync source, compile on a GB300 node | `cw.sh sync && cw.sh build-ao` |
| PyTorch | yes | nightly wheel, or compile on a GB300 node | `cw.sh setup-env` / `cw.sh build-torch` |

Nothing is ever pip-installed from a synced tree. The trees go on `PYTHONPATH`
and stay editable, because a copy in `site-packages` silently wins over
`PYTHONPATH` and you end up debugging a build you replaced an hour ago.

### Does my devserver have to match the cluster?

Only in `CW_TORCH_MODE=wheel`. Everywhere else the devserver is a control node:
it runs rsync, renders a batch script, and holds an SSH socket. All compilation
happens on a GB300 under `srun`, so an x86 H100 box works exactly as well as a
GB200 one. `cw.sh doctor` reports which case you are in.

What an H100 devserver costs you is local pre-flight, not correctness. MXFP8 and
FA4 need Blackwell, so nothing using them can be smoke-tested before submitting,
and large FakePG memory probes need more GPU memory than an H100 has. Every
iteration goes through the Slurm queue.

### TorchTitan

Pure Python, so `cw.sh sync` is the whole story. Edit locally, `cw.sh sync`,
submit. Two things the sync handles for you:

- Bytecode is precompiled once, from one process, and jobs run with
  `PYTHONDONTWRITEBYTECODE=1`. Otherwise every rank races to write `__pycache__`
  onto shared NFS and some get `OSError: [Errno 116] Stale file handle`. This
  first shows up around 16 ranks.
- `.so`, `build/` and `__pycache__` are excluded, so a locally compiled artifact
  can never reach the cluster and get imported for the wrong architecture.

### torchao and other native extensions

These cannot be built on the devserver and copied over, and this is the part
people lose a day to. Run `cw.sh doctor` to see why in your specific case:

```
local
  arch          x86_64
  glibc         2.34
  gpu cap       9.0
```

An x86 devserver cannot produce an aarch64 binary. Even on a GB200 devserver,
which is aarch64, the GPU is sm_100: an sm_100 cubin does not load on sm_103, and
torchao's `compute_120` PTX cannot JIT down to sm_103 either. So the extension
must be compiled on the cluster. `cw.sh build-ao` does that on a GB300 node and
handles the three traps:

1. `setup.py` hardcodes its gencode list and stops at sm_100. The script inserts
   `-gencode=arch=compute_103,code=sm_103` (idempotent).
2. `third_party/cutlass` is not materialized in fbsource, and `setup.py` calls
   `sys.exit(1)` without it. The script clones it at the pinned revision.
3. `uv` serves a cached wheel that skips compilation entirely and looks like a
   clean build. The script clears the cache and greps the log for
   `Building mxfp8_cuda extension`.

Verification runs from `/tmp`, not from inside the source tree, and imports
`torchao.prototype.mx_formats.kernels` before checking for the ops. Both matter:
a tree on the cwd shadows the real package, and bare `import torchao` reports the
ops as missing even when they registered fine.

### PyTorch

Pick a mode with `CW_TORCH_MODE` in `config.sh`.

**`nightly` (default).** `uv pip install --pre torch --index-url .../nightly/cu130`
on the cluster. Use this whenever you have no local PyTorch changes. The
validated runs used `2.15.0.dev+cu130`.

**`source`.** `cw.sh build-torch` rsyncs `CW_TORCH_SRC` and builds it on a GB300
node with `TORCH_CUDA_ARCH_LIST=10.3`. This is the only correct way to ship local
PyTorch changes from a devserver that does not match the cluster. Budget one to
two hours for a cold build. The rsync drops `.git`, so submodules must already be
checked out locally (`git submodule update --init --recursive`) before you sync.

**`wheel`.** Install a wheel you built yourself. Only valid when *all* of these
hold, and `cw.sh doctor` checks the first three:

1. local arch is `aarch64`
2. local glibc <= the cluster's
3. same Python minor version as the remote venv
4. same CUDA major version (13)
5. built with `TORCH_CUDA_ARCH_LIST` including `10.3`

Point 5 is the one that bites on a GB200 devserver, where 1 through 4 already
hold: the default arch list follows the local GPU, so it produces sm_100 and
nothing else. Build with `TORCH_CUDA_ARCH_LIST="10.0;10.3"`.

If you only changed Python files under `torch/` and the installed wheel is from
the same commit base, you can overlay them instead of rebuilding:

```bash
./cw.sh sh 'echo $(readlink -f ~/venvs/tt/lib/python3.12/site-packages/torch)'
rsync -az --include='*/' --include='*.py' --exclude='*' \
  "$CW_TORCH_SRC/torch/" "<that path>/"
```

This is a shortcut, not a supported path. Any change to C++, CUDA or a `.pyi`
contract needs a real build.

## Quick start

```bash
./cw.sh connect       # prints the one-time Duo instructions if needed
./cw.sh doctor        # entitlement, remote arch, portability verdict
./cw.sh sync          # push TorchTitan and torchao source
./cw.sh setup-env     # venv + torch + the tree's requirements.txt
./cw.sh build-ao      # compile torchao for sm_103, only if you need it
```

`sync` comes before `setup-env`: dependencies are installed from the synced
tree's own `requirements.txt`, which pins versions a hand-written list gets
wrong (`spmd_types` in particular).

Prove the harness with a cheap run before anything expensive:

```bash
job=$(./cw.sh submit --nodes 1 --time 00:20:00 --name smoke-llama3 -- \
  --module llama3 --config llama3_debugmodel \
  --training.steps 10 --metrics.log-freq 1 --checkpoint.no-enable)
./cw.sh status "$job"      # also prints the FAIR Hub link
./cw.sh logs "$job"
./cw.sh url "$job"         # https://www.internalfb.com/fair_hub/job/FAIR_CW_USE2_1/<id>/details
```

Then the real thing:

```bash
./cw.sh submit --nodes 16 --time 00:30:00 --name dsv3-16b -- \
  --module deepseek_v3 --config deepseek_v3_16b \
  --parallelism.expert-parallel-degree 64 \
  --training.steps 10 --dataloader.dataset c4_test \
  --metrics.log-freq 1 --checkpoint.no-enable
```

After the first setup, the edit-run loop is just `./cw.sh sync && ./cw.sh submit`.
Add `--dry-run` to print the batch script without submitting; that works offline.

## Profiling and getting traces back

```bash
job=$(./cw.sh submit --nodes 2 --name trace -- \
  --module llama3 --config llama3_debugmodel --training.steps 10 \
  --profiler.enable-profiling --profiler.profile-freq 5)
./cw.sh fetch "$job"
```

`fetch` pulls the whole output directory to `$CW_LOCAL_OUTPUT_DIR/<job>` and
lists the traces. They land at
`profiling/traces/iteration_<step>/rank<n>_trace.json.gz`, one gzipped Chrome
trace per rank per profiled step, loadable in Perfetto or `chrome://tracing`.

### Why job outputs cannot be lost

TorchTitan defaults `--dump-folder` to `./outputs`, relative to the job's cwd,
which is the synced TorchTitan tree. `cw.sh sync` rsyncs with `--delete`, so that
default puts every trace one routine sync away from deletion. Three independent
guards, because one is not enough:

1. **Outputs live at `$CW_OUTPUT_DIR/<job id>`, outside every synced tree.** The
   batch template passes this absolute path before your arguments, so you can
   still override it. The job id only separates runs from each other -- it is the
   *location* that protects them. `outputs/<job id>/` inside the tree would still
   be deleted, since `--delete` removes the whole subtree regardless of names.
2. **`CW_RSYNC_PROTECT` shields artifacts even inside a tree.** Traces,
   checkpoints and `structured_logs` survive a sync wherever they are written.
   `--delete` still removes stale `.py` files, which it must: a renamed module
   left behind stays importable and shadows the real one.
3. **Deletions are never silent.** Anything `--delete` removes is printed:

   ```
   !!  1 file(s) deleted under /home/chienchin/cw/torchtitan/:
       *deleting   DELETE_ME_scratch.py
   ```

Verified by planting a trace, a `.pt` and a scratch `.py` in the synced tree and
syncing: the first two were kept, the third deleted and reported.

**`CW_LOCAL_OUTPUT_DIR` also defaults outside this repo** (`$HOME/cw-outputs`).
Fetched traces are multi-megabyte binaries and this directory is inside an
fbsource checkout.

Import errors are much cheaper to find on the login node than in a job:

```bash
./cw.sh sh 'cd /tmp && PYTHONPATH=$HOME/cw/ao:$HOME/cw/torchtitan \
  $HOME/venvs/cw/bin/python -c "import torchtitan.train"'
```

## Files

| | |
|---|---|
| `config.sh` | cluster, entitlement, remote paths, PyTorch mode, source trees |
| `cw.sh` | the entrypoint; run it with no arguments for the command list |
| `lib.sh` | SSH multiplexing, rsync, remote path resolution |
| `remote/setup_env.sh` | builds the cluster venv |
| `remote/build_torchao.sh` | sm_103 extension build |
| `remote/build_torch.sh` | PyTorch source build |
| `templates/torchtitan.sbatch` | job template |

Everything in `config.sh` can be overridden from the environment:

```bash
CW_CLUSTER=fair-cw-use2-3 CW_TORCH_MODE=source ./cw.sh setup-env
```

## Megatron-Bridge as a baseline

NVIDIA publishes GB300 pre-training throughput for Megatron-Bridge at
[the performance summary](https://docs.nvidia.com/nemo/megatron-bridge/latest/performance-summary.html),
which makes it the natural baseline for TorchTitan numbers from this cluster.
The published table reports **Tokens/sec/GPU** and **Model TFLOP/sec/GPU** only
(no MFU), on DGX-GB300 and DGX-GB200, mostly MXFP8, for DeepSeek V3, GPT OSS
120B, Qwen3, and the Nemotron family.

For the DeepSeek V3 shape this work targets, their 26.08 entry is 256 GB300
GPUs, MXFP8, PP=2, EP=32, at **6288 tokens/sec/GPU and 1635 TFLOP/sec/GPU**.

Megatron-Bridge ships *inside* the NeMo container rather than as a pip package,
which is lucky: it sidesteps building Transformer Engine and Megatron-Core on
aarch64. This cluster has **enroot 4.0.1 and the pyxis SPANK plugin**, so
`srun --container-image=...` works, and `nvcr.io/nvidia/nemo` pulls anonymously
with no NGC API key. The image is multi-arch and has an arm64 variant.

```bash
./cw.sh setup-bridge    # import the container (18 GiB) + clone the launcher repo
./cw.sh bridge-info     # versions the image actually ships
```

`setup-bridge` writes a squashfs to `$CW_CONTAINER_DIR/nemo-<tag>.sqsh` (30 GB
for 26.08) and clones the repo to `$CW_BRIDGE_REPO`. Both are idempotent. It
runs on a **compute node**, not the login pod, for two reasons: the login pod is
a Kubernetes container with no `CAP_MKNOD`, so converting the image's AUFS
whiteouts to overlayfs ones fails half way with `failed to create ovlfs
whiteout: Operation not permitted`; and the node must be aarch64 or enroot
resolves the multi-arch tag to amd64.

Verified with `nemo:26.08`, which ships:

```
megatron.bridge     0.6.0+c93251151
megatron.core       0.19.0+16ad357ee
transformer_engine  2.17.1+4329ff84
torch               2.13.0a0+8145d630e8.nv26.06
```

so `CW_BRIDGE_BRANCH=r0.6.0` is the branch that matches this image.

### Every container step needs these flags

Slurm's `TaskProlog` here runs *inside* the container and is not
container-aware. It runs under `set -e -o pipefail` and pipes two host sidecars
into `tee` with no `|| true`, so a plain `srun --container-image=...` dies with
`TaskProlog failed status=1` before your command ever starts. The sidecars want
`jq`, `/public/meta/bin/scuba_cat` and a BBFS fuse mount, none of which exist in
the NeMo image.

`CW_CONTAINER_MOUNTS` and `CW_CONTAINER_EXPORT` carry the fix, and
`cw_container_args` in `cw.sh` applies it:

```
--container-mounts=/etc/slurm:/etc/slurm:ro,/public:/public:ro,/engshare:/engshare:ro
--export=ALL,DISABLE_PODMAN=1,AIRSTORE_BBFS_FUSE_DISABLED=1
--container-writable
```

`DISABLE_PODMAN` and `AIRSTORE_BBFS_FUSE_DISABLED` are opt-outs the prolog
documents itself. `--container-writable` is needed because Enroot's rootfs is
read-only by default. Reuse these for any container work on this cluster, not
just Megatron-Bridge.

### Reproduced: Qwen3-30B-A3B on 8 GB300

[Job 1076084](https://www.internalfb.com/fair_hub/job/FAIR_CW_USE2_1/1076084/details),
`qwen3_30b_a3b_pretrain_8gpu_gb300_fp8mx_config`, 2 nodes x 4 GPUs,
mock data, 25 steps, `COMPLETED`:

| | NVIDIA 26.08 | This cluster | |
|---|---:|---:|---:|
| Model TFLOP/s/GPU | 1029 | **1033.3** | 100.4% |
| Tokens/s/GPU | 44544 | **~44914** | 100.8% |

Steady step time 5836 ms over the last 8 iterations, flat within 0.3%. The
published number reproduces on this hardware, which validates the container, the
prolog workaround and the topology handling in one go.

**Discard the early steps.** Iteration 1 took 85 s and iteration 2 took 242 s --
CUDA graph capture, Transformer Engine autotuning and Triton compilation. Step 2
being slower than step 1 is normal here and is not a regression. Throughput only
settles from roughly iteration 10, so a short run reports nonsense.

```bash
cd $HOME/cw/megatron-bridge
uv run --no-project --with nemo-run==0.10.0 python scripts/performance/setup_experiment.py \
  -m qwen -mr qwen3_30b_a3b -g gb300 -c fp8_mx -ng 8 \
  -a faircw-pytorch-access -p g3 -t 00:40:00 --gres gpu:4 \
  --additional_slurm_params qos=g3_lowest \
  -i "$HOME/cw/containers/nemo-26.08.sqsh" \
  -cm /etc/slurm:/etc/slurm:ro,/public:/public:ro,/engshare:/engshare:ro \
  -E DISABLE_PODMAN=1 -E AIRSTORE_BBFS_FUSE_DISABLED=1 \
  --data mock -ms 25 --detach True
```

Four flags are cluster-specific and not in NVIDIA's documented invocation:
`--gres gpu:4` (nodes here have 4 GPUs, and without it Slurm rejects the job as
"submitting to the 'g3' partition without requesting a GPU"),
`--additional_slurm_params qos=g3_lowest` (omitting it gives
`Invalid qos specification`), and the two prolog workarounds. Recipe selection is
composed from `-m`/`-mr`/`-g`/`-c`/`-ng`, so those five flags pick
`qwen3_30b_a3b_pretrain_8gpu_gb300_fp8mx_config`; add `-cv large_scale` for a
named variant.

`Failed to import Triton kernels ... triton_kernels.matmul_ogs` appears
throughout the log and is harmless -- the run still hit the published number.

### Reproduced: DeepSeek V3 on 256 GB300

[Job 1076664](https://www.internalfb.com/fair_hub/job/FAIR_CW_USE2_1/1076664/details),
`deepseek_v3_pretrain_256gpu_gb300_fp8mx_config`, 64 nodes x 4 GPUs,
mock data, 50 steps, `COMPLETED`:

| | NVIDIA 26.08 | This cluster | |
|---|---:|---:|---:|
| Model TFLOP/s/GPU | 1635 | **1658.3** | 101.4% |
| Tokens/s/GPU | 6288 | **6378** | 101.4% |

Steady step 10275.6 ms over the last 10 iterations, 244 ms spread. Topology
verified `PROPERLY PACKED`, 4 blocks of 16 nodes. The resolved config matches
both the published table and NVIDIA's own reproduction repo for this shape
(TP1, PP2, CP1, EP32, DP128, VP8, MBS1, GBS4096, seq 4096).

**One deviation was required:** `model.moe_paged_stash=False`. Without it every
attempt died. A BF16 reference on the same 64 nodes, job 1076638, ran unmodified
and gives 3980 tokens/s/GPU at 1035.2 TFLOP/s/GPU, so MXFP8 is worth about 1.60x
here.

```bash
cd $HOME/cw/megatron-bridge
uv run --no-project --with nemo-run==0.10.0 python scripts/performance/setup_experiment.py \
  -m deepseek -mr deepseek_v3 -g gb300 -c fp8_mx -ng 256 \
  -a faircw-pytorch-access -p g3 -t 01:30:00 --gres gpu:4 \
  --additional_slurm_params qos=g3_lowest \
  -i "$HOME/cw/containers/nemo-26.08.sqsh" \
  -cm /etc/slurm:/etc/slurm:ro,/public:/public:ro,/engshare:/engshare:ro \
  -E DISABLE_PODMAN=1 -E AIRSTORE_BBFS_FUSE_DISABLED=1 \
  --data mock -ms 50 --detach True \
  model.moe_paged_stash=False
```

#### Profiling a Megatron-Bridge run

`submit-bridge` owns the cluster-specific flags and pins the launcher's
`--log_dir` under `$CW_OUTPUT_DIR/bridge/<name>`, so `bridge-fetch` knows where
to look. This is the Megatron equivalent of `submit` + `fetch`.

```bash
./cw.sh submit-bridge --name dsv3-256-fp8mx --profile --profile-steps 45:50 -- \
  -m deepseek -mr deepseek_v3 -g gb300 -c fp8_mx -ng 256 \
  -t 01:30:00 --data mock -ms 50 --detach True \
  model.moe_paged_stash=False

./cw.sh bridge-fetch dsv3-256-fp8mx
```

`--profile` maps to `-pyp True` plus `--profiling_start_step` /
`--profiling_stop_step`. Profile late: steps 45-50 matches what NVIDIA's own
benchmarking repo uses, and the profiled step costs about 5x normal wall time
(57 s against a 10.2 s steady step), so profiling early would also distort the
throughput you are trying to measure.

[Job 1076681](https://www.internalfb.com/fair_hub/job/FAIR_CW_USE2_1/1076681/details)
produced, from a run whose unprofiled steps held 1674-1679 TFLOP/s/GPU:

| Artifact | Size | |
|---|---:|---|
| `torch_profile/rank-0.json.gz` | 69 MB | 2,138,050 events, 1,001,240 GPU kernels |
| `pytorch_profile/snapshot_0.pickle` | 14 MB | memory snapshot |

The trace loads in Perfetto and reports `deviceProperties` naming four NVIDIA
GB300, with `ncclDevKernel_AllGather_RING_LL` and
`cudnn_generated_fort_native_sdpa_sm100_flash_bprop_mxfp8_*` among the top
kernels by time, so both the EP fabric traffic and the MXFP8 attention path are
captured.

`bridge-fetch` excludes `code/` and `*.tar.gz`. nemo-run's git packager copies
the entire Megatron-Bridge checkout into the results tree -- over 1600 `.py`
files plus docs and gifs -- which is 238 MB of nothing worth analysing. With the
excludes the transfer is 59 MB.

#### Getting there: what failed and why

**Do not pass `-cv large_scale`.** Its own docstring calls it a *"large-scale
proxy (PP=4, VP=4, EP=64)"* and it hard-sets `global_batch_size = 256`. The
suffix-less canonical recipe is the benchmarked one.

`paged_stash` is enabled by `_enable_deepseek_full_iteration_mxfp8` and allocates
five buffers per rank *after* iteration 1, sized from the routing statistics that
iteration observed. That is why every failure looked identical: iteration 1
completes with a sane loss, then a bare `SIGKILL` with no Python traceback and no
CUDA allocator error. Note `--kill-on-bad-exit=1` is in the srun args, so
hundreds of `Killed` lines are one failure cascading.

| Attempt | Change | Result |
|---|---|---|
| 1076143 | `-cv large_scale` | wrong config (GBS 256); died at stash alloc |
| 1076223 | canonical | correct config; died at stash alloc |
| 1076581 | `..._factor_cpu=0.5` | host halved 24.4 -> 12.2 GiB/rank; still died |
| 1076639 | `..._factor_cuda=0.6` | still died |
| 1076664 | `moe_paged_stash=False` | **ran, 101.4% of published** |

Neither buffer-size knob helped, so the sizing was never the problem -- host use
was only 49 GiB/node against `RealMemory=978522` MB. Something about the
`full_iteration` CUDA-graph stash path itself is incompatible with this cluster.
Since disabling it lands *above* the published throughput, the stash is not
buying anything here, but the root cause is still unexplained and worth raising
with cluster support or NVIDIA. NVIDIA's reproduction repo exposes a `DISABLE_CG`
knob for the same area, which suggests this path is known to be fragile.

### Two ways to launch, and the tradeoff

**NVIDIA's own launcher** gives maximum fidelity to the published numbers:

```bash
./cw.sh sh 'cd $HOME/cw/megatron-bridge && uv run python scripts/performance/setup_experiment.py \
  --account faircw-pytorch-access --partition g3 --gpu gb300 \
  --model_family_name deepseek --model_recipe_name deepseek_v3 \
  -ng 256 -c fp8_mx --container_image <sqsh>'
```

It drives NeMo-Run, which generates and submits its own sbatch. **The open
question is topology:** this cluster needs `--segment` to pack a job into one
NVL72 domain, and a NeMo-Run sbatch that omits it can scatter across racks and
report throughput that is worse than the hardware can do. Check the generated
script before trusting any number from it.

**Direct pyxis** keeps `--segment`, the log layout and `$CW_OUTPUT_DIR` from the
rest of this directory, at the cost of reproducing NVIDIA's recipe arguments by
hand. Prefer this only once you have confirmed the argument set matches.

Whichever you pick, the launcher must carry the container flags above, and
`setup_experiment.py` has not been run here yet.

Pin the container tag you are comparing against; NVIDIA quotes throughput per
release. `bridge-info` reports what the image ships so the branch is not a
guess.

### Keeping the comparison honest

- Match sequence length, global batch size, and parallelism, not just GPU count.
  NVIDIA's DeepSeek V3 rows use seq 4096 with PP=2/EP=32; the TorchTitan runs in
  P2469701911 use PP=1 with gradient accumulation.
- **All NVIDIA MoE numbers force-balance expert routing and are token-dropless.**
  A TorchTitan run with real routing is doing strictly more work. Match the
  routing policy or say plainly that you did not.
- Quote a warmed window. Step 1 is CUDA-graph capture, not compute.
- NVIDIA reports no MFU, so compare tokens/sec/GPU and TFLOP/sec/GPU.

An earlier Megatron-LM / Megatron-Bridge / TorchTitan comparison exists at
`fb/megatron/README.md`, but it runs on MAST via Docker and CVT. The model and
data preparation notes there are useful; the launch mechanics do not transfer to
Slurm.

## Topology

`--segment` asks Slurm to pack nodes into one NVL72 domain. `cw.sh submit`
defaults to `min(nodes, 16)` and rejects a node count that is not a multiple of
the segment. Never ask for more than 16: a rack has 18 nodes but is rarely fully
free, so larger segments may never schedule.

| GPUs | Nodes | Segment |
|---:|---:|---:|
| 4 | 1 | 1 |
| 16 | 4 | 4 |
| 64 | 16 | 16 |
| 128 | 32 | 16 (two blocks) |

`cw.sh status` runs `/engshare/bin/check_job_packing.sh`, which should report
`PROPERLY PACKED`.

## Traps worth knowing before you hit them

Everything here is already handled by the scripts, listed so the symptoms are
searchable.

**Environment**

- The system `python3` has no `Python.h`, and Triton JIT-compiles a driver shim
  against it on first use, so every rank dies at startup. `setup_env.sh` uses a
  uv-managed CPython, which ships headers.
- The login shell is zsh with `noclobber`, so `>` onto an existing file fails
  silently. The scripts use `>|` and force `bash -lc` for remote commands.

**Job scripts**

- `export` in the batch script does not reach the ranks without `--export=ALL`
  on `srun`. This is why a `NCCL_DEBUG=INFO` that "does nothing" does nothing.
- `srun` needs `--chdir`, or relative paths like `./tests/assets/tokenizer` fail
  on every node but the first.
- Every cache a rank writes must be node-local: `TRITON_CACHE_DIR`,
  `CUDA_CACHE_PATH`, `HF_HOME`, `XDG_CACHE_HOME`. `XDG_CACHE_HOME` does not cover
  Triton, which defaults to `~/.triton` on NFS and corrupts at 32 ranks with
  `AttributeError: 'CompiledKernel' object has no attribute 'module'`.

**Slurm**

- `QOSGrpGRES` means the account/QoS has no GPU entitlement. Check FAIR Passport.
- `cw.sh submit` runs `sbatch --test-only` first, so a bad account, QoS or
  segment fails immediately instead of sitting in the queue.

**Benchmarking**

- Discard step 1. With CUDA graphs it is capture, not compute: 322 tps/GPU
  against a 1,853 tps/GPU steady state in one measured run.
- Throughput is not flat at scale. It was stable to EP8 but decayed *within* a
  run at EP16 (-25% from step 2 to step 10) and EP32 (-35%). Quote a warmed
  window and say which one.
- FakePG numerics are meaningless by construction. Loss and grad norm from a
  FakePG run prove nothing.

**Model configuration**

- `max_routing_imbalance_factor` must equal the EP degree. It is a recipe field,
  not a CLI flag, so overriding EP on the command line leaves it stale, the
  DistMoE receive buffers under-provisioned, and routing overflow surfaces as
  `CUDA error: unspecified launch failure` mid-training. It fails sooner as EP
  widens, which looks exactly like an EP-scaling kernel bug and is not one.

## Status

Verified end to end on `fair-cw-use2-1` on 2026-08-20 with torch
`2.15.0.dev20260820+cu130` on CPython 3.12.14 aarch64. Both runs used the llama3
debug model for 10 steps and finished `COMPLETED` with exit `0:0`:

| Job | Shape | Result |
|---|---|---|
| 1075673 | 1 node, 4 GPUs, segment 1 | loss 8.12 -> 4.04, 4/4 ranks |
| 1075682 | 2 nodes, 8 GPUs, segment 2 | loss 8.22 -> 4.06, 8/8 ranks, `PROPERLY PACKED` |
| 1075690 | 2 nodes, 8 GPUs, profiling | 16 traces fetched and parsed |

The 2-node runs prove the parts that only break at scale: c10d rendezvous off
the head node, `--export=ALL`, `--chdir`, and node-local caches. No run produced
a stale NFS handle or a Triton cache error. A fetched trace parses clean, with
`deviceProperties[0].name == "NVIDIA GB300"`, 925 GPU kernels and 31 NCCL
kernels including `ncclDevKernel_AllReduce_Sum_u64_RING_LL`.

Megatron-Bridge staging is verified: `setup-bridge` imported the arm64
`nemo:26.08` image (30 GB squashfs) and cloned `r0.6.0`, and `bridge-info` ran
inside the container on a GB300 and reported its versions.

**TODO, unverified:** `build-ao` and `build-torch`. Both need a GB300
allocation for a compile and neither has been run. Treat the sm_103 patch, the
cutlass clone and the uv cache handling in `remote/build_torchao.sh` as
transcribed from P2469701911, not as tested code.

Megatron-Bridge baseline is reproduced at both scales: Qwen3-30B-A3B on 8 GPUs
at 100.4% of published, and DeepSeek V3 on 256 GPUs at 101.4%. The DeepSeek run
needs `model.moe_paged_stash=False`; the reason that path fails here is still
unexplained.

Also not exercised: TorchTitan beyond 2 nodes, and any non-default TorchTitan parallelism.
`hostname --ip-address` returns IPv6 here and c10d accepts it unbracketed; worth
remembering if rendezvous ever fails to parse an endpoint.
