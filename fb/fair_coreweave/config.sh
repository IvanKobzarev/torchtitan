# Settings for the FAIR CoreWeave helper scripts. Sourced by lib.sh.
# Override any value from the environment: CW_CLUSTER=fair-cw-use2-3 ./cw.sh doctor
# shellcheck disable=SC2034  # consumed by lib.sh and cw.sh

# ---------------------------------------------------------------- cluster ---

: "${CW_CLUSTER:=fair-cw-use2-1}"
: "${CW_USER:=$USER}"

# Slurm entitlement. Confirm yours on the cluster with:
#   sacctmgr show assoc where user="$USER" format=Cluster,Account,Partition,QOS%40
: "${CW_ACCOUNT:=faircw-pytorch-access}"
: "${CW_QOS:=g3_lowest}"
: "${CW_PARTITION:=g3}"

: "${CW_GPUS_PER_NODE:=4}"
: "${CW_CPUS_PER_TASK:=128}"

# GB300 compute capability. Native extensions must target this explicitly.
: "${CW_CUDA_ARCH:=10.3}"

# ------------------------------------------------------- remote workspace ---

# Everything below lives on the shared home filesystem, visible from every node.
: "${CW_REMOTE_ROOT:=\$HOME/cw}"
: "${CW_VENV:=\$HOME/venvs/cw}"
: "${CW_PYTHON_OVERLAY:=\$HOME/cw/overlays/torchtitan-public}"
: "${CW_LOG_DIR:=\$HOME/cw/logs}"

# Job outputs (traces, memory snapshots, checkpoints) live outside the synced
# trees. `cw.sh sync` rsyncs with --delete, so anything written inside a tree is
# destroyed by the next sync.
: "${CW_OUTPUT_DIR:=\$HOME/cw/outputs}"
# Outside the repo on purpose: traces are multi-megabyte binaries and this
# directory sits inside an fbsource checkout.
: "${CW_LOCAL_OUTPUT_DIR:=$HOME/cw-outputs}"
: "${CW_PYTHON_VERSION:=3.12}"

# ----------------------------------------------------------------- pytorch ---

# nightly     download a public wheel on the cluster (fast, no local torch changes)
# source      rsync CW_TORCH_SRC and build it on a GB300 node (slow, ships your changes)
# wheel       install the prebuilt CW_TORCH_WHEEL (only if it matches the cluster ABI)
: "${CW_TORCH_MODE:=nightly}"
: "${CW_TORCH_NIGHTLY_INDEX:=https://download.pytorch.org/whl/nightly/cu130}"
: "${CW_TORCH_SRC:=/data/users/$USER/fbsource/fbcode/caffe2}"
: "${CW_TORCH_WHEEL:=}"

# ------------------------------------------------------------ source trees ---

# Pure Python trees. Synced as-is and prepended to PYTHONPATH in the job, in the
# order listed. Never pip-install these too; a site-packages copy shadows them.
#   "<local path>:<name under CW_REMOTE_ROOT>"
: "${CW_TORCHTITAN_SRC:=/home/$USER/local/c/torchtitan}"
: "${CW_DIST_MOE_SRC:=/data/users/$USER/fbsource/genai/msl/dist_moe}"
CW_PYTHONPATH_TREES=(
  "$CW_TORCHTITAN_SRC:torchtitan"
  "$CW_DIST_MOE_SRC:dist_moe"
)

# Trees with native code. Synced without build artifacts, then compiled on a
# GB300 node by `cw.sh build-ao`. Also prepended to PYTHONPATH.
CW_BUILD_TREES=(
  "/home/$USER/fbsource/fbcode/pytorch/ao:ao"
)

# torchao vendors cutlass as a submodule that fbsource does not materialize, so
# the build script clones it at this revision.
: "${CW_CUTLASS_URL:=https://github.com/NVIDIA/cutlass.git}"
: "${CW_CUTLASS_SHA:=e51efbfe18fe4f4cbb66ab814c55bf4aa0185491}"

# -------------------------------------------------- megatron bridge baseline ---

# NVIDIA ships Megatron-Bridge inside the NeMo container and quotes its published
# throughput per container release, so pin the tag you are comparing against.
# The image is multi-arch; enroot pulls the arm64 variant that GB300 needs.
: "${CW_BRIDGE_TAG:=26.08}"
: "${CW_BRIDGE_IMAGE:=nvcr.io#nvidia/nemo:$CW_BRIDGE_TAG}"
: "${CW_CONTAINER_DIR:=\$HOME/cw/containers}"

# The repo is only needed for the performance recipe launchers; the library
# itself lives in the image. Branch must match the container: 26.04 pairs with
# r0.4.0. Run `cw.sh bridge-info` to see what the image actually ships.
: "${CW_BRIDGE_REPO:=\$HOME/cw/megatron-bridge}"
: "${CW_BRIDGE_REPO_URL:=https://github.com/NVIDIA-NeMo/Megatron-Bridge.git}"
: "${CW_BRIDGE_BRANCH:=r0.6.0}"

# Slurm's TaskProlog runs inside the container and is not container-aware. It
# runs under `set -e -o pipefail` and shells out to host sidecars that need jq,
# scuba_cat and a fuse mount, so every containerized step dies with
# "TaskProlog failed status=1" before your command runs. These are the two
# opt-outs the prolog documents itself, plus the host paths it reads.
: "${CW_CONTAINER_MOUNTS:=/etc/slurm:/etc/slurm:ro,/public:/public:ro,/engshare:/engshare:ro}"
: "${CW_CONTAINER_EXPORT:=ALL,DISABLE_PODMAN=1,AIRSTORE_BBFS_FUSE_DISABLED=1}"

# ------------------------------------------------------------------- sync ---

# Excluding build output is not an optimization. A locally built .so is compiled
# for this devserver's CPU and GPU; letting one reach the cluster produces import
# errors or silent wrong-architecture kernels.
# Never deleted from the cluster even when absent locally. --delete is required
# so a renamed or removed module cannot linger and still be importable, but it
# must not be able to destroy anything a job produced.
CW_RSYNC_PROTECT=(
  "outputs/"
  "outputs/**"
  "profiling/"
  "profiling/**"
  "checkpoint/"
  "checkpoint/**"
  "structured_logs/**"
  "*.json.gz"
  "*.pt"
  "*.safetensors"
)

CW_RSYNC_EXCLUDES=(
  ".git" ".sl" ".jj"
  "__pycache__" "*.pyc" "*.pyo"
  "*.so" "*.o" "*.a" "*.dylib"
  "build/" "dist/" "*.egg-info" ".eggs"
  ".pytest_cache" ".mypy_cache" ".ruff_cache"
  "agent_space/" "outputs/" "wandb/" "*.pt" "*.safetensors"
)
