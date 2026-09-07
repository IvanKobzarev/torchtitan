#!/usr/bin/env bash
# Drive TorchTitan runs on the FAIR CoreWeave GB300 clusters. See README.md.

set -euo pipefail

CW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$CW_DIR/lib.sh"

usage() {
  cat <<EOF
usage: cw.sh <command> [args]

connection
  connect              open the Duo-backed SSH control master
  disconnect           close it
  doctor               report local/remote compatibility and entitlement
  sh <cmd>...          run a command on the login node

stack
  sync                 rsync the configured source trees to the cluster
  setup-env            create or refresh the cluster venv
  setup-overlay        install external TorchTitan-only Python dependencies
  build-torch          build PyTorch from the synced source on a GB300 node  [TODO: unverified]
  build-ao             build torchao's extensions for sm_${CW_CUDA_ARCH//./}         [TODO: unverified]

megatron bridge baseline
  setup-bridge         import the NeMo container and clone the launcher repo
  bridge-info          print the versions and recipes the container ships
  submit-bridge [opts] -- <launcher args>   run a Megatron-Bridge benchmark
      --name NAME      log directory under CW_OUTPUT_DIR, default bridge
      --profile        PyTorch profiler over --profile-steps
      --profile-steps A:B   default 45:50
      --dry-run        print the launcher command instead of running it
  bridge-fetch <name>  pull a Megatron-Bridge run's logs and traces back

jobs
  submit [opts] -- <train args>   render and submit a job
      --nodes N        default 1
      --time HH:MM:SS  default 00:30:00
      --name NAME      default torchtitan
      --segment N      default min(nodes, 16)
      --dry-run        print the batch script instead of submitting
  status <job>         queue state, placement and accounting
  logs <job>           follow stdout and stderr
  fetch <job>          pull traces and outputs back to CW_LOCAL_OUTPUT_DIR
  cancel <job>         scancel
  url <job>            print the FAIR Hub link for a job

cluster: $CW_CLUSTER   account: $CW_ACCOUNT   qos: $CW_QOS
EOF
}

# ------------------------------------------------------------------ doctor ---

cmd_doctor() {
  local local_arch local_glibc local_cap
  local_arch="$(uname -m)"
  local_glibc="$(ldd --version | head -n 1 | awk '{print $NF}')"
  local_cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null |
    head -n 1 || echo none)"

  echo "local"
  echo "  arch          $local_arch"
  echo "  glibc         $local_glibc"
  echo "  gpu cap       ${local_cap:-none}"

  # aarch64 unless the login pod says otherwise; only the wheel path needs the
  # exact remote values, and it is worth reporting the rest offline.
  local remote_arch="aarch64" remote_glibc="(unknown)"
  if cw_master_alive; then
    cw_resolve_paths
    local remote
    remote="$(cw_bash '
      echo "arch=$(uname -m)"
      echo "glibc=$(ldd --version | head -n 1 | awk "{print \$NF}")"
    ')"
    remote_arch="$(sed -n 's/^arch=//p' <<<"$remote")"
    remote_glibc="$(sed -n 's/^glibc=//p' <<<"$remote")"

    echo "remote ($CW_CLUSTER login pod)"
    echo "  arch          $remote_arch"
    echo "  glibc         $remote_glibc"
    echo "  venv          $CW_VENV"
    cw_bash "test -x '$CW_VENV/bin/python' &&
      cd /tmp && '$CW_VENV/bin/python' -c \
      'import torch;print(\"  torch         \",torch.__version__, torch.cuda.get_arch_list())' \
      2>/dev/null" || echo "  torch         (not installed)"

    echo "entitlement"
    cw_bash "sacctmgr -n show assoc where user=\$USER format=Account%30,Partition%12,QOS%30" ||
      warn "sacctmgr query failed"
  else
    warn "no control master; skipping the remote half. Run 'cw.sh connect'."
  fi

  echo "binary portability (CW_TORCH_MODE=$CW_TORCH_MODE)"
  echo "  torchao always compiles on a GB300 node, whatever this devserver is."
  if [ "$CW_TORCH_MODE" != wheel ]; then
    echo "  torch ships as a wheel built elsewhere, so nothing is compiled here and"
    echo "  this devserver's $local_arch / GPU $local_cap does not have to match."
  elif [ "$local_arch" != "$remote_arch" ]; then
    warn "local $local_arch != remote $remote_arch: nothing compiled here runs on"
    warn "the cluster. Use CW_TORCH_MODE=nightly or CW_TORCH_MODE=source."
  else
    echo "  arch matches; a local wheel is portable if its glibc is <= $remote_glibc"
    echo "  and it was built for Python $CW_PYTHON_VERSION."
    if [ "$local_cap" != "$CW_CUDA_ARCH" ]; then
      warn "local GPU is $local_cap but the cluster is $CW_CUDA_ARCH: build with"
      warn "TORCH_CUDA_ARCH_LIST=\"$local_cap;$CW_CUDA_ARCH\" or sm_${CW_CUDA_ARCH//./} will be missing."
    fi
  fi
}

# -------------------------------------------------------------------- sync ---

cmd_sync() {
  cw_require_master
  cw_resolve_paths
  cw_bash "mkdir -p '$CW_REMOTE_ROOT' '$CW_LOG_DIR'"

  local entry src name dst
  for entry in "${CW_PYTHONPATH_TREES[@]}" "${CW_BUILD_TREES[@]}"; do
    src="${entry%:*}"
    name="${entry##*:}"
    [ -d "$src" ] || die "local tree not found: $src"
    dst="$CW_REMOTE_ROOT/$name"
    log "sync $src -> $dst"
    cw_rsync "$src/" "$dst/"
  done

  cw_compile_trees
}

# Precompile once from a single process. Ranks cannot do it themselves because
# PYTHONDONTWRITEBYTECODE is set in the job to keep them off the shared home.
cw_compile_trees() {
  if ! cw_bash "test -x '$CW_VENV/bin/python'"; then
    warn "no venv yet; skipping bytecode. Run 'cw.sh setup-env'."
    return 0
  fi
  log "precompiling bytecode"
  local entry name
  for entry in "${CW_PYTHONPATH_TREES[@]}" "${CW_BUILD_TREES[@]}"; do
    name="${entry##*:}"
    cw_bash "cd '$CW_REMOTE_ROOT/$name' &&
      '$CW_VENV/bin/python' -m compileall -q . >/dev/null 2>&1 || true"
  done
}

cmd_sync_torch() {
  cw_require_master
  cw_resolve_paths
  [ -d "$CW_TORCH_SRC" ] || die "CW_TORCH_SRC not found: $CW_TORCH_SRC"
  local dst="$CW_REMOTE_ROOT/pytorch"
  log "sync $CW_TORCH_SRC -> $dst (submodules included, build output excluded)"
  cw_bash "mkdir -p '$dst'"
  cw_rsync "$CW_TORCH_SRC/" "$dst/"
}

# ------------------------------------------------------------------- stack ---

cw_run_remote_script() {
  local script="$1"
  shift
  local remote
  remote="$CW_REMOTE_ROOT/.scripts/$(basename "$script")"
  cw_bash "mkdir -p '$CW_REMOTE_ROOT/.scripts'"
  cw_push_file "$CW_DIR/remote/$script" "$remote"
  cw_bash "chmod +x '$remote' && $* '$remote'"
}

cmd_setup_env() {
  cw_require_master
  cw_resolve_paths
  cw_run_remote_script setup_env.sh \
    "CW_VENV='$CW_VENV'" \
    "CW_PYTHON_VERSION='$CW_PYTHON_VERSION'" \
    "CW_TORCH_MODE='$CW_TORCH_MODE'" \
    "CW_TORCH_NIGHTLY_INDEX='$CW_TORCH_NIGHTLY_INDEX'" \
    "CW_TORCH_WHEEL='$CW_TORCH_WHEEL'" \
    "CW_REQUIREMENTS='$(cw_workdir)/requirements.txt'" \
    bash
  cw_compile_trees
}

cmd_setup_overlay() {
  cw_require_master
  cw_resolve_paths
  cw_run_remote_script setup_runtime_overlay.sh \
    "CW_VENV='$CW_VENV'" \
    "CW_PYTHON_OVERLAY='$CW_PYTHON_OVERLAY'" \
    "CW_OVERLAY_REQUIREMENTS='$(cw_workdir)/fb/fair_coreweave/requirements-runtime-overlay.txt'" \
    bash
}

# Compilation needs a GB300 so nvcc and torch see the real compute capability.
cw_gpu_srun() {
  printf "srun --account='%s' --qos='%s' --partition='%s' --nodes=1 --ntasks=1 \
--gpus-per-node=1 --cpus-per-task='%s' --time='%s'" \
    "$CW_ACCOUNT" "$CW_QOS" "$CW_PARTITION" "$CW_CPUS_PER_TASK" "${CW_BUILD_TIME:-04:00:00}"
}

# The job's cwd, so relative paths such as ./tests/assets/tokenizer resolve.
cw_workdir() {
  local entry
  for entry in "${CW_PYTHONPATH_TREES[@]}"; do
    case "${entry##*:}" in torchtitan) printf '%s' "$CW_REMOTE_ROOT/torchtitan"; return ;; esac
  done
  printf '%s' "$CW_REMOTE_ROOT/${CW_PYTHONPATH_TREES[0]##*:}"
}

cmd_build_ao() {
  cw_require_master
  cw_resolve_paths
  local ao_dir="" entry
  for entry in "${CW_BUILD_TREES[@]}"; do
    case "${entry##*:}" in ao|torchao) ao_dir="$CW_REMOTE_ROOT/${entry##*:}" ;; esac
  done
  [ -n "$ao_dir" ] || die "no 'ao' entry in CW_BUILD_TREES"
  cw_run_remote_script build_torchao.sh \
    "$(cw_gpu_srun)" \
    "/usr/bin/env CW_VENV='$CW_VENV' CW_AO_DIR='$ao_dir' CW_CUDA_ARCH='$CW_CUDA_ARCH'" \
    "CW_CUTLASS_URL='$CW_CUTLASS_URL' CW_CUTLASS_SHA='$CW_CUTLASS_SHA'" \
    bash
}

# ------------------------------------------------------- megatron bridge ---

cw_bridge_sqsh() {
  printf '%s/nemo-%s.sqsh' "$CW_CONTAINER_DIR" "$CW_BRIDGE_TAG"
}

cmd_setup_bridge() {
  cw_require_master
  cw_resolve_paths
  cw_run_remote_script setup_megatron_bridge.sh \
    "$(cw_gpu_srun)" \
    "/usr/bin/env CW_BRIDGE_IMAGE='$CW_BRIDGE_IMAGE'" \
    "CW_BRIDGE_SQSH='$(cw_bridge_sqsh)'" \
    "CW_BRIDGE_REPO='$CW_BRIDGE_REPO'" \
    "CW_BRIDGE_REPO_URL='$CW_BRIDGE_REPO_URL'" \
    "CW_BRIDGE_BRANCH='$CW_BRIDGE_BRANCH'" \
    bash
}

# Container flags that make a pyxis step survive this cluster's TaskProlog.
cw_container_args() {
  printf -- "--container-image='%s' --container-writable --container-mounts='%s' --export=%s" \
    "$(cw_bridge_sqsh)" "$CW_CONTAINER_MOUNTS" "$CW_CONTAINER_EXPORT"
}

# Runs inside the container on a GPU node, which is the only place the shipped
# versions can be read authoritatively.
cmd_bridge_info() {
  cw_require_master
  cw_resolve_paths
  local sqsh
  sqsh="$(cw_bridge_sqsh)"
  cw_bash "test -f '$sqsh'" || die "no container at $sqsh; run 'cw.sh setup-bridge'"
  cw_bash "$(cw_gpu_srun) $(cw_container_args) python -c \"
import torch, importlib
print('torch          ', torch.__version__)
print('arch list      ', torch.cuda.get_arch_list())
for m in ('megatron.bridge', 'megatron.core', 'transformer_engine'):
    try:
        mod = importlib.import_module(m)
        print(f'{m:18}', getattr(mod, '__version__', 'installed'))
    except Exception as e:
        print(f'{m:18} FAILED:', type(e).__name__)
\" 2>&1 | grep -vE 'UserWarning|warnings.warn|registry.py|^  '"
}

# Megatron-Bridge's launcher owns job submission, so the cluster-specific flags
# live here rather than in templates/torchtitan.sbatch.
cw_bridge_launcher_args() {
  printf -- "-a %s -p %s --gres gpu:%s --additional_slurm_params qos=%s -i %s \
-cm %s -E DISABLE_PODMAN=1 -E AIRSTORE_BBFS_FUSE_DISABLED=1" \
    "$CW_ACCOUNT" "$CW_PARTITION" "$CW_GPUS_PER_NODE" "$CW_QOS" \
    "$(cw_bridge_sqsh)" "$CW_CONTAINER_MOUNTS"
}

cmd_submit_bridge() {
  local name=bridge profile=0 steps="45:50" dry=0
  while [ $# -gt 0 ]; do
    case "$1" in
      --name)          name="$2"; shift 2 ;;
      --profile)       profile=1; shift ;;
      --profile-steps) steps="$2"; shift 2 ;;
      --dry-run)       dry=1; shift ;;
      --)              shift; break ;;
      *)               die "unknown submit-bridge option: $1" ;;
    esac
  done
  [ $# -gt 0 ] || die "no launcher arguments; pass them after --"

  cw_require_master
  cw_resolve_paths
  local log_dir="$CW_OUTPUT_DIR/bridge/$name"
  local extra=""
  if [ "$profile" = 1 ]; then
    extra="-pyp True --profiling_start_step ${steps%%:*} --profiling_stop_step ${steps##*:}"
  fi

  local cmd
  cmd="cd '$CW_BRIDGE_REPO' && export PATH=\$HOME/.local/bin:\$PATH &&
    mkdir -p '$log_dir' &&
    uv run --no-project --with nemo-run==0.10.0 python \
      scripts/performance/setup_experiment.py \
      $(cw_bridge_launcher_args) -l '$log_dir' $extra $*"

  if [ "$dry" = 1 ]; then printf '%s\n' "$cmd"; return 0; fi
  log "submit-bridge '$name' -> $log_dir"
  cw_bash "$cmd"
}

# nemo-run buries results under <log_dir>/experiments/<exp>/<run>/<task>/, so pull
# the whole tree rather than guessing the path.
cmd_bridge_fetch() {
  local name="${1:-bridge}"
  cw_require_master
  cw_resolve_paths
  local remote="$CW_OUTPUT_DIR/bridge/$name"
  cw_bash "test -d '$remote'" || die "nothing at $remote; check --name"
  local dest="$CW_LOCAL_OUTPUT_DIR/bridge-$name"
  mkdir -p "$dest"
  log "fetch $remote -> $dest"
  # nemo-run's git packager snapshots the whole repo into the results tree; that
  # is over a thousand files of source and docs with nothing to analyse in them.
  cw_rsync_from "$remote/" "$dest/" --exclude 'code/' --exclude '*.tar.gz'
  find "$dest" -type f \( -name '*.json.gz' -o -name '*.nsys-rep' \
    -o -name '*.pickle' \) -printf '%10s  %p\n' | sort -k2
}

cmd_build_torch() {
  cw_require_master
  cw_resolve_paths
  cmd_sync_torch
  cw_run_remote_script build_torch.sh \
    "$(cw_gpu_srun)" \
    "/usr/bin/env CW_VENV='$CW_VENV' CW_TORCH_DIR='$CW_REMOTE_ROOT/pytorch'" \
    "CW_CUDA_ARCH='$CW_CUDA_ARCH'" \
    bash
}

# -------------------------------------------------------------------- jobs ---

cmd_submit() {
  local nodes=1 time="00:30:00" name="torchtitan" segment="" dry=0
  while [ $# -gt 0 ]; do
    case "$1" in
      --nodes)   nodes="$2"; shift 2 ;;
      --time)    time="$2"; shift 2 ;;
      --name)    name="$2"; shift 2 ;;
      --segment) segment="$2"; shift 2 ;;
      --dry-run) dry=1; shift ;;
      --)        shift; break ;;
      *)         die "unknown submit option: $1" ;;
    esac
  done
  [ $# -gt 0 ] || die "no training arguments; pass them after --"

  # A rack holds 18 nodes but is never fully free, so 16 is the largest segment
  # that schedules reliably.
  if [ -z "$segment" ]; then
    segment=$(( nodes < 16 ? nodes : 16 ))
  fi
  [ $(( nodes % segment )) -eq 0 ] ||
    die "--nodes ($nodes) must be a multiple of --segment ($segment)"

  # A dry run only renders, so it works without a cluster connection. The paths
  # it prints keep a literal $HOME, which Slurm does not expand in #SBATCH lines.
  if [ "$dry" = 0 ] || cw_master_alive; then
    cw_require_master
    cw_resolve_paths
  fi

  local rendered
  rendered="$(python3 - "$CW_DIR/templates/torchtitan.sbatch" "$@" <<PY
import shlex, sys

template, *train_args = sys.argv[1:]
with open(template) as f:
    text = f.read()


def render_args(args):
    """One '--flag value' pair per continued line."""
    lines, i = [], 0
    while i < len(args):
        chunk = [args[i]]
        if args[i].startswith("-") and i + 1 < len(args) and not args[i + 1].startswith("-"):
            chunk.append(args[i + 1])
            i += 1
        lines.append(" ".join(shlex.quote(a) for a in chunk))
        i += 1
    return " \\\\\n    ".join(lines)

subs = {
    "JOB_NAME": "$name",
    "ACCOUNT": "$CW_ACCOUNT",
    "QOS": "$CW_QOS",
    "PARTITION": "$CW_PARTITION",
    "NODES": "$nodes",
    "GPUS_PER_NODE": "$CW_GPUS_PER_NODE",
    "CPUS_PER_TASK": "$CW_CPUS_PER_TASK",
    "TIME": "$time",
    "SEGMENT": "$segment",
    "LOG_DIR": "$CW_LOG_DIR",
    "OUTPUT_DIR": "$CW_OUTPUT_DIR",
    "WORKDIR": "$(cw_workdir)",
    "VENV": "$CW_VENV",
    "PYTHONPATH": "$(cw_remote_pythonpath)",
    "TRAIN_ARGS": render_args(train_args),
}
for key, value in subs.items():
    text = text.replace(f"@{key}@", value)
sys.stdout.write(text)
PY
)"

  if [ "$dry" = 1 ]; then
    printf '%s\n' "$rendered"
    return 0
  fi

  local remote="$CW_REMOTE_ROOT/.jobs/$name.sbatch"
  cw_bash "mkdir -p '$CW_REMOTE_ROOT/.jobs' '$CW_LOG_DIR' '$CW_OUTPUT_DIR'"
  printf '%s\n' "$rendered" | cw_ssh "cat >| '$remote'"

  log "test-only scheduling check"
  cw_bash "sbatch --test-only '$remote'" || die "Slurm rejected the request"

  local job_id
  job_id="$(cw_bash "sbatch --parsable '$remote'" | tr -d '\r')"
  log "submitted job $job_id"
  log "$(cw_job_url "$job_id")"
  echo "$job_id"
}

cw_job_id() {
  [ $# -ge 1 ] && [ -n "$1" ] || die "missing job id"
  printf '%s' "$1"
}

cmd_status() {
  local job
  job="$(cw_job_id "$@")"
  cw_require_master
  echo "$(cw_job_url "$job")"
  cw_bash "squeue -j '$job' -o '%.18i %.2t %.10M %.6D %R' || true
    /engshare/bin/check_job_packing.sh '$job' 2>/dev/null || true
    sacct -j '$job' --format=JobID,JobName%30,State,ExitCode,Elapsed,AllocTRES%60"
}

cmd_logs() {
  local job
  job="$(cw_job_id "$@")"
  cw_require_master
  cw_resolve_paths
  cw_bash "tail -F '$CW_LOG_DIR'/*-'$job'.out '$CW_LOG_DIR'/*-'$job'.err"
}

# Traces are one gzipped JSON per rank per profiled step, so pull the whole
# output directory rather than guessing paths.
cmd_fetch() {
  local job
  job="$(cw_job_id "$@")"
  cw_require_master
  cw_resolve_paths
  local remote="$CW_OUTPUT_DIR/$job"
  if ! cw_bash "test -d '$remote'"; then
    die "no outputs at $remote (did the job enable profiling?)"
  fi
  local dest="$CW_LOCAL_OUTPUT_DIR/$job"
  mkdir -p "$dest"
  log "fetch $remote -> $dest"
  cw_rsync_from "$remote/" "$dest/"
  find "$dest" -name '*trace.json.gz' | sort
}

cmd_cancel() {
  local job
  job="$(cw_job_id "$@")"
  cw_require_master
  cw_bash "scancel '$job'"
}

# -------------------------------------------------------------------------- --

case "${1:-}" in
  connect)     shift; cw_connect ;;
  disconnect)  shift; cw_disconnect ;;
  doctor)      shift; cmd_doctor ;;
  sh)          shift; cw_require_master; cw_bash "$*" ;;
  sync)        shift; cmd_sync ;;
  setup-env)   shift; cmd_setup_env ;;
  setup-overlay) shift; cmd_setup_overlay ;;
  build-torch) shift; cmd_build_torch ;;
  build-ao)    shift; cmd_build_ao ;;
  setup-bridge) shift; cmd_setup_bridge ;;
  bridge-info) shift; cmd_bridge_info ;;
  submit-bridge) shift; cmd_submit_bridge "$@" ;;
  bridge-fetch) shift; cmd_bridge_fetch "$@" ;;
  submit)      shift; cmd_submit "$@" ;;
  status)      shift; cmd_status "$@" ;;
  logs)        shift; cmd_logs "$@" ;;
  fetch)       shift; cmd_fetch "$@" ;;
  cancel)      shift; cmd_cancel "$@" ;;
  url)         shift; cw_job_url "$@"; echo ;;
  ""|-h|--help|help) usage ;;
  *)           usage; exit 1 ;;
esac
