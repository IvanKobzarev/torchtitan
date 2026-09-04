#!/usr/bin/env bash
# Build PyTorch from the synced source tree and install it into the cluster
# venv. Runs on a GPU node under srun, launched by cw.sh. Expect one to two
# hours from scratch; ccache makes later builds much cheaper.
#
# Inputs: CW_VENV, CW_TORCH_DIR, CW_CUDA_ARCH.

set -euo pipefail

venv="${CW_VENV:?}"
torch_dir="${CW_TORCH_DIR:?}"
arch="${CW_CUDA_ARCH:?}"
python="$venv/bin/python"

cd "$torch_dir"

echo "==> installing build dependencies"
VIRTUAL_ENV="$venv" uv pip install -r requirements.txt
VIRTUAL_ENV="$venv" uv pip install cmake ninja setuptools wheel

# The rsync from the devserver drops .git, so submodules cannot be restored on
# the cluster. They must already be checked out locally before `cw.sh sync`.
test -f third_party/pybind11/CMakeLists.txt ||
  { echo "third_party is empty: run 'git submodule update --init --recursive' locally, then re-sync" >&2; exit 1; }

echo "==> building for sm_${arch//./}"
export TORCH_CUDA_ARCH_LIST="$arch"
export USE_CUDA=1
export BUILD_TEST=0
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"
export CMAKE_C_COMPILER_LAUNCHER=ccache
export CMAKE_CXX_COMPILER_LAUNCHER=ccache
export CMAKE_CUDA_COMPILER_LAUNCHER=ccache

rm -rf dist
"$python" setup.py bdist_wheel

wheel=$(find dist -name 'torch-*.whl' -print -quit)
echo "==> installing $wheel"
VIRTUAL_ENV="$venv" uv pip install --force-reinstall --no-deps "$wheel"

cd /tmp
"$python" - <<'PY'
import torch

print("torch        ", torch.__version__)
print("arch list    ", torch.cuda.get_arch_list())
print("device       ", torch.cuda.get_device_capability())
PY
