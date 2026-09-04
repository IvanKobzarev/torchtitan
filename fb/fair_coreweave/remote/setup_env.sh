#!/usr/bin/env bash
# Build the cluster virtualenv. Runs on the login node, pushed there by cw.sh.
#
# Inputs come from the environment: CW_VENV, CW_PYTHON_VERSION, CW_TORCH_MODE,
# CW_TORCH_NIGHTLY_INDEX, CW_TORCH_WHEEL, CW_REMOTE_ROOT, CW_PYTHONPATH.

set -euo pipefail

venv="${CW_VENV:?}"
py_version="${CW_PYTHON_VERSION:?}"
torch_mode="${CW_TORCH_MODE:?}"

export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
  echo "==> installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# The system interpreter ships no Python.h, and Triton JIT-compiles a driver
# shim against it on first use, so every rank dies at startup. A uv-managed
# CPython includes the headers.
echo "==> provisioning CPython $py_version"
uv python install "$py_version"

if [ ! -x "$venv/bin/python" ]; then
  echo "==> creating $venv"
  uv venv --python "$py_version" --python-preference only-managed "$venv"
fi

python="$venv/bin/python"
uvpip() { VIRTUAL_ENV="$venv" uv pip "$@"; }

echo "==> installing torch ($torch_mode)"
case "$torch_mode" in
  nightly)
    uvpip install --pre torch --index-url "${CW_TORCH_NIGHTLY_INDEX:?}"
    ;;
  wheel)
    uvpip install --force-reinstall "${CW_TORCH_WHEEL:?}"
    ;;
  source)
    if "$python" -c 'import torch' 2>/dev/null; then
      echo "    torch already present; run 'cw.sh build-torch' to rebuild"
    else
      echo "    no torch yet; run 'cw.sh build-torch'"
    fi
    ;;
  *)
    echo "unknown CW_TORCH_MODE: $torch_mode" >&2
    exit 1
    ;;
esac

# uv venv seeds nothing, so `setup.py build_ext` for torchao would not find
# setuptools. ninja keeps the nvcc fan-out parallel.
echo "==> installing build tooling"
uvpip install setuptools wheel ninja cmake

# The tree pins its own versions, including spmd_types, which a hand-written
# list gets wrong.
requirements="${CW_REQUIREMENTS:-}"
if [ -n "$requirements" ] && [ -f "$requirements" ]; then
  echo "==> installing dependencies from $requirements"
  uvpip install -r "$requirements"
else
  echo "!!  no requirements.txt at '$requirements'; run 'cw.sh sync' first"
fi

# torchao comes from the source tree on PYTHONPATH. A PyPI copy in site-packages
# wins over PYTHONPATH and silently replaces the extension built for sm_103.
if uvpip show torchao >/dev/null 2>&1; then
  echo "==> removing site-packages torchao so the source tree is used"
  uvpip uninstall torchao
fi

echo "==> environment"
cd /tmp  # a source tree on the cwd shadows the installed package
"$python" - <<'PY'
import platform
import torch

print("python       ", platform.python_version(), platform.machine())
print("torch        ", torch.__version__)
print("torch cuda   ", torch.version.cuda)
print("arch list    ", torch.cuda.get_arch_list())
PY
