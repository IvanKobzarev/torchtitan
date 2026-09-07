#!/usr/bin/env bash

set -euo pipefail

venv="${CW_VENV:?}"
overlay="${CW_PYTHON_OVERLAY:?}"
requirements="${CW_OVERLAY_REQUIREMENTS:?}"
python="$venv/bin/python"
compat="$venv/lib/cuda-compat-13-1"

export LD_LIBRARY_PATH="$venv/lib:${LD_LIBRARY_PATH:-}"
if [ -d "$compat" ]; then
  export LD_PRELOAD="$compat/libcuda.so.1:$compat/libnvidia-ptxjitcompiler.so.1"
fi

if PYTHONPATH="$overlay" "$python" -c 'import grain.python' 2>/dev/null; then
  echo "==> runtime dependency overlay already available at $overlay"
  exit 0
fi

mkdir -p "$overlay"
"$python" -m pip install \
  --disable-pip-version-check \
  --no-deps \
  --target "$overlay" \
  --requirement "$requirements"

PYTHONPATH="$overlay" "$python" - <<'PY'
import grain

print("grain", grain.__version__)
PY
