#!/usr/bin/env bash
# Compile torchao's CUDA extensions for GB300 in the synced source tree.
# Runs on a GPU node under srun, launched by cw.sh.
#
# Inputs: CW_VENV, CW_AO_DIR, CW_CUDA_ARCH, CW_CUTLASS_URL, CW_CUTLASS_SHA.

set -euo pipefail

venv="${CW_VENV:?}"
ao_dir="${CW_AO_DIR:?}"
arch="${CW_CUDA_ARCH:?}"
python="$venv/bin/python"

cd "$ao_dir"

# fbsource does not materialize this submodule, so setup.py exits immediately.
cutlass="third_party/cutlass"
if [ ! -f "$cutlass/include/cutlass/cutlass.h" ]; then
  echo "==> cloning cutlass at ${CW_CUTLASS_SHA:?}"
  rm -rf "$cutlass"
  git clone --filter=blob:none "${CW_CUTLASS_URL:?}" "$cutlass"
  git -C "$cutlass" checkout --detach "$CW_CUTLASS_SHA"
fi

# The mxfp8 extension hardcodes its gencode list and stops at sm_100. An sm_100
# cubin does not load on sm_103, and the compute_120 PTX cannot JIT down to it,
# so without this the kernels are simply absent on GB300.
sm="${arch//./}"
if ! grep -q "code=sm_$sm" setup.py; then
  echo "==> adding sm_$sm to setup.py"
  "$python" - "$sm" <<'PY'
import sys

sm = sys.argv[1]
anchor = '"-gencode=arch=compute_100,code=sm_100",\n'
added = f'"-gencode=arch=compute_{sm},code=sm_{sm}",\n'

with open("setup.py") as f:
    text = f.read()

stripped = anchor.strip()
out = []
for line in text.splitlines(keepends=True):
    out.append(line)
    if line.strip() == stripped:
        indent = line[: len(line) - len(line.lstrip())]
        out.append(indent + added)

if len(out) == len(text.splitlines(keepends=True)):
    raise SystemExit("gencode anchor not found in setup.py; patch it by hand")

with open("setup.py", "w") as f:
    f.writelines(out)
PY
fi

# uv happily serves a cached wheel built without the patch above, and the build
# then appears to succeed while compiling nothing.
echo "==> clearing cached torchao wheels"
uv cache clean torchao || true

echo "==> building extensions in place"
export TORCH_CUDA_ARCH_LIST="$arch"
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"
export USE_CPP=1
"$python" setup.py build_ext --inplace 2>&1 | tee /tmp/torchao-build.log

grep -q "Building mxfp8_cuda extension" /tmp/torchao-build.log ||
  echo "!!  'Building mxfp8_cuda extension' missing from the log; nothing compiled"

echo "==> verifying"
cd /tmp  # verifying from inside the tree passes even when the build failed
PYTHONPATH="$ao_dir" "$python" - <<'PY'
import torch
import torchao

# Registration happens on import of the kernels module; plain `import torchao`
# reports the ops as missing even when they are there.
import torchao.prototype.mx_formats.kernels  # noqa: F401

print("torchao      ", torchao.__file__)
print("device       ", torch.cuda.get_device_capability())
for op in ("mxfp8_quantize", "mxfp8_quantize_out"):
    print(f"{op:13}", hasattr(torch.ops.torchao, op))
PY
