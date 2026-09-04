#!/usr/bin/env bash
# Stage the Megatron-Bridge baseline: import the NeMo container as a squashfs
# image and clone the matching launcher repo.
#
# Runs on a compute node under srun, not on the login node. Converting the
# image's AUFS whiteouts to overlayfs ones needs mknod, and the login pod is a
# Kubernetes container without CAP_MKNOD, so the import dies part way with
# "failed to create ovlfs whiteout: Operation not permitted". The node must be
# aarch64 (partition g3) or enroot resolves the multi-arch tag to amd64.
#
# Inputs: CW_BRIDGE_IMAGE, CW_BRIDGE_SQSH, CW_BRIDGE_REPO, CW_BRIDGE_REPO_URL,
#         CW_BRIDGE_BRANCH.

set -euo pipefail

image="${CW_BRIDGE_IMAGE:?}"
sqsh="${CW_BRIDGE_SQSH:?}"
repo="${CW_BRIDGE_REPO:?}"

mkdir -p "$(dirname "$sqsh")"

case "$(uname -m)" in
  aarch64) ;;
  *) echo "refusing to import on $(uname -m); need an aarch64 node" >&2; exit 1 ;;
esac

# Keep the multi-gigabyte unpack off the shared home filesystem.
export ENROOT_TEMP_PATH="${ENROOT_TEMP_PATH:-/tmp/enroot-$USER}"
export ENROOT_CACHE_PATH="${ENROOT_CACHE_PATH:-/tmp/enroot-cache-$USER}"
mkdir -p "$ENROOT_TEMP_PATH" "$ENROOT_CACHE_PATH"

if [ -f "$sqsh" ]; then
  echo "==> image already present: $sqsh ($(du -h "$sqsh" | cut -f1))"
else
  echo "==> importing docker://$image"
  echo "    18 GiB compressed; expect this to take a while"
  # nvcr.io serves the NeMo image anonymously, so no NGC key is needed.
  enroot import -o "$sqsh" "docker://$image"
  echo "==> imported $sqsh ($(du -h "$sqsh" | cut -f1))"
fi

# Only the performance launchers come from git; the library is in the image.
if [ -d "$repo/.git" ]; then
  echo "==> repo already cloned: $repo"
  git -C "$repo" fetch --quiet origin || true
else
  echo "==> cloning ${CW_BRIDGE_REPO_URL:?}"
  git clone --quiet --filter=blob:none "$CW_BRIDGE_REPO_URL" "$repo"
fi

branch="${CW_BRIDGE_BRANCH:?}"
if git -C "$repo" rev-parse --verify --quiet "origin/$branch" >/dev/null; then
  git -C "$repo" checkout --quiet -B "$branch" "origin/$branch"
  echo "==> repo on $branch ($(git -C "$repo" rev-parse --short HEAD))"
else
  echo "!!  branch '$branch' not found; leaving repo on $(git -C "$repo" rev-parse --abbrev-ref HEAD)"
  echo "    available:"
  git -C "$repo" branch -r --list 'origin/r*' | tail -8
fi

echo
echo "Next: cw.sh bridge-info   (reads the versions the image actually ships)"
