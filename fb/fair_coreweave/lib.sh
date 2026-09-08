# Shared helpers for the FAIR CoreWeave scripts. Source this, do not run it.

set -euo pipefail

CW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.sh
source "$CW_DIR/config.sh"

CW_BASTION="$CW_USER@bastion.$CW_CLUSTER.cw.metafb.cloud"
CW_LOGIN="$CW_USER@$CW_USER.login.$CW_CLUSTER.cw.metafb.cloud"
CW_CTL="/tmp/cw-$CW_CLUSTER-$CW_USER.ctl"

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*" >&2; }
warn() { printf '\033[1;33m!!\033[0m  %s\n' "$*" >&2; }
die()  { printf '\033[1;31mxx\033[0m  %s\n' "$*" >&2; exit 1; }

# `cloud hpc login` puts the minted certificate into an ssh-agent and nowhere
# else, so with no agent running the login fails and leaves no credential behind.
# Devservers ship the agent as an inactive systemd user unit.
cw_ensure_agent() {
  local sock="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/ssh-agent.socket"
  if [ -n "${SSH_AUTH_SOCK:-}" ] && ssh-add -l >/dev/null 2>&1; then
    return 0
  fi
  if [ ! -S "$sock" ]; then
    log "starting the ssh-agent systemd user unit"
    systemctl --user start ssh-agent.service ||
      die "could not start ssh-agent; see https://fburl.com/ssh-agent-troubleshooting"
  fi
  export SSH_AUTH_SOCK="$sock"
}

# FAIR Hub names clusters in upper snake case: fair-cw-use2-1 -> FAIR_CW_USE2_1.
cw_job_url() {
  local cluster="${CW_CLUSTER//-/_}"
  printf 'https://www.internalfb.com/fair_hub/job/%s/%s/details' \
    "${cluster^^}" "${1:?job id}"
}

cw_ssh_opts() {
  printf '%s\n' \
    -F /etc/ssh/ssh_config \
    -o StrictHostKeyChecking=accept-new \
    -o PasswordAuthentication=no \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -o ConnectTimeout=60 \
    -o ControlMaster=auto \
    -o "ControlPath=$CW_CTL" \
    -o ControlPersist=yes \
    -A \
    -J "$CW_BASTION"
}

cw_master_alive() {
  ssh -o "ControlPath=$CW_CTL" -O check "$CW_LOGIN" >/dev/null 2>&1
}

# The bastion runs AuthenticationMethods publickey,keyboard-interactive: the
# certificate only gets partial success, and the Duo half needs a real TTY. So
# the master has to be opened from a terminal, once. Everything afterwards rides
# the socket and never re-authenticates.
cw_master_cmd() {
  cat <<EOF
SSH_AUTH_SOCK=${SSH_AUTH_SOCK:-\$XDG_RUNTIME_DIR/ssh-agent.socket} \\
ssh -F /etc/ssh/ssh_config \\
    -o StrictHostKeyChecking=accept-new \\
    -o ControlMaster=yes \\
    -o ControlPath=$CW_CTL \\
    -o ControlPersist=yes \\
    -A -J $CW_BASTION \\
    -N -f $CW_LOGIN
EOF
}

# Open the multiplexed master. The SSH certificate behind it comes from
# `cloud hpc login`, which needs a Duo approval that no script can supply.
cw_connect() {
  if cw_master_alive; then
    log "control master already up ($CW_CTL)"
    return 0
  fi
  cw_ensure_agent
  local opts=()
  mapfile -t opts < <(cw_ssh_opts)
  if ssh "${opts[@]}" -o BatchMode=yes -N -f "$CW_LOGIN" 2>/dev/null &&
     cw_master_alive; then
    log "control master established ($CW_CTL)"
    return 0
  fi
  cat >&2 <<EOF

Cannot open the control master for $CW_CLUSTER non-interactively.

The bastion needs publickey AND keyboard-interactive (Duo), and the Duo half
needs a real terminal. Do this once, in a terminal on this host, NOT through an
agent and NOT via Claude Code's '!' prefix, which has no TTY:

1. Mint the certificate if 'ssh-add -l' shows none:

     SSH_AUTH_SOCK=${SSH_AUTH_SOCK:-\$XDG_RUNTIME_DIR/ssh-agent.socket} cloud hpc login $CW_CLUSTER

   It will fail to open a shell if run without a TTY, which is fine; the
   certificate is minted into the ssh-agent either way.

2. Open the master and answer the Duo prompt. It backgrounds itself:

$(cw_master_cmd | sed 's/^/     /')

Then re-run: $0 connect
EOF
  return 1
}

cw_disconnect() {
  cw_master_alive || { log "no control master to close"; return 0; }
  ssh -o "ControlPath=$CW_CTL" -O exit "$CW_LOGIN" >/dev/null 2>&1 || true
  log "control master closed"
}

cw_require_master() {
  cw_master_alive || cw_connect || die "cannot reach $CW_CLUSTER"
}

# Run a command on the login node. Arguments are joined into one remote shell
# command; quote them the way you would for `bash -c`.
cw_ssh() {
  local opts=()
  mapfile -t opts < <(cw_ssh_opts)
  ssh "${opts[@]}" "$CW_LOGIN" "$@"
}

# The login shell is zsh with noclobber set, so `>` on an existing file fails
# silently. Force bash for anything scripted.
cw_bash() {
  cw_ssh "bash -lc $(printf '%q' "$*")"
}

# Deletions on the receiver are never silent: anything --delete removes is
# reported, so a surprise is visible in the log instead of discovered later.
cw_rsync() {
  local src="$1" dst="$2"
  shift 2
  local opts=() filters=() e
  mapfile -t opts < <(cw_ssh_opts)
  for e in "${CW_RSYNC_PROTECT[@]}"; do filters+=(--filter "P $e"); done
  for e in "${CW_RSYNC_EXCLUDES[@]}"; do filters+=(--exclude "$e"); done

  local out
  out="$(rsync -az --delete --itemize-changes --info=stats1 \
    -e "ssh $(printf '%s ' "${opts[@]}")" \
    "${filters[@]}" "$@" \
    "$src" "$CW_LOGIN:$dst")"

  local deleted
  deleted="$(grep -c '^\*deleting' <<<"$out" || true)"
  if [ "$deleted" -gt 0 ]; then
    warn "$deleted file(s) deleted under $dst:"
    grep '^\*deleting' <<<"$out" | head -20 | sed 's/^/    /' >&2
    [ "$deleted" -gt 20 ] && warn "    ... and $((deleted - 20)) more"
  fi
  grep -E '^(sent|total size)' <<<"$out" || true
}

cw_rsync_from() {
  local src="$1" dst="$2"
  shift 2
  local opts=()
  mapfile -t opts < <(cw_ssh_opts)
  rsync -az --info=stats1 \
    -e "ssh $(printf '%s ' "${opts[@]}")" \
    "$@" "$CW_LOGIN:$src" "$dst"
}

cw_push_file() {
  local src="$1" dst="$2"
  local opts=()
  mapfile -t opts < <(cw_ssh_opts)
  rsync -az -e "ssh $(printf '%s ' "${opts[@]}")" "$src" "$CW_LOGIN:$dst"
}

# rsync 3.2.4 and later stop the remote shell from expanding path arguments, so
# resolve $HOME once here and substitute it into the configured remote paths.
cw_remote_home() {
  if [ -z "${_CW_REMOTE_HOME:-}" ]; then
    _CW_REMOTE_HOME="$(cw_ssh 'echo $HOME' | tr -d '\r')"
    [ -n "$_CW_REMOTE_HOME" ] || die "could not resolve remote \$HOME"
  fi
  printf '%s' "$_CW_REMOTE_HOME"
}

cw_expand() {
  local path="$1" home
  # The patterns below are literals to match, not paths to expand locally.
  # shellcheck disable=SC2088
  case "$path" in
    '$HOME'*|'${HOME}'*|'~/'*)
      home="$(cw_remote_home)"
      path="${path/#\$HOME/$home}"
      path="${path/#\$\{HOME\}/$home}"
      path="${path/#\~/$home}"
      ;;
  esac
  printf '%s' "$path"
}

# Expand every configured remote path in place. Call once, after the master is up.
cw_resolve_paths() {
  CW_REMOTE_ROOT="$(cw_expand "$CW_REMOTE_ROOT")"
  CW_VENV="$(cw_expand "$CW_VENV")"
  CW_PYTHON_OVERLAY="$(cw_expand "$CW_PYTHON_OVERLAY")"
  CW_LOG_DIR="$(cw_expand "$CW_LOG_DIR")"
  CW_OUTPUT_DIR="$(cw_expand "$CW_OUTPUT_DIR")"
  CW_CONTAINER_DIR="$(cw_expand "$CW_CONTAINER_DIR")"
  CW_BRIDGE_REPO="$(cw_expand "$CW_BRIDGE_REPO")"
}

# Space-separated remote PYTHONPATH entries. Immutable runtimes already contain
# their matching compiled build trees, so exposing a synced source-only copy
# would shadow native extensions such as torchao._C_mxfp8.
cw_remote_pythonpath() {
  local entry name parts=()
  if [[ "$CW_VENV" != "$CW_REMOTE_ROOT"/runtimes/*/conda ]]; then
    for entry in "${CW_BUILD_TREES[@]}"; do
      name="${entry##*:}"
      parts+=("$CW_REMOTE_ROOT/$name")
    done
  fi
  for entry in "${CW_PYTHONPATH_TREES[@]}"; do
    name="${entry##*:}"
    parts+=("$CW_REMOTE_ROOT/$name")
  done
  parts+=("$CW_PYTHON_OVERLAY")
  local IFS=:
  printf '%s' "${parts[*]}"
}
