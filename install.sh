#!/usr/bin/env bash
#
# Install `ysu` (yale-slurm-utils) so it is available from anywhere on the
# cluster — every login and compute node, from any directory — without having
# to activate a virtual environment first.
#
# It uses `uv tool install`, which builds the package into its own isolated
# environment and drops the `ysu` / `yale-slurm` launchers into ~/.local/bin.
# On an HPC your home directory is shared across nodes, so installing once
# makes `ysu` work everywhere.
#
# Usage:
#   ./install.sh            # install / reinstall from this checkout
#   ./install.sh --upgrade  # reinstall after pulling new changes
#
set -euo pipefail

# Resolve the directory this script lives in, so it works no matter where it is
# invoked from.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v uv >/dev/null 2>&1; then
  echo "error: 'uv' was not found on your PATH." >&2
  echo "Install it first, e.g.:  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

echo "Installing ysu from: $REPO_DIR"
# --force makes this safe to re-run: it replaces any previous (or broken)
# install rather than erroring out.
# --reinstall --refresh force a fresh build from the current source. Without
# them, uv can reuse a cached wheel for the same version number and silently
# install stale code when you've edited the source without bumping the version.
uv tool install --force --reinstall --refresh "$REPO_DIR"

# Make sure ~/.local/bin is on PATH in your shell profile. This is a no-op if
# it is already configured.
uv tool update-shell || true

echo
echo "Done. 'ysu' is installed in ~/.local/bin."
if command -v ysu >/dev/null 2>&1; then
  echo "It is already on your PATH — try:  ysu free"
else
  echo "Open a new shell (or 'source ~/.bashrc') and then try:  ysu free"
fi
