#!/bin/zsh
set -e
cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-semantic.txt
corepack pnpm install
corepack pnpm build
SPLAT_PYTHON="$PWD/.venv/bin/python" corepack pnpm desktop
