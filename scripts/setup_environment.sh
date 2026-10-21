#!/usr/bin/env bash
# Install the research workflow without running training or paid inference.
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
venv_dir=${JEV_VENV_DIR:-"$repo_dir/.venv"}
python3 -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 is required"'
if [[ ! -x "$venv_dir/bin/python" ]]; then
    python3 -m venv "$venv_dir"
fi
source "$venv_dir/bin/activate"
config_dir="$repo_dir/artifacts/environment"
mkdir -p "$config_dir/cache" "$config_dir/matplotlib" "$config_dir/ipython" "$config_dir/jupyter" "$config_dir/runtime"
chmod 700 "$config_dir/runtime"
export PIP_CACHE_DIR="$config_dir/cache/pip"
export UV_CACHE_DIR="$config_dir/cache/uv"
export MPLCONFIGDIR="$config_dir/matplotlib"
export IPYTHONDIR="$config_dir/ipython"
export JUPYTER_CONFIG_DIR="$config_dir/jupyter"
export JUPYTER_RUNTIME_DIR="$config_dir/runtime"
if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$venv_dir/bin/python" -e "${repo_dir}[notebook]"
else
    python -m pip install -e "${repo_dir}[notebook]"
fi
python -m pip check
cd "$repo_dir"
python -m jev_research.cli --help >/dev/null
printf '%s\n' 'Research dependencies installed; no training or paid inference was run.'
