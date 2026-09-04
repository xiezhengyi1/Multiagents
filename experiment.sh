#!/usr/bin/env bash
set -euo pipefail

uv run python -m experiments.scripts.run_experiment "$@"
