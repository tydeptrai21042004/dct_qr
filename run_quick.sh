#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python scripts/run_unified_benchmark.py --protocol configs/benchmark/quick.json
