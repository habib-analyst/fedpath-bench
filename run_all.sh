#!/usr/bin/env bash
# fedpath-bench pilot: tests, then the preregistered run.
# Data must be extracted first (see README):
#   python scripts/extract_subset.py --data-dir /path/to/pcam/h5 --out-dir data_cache
set -euo pipefail
cd "$(dirname "$0")"
python -m pytest tests/ -q
python scripts/run_pilot.py --cache-dir data_cache --out-dir results
