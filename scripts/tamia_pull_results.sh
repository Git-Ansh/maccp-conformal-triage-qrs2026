#!/bin/bash
# Pull experiment results from TamIA back to local
# Usage: ./scripts/tamia_pull_results.sh
set -e

REMOTE_PROJ='~/links/projects/aip-rnishat/shared/perf-regression-ci'

echo "=== Pulling conformal_outputs ==="
rsync -av tamia:$REMOTE_PROJ/conformal_outputs/ ./conformal_outputs/ 2>/dev/null || \
rsync -av tamia:'$SCRATCH/perf-regression-ci-outputs/' ./conformal_outputs/

echo "=== Pulling cascade_outputs ==="
rsync -av tamia:$REMOTE_PROJ/cascade_outputs/ ./cascade_outputs/ 2>/dev/null || \
rsync -av tamia:'$SCRATCH/perf-regression-ci-outputs/mozilla/' ./cascade_outputs/

echo "=== Done ==="
