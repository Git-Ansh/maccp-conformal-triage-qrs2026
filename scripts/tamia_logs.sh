#!/bin/bash
# View SLURM job output logs on TamIA
# Usage: ./scripts/tamia_logs.sh <job_id>
#        ./scripts/tamia_logs.sh          (shows latest)
set -e

REMOTE_PROJ='~/links/projects/aip-rnishat/shared/perf-regression-ci'

if [ -z "$1" ]; then
    echo "=== Latest job outputs ==="
    ssh tamia "ls -lt $REMOTE_PROJ/jobs/*.out 2>/dev/null | head -5"
    echo ""
    echo "=== Latest output ==="
    ssh tamia "ls -t $REMOTE_PROJ/jobs/*.out 2>/dev/null | head -1 | xargs tail -50"
else
    echo "=== Job $1 output ==="
    ssh tamia "cat $REMOTE_PROJ/jobs/*_$1.out 2>/dev/null"
    echo ""
    echo "=== Job $1 errors ==="
    ssh tamia "cat $REMOTE_PROJ/jobs/*_$1.err 2>/dev/null"
fi
