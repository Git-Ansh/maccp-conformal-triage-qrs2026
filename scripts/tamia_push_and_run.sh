#!/bin/bash
# Push code to GitHub and submit a SLURM job on TamIA
# Usage: ./scripts/tamia_push_and_run.sh jobs/run_eclipse_s2.sh
set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <job_script>"
    echo "Example: $0 jobs/run_eclipse_s2.sh"
    exit 1
fi

REMOTE_PROJ='~/links/projects/aip-rnishat/shared/perf-regression-ci'

echo "=== Pushing to GitHub ==="
git push origin main

echo "=== Pulling on TamIA ==="
ssh tamia "cd $REMOTE_PROJ && git pull"

echo "=== Submitting job: $1 ==="
ssh tamia "cd $REMOTE_PROJ && sbatch $1"

echo "=== Current queue ==="
ssh tamia "squeue -u anshshah --format='%.10i %.30j %.8T %.10M %.6D %R'"
